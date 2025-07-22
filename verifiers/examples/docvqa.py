from datasets import load_dataset
import verifiers as vf
from openai import AsyncOpenAI
import os
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
DocVQA evaluation example for both API and local models.

For API evaluation:
    export OPENAI_API_KEY="your-api-key"
    uv run python docvqa.py

For local model training (1+1 GPUs):
    # Install qwen stuff
    uv pip install qwen-vl-utils
    # Inference
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen2.5-VL-3B-Instruct' --max-model-len 32768
    # Train
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --config-file configs/single_gpu.yaml verifiers/examples/docvqa.py
"""

# Data collator for API models
def api_data_collator(batch: list[dict]) -> list[dict]:
    """Format data for API models - images are passed separately."""
    processed_samples = []
    for sample in batch:
        # For API models, we need to separate text prompts and images
        # The environment's format_oai_chat_msg will handle image conversion
        
        # Create text-only prompt with placeholder for image
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image"}  # Placeholder - actual image handled by format_oai_chat_msg
            ]}
        ]
        
        sample["prompt"] = messages
        sample["images"] = [sample["image"]]  # Single list - one image per message
        sample["answer"] = sample["answers"]  # Keep as list for scoring
        processed_samples.append(sample)
    return processed_samples

# Data collator for local Qwen models
def qwen_data_collator(batch: list[dict]) -> list[dict]:
    """Format data for Qwen VLM."""
    from qwen_vl_utils import process_vision_info
    
    processed_samples = []
    for sample in batch:
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        content_block = []
        content_block.append({"type": "text", "text": sample["question"]})
        content_block.append(
            {
                "type": "image",
                "image": sample["image"],
                "resized_height": 384,  # Reduce resolution for memory
                "resized_width": 512,
            }
        )
        messages.append({"role": "user", "content": content_block})
        processed_images, *_ = process_vision_info(messages.copy())
        
        sample["prompt"] = messages
        sample["images"] = processed_images
        sample["answer"] = sample["answers"]
        processed_samples.append(sample)
    return processed_samples


dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[10%:]")
eval_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[:20]")  # Only use 20 examples for evaluation

parser = vf.XMLParser(["think", "answer"], answer_field="answer")
system_prompt = f"""Answer the questions.

Respond in the following format:
{parser.get_format_str()}"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Custom judge prompt for that returns a score
judge_prompt_score = """You are an expert judge evaluating answers.

Question:
```
{question}
```

Ground Truth Answers:
```
{answer}
```

Predicted Answer:
```
{response}
```

Score the predicted answer on a scale of 0-1:
- 1.0: Perfect match or equivalent meaning
- 0.7-0.9: Correct with minor differences  
- 0.4-0.6: Partially correct
- 0.1-0.3: Wrong but related
- 0.0: Completely wrong

Respond with ONLY a number between 0 and 1, nothing else."""

# Initialize OpenAI client for judge
from openai import OpenAI

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for judge evaluation")

judge_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("Using OpenAI judge for evaluation")

# Create JudgeRubric with numeric scoring  
judge_rubric = vf.JudgeRubric(
    parser=parser,
    judge_client=judge_client,
    judge_model="gpt-4.1-nano",
    judge_prompt=judge_prompt_score,
    judge_sampling_args={"temperature": 0.0, "max_tokens": 10}
)

def docvqa_judge_score(completion, **kwargs):
    # Get the judge's response using the judge method
    state = {}
    prompt = kwargs.get("prompt", "")
    answer = kwargs.get("answer", [])
    
    # Format answer for judge
    if isinstance(answer, list):
        answer_str = ", ".join(answer)
    else:
        answer_str = str(answer)
    
    logger.info(f"Ground truth answers: {answer_str}")
    
    # Try to extract the model's answer
    try:
        parsed_answer = parser.parse_answer(completion)
        logger.info(f"Parser extracted answer: '{parsed_answer}'")
    except Exception as e:
        logger.error(f"Parser failed: {e}")
        parsed_answer = None
    
    # If parser failed, try manual extraction
    if not parsed_answer:
        logger.warning("Parser returned empty answer, trying manual extraction")
        for msg in completion:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Try to extract answer from XML tags (handle missing closing tag)
                # First try with closing tag
                match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                if not match:
                    # Try without closing tag
                    match = re.search(r"<answer>(.*)$", content, re.DOTALL | re.MULTILINE)
                if match:
                    parsed_answer = match.group(1).strip()
                    logger.info(f"Manually extracted answer from XML: '{parsed_answer}'")
                else:
                    # No XML format at all - try to extract the actual answer
                    logger.warning("No XML format found, attempting smart extraction")
                    
                    # Remove any <think> content first
                    content_no_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                    content_no_think = content_no_think.strip()
                    
                    # Common patterns for answers without XML
                    patterns = [
                        r"(?:The answer is|Answer:|A:)\s*(.+?)(?:\.|$)",  # "The answer is X" or "Answer: X"
                        r"(?:It is|It's|They are|These are)\s*(.+?)(?:\.|$)",  # "It is X" or "They are X"
                        r"^([^.!?]+)(?:[.!?]|$)",  # First sentence if nothing else matches
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, content_no_think, re.IGNORECASE | re.MULTILINE)
                        if match:
                            parsed_answer = match.group(1).strip()
                            logger.info(f"Smart extraction found: '{parsed_answer}'")
                            break
                    
                    # Last resort - if content is short and no patterns match, use the whole thing
                    if not parsed_answer and content_no_think and len(content_no_think) < 100:
                        parsed_answer = content_no_think
                        logger.info(f"Using entire response as answer: '{parsed_answer}'")
                break
    
    # Check if we have an answer to judge
    if not parsed_answer:
        logger.error("No answer found to judge")
        return 0.0
    
    # Call the judge method with the parsed answer
    # Update the judge prompt to use the parsed answer
    judge_prompt = judge_prompt_score.format(
        question=prompt[-1]["content"][0]["text"] if isinstance(prompt, list) else prompt,
        answer=answer_str,
        response=parsed_answer
    )
    
    logger.debug(f"Judge prompt: {judge_prompt[:200]}...")
    
    judge_response = judge_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=10
    )
    
    judge_result = judge_response.choices[0].message.content.strip() # type: ignore
    logger.info(f"Judge response: {judge_result}")
    
    try:
        # Extract numeric score
        score = float(judge_result)
        logger.info(f"Parsed score: {score}")
        return score
    except ValueError:
        logger.error(f"Could not parse judge response as float: {judge_result}")
        # Fallback to binary scoring
        if "yes" in judge_result.lower() or "1" in judge_result:
            return 1.0
        else:
            return 0.0

# Combine format checking and judge scoring
rubric = vf.Rubric(
    funcs=[
        parser.get_format_reward_func(),
        docvqa_judge_score,
    ]
)

# Create environment for training or evaluation
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    data_collator=qwen_data_collator,
    rollouts_per_sample=4,  # Must match num_generations
)

# Evaluation example using API models
# api_key = 'token-abc123' # vllm
# client = AsyncOpenAI(api_key=api_key, base_url="http://localhost:8000/v1")

# model = "Qwen/Qwen2.5-VL-7B-Instruct"

# # Sampling parameters
# sampling_args = {
#     "temperature": 0.0, 
#     "max_tokens": 500,
# }

# # Run evaluation
# print(f"\nEvaluating {model} on DocVQA...")
# results = vf_env.evaluate(
#     client=client,
#     model=model,
#     sampling_args=sampling_args,
#     num_examples=10,  # Evaluate on 10 examples
#     rollouts_per_example=1,
# )

# # Analyze results
# print("\nEvaluation Results:")
# print("-" * 50)

# # Extract scores
# format_scores = results.get("format_reward_func", [])
# correctness_scores = results.get("docvqa_judge_score", [])
# total_rewards = results.get("reward", [])

# # Calculate statistics
# def calculate_stats(scores):
#     if not scores:
#         return {"mean": 0, "min": 0, "max": 0}
#     return {
#         "mean": sum(scores) / len(scores),
#         "min": min(scores),
#         "max": max(scores),
#     }

# format_stats = calculate_stats(format_scores)
# correctness_stats = calculate_stats(correctness_scores)
# total_stats = calculate_stats(total_rewards)

# print(f"Format Score - Mean: {format_stats['mean']:.3f}, Min: {format_stats['min']:.3f}, Max: {format_stats['max']:.3f}")
# print(f"Correctness Score - Mean: {correctness_stats['mean']:.3f}, Min: {correctness_stats['min']:.3f}, Max: {correctness_stats['max']:.3f}")
# print(f"Total Reward - Mean: {total_stats['mean']:.3f}, Min: {total_stats['min']:.3f}, Max: {total_stats['max']:.3f}")

# # Show some example completions
# print("\nExample Completions:")
# print("-" * 50)

# for i in range(len(results["completion"])):
#     print(f"\nExample {i+1}:")
    
#     # Extract question from prompt
#     prompt = results['prompt'][i]
#     if isinstance(prompt, list) and len(prompt) > 0:
#         # Find the user message with the question
#         for msg in prompt:
#             if msg.get('role') == 'user':
#                 content = msg.get('content', [])
#                 for item in content:
#                     if item.get('type') == 'text':
#                         print(f"Question: {item.get('text', '')}")
#                         break
#                 break
    
#     # Extract ground truth
#     answer = results['answer'][i]
#     print(f"Ground Truth: {answer}")
    
#     # Extract assistant response
#     completion = results['completion'][i]
#     assistant_msg = next((msg for msg in completion if msg['role'] == 'assistant'), None)
#     if assistant_msg:
#         print(f"Model Response: {assistant_msg['content']}")
    
#     print(f"Correctness Score: {correctness_scores[i]:.3f}")
#     print(f"Total Reward: {total_rewards[i]:.3f}")


model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model, processor = vf.get_model_and_tokenizer(model_name)
run_name = "docvqa_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.learning_rate = 3e-6
training_args.max_steps = 100  # Limit steps for testing
training_args.eval_strategy = "steps"
training_args.eval_steps = 10  # Evaluate every 10 steps
training_args.gradient_checkpointing_kwargs = {
    "use_reentrant": False,
}

# GRPO specific settings
training_args.num_generations = 4  # Number of generations per prompt
training_args.per_device_train_batch_size = 4  # Increased from 2
training_args.gradient_accumulation_steps = 2  # Effective batch size = 8

# Memory optimization settings
training_args.fp16 = True  # Use mixed precision training
training_args.optim = "adamw_8bit"  # Use 8-bit optimizer
training_args.gradient_checkpointing = True  # Enable gradient checkpointing

# Generation settings for memory
training_args.temperature = 0.7
training_args.max_new_tokens = 200  # Limit generation length

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=processor,
    env=vf_env,
    args=training_args,
)
trainer.train()