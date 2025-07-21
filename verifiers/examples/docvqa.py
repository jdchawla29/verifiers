import re
from datasets import load_dataset
import verifiers as vf

"""
DocVQA evaluation example for both API and local models.

For API evaluation:
    export OPENAI_API_KEY="your-api-key"
    uv run python docvqa.py

For local model training:
    # Install qwen stuff
    uv pip install qwen-vl-utils
    # Inference
    CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-VL-7B-Instruct' --max-model-len 32000 --tensor_parallel_size 4 
    # Train
    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/docvqa.py
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
    """Format data for local Qwen VLM training."""
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
                "resized_height": 768,  # XGA resolution
                "resized_width": 1024,
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
eval_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[:10%]")

parser = vf.XMLParser(["think", "answer"], answer_field="answer")
system_prompt = f"""Answer the questions.

Respond in the following format:
{parser.get_format_str()}"""


def correctness_reward_func(completion: list[dict[str, str]], **kwargs) -> float:
    def get_assistant_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [msg for msg in messages if msg.get("role") == "assistant"]

    def parse_xml_content(text: str, tag: str, strip: bool = True) -> str | None:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            return content.strip() if strip else content
        return None

    assistant_messages = get_assistant_messages(completion)
    if assistant_messages is None:
        return 0.0
    msgs_scores = []
    for msg in assistant_messages:
        content = msg.get("content", "")
        answer = parse_xml_content(content, "answer")
        if answer is None:
            continue
        gt_answers = kwargs["answer"]
        mean_gt_len = sum([len(gt_answer) for gt_answer in gt_answers]) / len(
            gt_answers
        )
        if len(answer) > 0:
            diff_from_mean = min(mean_gt_len / len(answer), 1.0)  # penalize long answers
        else:
            diff_from_mean = 0.0
        if answer in gt_answers:
            msgs_scores.append(2.0)
        elif answer.lower() in [ans.lower() for ans in gt_answers]:
            msgs_scores.append(1.0)
        elif any(ans.lower() in answer.lower() for ans in gt_answers):
            msgs_scores.append(diff_from_mean)
    if msgs_scores == []:
        return 0.0
    else:
        return sum(msgs_scores) / len(msgs_scores) / 2.0


rubric = vf.Rubric(
    funcs=[
        parser.get_format_reward_func(),
        correctness_reward_func,
    ]
)

# Create environment for API evaluation
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    data_collator=api_data_collator,
)

# Evaluation example using API models
from openai import AsyncOpenAI
client = AsyncOpenAI()  # Uses OPENAI_API_KEY from environment
model = "gpt-4o"

# Sampling parameters
sampling_args = {
    "temperature": 0.0, 
    "max_tokens": 500,
}

# Run evaluation
print(f"\nEvaluating {model} on DocVQA...")
results = vf_env.evaluate(
    client=client,
    model=model,
    sampling_args=sampling_args,
    num_examples=10,  # Evaluate on 10 examples
    rollouts_per_example=1,
)

# Analyze results
print("\nEvaluation Results:")
print("-" * 50)

# Extract scores
format_scores = results.get("format_reward_func", [])
correctness_scores = results.get("correctness_reward_func", [])
total_rewards = results.get("reward", [])

# Calculate statistics
def calculate_stats(scores):
    if not scores:
        return {"mean": 0, "min": 0, "max": 0}
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }

format_stats = calculate_stats(format_scores)
correctness_stats = calculate_stats(correctness_scores)
total_stats = calculate_stats(total_rewards)

print(f"Format Score - Mean: {format_stats['mean']:.3f}, Min: {format_stats['min']:.3f}, Max: {format_stats['max']:.3f}")
print(f"Correctness Score - Mean: {correctness_stats['mean']:.3f}, Min: {correctness_stats['min']:.3f}, Max: {correctness_stats['max']:.3f}")
print(f"Total Reward - Mean: {total_stats['mean']:.3f}, Min: {total_stats['min']:.3f}, Max: {total_stats['max']:.3f}")

# Show some example completions
print("\nExample Completions:")
print("-" * 50)

for i in range(min(3, len(results["completion"]))):
    print(f"\nExample {i+1}:")
    
    # Extract question from prompt
    prompt = results['prompt'][i]
    if isinstance(prompt, list) and len(prompt) > 0:
        # Find the user message with the question
        for msg in prompt:
            if msg.get('role') == 'user':
                content = msg.get('content', [])
                for item in content:
                    if item.get('type') == 'text':
                        print(f"Question: {item.get('text', '')}")
                        break
                break
    
    # Extract ground truth
    answer = results['answer'][i]
    print(f"Ground Truth: {answer}")
    
    # Extract assistant response
    completion = results['completion'][i]
    assistant_msg = next((msg for msg in completion if msg['role'] == 'assistant'), None)
    if assistant_msg:
        print(f"Model Response: {assistant_msg['content']}")
    
    print(f"Correctness Score: {correctness_scores[i]:.3f}")
    print(f"Total Reward: {total_rewards[i]:.3f}")


# model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
# model, processor = vf.get_model_and_tokenizer(model_name)
# run_name = "docvqa_" + model_name.split("/")[-1].lower()

# training_args = vf.grpo_defaults(run_name=run_name)
# training_args.learning_rate = 3e-6
# training_args.max_steps = -1
# training_args.eval_strategy = "steps"
# training_args.eval_steps = 100
# training_args.gradient_checkpointing_kwargs = {
#     "use_reentrant": False,
# }

# trainer = vf.GRPOTrainer(
#     model=model,
#     processing_class=processor,
#     env=vf_env,
#     args=training_args,
# )
# trainer.train()