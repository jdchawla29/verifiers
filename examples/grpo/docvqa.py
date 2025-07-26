from datasets import load_dataset
import verifiers as vf
from openai import OpenAI, AsyncOpenAI
import os
import logging

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
judge_prompt = """You are an expert judge evaluating answers.

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
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required for judge evaluation")

judge_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("Using OpenAI judge for evaluation")

# Custom JudgeRubric that returns numeric scores
class NumericJudgeRubric(vf.JudgeRubric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add the scoring function to the rubric
        self.add_reward_func(self.numeric_judge_score)
    
    def numeric_judge_score(self, completion, answer, state, **kwargs):
        """Scoring function that calls judge and converts to float."""
        # Call the parent judge method
        prompt = kwargs.get("prompt", "")
        judge_response = self.judge(
            prompt,
            completion,
            answer,
            state
        )
        
        try:
            # Extract numeric score
            score = float(judge_response.strip())
            logger.info(f"Parsed score: {score}")
            return score
        except ValueError:
            logger.error(f"Could not parse judge response as float: {judge_response}")
            # Fallback to binary scoring
            if "yes" in judge_response.lower() or "1" in judge_response:
                return 1.0
            else:
                return 0.0

# Create the numeric judge rubric
judge_rubric = NumericJudgeRubric(
    parser=parser,
    judge_client=judge_client,
    judge_model="gpt-4.1-nano",
    judge_prompt=judge_prompt,
    judge_sampling_args={"temperature": 0.0, "max_tokens": 10}
)

# Combine format checking and judge scoring
rubric = vf.RubricGroup(rubrics=[
    vf.Rubric(funcs=[parser.get_format_reward_func()]),
    judge_rubric
])

# Configuration flags
MODE = "eval"  # "train" or "eval"
API_TYPE = "openai"  # "openai" or "vllm" (only used when MODE="eval")

client = None  # Will be initialized based on API_TYPE

# Model and client configuration
if MODE == "eval":
    if API_TYPE == "vllm":
        # vLLM API configuration
        client = AsyncOpenAI(
            base_url="http://localhost:8000/v1"
        )
        models = client.models.list()
        if not models:
            raise ValueError("No models found in vLLM server. Make sure the server is running and models are loaded.")
        MODEL_NAME = models[0].id  # type: ignore
        logger.info(f"Using vLLM model: {MODEL_NAME}")
    else:
        MODEL_NAME = "gpt-4.1-mini"
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    # Training uses local model
    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Evaluation settings
EVAL_NUM_EXAMPLES = 10
EVAL_ROLLOUTS_PER_EXAMPLE = 1

# Training settings
TRAIN_MAX_STEPS = 100
TRAIN_EVAL_STEPS = 10
TRAIN_BATCH_SIZE = 4
TRAIN_GRADIENT_ACCUMULATION_STEPS = 2
TRAIN_NUM_GENERATIONS = 4

# Select appropriate data collator
data_collator = api_data_collator if MODE == "eval" else qwen_data_collator

# Create environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    data_collator=data_collator,
    rollouts_per_sample=TRAIN_NUM_GENERATIONS if MODE == "train" else EVAL_ROLLOUTS_PER_EXAMPLE,
)

if MODE == "eval":
    logger.info(f"Running evaluation with {API_TYPE.upper()} API model: {MODEL_NAME}")
    
    sampling_args = {
        "temperature": 0.0,
        "max_tokens": 500,
    }
    results = vf_env.evaluate(
        client=client,  # type: ignore
        model=MODEL_NAME,
        sampling_args=sampling_args,
        num_examples=EVAL_NUM_EXAMPLES,
        rollouts_per_example=EVAL_ROLLOUTS_PER_EXAMPLE,
    )
    
    # Analyze results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    
    # Extract scores
    format_scores = results.get("format_reward_func", [])
    correctness_scores = results.get("numeric_judge_score", [])
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
    
    logger.info(f"Format Score - Mean: {format_stats['mean']:.3f}, Min: {format_stats['min']:.3f}, Max: {format_stats['max']:.3f}")
    logger.info(f"Correctness Score - Mean: {correctness_stats['mean']:.3f}, Min: {correctness_stats['min']:.3f}, Max: {correctness_stats['max']:.3f}")
    logger.info(f"Total Reward - Mean: {total_stats['mean']:.3f}, Min: {total_stats['min']:.3f}, Max: {total_stats['max']:.3f}")
    
    # Show some example completions
    logger.info("\nExample Completions:")
    logger.info("-" * 50)
    
    for i in range(min(3, len(results["completion"]))):  # Show first 3 examples
        logger.info(f"\nExample {i+1}:")
        
        # Extract question from prompt
        prompt = results['prompt'][i]
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in prompt:
                if msg.get('role') == 'user':
                    content = msg.get('content', [])
                    for item in content:
                        if item.get('type') == 'text':
                            logger.info(f"Question: {item.get('text', '')}")
                            break
                    break
        
        # Extract ground truth
        answer = results['answer'][i]
        logger.info(f"Ground Truth: {answer}")
        
        # Extract assistant response
        completion = results['completion'][i]
        assistant_msg = next((msg for msg in completion if msg['role'] == 'assistant'), None)
        if assistant_msg:
            logger.info(f"Model Response: {assistant_msg['content'][:200]}...")  # Truncate long responses
        
        if i < len(correctness_scores):
            logger.info(f"Correctness Score: {correctness_scores[i]:.3f}")
        if i < len(total_rewards):
            logger.info(f"Total Reward: {total_rewards[i]:.3f}")

elif MODE == "train":
    logger.info(f"Running training with local model: {MODEL_NAME}")
    
    model, processor = vf.get_model_and_tokenizer(MODEL_NAME)
    run_name = "docvqa_" + MODEL_NAME.split("/")[-1].lower()
    
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.learning_rate = 3e-6
    training_args.max_steps = TRAIN_MAX_STEPS
    training_args.eval_strategy = "steps"
    training_args.eval_steps = TRAIN_EVAL_STEPS
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False,
    }
    
    # GRPO specific settings
    training_args.num_generations = TRAIN_NUM_GENERATIONS
    training_args.per_device_train_batch_size = TRAIN_BATCH_SIZE
    training_args.gradient_accumulation_steps = TRAIN_GRADIENT_ACCUMULATION_STEPS
    
    # Memory optimization settings
    training_args.fp16 = True
    training_args.optim = "adamw_8bit"
    training_args.gradient_checkpointing = True
    
    # Generation settings for memory
    training_args.temperature = 0.7
    training_args.max_new_tokens = 200
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=processor,
        env=vf_env,
        args=training_args,
    )
    trainer.train()
    
else:
    raise ValueError(f"Invalid MODE: {MODE}. Must be 'train' or 'eval'")