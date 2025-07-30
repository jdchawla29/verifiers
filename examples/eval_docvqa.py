"""
Example evaluation script for DocVQA environment.

For API evaluation:
    export OPENAI_API_KEY="your-api-key"
    uv run python examples/eval_docvqa.py

For local model evaluation:
    # Start vLLM server with a multimodal model
    CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen2-VL-2B-Instruct' --max-model-len 32768
    # Run evaluation
    uv run python examples/eval_docvqa.py
"""

import logging
import os
from openai import AsyncOpenAI
import verifiers as vf

# Import the environment
import sys
sys.path.append('environments/vf_docvqa')
from vf_docvqa import load_environment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configuration
    API_TYPE = "vllm"  # "openai" or "vllm"
    EVAL_NUM_EXAMPLES = 5  # Number of examples to evaluate
    
    # Load environment
    vf_env = load_environment(num_eval_examples=EVAL_NUM_EXAMPLES)
    
    # Set up client and model based on API type
    if API_TYPE == "vllm":
        # vLLM API configuration
        client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="token")
        MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"Using vLLM model: {MODEL_NAME}")
    else:
        # OpenAI API configuration
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        MODEL_NAME = "gpt-4o"  # Use GPT-4o for multimodal
        logger.info(f"Using OpenAI model: {MODEL_NAME}")
    
    # Sampling arguments
    sampling_args = {
        "temperature": 0.0,
        "max_tokens": 500,
    }
    
    # Run evaluation
    logger.info(f"Running evaluation on {EVAL_NUM_EXAMPLES} examples...")
    results = vf_env.evaluate(
        client=client,
        model=MODEL_NAME,
        sampling_args=sampling_args,
        num_examples=EVAL_NUM_EXAMPLES,
        rollouts_per_example=1,
    )
    
    # Analyze results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 50)
    
    # Extract scores
    answer_scores = results.metrics.get("answer_match_reward", [])
    format_scores = results.metrics.get("format_reward_func", [])
    total_rewards = results.reward
    
    # Calculate statistics
    def calculate_stats(scores):
        if not scores:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
        }
    
    answer_stats = calculate_stats(answer_scores)
    format_stats = calculate_stats(format_scores)
    total_stats = calculate_stats(total_rewards)
    
    logger.info(f"Answer Accuracy - Mean: {answer_stats['mean']:.3f}, Min: {answer_stats['min']:.3f}, Max: {answer_stats['max']:.3f}")
    logger.info(f"Format Score - Mean: {format_stats['mean']:.3f}, Min: {format_stats['min']:.3f}, Max: {format_stats['max']:.3f}")
    logger.info(f"Total Reward - Mean: {total_stats['mean']:.3f}, Min: {total_stats['min']:.3f}, Max: {total_stats['max']:.3f}")
    
    # Show some example completions
    logger.info("\nExample Completions:")
    logger.info("-" * 50)
    
    for i in range(min(3, len(results.completion))):
        logger.info(f"\nExample {i+1}:")
        
        # Extract question from prompt
        prompt = results.prompt[i]
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in prompt:
                if msg.get('role') == 'user':
                    content = msg.get('content', [])
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            logger.info(f"Question: {item.get('text', '')}")
                            break
                    break
        
        # Show ground truth and response
        answer = results.answer[i]
        logger.info(f"Acceptable Answers: {answer}")
        
        completion = results.completion[i]
        assistant_msg = next((msg for msg in completion if msg['role'] == 'assistant'), None)
        if assistant_msg:
            logger.info(f"Model Response: {assistant_msg['content'][:200]}...")
        
        if i < len(answer_scores):
            logger.info(f"Answer Score: {answer_scores[i]:.3f}")
        if i < len(total_rewards):
            logger.info(f"Total Reward: {total_rewards[i]:.3f}")


if __name__ == "__main__":
    main()