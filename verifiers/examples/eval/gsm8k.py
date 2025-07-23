import os
import logging
from openai import OpenAI

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer

logger = logging.getLogger(__name__)

dataset = load_example_dataset("gsm8k").select(range(100))

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0
rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

def main(num_examples: int, rollouts_per_example: int, max_tokens: int):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1-nano" 
    client = OpenAI(api_key=api_key)
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    results = vf_env.evaluate(
        client=client,
        model=model_name, 
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
    )
    
    logger.info("--- Example ---")
    logger.info("Prompt: %s", results['prompt'][0])
    logger.info("Completion: %s", results['completion'][0])
    logger.info("Answer: %s", results['answer'][0])
    logger.info("Reward: %s", results['reward'][0])
    logger.info("--- All ---")
    logger.info("Rewards:")
    for k, v in results.items():
        if 'reward' in k:
            logger.info('%s - %s', k, sum(v) / len(v))

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num-examples", "-n", type=int, default=-1)
    argparser.add_argument("--rollouts-per-example", "-r", type=int, default=1)
    argparser.add_argument("--max-tokens", "-t", type=int, default=2048)
    args = argparser.parse_args()
    main(args.num_examples, args.rollouts_per_example, args.max_tokens)