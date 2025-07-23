import os
import logging
from openai import OpenAI

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer

logger = logging.getLogger(__name__)

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0
rubric1 = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

dataset1 = load_example_dataset("gsm8k", split="train").select(range(10))
env1 = vf.SingleTurnEnv(
    dataset=dataset1,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric1,
)
def incorrect_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 0.0 if response == answer else 1.0
rubric2 = vf.Rubric(funcs=[
    incorrect_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

dataset2 = load_example_dataset("math", split="train").select(range(10))
env2 = vf.SingleTurnEnv(
    dataset=dataset2,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric2,
)

vf_env = vf.EnvGroup([env1, env2], env_names=["gsm8k", "math"])

def main(num_examples: int, rollouts_per_example: int, max_tokens: int):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1" 
    client = OpenAI(api_key=api_key)
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    results = vf_env.evaluate(
        client=client, model=model_name, 
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
    )
    logger.info("--- Example ---")
    logger.info("Prompt: %s", results['prompt'][0])
    logger.info("Completion: %s", results['completion'][0])
    logger.info("Answer: %s", results['answer'][0])
    logger.info("Task: %s", results['task'][0])
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