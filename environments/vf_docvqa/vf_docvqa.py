import verifiers as vf
from datasets import load_dataset


def load_environment(num_train_examples=-1, num_eval_examples=-1):
    """Load DocVQA environment for document visual question answering.

    This environment supports multimodal inputs (text + images) for document QA tasks.
    """
    # Load datasets
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[10%:]")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))

    eval_dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation[:10%]")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    # System prompt
    system_prompt = """Answer the questions about the document image.

Respond in the following format:
<think>
[Your reasoning here]
</think>
<answer>
[Your concise answer here]
</answer>"""

    # Parser
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")

    # Data collator for multimodal inputs
    def data_collator(batch):
        """Format data for multimodal models - images are passed separately.

        When used with dataset.map(batched=True), batch is a dict with lists of values.
        Returns a dict with the same structure.
        """
        prompts = []
        images = []
        answers = []

        for i in range(len(batch["question"])):
            # Create multimodal prompt with image placeholder
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": batch["question"][i]},
                        {
                            "type": "image"
                        },  # Placeholder - actual image handled by format_oai_chat_msg
                    ],
                },
            ]

            prompts.append(messages)
            images.append([batch["image"][i]])  # Single image per question

            # Convert list of answers to a single string for compatibility
            answer = batch["answers"][i]
            if isinstance(answer, list):
                answer = "|".join(answer)
            answers.append(answer)

        # Return updated batch dict
        result = dict(batch)  # Copy all existing columns
        result["prompt"] = prompts
        result["images"] = images
        result["answer"] = answers
        return result

    # Rubric with format checking and flexible answer matching
    def answer_match_reward(completion, answer, **kwargs):
        """Check if the response matches any of the acceptable answers."""
        response = parser.parse_answer(completion) or ""
        response = response.strip().lower()

        # Split answer by | to get list of acceptable answers
        acceptable_answers = answer.split("|") if isinstance(answer, str) else [answer]

        for acceptable in acceptable_answers:
            if (
                acceptable.strip().lower() in response
                or response in acceptable.strip().lower()
            ):
                return 1.0
        return 0.0

    rubric = vf.Rubric(
        funcs=[answer_match_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.1],  # Prioritize correctness over format
    )

    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
        data_collator=data_collator,
        # Don't set system_prompt here since it's included in data_collator
    )

    return vf_env
