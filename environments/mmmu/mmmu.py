import verifiers as vf
from datasets import load_dataset, concatenate_datasets


def load_environment(num_train_examples=-1, num_eval_examples=-1):
    """Load MMMU environment for multimodal multi-discipline reasoning.

    MMMU (Massive Multi-discipline Multimodal Understanding) is a benchmark
    covering multiple subjects requiring vision-language understanding.

    Uses MMMU's validation set (900 samples) for training and dev set (150 samples) for evaluation.
    """
    # All available subject configs in MMMU
    subjects = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]

    # Load and concatenate all subjects for validation set (training data)
    val_datasets = []
    for subject in subjects:
        val_ds = load_dataset("MMMU/MMMU", subject, split="validation")
        # Filter for multiple-choice questions only
        val_ds = val_ds.filter(lambda x: x["question_type"] == "multiple-choice")
        val_datasets.append(val_ds)
    dataset = concatenate_datasets(val_datasets)

    # Shuffle the combined dataset for better training
    dataset = dataset.shuffle(seed=42)

    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))

    # Load and concatenate all subjects for dev set (evaluation data)
    dev_datasets = []
    for subject in subjects:
        dev_ds = load_dataset("MMMU/MMMU", subject, split="dev")
        # Filter for multiple-choice questions only
        dev_ds = dev_ds.filter(lambda x: x["question_type"] == "multiple-choice")
        dev_datasets.append(dev_ds)
    eval_dataset = concatenate_datasets(dev_datasets)

    # Shuffle the combined eval dataset
    eval_dataset = eval_dataset.shuffle(seed=42)

    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    # System prompt for MMMU multiple-choice questions
    system_prompt = """You are solving multiple-choice questions from the MMMU benchmark.

You MUST respond using ONLY the following XML format:

<think>
Step-by-step reasoning about the question and images
</think>
<answer>
A
</answer>

Important:
- The <answer> tag must contain ONLY a single letter (A, B, C, D, etc.)
- Do NOT include any text outside these XML tags
- Do NOT explain your answer outside the tags"""

    # Parser
    parser = vf.XMLParser(["think", "answer"], answer_field="answer")

    # Data collator for multimodal inputs
    def data_collator(batch):
        """Format MMMU data for multimodal models.

        MMMU can have multiple images per question stored in image_1 through image_7.
        """
        import ast

        prompts = []
        images = []

        for i in range(len(batch["question"])):
            # Build the question with options
            question_text = batch["question"][i]
            options_str = batch["options"][i]

            # Parse options - they come as a string representation of a list
            if options_str and options_str != "[]":
                try:
                    options = ast.literal_eval(options_str)
                except Exception:
                    options = []
            else:
                options = []

            # Format options with letters (all questions are now multiple-choice)
            formatted_options = []
            for j, option in enumerate(options):
                letter = chr(65 + j)  # A, B, C, D...
                formatted_options.append(f"{letter}. {option}")

            full_question = f"{question_text}\n\nOptions:\n" + "\n".join(
                formatted_options
            )

            # Collect all images for this question
            question_images = []
            for img_num in range(1, 8):  # MMMU has image_1 through image_7
                img_key = f"image_{img_num}"
                if img_key in batch and batch[img_key][i] is not None:
                    question_images.append(batch[img_key][i])

            prompt = batch["prompt"][i]
            new_messages = []

            for msg in prompt:
                if msg["role"] == "user":
                    # Parse and interleave images at their marked positions
                    import re

                    content = []

                    # Split text by image placeholders while keeping track of which image goes where
                    parts = re.split(r"(<image \d+>)", full_question)

                    for part in parts:
                        if part.startswith("<image ") and part.endswith(">"):
                            # Extract image number (e.g., "<image 1>" -> 1)
                            img_num = int(re.search(r"\d+", part).group()) - 1
                            if img_num < len(question_images):
                                content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "placeholder://image"},
                                    }
                                )
                        elif part.strip():  # Add non-empty text parts
                            content.append({"type": "text", "text": part.strip()})

                    new_messages.append({"role": "user", "content": content})
                else:
                    new_messages.append(
                        {
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"]}],
                        }
                    )

            prompts.append(new_messages)
            images.append(question_images)

        # Process answers - MMMU uses letter answers
        answers = (
            batch["answer"] if "answer" in batch else [""] * len(batch["question"])
        )

        # Return all columns including new ones
        result = dict(batch)
        result["prompt"] = prompts
        result["images"] = images
        result["answer"] = answers
        return result

    # Rubric for evaluating MMMU multiple-choice responses
    def answer_match_reward(completion, answer, **kwargs):
        """Check if the response matches the correct answer letter."""
        response = parser.parse_answer(completion) or ""
        response = response.strip().upper()

        # Extract just the letter if the response includes more
        if response and response[0] in "ABCDEFGHIJ":
            response_letter = response[0]
        else:
            response_letter = response

        correct_answer = answer.strip().upper()

        return 1.0 if response_letter == correct_answer else 0.0

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
        system_prompt=system_prompt,
    )

    return vf_env
