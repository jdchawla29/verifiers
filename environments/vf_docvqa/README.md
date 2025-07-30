# DocVQA Environment

Document Visual Question Answering (DocVQA) environment for multimodal document understanding tasks.

## Dataset

Uses the [DocVQA dataset](https://huggingface.co/datasets/lmms-lab/DocVQA) which contains:
- Document images (forms, receipts, letters, etc.)
- Questions about the documents
- Multiple acceptable answers per question

## Task Description

Given a document image and a question about it, the model must extract and provide the correct answer from the document.

## Evaluation

The environment uses:
- **Answer matching**: Checks if the model's response contains any of the acceptable answers (case-insensitive, partial matching)
- **Format checking**: Ensures responses follow the think/answer XML format

## Usage

```python
from vf_docvqa import load_environment

env = load_environment(
    num_train_examples=1000,  # -1 for full dataset
    num_eval_examples=100     # -1 for full eval set
)
```

## Multimodal Support

This environment supports multimodal inputs (text + images). The data collator formats prompts with image placeholders that are replaced with actual images during generation.