# MMMU Environment

Environment for training and evaluating multimodal models on the MMMU benchmark multiple-choice questions.

## Overview

MMMU is a challenging multimodal benchmark that tests models across multiple academic disciplines including:
- Art & Design
- Business
- Science
- Health & Medicine
- Humanities & Social Science
- Tech & Engineering

This environment filters for multiple-choice questions only, where each question:
- May contain up to 7 images
- Has 2-10 answer options labeled A, B, C, etc.
- Requires reasoning across both visual and textual modalities

## Installation

```bash
vf-install mmmu
```

## Usage

```python
from mmmu import load_environment

# Load environment with custom dataset sizes
vf_env = load_environment(
    num_train_examples=500,  # Use 500 training examples
    num_eval_examples=100      # Use 100 evaluation examples
)
```

## Dataset

This environment uses the MMMU dataset from HuggingFace, filtered for multiple-choice questions only:
- Training: Validation set multiple-choice questions (filtered from 900 total samples)
- Evaluation: Dev set multiple-choice questions (filtered from 150 total samples)

The test set is held out and not used for training.
Open-ended questions are excluded to simplify training and evaluation.