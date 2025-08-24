"""
GRPO training script for DocVQA environment.

For local multimodal model training:
    # Run training (requires 2 GPUs)
    # GPU 0: Inference server
    CUDA_VISIBLE_DEVICES=0 vf-vllm --model 'Qwen/Qwen2.5-VL-3B-Instruct' --max-model-len 16384

    # GPU 1: Training
    CUDA_VISIBLE_DEVICES=1 python examples/grpo/train_docvqa.py
"""

import argparse
import logging
import sys
import verifiers as vf

# Import the environment
sys.path.append("environments/docvqa")
from docvqa import load_environment

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train multimodal models on DocVQA")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model to train (must match the vLLM server model)",
    )
    parser.add_argument(
        "--train-samples", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--eval-samples", type=int, default=100, help="Number of evaluation samples"
    )
    args = parser.parse_args()

    # Model configuration
    MODEL_NAME = args.model

    # Training configuration
    TRAIN_SAMPLES = args.train_samples
    EVAL_SAMPLES = args.eval_samples
    MAX_STEPS = 200
    EVAL_STEPS = 20
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_GENERATIONS = 8

    vf_env = load_environment(
        num_train_examples=TRAIN_SAMPLES, num_eval_examples=EVAL_SAMPLES
    )

    # Load model and processor
    model, processor = vf.get_model_and_tokenizer(MODEL_NAME)

    # Training arguments
    run_name = f"docvqa_{MODEL_NAME.split('/')[-1].lower()}"
    training_args = vf.grpo_defaults(run_name=run_name)

    # Customize training arguments
    training_args.learning_rate = 1e-6
    training_args.max_steps = MAX_STEPS
    training_args.eval_strategy = "steps"
    training_args.eval_steps = EVAL_STEPS
    training_args.save_steps = EVAL_STEPS
    training_args.eval_on_start = True
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False,
    }

    # GRPO specific settings
    training_args.num_generations = NUM_GENERATIONS
    training_args.per_device_train_batch_size = BATCH_SIZE
    training_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS

    training_args.gradient_checkpointing = True

    # Generation settings
    training_args.temperature = 0.7
    training_args.max_tokens = 200
    training_args.max_seq_len = 16384

    # Create trainer
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=processor,
        env=vf_env,
        args=training_args,
    )

    logger.info(f"Starting GRPO training for {MODEL_NAME} on DocVQA")
    logger.info(f"Train samples: {TRAIN_SAMPLES}, Eval samples: {EVAL_SAMPLES}")
    logger.info(
        f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}"
    )
    logger.info(
        f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GENERATIONS}"
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model()
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
