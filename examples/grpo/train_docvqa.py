"""
GRPO training script for DocVQA environment.

For local multimodal model training:
    # Install model requirements
    uv pip install qwen-vl-utils
    
    # Run training (requires 2 GPUs)
    # GPU 0: Inference server
    CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model 'Qwen/Qwen2-VL-2B-Instruct' --max-model-len 16384
    
    # GPU 1: Training
    CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --config_file configs/single_gpu.yaml examples/grpo/train_docvqa.py
"""

import logging
import sys
import verifiers as vf

# Import the environment
sys.path.append('environments/vf_docvqa')
from vf_docvqa import load_environment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def qwen_data_collator(batch):
    """Format data for Qwen VLM models."""
    from qwen_vl_utils import process_vision_info
    
    processed_samples = []
    for sample in batch:
        messages = []
        # System prompt
        messages.append({
            "role": "system", 
            "content": """Answer the questions about the document image.

Respond in the following format:
<think>
[Your reasoning here]
</think>
<answer>
[Your concise answer here]
</answer>"""
        })
        
        # User message with image
        content_block = []
        content_block.append({"type": "text", "text": sample["question"]})
        content_block.append({
            "type": "image",
            "image": sample["image"],
            "resized_height": 768,  # Higher resolution for document images
            "resized_width": 768,
        })
        messages.append({"role": "user", "content": content_block})
        
        # Process images for Qwen
        processed_images, *_ = process_vision_info(messages.copy())
        
        sample["prompt"] = messages
        sample["images"] = processed_images
        sample["answer"] = sample["answers"]
        processed_samples.append(sample)
    return processed_samples


def main():
    # Model configuration
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Training configuration
    TRAIN_SAMPLES = 1000  # Number of training samples
    EVAL_SAMPLES = 100    # Number of eval samples
    MAX_STEPS = 200
    EVAL_STEPS = 20
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_GENERATIONS = 4
    
    # Load environment with custom data collator for training
    vf_env = load_environment(
        num_train_examples=TRAIN_SAMPLES,
        num_eval_examples=EVAL_SAMPLES
    )
    
    # Override data collator for Qwen model
    vf_env.data_collator = qwen_data_collator
    
    # Load model and processor
    model, processor = vf.get_model_and_tokenizer(MODEL_NAME)
    
    # Training arguments
    run_name = f"docvqa_{MODEL_NAME.split('/')[-1].lower()}"
    training_args = vf.grpo_defaults(run_name=run_name)
    
    # Customize training arguments
    training_args.learning_rate = 1e-6  # Lower LR for vision models
    training_args.max_steps = MAX_STEPS
    training_args.eval_strategy = "steps"
    training_args.eval_steps = EVAL_STEPS
    training_args.save_steps = EVAL_STEPS
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False,
    }
    
    # GRPO specific settings
    training_args.num_generations = NUM_GENERATIONS
    training_args.per_device_train_batch_size = BATCH_SIZE
    training_args.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    
    # Memory optimization for multimodal training
    training_args.fp16 = True
    training_args.optim = "adamw_8bit"
    training_args.gradient_checkpointing = True
    
    # Generation settings
    training_args.temperature = 0.7
    training_args.max_new_tokens = 200
    
    # Create trainer
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=processor,
        env=vf_env,
        args=training_args,
    )
    
    logger.info(f"Starting GRPO training for {MODEL_NAME} on DocVQA")
    logger.info(f"Train samples: {TRAIN_SAMPLES}, Eval samples: {EVAL_SAMPLES}")
    logger.info(f"Batch size: {BATCH_SIZE}, Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GENERATIONS}")
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    logger.info(f"Training completed! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()