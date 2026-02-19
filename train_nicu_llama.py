"""
NICU-GPT: Fine-tuning Llama-3-8B using QLoRA with Unsloth
Author: ENTC Undergraduate
Description: Clinical Decision Support LLM for NICU Protocols
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for NICU-Llama fine-tuning"""
    
    # Model Configuration
    MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 2048  # Llama-3 supports up to 8k, but 2k is efficient
    LOAD_IN_4BIT = True
    
    # LoRA Configuration
    LORA_R = 16  # Rank of LoRA adapters (higher = more parameters)
    LORA_ALPHA = 16  # Scaling factor
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
    
    # Training Configuration
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 2 * 4 = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 5
    LOGGING_STEPS = 1
    SAVE_STEPS = 50
    
    # Data Configuration
    DATA_PATH = "data/processed/nicu_training_data.json"
    OUTPUT_DIR = "output/nicu-llama-qlora"
    
    # Wandb Configuration
    USE_WANDB = True
    WANDB_PROJECT = "nicu-gpt"
    WANDB_RUN_NAME = f"llama3-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_nicu_prompt(instruction, input_text, output_text):
    """
    Format data in Alpaca-style instruction format for clinical scenarios
    
    Args:
        instruction: The task instruction (e.g., "Analyze vital signs and suggest next steps")
        input_text: Patient vital signs and context
        output_text: Expected clinical recommendation
    
    Returns:
        Formatted prompt string
    """
    alpaca_prompt = """Below is an instruction that describes a clinical task, paired with an input that provides patient context. Write a response that appropriately completes the request based on AAP guidelines.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    return alpaca_prompt.format(instruction, input_text, output_text)


def formatting_prompts_func(examples):
    """
    Batch formatting function for dataset
    
    Args:
        examples: Dictionary with 'instruction', 'input', 'output' keys
    
    Returns:
        Dictionary with 'text' key containing formatted prompts
    """
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    
    texts = []
    for instruction, input_text, output_text in zip(instructions, inputs, outputs):
        text = format_nicu_prompt(instruction, input_text, output_text)
        texts.append(text)
    
    return {"text": texts}


# ============================================================================
# MODEL SETUP
# ============================================================================

def setup_model_and_tokenizer(config):
    """
    Load Llama-3-8B in 4-bit quantization with LoRA adapters
    
    Args:
        config: TrainingConfig instance
    
    Returns:
        model, tokenizer: Initialized model and tokenizer
    """
    print("=" * 80)
    print("LOADING LLAMA-3-8B WITH QLORA")
    print("=" * 80)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (Float16 for Tesla T4, V100, Bfloat16 for Ampere+)
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=3407,
        use_rslora=False,  # Rank-stabilized LoRA
        loftq_config=None,
    )
    
    print(f"\n✓ Model loaded successfully")
    print(f"✓ LoRA adapters applied with rank={config.LORA_R}")
    print(f"✓ Target modules: {config.TARGET_MODULES}")
    
    return model, tokenizer


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare_dataset(config):
    """
    Load and prepare the NICU training dataset
    
    Args:
        config: TrainingConfig instance
    
    Returns:
        train_dataset, eval_dataset: Prepared datasets
    """
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    # Load dataset
    dataset = load_dataset('json', data_files=config.DATA_PATH, split='train')
    
    # Split into train/eval (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"\n✓ Training samples: {len(train_dataset)}")
    print(f"✓ Evaluation samples: {len(eval_dataset)}")
    
    # Show example
    print("\n" + "-" * 80)
    print("SAMPLE DATA POINT:")
    print("-" * 80)
    example = train_dataset[0]
    print(f"Instruction: {example['instruction'][:100]}...")
    print(f"Input: {example['input'][:100]}...")
    print(f"Output: {example['output'][:100]}...")
    
    return train_dataset, eval_dataset


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    """
    Fine-tune the model using SFTTrainer
    
    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: TrainingConfig instance
    """
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    # Initialize Wandb if enabled
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            name=config.WANDB_RUN_NAME,
            config={
                "model": config.MODEL_NAME,
                "lora_r": config.LORA_R,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.NUM_EPOCHS,
            }
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config.LOGGING_STEPS,
        eval_steps=config.SAVE_STEPS,
        save_steps=config.SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb" if config.USE_WANDB else "none",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )
    
    # Train
    print("\n🚀 Training started...")
    trainer.train()
    
    print("\n✓ Training completed!")
    
    # Save final model
    final_output_dir = os.path.join(config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_output_dir)
    print(f"✓ Model saved to: {final_output_dir}")
    
    if config.USE_WANDB:
        wandb.finish()
    
    return trainer


# ============================================================================
# INFERENCE TESTING
# ============================================================================

def test_inference(model, tokenizer):
    """
    Test the fine-tuned model with a sample NICU scenario
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
    """
    print("\n" + "=" * 80)
    print("TESTING INFERENCE")
    print("=" * 80)
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    # Sample clinical scenario
    test_instruction = "Analyze the neonatal vital signs and provide clinical recommendations based on AAP guidelines."
    test_input = """
Patient: Neonate, 2 days old, 36 weeks gestational age
Vital Signs:
- Heart Rate: 175 bpm
- SpO2: 88%
- Respiratory Rate: 68 breaths/min
- Temperature: 36.2°C
- Blood Pressure: 55/30 mmHg

Clinical Context: Infant appears dusky, mild intercostal retractions observed.
"""
    
    prompt = format_nicu_prompt(test_instruction, test_input, "")
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("\n" + "-" * 80)
    print("TEST SCENARIO:")
    print(test_input)
    print("\n" + "-" * 80)
    print("MODEL RESPONSE:")
    print(response.split("### Response:")[-1].strip())
    print("-" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(config)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, config)
    
    # Test inference
    test_inference(model, tokenizer)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {config.OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Run evaluation: python evaluate_model.py")
    print("2. Export to GGUF: python export_to_gguf.py")
    print("3. Test with LM Studio or Ollama")


if __name__ == "__main__":
    main()
