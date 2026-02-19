"""
Export NICU-Llama to GGUF Format
For use with LM Studio, Ollama, and llama.cpp
"""

import os
import sys
from pathlib import Path
import torch
from unsloth import FastLanguageModel
import subprocess


# ============================================================================
# CONFIGURATION
# ============================================================================

class ExportConfig:
    """Configuration for model export"""
    
    # Model paths
    FINETUNED_MODEL_PATH = "output/nicu-llama-qlora/final_model"
    MERGED_OUTPUT_PATH = "output/nicu-llama-merged"
    GGUF_OUTPUT_PATH = "output/nicu-llama-gguf"
    
    # Quantization options
    # Available: q4_k_m (recommended), q5_k_m, q8_0, f16
    QUANTIZATION_METHODS = ["q4_k_m", "q5_k_m", "q8_0"]
    
    # Model parameters
    MAX_SEQ_LENGTH = 2048
    DTYPE = None  # Auto-detect


# ============================================================================
# STEP 1: MERGE LORA ADAPTERS
# ============================================================================

def merge_lora_adapters(config: ExportConfig):
    """
    Merge LoRA adapters back into base model
    
    Args:
        config: ExportConfig instance
    """
    print("=" * 80)
    print("STEP 1: MERGING LORA ADAPTERS WITH BASE MODEL")
    print("=" * 80)
    
    print(f"\nLoading fine-tuned model from: {config.FINETUNED_MODEL_PATH}")
    
    # Load the fine-tuned model with adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.FINETUNED_MODEL_PATH,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=False,  # Load in full precision for merging
    )
    
    print("✓ Model loaded successfully")
    
    # Merge LoRA weights into base model
    print("\nMerging LoRA adapters...")
    model = model.merge_and_unload()
    
    print("✓ LoRA adapters merged")
    
    # Save merged model
    print(f"\nSaving merged model to: {config.MERGED_OUTPUT_PATH}")
    os.makedirs(config.MERGED_OUTPUT_PATH, exist_ok=True)
    
    model.save_pretrained(config.MERGED_OUTPUT_PATH)
    tokenizer.save_pretrained(config.MERGED_OUTPUT_PATH)
    
    print("✓ Merged model saved successfully")
    
    return model, tokenizer


# ============================================================================
# STEP 2: EXPORT TO GGUF
# ============================================================================

def export_to_gguf_unsloth(config: ExportConfig):
    """
    Export model to GGUF format using Unsloth
    
    Args:
        config: ExportConfig instance
    """
    print("\n" + "=" * 80)
    print("STEP 2: EXPORTING TO GGUF FORMAT")
    print("=" * 80)
    
    print(f"\nLoading merged model from: {config.MERGED_OUTPUT_PATH}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MERGED_OUTPUT_PATH,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=False,
    )
    
    print("✓ Merged model loaded")
    
    # Create output directory
    os.makedirs(config.GGUF_OUTPUT_PATH, exist_ok=True)
    
    # Export to different quantization levels
    for quant_method in config.QUANTIZATION_METHODS:
        print(f"\n{'-' * 80}")
        print(f"Exporting with quantization: {quant_method}")
        print(f"{'-' * 80}")
        
        output_filename = f"nicu-llama-{quant_method}.gguf"
        output_path = os.path.join(config.GGUF_OUTPUT_PATH, output_filename)
        
        try:
            model.save_pretrained_gguf(
                output_path,
                tokenizer,
                quantization_method=quant_method,
            )
            print(f"✓ Exported to: {output_path}")
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 ** 3)  # GB
            print(f"  File size: {file_size:.2f} GB")
            
        except Exception as e:
            print(f"✗ Error exporting {quant_method}: {e}")
    
    print("\n✓ GGUF export completed!")


# ============================================================================
# STEP 3: CREATE MODELFILE FOR OLLAMA
# ============================================================================

def create_ollama_modelfile(config: ExportConfig):
    """
    Create Modelfile for Ollama deployment
    
    Args:
        config: ExportConfig instance
    """
    print("\n" + "=" * 80)
    print("STEP 3: CREATING OLLAMA MODELFILE")
    print("=" * 80)
    
    # Find the q4_k_m model (recommended for Ollama)
    gguf_file = os.path.join(config.GGUF_OUTPUT_PATH, "nicu-llama-q4_k_m.gguf")
    
    if not os.path.exists(gguf_file):
        print(f"⚠ GGUF file not found: {gguf_file}")
        print("  Skipping Ollama Modelfile creation")
        return
    
    modelfile_content = f"""# NICU-Llama Modelfile for Ollama
# Fine-tuned Llama-3-8B for Neonatal Clinical Decision Support

FROM {gguf_file}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt
SYSTEM \"\"\"You are NICU-Llama, a specialized clinical decision support assistant for neonatal intensive care. You provide evidence-based recommendations based on AAP (American Academy of Pediatrics) guidelines and current best practices in neonatal medicine.

Your responses should:
- Be based on established clinical guidelines (especially AAP)
- Prioritize patient safety
- Be clear and actionable for healthcare providers
- Include relevant clinical reasoning
- Acknowledge limitations and when to seek additional expertise

You assist with interpreting vital signs, suggesting management strategies, and providing clinical context for NICU scenarios.\"\"\"

# Template (Alpaca format)
TEMPLATE \"\"\"Below is an instruction that describes a clinical task, paired with an input that provides patient context. Write a response that appropriately completes the request based on AAP guidelines.

### Instruction:
{{ .Prompt }}

### Input:
{{ .Input }}

### Response:
\"\"\"
"""
    
    modelfile_path = os.path.join(config.GGUF_OUTPUT_PATH, "Modelfile")
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✓ Modelfile created: {modelfile_path}")
    
    print("\n" + "-" * 80)
    print("To use with Ollama:")
    print("-" * 80)
    print(f"1. Navigate to: {config.GGUF_OUTPUT_PATH}")
    print("2. Run: ollama create nicu-llama -f Modelfile")
    print("3. Run: ollama run nicu-llama")
    print("-" * 80)


# ============================================================================
# STEP 4: CREATE LM STUDIO INSTRUCTIONS
# ============================================================================

def create_lm_studio_instructions(config: ExportConfig):
    """
    Create instructions for using model in LM Studio
    
    Args:
        config: ExportConfig instance
    """
    print("\n" + "=" * 80)
    print("STEP 4: LM STUDIO SETUP INSTRUCTIONS")
    print("=" * 80)
    
    instructions = f"""
# Using NICU-Llama with LM Studio

## Files Available:
{config.GGUF_OUTPUT_PATH}/
├── nicu-llama-q4_k_m.gguf  (Recommended - 4.5GB, good balance)
├── nicu-llama-q5_k_m.gguf  (Higher quality - 5.5GB)
└── nicu-llama-q8_0.gguf    (Highest quality - 8.5GB)

## Setup Steps:

1. **Download LM Studio**
   - Visit: https://lmstudio.ai/
   - Download and install for your OS

2. **Load NICU-Llama**
   - Open LM Studio
   - Click "Local Models" 
   - Click "Load Model"
   - Navigate to: {os.path.abspath(config.GGUF_OUTPUT_PATH)}
   - Select: nicu-llama-q4_k_m.gguf

3. **Configure Settings**
   - Temperature: 0.7
   - Top P: 0.9
   - Context Length: 2048
   - GPU Layers: Auto (or adjust based on your GPU)

4. **System Prompt**
   Use this system prompt:
   
   "You are NICU-Llama, a specialized clinical decision support assistant 
   for neonatal intensive care. Provide evidence-based recommendations 
   based on AAP guidelines and current best practices."

5. **Test Query**
   Try this example:
   
   Instruction: Analyze vital signs and recommend management
   
   Input: Neonate, 48 hours old. HR: 180 bpm, SpO2: 88%, RR: 72, 
   Temperature: 36.1°C. Mild retractions noted.

## Model Sizes:
- q4_k_m: ~4.5 GB (Recommended for most users)
- q5_k_m: ~5.5 GB (Better quality, needs more VRAM)
- q8_0: ~8.5 GB (Best quality, needs significant VRAM)

## Performance Tips:
- For faster inference: Use q4_k_m and increase GPU layers
- For better quality: Use q5_k_m or q8_0
- If running on CPU: q4_k_m is most efficient
"""
    
    instructions_path = os.path.join(config.GGUF_OUTPUT_PATH, "LM_STUDIO_GUIDE.txt")
    
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"✓ Instructions saved to: {instructions_path}")
    print("\n" + instructions)


# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================

def verify_gguf_files(config: ExportConfig):
    """
    Verify exported GGUF files
    
    Args:
        config: ExportConfig instance
    """
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    print("\nExported models:")
    print("-" * 80)
    
    total_size = 0
    
    for quant_method in config.QUANTIZATION_METHODS:
        filename = f"nicu-llama-{quant_method}.gguf"
        filepath = os.path.join(config.GGUF_OUTPUT_PATH, filename)
        
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024 ** 3)
            total_size += size_gb
            print(f"✓ {filename:<30} {size_gb:>8.2f} GB")
        else:
            print(f"✗ {filename:<30} NOT FOUND")
    
    print("-" * 80)
    print(f"Total size: {total_size:.2f} GB")
    
    # Check for additional files
    modelfile = os.path.join(config.GGUF_OUTPUT_PATH, "Modelfile")
    instructions = os.path.join(config.GGUF_OUTPUT_PATH, "LM_STUDIO_GUIDE.txt")
    
    print("\nAdditional files:")
    print(f"✓ Modelfile (Ollama): {os.path.exists(modelfile)}")
    print(f"✓ LM Studio Guide: {os.path.exists(instructions)}")


# ============================================================================
# MAIN EXPORT PIPELINE
# ============================================================================

def main():
    """Main export pipeline"""
    
    config = ExportConfig()
    
    print("=" * 80)
    print("NICU-LLAMA GGUF EXPORT PIPELINE")
    print("=" * 80)
    print("\nThis will convert your fine-tuned model to GGUF format for")
    print("local deployment with LM Studio, Ollama, or llama.cpp")
    print("\nSteps:")
    print("1. Merge LoRA adapters with base model")
    print("2. Export to GGUF with multiple quantization levels")
    print("3. Create Ollama Modelfile")
    print("4. Generate LM Studio instructions")
    print("5. Verify outputs")
    
    # Check if fine-tuned model exists
    if not os.path.exists(config.FINETUNED_MODEL_PATH):
        print(f"\n✗ ERROR: Fine-tuned model not found at: {config.FINETUNED_MODEL_PATH}")
        print("\nPlease run training first: python train_nicu_llama.py")
        return
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        # Step 1: Merge LoRA adapters
        merge_lora_adapters(config)
        
        # Step 2: Export to GGUF
        export_to_gguf_unsloth(config)
        
        # Step 3: Create Ollama Modelfile
        create_ollama_modelfile(config)
        
        # Step 4: Create LM Studio instructions
        create_lm_studio_instructions(config)
        
        # Step 5: Verify
        verify_gguf_files(config)
        
        print("\n" + "=" * 80)
        print("✅ EXPORT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nYour GGUF models are ready at: {config.GGUF_OUTPUT_PATH}")
        print("\nNext steps:")
        print("1. For Ollama: Follow instructions in Modelfile")
        print("2. For LM Studio: See LM_STUDIO_GUIDE.txt")
        print("3. For llama.cpp: Use any .gguf file directly")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Export cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ ERROR: Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
