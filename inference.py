"""
Simple inference script for testing NICU-Llama
"""

import torch
from unsloth import FastLanguageModel
from train_nicu_llama import format_nicu_prompt


def load_nicu_model(model_path: str = "output/nicu-llama-qlora/final_model"):
    """Load the fine-tuned NICU-Llama model"""
    
    print("Loading NICU-Llama model...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully!\n")
    
    return model, tokenizer


def get_clinical_advice(
    model,
    tokenizer,
    instruction: str,
    patient_info: str,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """
    Get clinical advice from NICU-Llama
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        instruction: Clinical instruction/question
        patient_info: Patient vital signs and context
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.7 = balanced)
    
    Returns:
        Clinical recommendation text
    """
    
    # Format prompt
    prompt = format_nicu_prompt(instruction, patient_info, "")
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True,
    )
    
    # Decode
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    """Interactive chat mode for NICU-Llama"""
    
    print("=" * 80)
    print("NICU-LLAMA INTERACTIVE MODE")
    print("=" * 80)
    print("\nType 'exit' to quit, 'example' for a sample query\n")
    
    while True:
        print("-" * 80)
        
        # Get instruction
        instruction = input("Instruction (e.g., 'Analyze vital signs'): ").strip()
        
        if instruction.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if instruction.lower() == 'example':
            instruction = "Analyze vital signs and provide clinical recommendations"
            patient_info = """
Patient: Neonate, 48 hours old, term (39 weeks)
Vital Signs:
- Heart Rate: 165 bpm
- SpO2: 95% on room air
- Respiratory Rate: 52 breaths/min
- Temperature: 36.8°C
- Blood Pressure: 65/38 mmHg

Clinical Context: Vigorous, feeding well, normal activity
"""
            print(f"Using example:\n{patient_info}")
        
        else:
            # Get patient info
            print("\nEnter patient information (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line.strip() == "" and lines:
                    break
                lines.append(line)
            patient_info = "\n".join(lines)
        
        if not patient_info.strip():
            print("⚠ No patient information provided")
            continue
        
        # Get response
        print("\n🤖 NICU-Llama is thinking...\n")
        
        response = get_clinical_advice(
            model,
            tokenizer,
            instruction,
            patient_info
        )
        
        print("=" * 80)
        print("CLINICAL RECOMMENDATION:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        print()


def main():
    """Main entry point"""
    
    # Load model
    model, tokenizer = load_nicu_model()
    
    # Quick test
    print("Running quick test...\n")
    
    test_instruction = "Assess neonatal jaundice and recommend management"
    test_patient = """
Patient: 72 hours old, term neonate
Laboratory: Total Serum Bilirubin 16.5 mg/dL
Risk Factors: Breastfeeding, 37 weeks gestation
"""
    
    response = get_clinical_advice(model, tokenizer, test_instruction, test_patient)
    
    print("=" * 80)
    print("TEST QUERY:")
    print("=" * 80)
    print(f"Instruction: {test_instruction}")
    print(f"\nInput:\n{test_patient}")
    print("\n" + "=" * 80)
    print("RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # Ask if user wants interactive mode
    print("\n")
    choice = input("Enter interactive mode? (y/n): ").strip().lower()
    
    if choice == 'y':
        interactive_mode(model, tokenizer)
    else:
        print("\nDone!")


if __name__ == "__main__":
    main()
