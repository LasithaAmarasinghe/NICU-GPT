"""
GPT-4 as a Judge: Evaluation System for NICU-Llama
Compares base model vs fine-tuned model against AAP Golden Set
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # API Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GPT_JUDGE_MODEL: str = "gpt-4o"  # or "gpt-4-turbo"
    
    # Evaluation Data
    TEST_SET_PATH: str = "data/evaluation/golden_set.json"
    
    # Model Paths
    BASE_MODEL_PATH: str = "unsloth/llama-3-8b-Instruct-bnb-4bit"
    FINETUNED_MODEL_PATH: str = "output/nicu-llama-qlora/final_model"
    
    # Output
    RESULTS_DIR: str = "results"
    TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Evaluation Parameters
    TEMPERATURE: float = 0.0  # Deterministic for judge
    MAX_TOKENS: int = 1000


# ============================================================================
# JUDGE PROMPT
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert neonatologist and clinical evaluator. Your task is to evaluate AI-generated clinical recommendations for neonatal intensive care scenarios against gold standard answers based on AAP (American Academy of Pediatrics) guidelines.

Evaluate each response on the following criteria:

1. **Clinical Accuracy** (0-10): Are the recommendations medically correct and aligned with AAP guidelines?
2. **Completeness** (0-10): Does the response cover all necessary aspects of patient management?
3. **Safety** (0-10): Are the recommendations safe? Are dangerous omissions or errors present?
4. **Clarity** (0-10): Is the response clear, well-structured, and actionable for clinicians?
5. **Evidence-Based** (0-10): Are recommendations based on established guidelines and evidence?

Provide:
- Individual scores for each criterion (0-10)
- Overall score (average of the 5 criteria)
- Brief justification for your scoring
- Critical errors (if any)

Be strict and thorough. Patient safety is paramount."""


def create_judge_prompt(
    scenario: str,
    golden_answer: str,
    model_answer: str,
    model_name: str
) -> str:
    """
    Create evaluation prompt for GPT-4 judge
    
    Args:
        scenario: Clinical scenario (instruction + input)
        golden_answer: Gold standard answer from AAP guidelines
        model_answer: Model's generated answer
        model_name: Name of model being evaluated
    
    Returns:
        Formatted prompt for GPT-4
    """
    return f"""**CLINICAL SCENARIO:**
{scenario}

---

**GOLD STANDARD ANSWER (AAP Guidelines):**
{golden_answer}

---

**MODEL ANSWER ({model_name}):**
{model_answer}

---

**TASK:** Evaluate the MODEL ANSWER against the GOLD STANDARD using the criteria provided. The model answer should align with AAP guidelines and provide safe, accurate clinical recommendations.

Provide your evaluation in the following JSON format:
{{
    "clinical_accuracy": <score 0-10>,
    "completeness": <score 0-10>,
    "safety": <score 0-10>,
    "clarity": <score 0-10>,
    "evidence_based": <score 0-10>,
    "overall_score": <average of above>,
    "justification": "<2-3 sentences explaining your scoring>",
    "critical_errors": ["<list any critical errors, or empty list if none>"],
    "strengths": ["<key strengths of the response>"],
    "weaknesses": ["<key weaknesses or missing elements>"]
}}

Return ONLY the JSON object, no additional text."""


# ============================================================================
# GPT-4 JUDGE
# ============================================================================

class GPT4Judge:
    """GPT-4 based evaluator for clinical responses"""
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize GPT-4 judge
        
        Args:
            config: EvaluationConfig instance
        """
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.config = config
        self.call_count = 0
    
    def evaluate_response(
        self,
        scenario: str,
        golden_answer: str,
        model_answer: str,
        model_name: str,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate a single model response
        
        Args:
            scenario: Clinical scenario
            golden_answer: Gold standard answer
            model_answer: Model's answer
            model_name: Name of model
            retry_count: Number of retries on failure
        
        Returns:
            Evaluation results as dictionary
        """
        prompt = create_judge_prompt(scenario, golden_answer, model_answer, model_name)
        
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.GPT_JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                )
                
                self.call_count += 1
                
                # Parse JSON response
                result_text = response.choices[0].message.content.strip()
                
                # Extract JSON if wrapped in code blocks
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                evaluation = json.loads(result_text)
                
                # Add metadata
                evaluation["model_name"] = model_name
                evaluation["timestamp"] = datetime.now().isoformat()
                
                return evaluation
                
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"⚠ Evaluation attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    print(f"✗ Evaluation failed after {retry_count} attempts: {e}")
                    return {
                        "error": str(e),
                        "model_name": model_name,
                        "clinical_accuracy": 0,
                        "completeness": 0,
                        "safety": 0,
                        "clarity": 0,
                        "evidence_based": 0,
                        "overall_score": 0,
                        "justification": f"Evaluation failed: {str(e)}",
                        "critical_errors": ["Evaluation system error"],
                        "strengths": [],
                        "weaknesses": []
                    }


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def load_model_for_inference(model_path: str, max_seq_length: int = 2048):
    """
    Load model for inference
    
    Args:
        model_path: Path to model
        max_seq_length: Maximum sequence length
    
    Returns:
        model, tokenizer
    """
    from unsloth import FastLanguageModel
    
    print(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 512
) -> str:
    """
    Generate response from model
    
    Args:
        model: Model
        tokenizer: Tokenizer
        instruction: Instruction text
        input_text: Input text
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated response
    """
    from train_nicu_llama import format_nicu_prompt
    
    prompt = format_nicu_prompt(instruction, input_text, "")
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def run_evaluation(config: EvaluationConfig) -> pd.DataFrame:
    """
    Run complete evaluation pipeline
    
    Args:
        config: EvaluationConfig instance
    
    Returns:
        DataFrame with evaluation results
    """
    print("=" * 80)
    print("NICU-LLAMA EVALUATION WITH GPT-4 AS JUDGE")
    print("=" * 80)
    
    # Load test set
    print(f"\nLoading test set from: {config.TEST_SET_PATH}")
    with open(config.TEST_SET_PATH, 'r') as f:
        test_set = json.load(f)
    
    print(f"✓ Loaded {len(test_set)} test cases")
    
    # Load models
    print("\n" + "-" * 80)
    print("LOADING MODELS")
    print("-" * 80)
    
    print("\n1. Base Model:")
    base_model, base_tokenizer = load_model_for_inference(config.BASE_MODEL_PATH)
    
    print("\n2. Fine-tuned Model:")
    finetuned_model, finetuned_tokenizer = load_model_for_inference(config.FINETUNED_MODEL_PATH)
    
    # Initialize judge
    print("\n" + "-" * 80)
    print("INITIALIZING GPT-4 JUDGE")
    print("-" * 80)
    judge = GPT4Judge(config)
    print(f"✓ Using {config.GPT_JUDGE_MODEL}")
    
    # Run evaluation
    print("\n" + "-" * 80)
    print("RUNNING EVALUATION")
    print("-" * 80)
    
    results = []
    
    for i, test_case in enumerate(tqdm(test_set, desc="Evaluating")):
        print(f"\n[{i+1}/{len(test_set)}] Processing test case...")
        
        instruction = test_case["instruction"]
        input_text = test_case["input"]
        golden_output = test_case["output"]
        
        scenario = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
        
        # Generate responses
        print("  - Generating base model response...")
        base_response = generate_response(base_model, base_tokenizer, instruction, input_text)
        
        print("  - Generating fine-tuned model response...")
        finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, instruction, input_text)
        
        # Evaluate both
        print("  - Evaluating with GPT-4...")
        base_eval = judge.evaluate_response(scenario, golden_output, base_response, "Base Llama-3-8B")
        time.sleep(1)  # Rate limiting
        
        finetuned_eval = judge.evaluate_response(scenario, golden_output, finetuned_response, "Fine-tuned NICU-Llama")
        time.sleep(1)  # Rate limiting
        
        # Store results
        results.append({
            "test_case_id": i,
            "instruction": instruction,
            "input": input_text,
            "golden_output": golden_output,
            "base_response": base_response,
            "finetuned_response": finetuned_response,
            "base_evaluation": base_eval,
            "finetuned_evaluation": finetuned_eval,
        })
    
    print("\n✓ Evaluation completed!")
    print(f"Total GPT-4 API calls: {judge.call_count}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


# ============================================================================
# ANALYSIS & REPORTING
# ============================================================================

def analyze_results(df: pd.DataFrame, config: EvaluationConfig):
    """
    Analyze and report evaluation results
    
    Args:
        df: DataFrame with evaluation results
        config: EvaluationConfig instance
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Extract scores
    base_scores = []
    finetuned_scores = []
    
    criteria = ["clinical_accuracy", "completeness", "safety", "clarity", "evidence_based", "overall_score"]
    
    for _, row in df.iterrows():
        base_eval = row["base_evaluation"]
        ft_eval = row["finetuned_evaluation"]
        
        base_scores.append([base_eval.get(c, 0) for c in criteria])
        finetuned_scores.append([ft_eval.get(c, 0) for c in criteria])
    
    base_scores = np.array(base_scores)
    finetuned_scores = np.array(finetuned_scores)
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("AVERAGE SCORES")
    print("-" * 80)
    
    print(f"\n{'Criterion':<20} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for i, criterion in enumerate(criteria):
        base_avg = base_scores[:, i].mean()
        ft_avg = finetuned_scores[:, i].mean()
        improvement = ft_avg - base_avg
        improvement_pct = (improvement / base_avg * 100) if base_avg > 0 else 0
        
        print(f"{criterion:<20} {base_avg:<15.2f} {ft_avg:<15.2f} {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    # Critical errors
    print("\n" + "-" * 80)
    print("CRITICAL ERRORS")
    print("-" * 80)
    
    base_critical = sum(1 for _, row in df.iterrows() if row["base_evaluation"].get("critical_errors", []))
    ft_critical = sum(1 for _, row in df.iterrows() if row["finetuned_evaluation"].get("critical_errors", []))
    
    print(f"Base Model: {base_critical} cases with critical errors")
    print(f"Fine-tuned Model: {ft_critical} cases with critical errors")
    print(f"Improvement: {base_critical - ft_critical} fewer critical errors")
    
    # Save results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    output_file = os.path.join(config.RESULTS_DIR, f"evaluation_{config.TIMESTAMP}.json")
    df.to_json(output_file, orient="records", indent=2)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Save summary
    summary = {
        "timestamp": config.TIMESTAMP,
        "test_cases": len(df),
        "base_model": config.BASE_MODEL_PATH,
        "finetuned_model": config.FINETUNED_MODEL_PATH,
        "average_scores": {
            "base": {criteria[i]: float(base_scores[:, i].mean()) for i in range(len(criteria))},
            "finetuned": {criteria[i]: float(finetuned_scores[:, i].mean()) for i in range(len(criteria))},
        },
        "critical_errors": {
            "base": base_critical,
            "finetuned": ft_critical,
        }
    }
    
    summary_file = os.path.join(config.RESULTS_DIR, f"summary_{config.TIMESTAMP}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {summary_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation pipeline"""
    
    config = EvaluationConfig()
    
    # Check API key
    if not config.OPENAI_API_KEY:
        print("⚠ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\nSet your API key:")
        print("  Windows: set OPENAI_API_KEY=sk-...")
        print("  Linux/Mac: export OPENAI_API_KEY=sk-...")
        return
    
    # Run evaluation
    results_df = run_evaluation(config)
    
    # Analyze results
    analyze_results(results_df, config)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
