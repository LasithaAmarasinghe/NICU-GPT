"""
Data Preprocessing Module for NICU-GPT
Converts raw clinical data into instruction-input-output format for fine-tuning
"""

import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import re


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class NICUDatapoint:
    """Represents a single NICU clinical scenario"""
    
    def __init__(self, instruction: str, input_data: str, output: str):
        self.instruction = instruction
        self.input = input_data
        self.output = output
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format"""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'NICUDatapoint':
        """Create from dictionary"""
        return cls(
            instruction=data["instruction"],
            input_data=data["input"],
            output=data["output"]
        )


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_datapoint(datapoint: Dict[str, str]) -> bool:
    """
    Validate a single datapoint for quality
    
    Args:
        datapoint: Dictionary with instruction, input, output
    
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ["instruction", "input", "output"]
    
    # Check all keys present
    if not all(key in datapoint for key in required_keys):
        return False
    
    # Check none are empty
    if not all(datapoint[key].strip() for key in required_keys):
        return False
    
    # Check reasonable length (outputs should be substantial)
    if len(datapoint["output"]) < 50:
        return False
    
    # Check input contains vital signs keywords
    input_lower = datapoint["input"].lower()
    vital_sign_keywords = ["heart rate", "spo2", "respiratory rate", "temperature", "bp", "blood pressure"]
    if not any(keyword in input_lower for keyword in vital_sign_keywords):
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n')
    
    return text


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_nicu_data(
    input_file: str,
    output_file: str,
    validate: bool = True
) -> None:
    """
    Preprocess raw NICU data into training format
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        validate: Whether to validate datapoints
    """
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✓ Loaded {len(raw_data)} raw datapoints")
    
    # Process each datapoint
    processed_data = []
    rejected_count = 0
    
    for i, item in enumerate(raw_data):
        # Clean text fields
        cleaned_item = {
            "instruction": clean_text(item.get("instruction", "")),
            "input": clean_text(item.get("input", "")),
            "output": clean_text(item.get("output", ""))
        }
        
        # Validate if requested
        if validate:
            if validate_datapoint(cleaned_item):
                processed_data.append(cleaned_item)
            else:
                rejected_count += 1
                print(f"⚠ Rejected datapoint {i}: Failed validation")
        else:
            processed_data.append(cleaned_item)
    
    print(f"\n✓ Processed {len(processed_data)} valid datapoints")
    if rejected_count > 0:
        print(f"⚠ Rejected {rejected_count} invalid datapoints")
    
    # Save processed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved processed data to: {output_file}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    avg_input_len = sum(len(d["input"]) for d in processed_data) / len(processed_data)
    avg_output_len = sum(len(d["output"]) for d in processed_data) / len(processed_data)
    
    print(f"Total samples: {len(processed_data)}")
    print(f"Average input length: {avg_input_len:.0f} characters")
    print(f"Average output length: {avg_output_len:.0f} characters")


# ============================================================================
# DATASET AUGMENTATION
# ============================================================================

def augment_vital_signs(datapoint: Dict[str, str], variation: float = 0.1) -> Dict[str, str]:
    """
    Create augmented version by slightly varying vital signs
    
    Args:
        datapoint: Original datapoint
        variation: Percentage variation (0.1 = 10%)
    
    Returns:
        Augmented datapoint
    """
    import random
    
    new_datapoint = datapoint.copy()
    input_text = new_datapoint["input"]
    
    # Find and vary numerical values (simple approach)
    def vary_number(match):
        num = float(match.group(0))
        variation_amount = num * variation * random.choice([-1, 1])
        new_num = num + variation_amount
        # Round appropriately
        if '.' in match.group(0):
            return f"{new_num:.1f}"
        else:
            return f"{int(new_num)}"
    
    # This is a simple regex-based approach
    # In production, you'd want more sophisticated parsing
    augmented_input = re.sub(r'\d+\.?\d*', vary_number, input_text)
    
    new_datapoint["input"] = augmented_input
    
    return new_datapoint


# ============================================================================
# SPECIALIZED FORMATTERS
# ============================================================================

def create_jaundice_scenario(
    age_hours: int,
    bilirubin_level: float,
    risk_factors: List[str]
) -> NICUDatapoint:
    """
    Create a neonatal jaundice scenario
    
    Args:
        age_hours: Age in hours
        bilirubin_level: Total serum bilirubin (mg/dL)
        risk_factors: List of risk factors
    
    Returns:
        NICUDatapoint
    """
    instruction = "Assess neonatal jaundice and recommend management based on AAP guidelines."
    
    input_data = f"""
Patient: Neonate, {age_hours} hours old
Laboratory Results:
- Total Serum Bilirubin: {bilirubin_level} mg/dL

Risk Factors:
{chr(10).join(f'- {rf}' for rf in risk_factors)}
"""
    
    # Simple rule-based output (in real scenario, use AAP nomogram)
    if bilirubin_level > 15:
        output = f"""
ASSESSMENT: Hyperbilirubinemia requiring intervention

RECOMMENDATIONS per AAP Guidelines:
1. Initiate phototherapy immediately
2. Recheck TSB in 4-6 hours
3. Ensure adequate hydration and feeding
4. Monitor for signs of acute bilirubin encephalopathy
5. Consider exchange transfusion if TSB > 20 mg/dL or signs of ABE
6. Order blood type, Coombs test if not already done

RATIONALE: TSB of {bilirubin_level} mg/dL at {age_hours} hours exceeds phototherapy threshold on AAP nomogram.
"""
    else:
        output = f"""
ASSESSMENT: Physiological jaundice within acceptable range

RECOMMENDATIONS per AAP Guidelines:
1. Continue monitoring clinically
2. Recheck TSB in 12-24 hours
3. Ensure adequate feeding (8-12 times per 24 hours)
4. Educate parents on jaundice monitoring
5. Schedule follow-up within 2-3 days

RATIONALE: TSB of {bilirubin_level} mg/dL at {age_hours} hours is below phototherapy threshold.
"""
    
    return NICUDatapoint(instruction, input_data.strip(), output.strip())


def create_respiratory_distress_scenario(
    gestational_age: int,
    respiratory_rate: int,
    spo2: int,
    fio2: float,
    retractions: bool
) -> NICUDatapoint:
    """
    Create a respiratory distress scenario
    
    Args:
        gestational_age: Gestational age in weeks
        respiratory_rate: Breaths per minute
        spo2: Oxygen saturation (%)
        fio2: Fraction of inspired oxygen (0-1)
        retractions: Presence of retractions
    
    Returns:
        NICUDatapoint
    """
    instruction = "Evaluate respiratory distress in a neonate and provide management recommendations."
    
    input_data = f"""
Patient: Neonate, gestational age {gestational_age} weeks
Vital Signs:
- Respiratory Rate: {respiratory_rate} breaths/min
- SpO2: {spo2}%
- FiO2: {fio2:.2f}
- Clinical: {"Intercostal and subcostal retractions present" if retractions else "No retractions"}
"""
    
    # Calculate severity
    severity_score = 0
    if respiratory_rate > 60:
        severity_score += 1
    if spo2 < 90:
        severity_score += 2
    if fio2 > 0.4:
        severity_score += 1
    if retractions:
        severity_score += 1
    
    if severity_score >= 3:
        output = f"""
ASSESSMENT: Moderate to severe respiratory distress

IMMEDIATE ACTIONS:
1. Increase respiratory support - consider CPAP or mechanical ventilation
2. Target SpO2 90-95% (adjust FiO2 as needed)
3. Obtain chest X-ray
4. Order arterial blood gas
5. Consider surfactant administration if preterm with RDS
6. Rule out pneumothorax, sepsis, TTN

MONITORING:
- Continuous pulse oximetry
- Respiratory rate and work of breathing every 15-30 minutes
- Blood gas in 1 hour

DIFFERENTIAL: RDS, TTN, pneumothorax, sepsis, PPHN
"""
    else:
        output = f"""
ASSESSMENT: Mild respiratory distress

MANAGEMENT:
1. Maintain current FiO2, target SpO2 90-95%
2. Monitor respiratory status closely
3. Consider chest X-ray if not improving
4. Evaluate for infection - CBC, blood culture if indicated
5. Supportive care

MONITORING:
- Pulse oximetry continuous
- Reassess in 1-2 hours
- Document trend in respiratory effort

DIFFERENTIAL: Mild RDS, TTN, early sepsis
"""
    
    return NICUDatapoint(instruction, input_data.strip(), output.strip())


# ============================================================================
# BATCH GENERATION
# ============================================================================

def generate_sample_dataset(num_samples: int = 100) -> List[Dict[str, str]]:
    """
    Generate sample NICU scenarios for demonstration
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        List of datapoint dictionaries
    """
    import random
    
    print("Generating sample dataset...")
    
    datapoints = []
    
    # Generate jaundice scenarios
    for _ in range(num_samples // 2):
        age = random.randint(24, 120)
        bili = random.uniform(5, 20)
        risks = random.sample([
            "Gestational age 35-36 weeks",
            "Exclusive breastfeeding",
            "Family history of jaundice",
            "Cephalohematoma",
            "East Asian ethnicity"
        ], k=random.randint(1, 3))
        
        dp = create_jaundice_scenario(age, bili, risks)
        datapoints.append(dp.to_dict())
    
    # Generate respiratory scenarios
    for _ in range(num_samples // 2):
        ga = random.randint(28, 40)
        rr = random.randint(40, 80)
        spo2 = random.randint(75, 98)
        fio2 = random.uniform(0.21, 0.8)
        retractions = random.choice([True, False])
        
        dp = create_respiratory_distress_scenario(ga, rr, spo2, fio2, retractions)
        datapoints.append(dp.to_dict())
    
    print(f"✓ Generated {len(datapoints)} sample scenarios")
    
    return datapoints


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    
    # Create directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Generate sample data (for demonstration)
    sample_data = generate_sample_dataset(num_samples=200)
    
    sample_file = "data/raw/sample_nicu_data.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved sample data to: {sample_file}")
    
    # Preprocess the data
    preprocess_nicu_data(
        input_file=sample_file,
        output_file="data/processed/nicu_training_data.json",
        validate=True
    )
    
    print("\n" + "=" * 80)
    print("✅ DATA PREPROCESSING COMPLETED")
    print("=" * 80)
    print("\nProcessed data is ready for training!")
    print("Run: python train_nicu_llama.py")


if __name__ == "__main__":
    main()
