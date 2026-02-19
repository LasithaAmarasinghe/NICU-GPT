# Data Synthesis Guide for NICU-GPT

## Overview

To create high-quality training data for NICU-Llama, you'll need to synthesize clinical scenarios from AAP guidelines and medical literature.

---

## Phase 1: Source Material Collection

### Recommended Sources

1. **AAP Clinical Practice Guidelines**
   - Neonatal Hyperbilirubinemia
   - Respiratory Support in Preterm Infants
   - Early-Onset Sepsis
   - Neonatal Resuscitation
   - Download from: [AAP Publications](https://publications.aap.org/)

2. **WHO Guidelines**
   - Pocket Book of Hospital Care for Children
   - Managing Newborn Problems
   - Available at: [WHO Publications](https://www.who.int/)

3. **PubMed Open Access**
   - Search: "neonatal intensive care" + "guideline"
   - Filter: Free full text
   - Focus: rPPG, vital signs monitoring

4. **Clinical Textbooks**
   - Avery's Neonatology
   - Cloherty and Stark's Manual of Neonatal Care

---

## Phase 2: Synthetic Data Generation

### Method 1: GPT-4 / Claude Synthesis

Use a large language model to convert guidelines into Q&A format.

#### Prompt Template

```
You are a neonatology expert. Based on the following AAP clinical 
practice guideline, create 20 diverse clinical scenarios in JSON format.

GUIDELINE:
[Paste AAP guideline text here]

For each scenario, provide:
1. instruction: The clinical task
2. input: Patient details, vital signs, labs
3. output: Step-by-step management per AAP guidelines

Requirements:
- Vary patient ages (hours to days)
- Include diverse vital sign ranges
- Cover mild, moderate, severe cases
- Include edge cases and contraindications
- Be specific with numbers and thresholds
- Reference AAP guidelines explicitly

Format as JSON array:
[
  {
    "instruction": "...",
    "input": "...",
    "output": "..."
  }
]
```

#### Example Execution

```python
import anthropic
import json

client = anthropic.Anthropic(api_key="your-key")

with open("aap_jaundice_guideline.txt", "r") as f:
    guideline = f.read()

prompt = f"""You are a neonatology expert... [template above]

GUIDELINE:
{guideline}
"""

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8000,
    messages=[{"role": "user", "content": prompt}]
)

scenarios = json.loads(response.content[0].text)

with open("jaundice_scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=2)
```

---

### Method 2: Manual Expert Creation

Work with neonatologists to create scenarios.

#### Template Form

```markdown
## Clinical Scenario Template

**Scenario ID**: [Unique ID]
**Topic**: [e.g., Hyperbilirubinemia, RDS, Sepsis]
**Complexity**: [Simple / Moderate / Complex]

---

**Instruction**:
[What should the model do? e.g., "Assess jaundice and recommend management"]

**Patient Information**:
- Age: [hours/days old]
- Gestational age: [weeks]
- Birth weight: [grams]
- Delivery type: [vaginal/C-section]

**Vital Signs**:
- Heart rate: [bpm]
- SpO2: [%]
- Respiratory rate: [breaths/min]
- Temperature: [°C]
- Blood pressure: [mmHg]

**Laboratory Results** (if applicable):
- [Test]: [Value]

**Clinical Context**:
[Additional observations, symptoms, history]

---

**Expected Output** (per AAP Guidelines):

**Assessment**:
[Clinical diagnosis/impression]

**Immediate Actions**:
1. [First action with rationale]
2. [Second action...]

**Monitoring**:
[What to monitor and when]

**Rationale**:
[Why these recommendations, reference to AAP guidelines]

---

**AAP Guideline Reference**: [Specific citation]
**Reviewer**: [Name, credentials]
**Date**: [YYYY-MM-DD]
```

---

### Method 3: Augmentation from Case Reports

Extract and augment from published case reports.

#### Python Script

```python
import json
import random

def augment_case(base_case):
    """Create variations of a base clinical case"""
    
    variations = []
    
    # Original
    variations.append(base_case)
    
    # Vary vital signs ±10%
    for _ in range(5):
        variant = base_case.copy()
        
        # Simple regex-based augmentation
        # In production, use proper parsing
        if "Heart rate:" in variant["input"]:
            hr = int(re.search(r"Heart rate: (\d+)", variant["input"]).group(1))
            new_hr = int(hr * random.uniform(0.9, 1.1))
            variant["input"] = re.sub(
                r"Heart rate: \d+",
                f"Heart rate: {new_hr}",
                variant["input"]
            )
        
        variations.append(variant)
    
    return variations
```

---

## Phase 3: Quality Control

### Validation Checklist

For each scenario, verify:

- [ ] **Medically Accurate**: Facts align with AAP guidelines
- [ ] **Complete**: All necessary information provided
- [ ] **Safe**: No dangerous recommendations
- [ ] **Specific**: Concrete numbers, not vague descriptions
- [ ] **Actionable**: Clear next steps for clinicians
- [ ] **Referenced**: Cites AAP guideline when appropriate

### Automated Validation

```python
def validate_scenario(scenario):
    """Automated validation checks"""
    
    errors = []
    
    # Check required fields
    required = ["instruction", "input", "output"]
    for field in required:
        if not scenario.get(field):
            errors.append(f"Missing {field}")
    
    # Check vital signs present
    input_text = scenario.get("input", "").lower()
    vital_signs = ["heart rate", "spo2", "respiratory rate", "temperature"]
    
    found_vitals = sum(1 for vs in vital_signs if vs in input_text)
    if found_vitals < 2:
        errors.append("Insufficient vital sign data")
    
    # Check output length (should be substantial)
    if len(scenario.get("output", "")) < 100:
        errors.append("Output too brief")
    
    # Check for AAP reference
    if "aap" not in scenario.get("output", "").lower():
        errors.append("No AAP guideline reference")
    
    return errors
```

---

## Phase 4: Dataset Organization

### Recommended Structure

```
data/
├── raw/
│   ├── aap_guidelines/
│   │   ├── jaundice.pdf
│   │   ├── respiratory.pdf
│   │   └── sepsis.pdf
│   └── synthesis_outputs/
│       ├── jaundice_scenarios.json
│       └── respiratory_scenarios.json
│
├── processed/
│   ├── nicu_training_data.json      # Final training set
│   └── metadata.json                 # Dataset statistics
│
└── evaluation/
    └── golden_set.json               # Expert-validated test set
```

### Combining Datasets

```python
import json
from pathlib import Path

def combine_datasets(input_dir, output_file):
    """Combine multiple scenario files"""
    
    all_scenarios = []
    
    for json_file in Path(input_dir).glob("*.json"):
        with open(json_file) as f:
            scenarios = json.load(f)
            all_scenarios.extend(scenarios)
    
    # Shuffle for diversity
    random.shuffle(all_scenarios)
    
    # Save combined
    with open(output_file, "w") as f:
        json.dump(all_scenarios, f, indent=2)
    
    print(f"Combined {len(all_scenarios)} scenarios")

combine_datasets("data/raw/synthesis_outputs", "data/processed/nicu_training_data.json")
```

---

## Best Practices

### 1. Diversity

- Cover full range of gestational ages (preterm to term)
- Include various severity levels (mild, moderate, severe)
- Vary vital sign values realistically
- Include edge cases and contraindications

### 2. Realism

- Use realistic vital sign ranges for age
- Include contextual information (delivery type, maternal history)
- Avoid oversimplified scenarios
- Include comorbidities when relevant

### 3. Guidelines Adherence

- Always reference specific AAP guidelines
- Include threshold values from nomograms
- Mention alternative approaches when appropriate
- Note when expert consultation is needed

### 4. Safety

- Never suggest harmful interventions
- Include monitoring parameters
- Mention contraindications
- Emphasize when immediate action is needed

---

## Example High-Quality Scenario

```json
{
  "instruction": "Evaluate this preterm neonate with suspected RDS and recommend respiratory management per AAP guidelines.",
  
  "input": "Patient: Preterm male, 30 weeks gestational age, 2 hours old\nBirth weight: 1450g\nDelivery: Emergency C-section for preeclampsia\nAntenatal steroids: Single dose 18 hours before delivery\n\nVital Signs:\n- Heart Rate: 168 bpm\n- Respiratory Rate: 78 breaths/min\n- SpO2: 84% on room air\n- Temperature: 36.2°C (warming under radiant heater)\n- Blood Pressure: 42/26 mmHg (mean 31)\n\nClinical Findings:\n- Nasal flaring present\n- Moderate intercostal and subcostal retractions\n- Grunting with each breath\n- Symmetric air entry bilaterally with fine crackles\n- Silverman-Andersen score: 7/10",
  
  "output": "ASSESSMENT: Respiratory Distress Syndrome (RDS) - Moderate severity\n\nRisk factors:\n- Extreme prematurity (30 weeks)\n- Incomplete antenatal steroid course\n- Typical presentation at 2 hours of life\n\nIMMEDIATE RESPIRATORY MANAGEMENT:\n\n1. **CPAP Initiation**\n   - Start nasal CPAP at 6 cmH2O\n   - FiO2 titrate to maintain SpO2 90-94%\n   - Prefer bubble CPAP if available\n   - Target per AAP/NRP: Avoid hyperoxia in preterm\n\n2. **Surfactant Administration - INDICATED**\n   - Criteria met: 30 weeks GA + RDS + Oxygen requirement\n   - Timing: Early administration (consider INSURE technique)\n   - Options: Poractant alfa 200 mg/kg or Beractant 100 mg/kg\n   - Technique: INSURE (Intubate-Surfactant-Extubate) if stable\n   - Post-surfactant: Back to CPAP, monitor for improvement\n\n3. **Diagnostic Studies**\n   - Chest X-ray STAT: Expect ground-glass appearance, air bronchograms\n   - Arterial/capillary blood gas\n   - Complete blood count, blood culture (rule out sepsis)\n   - Glucose monitoring\n\n4. **Supportive Care**\n   - Maintain thermal neutrality (current temp acceptable)\n   - IV access: Two peripheral IVs or UVC placement\n   - Fluids: 60-80 mL/kg/day D10W initially\n   - NPO, orogastric tube for decompression\n   - Minimal stimulation\n\n5. **Infection Prophylaxis**\n   - Consider empiric antibiotics: Ampicillin + Gentamicin\n   - RDS can present similarly to early-onset sepsis\n   - Discontinue at 48h if cultures negative and improving\n\n6. **Monitoring Plan**\n   - Continuous pulse oximetry and heart rate\n   - Respiratory rate and work of breathing every 15 min initially\n   - Blood pressure every 30 min × 4, then hourly\n   - Blood gas 30-60 min after CPAP initiation\n   - Chest X-ray 6-12 hours or if deterioration\n   - Serial exams for pneumothorax\n\n7. **Escalation Criteria**\n   - If FiO2 >0.4 on CPAP to maintain SpO2\n   - If persistent acidosis (pH <7.25, pCO2 >60)\n   - If increasing work of breathing despite CPAP\n   - Consider: Intubation and mechanical ventilation\n\nRATIONALE:\nThis infant has classic RDS presentation. AAP/NRP guidelines recommend:\n- Early CPAP for preterm infants with respiratory distress\n- Surfactant for infants <32 weeks with oxygen requirement\n- Target SpO2 90-94% (per SUPPORT trial)\n- Gentle ventilation strategies to minimize BPD risk\n\nIncomplete antenatal steroids increase RDS severity. Early surfactant via INSURE technique may prevent need for prolonged ventilation. Close monitoring essential as RDS typically worsens in first 48-72 hours before improvement.\n\nREFERENCE: AAP Committee on Fetus and Newborn, \"Respiratory Support in Preterm Infants at Birth\" (2014)"
}
```

---

## Troubleshooting

**Q: Generated scenarios seem too simple**
- Use more complex prompt templates
- Request multiple comorbidities
- Ask for edge cases specifically

**Q: Outputs lack clinical detail**
- Provide detailed examples in prompt
- Request specific vital sign ranges
- Ask for AAP guideline citations

**Q: Inconsistent quality**
- Implement automated validation
- Have clinical experts review samples
- Use temperature=0.7 for less randomness

---

**Need Help?** Consult with neonatologists to ensure medical accuracy.
