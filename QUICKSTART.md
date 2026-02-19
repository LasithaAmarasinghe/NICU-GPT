# Quick Start Guide for NICU-GPT

## 5-Minute Setup (Google Colab)

If you want to try NICU-GPT immediately without local setup:

### 1. Open Google Colab
- Go to [Google Colab](https://colab.research.google.com)
- Create new notebook
- Select GPU runtime: Runtime → Change runtime type → GPU (T4)

### 2. Install Dependencies

```python
!pip install unsloth[colab-new] torch transformers datasets trl
```

### 3. Generate Sample Data

```python
!git clone https://github.com/yourusername/nicu-gpt.git
%cd nicu-gpt
!python preprocess_data.py
```

### 4. Train the Model

```python
# This will take ~4 hours on T4 GPU
!python train_nicu_llama.py
```

### 5. Test Inference

```python
from train_nicu_llama import *

config = TrainingConfig()
model, tokenizer = setup_model_and_tokenizer(config)
test_inference(model, tokenizer)
```

---

## Local Setup (Windows/Linux)

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.1+

### Step-by-Step

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/nicu-gpt.git
cd nicu-gpt
```

#### 2. Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Generate Sample Data
```bash
python preprocess_data.py
```

Output: Creates 200 synthetic NICU scenarios in `data/processed/`

#### 5. Train Model (Optional - Skip if using pre-trained)
```bash
python train_nicu_llama.py
```

Training time:
- RTX 3090 (24GB): ~2.5 hours
- RTX 4090 (24GB): ~2 hours
- Tesla T4 (16GB): ~4 hours

#### 6. Export to GGUF (for local use)
```bash
python export_to_gguf.py
```

Output: GGUF files in `output/nicu-llama-gguf/`

#### 7. Run with Ollama (Easiest)
```bash
cd output/nicu-llama-gguf
ollama create nicu-llama -f Modelfile
ollama run nicu-llama
```

---

## Quick Test

### Test with Python

```python
from train_nicu_llama import format_nicu_prompt, setup_model_and_tokenizer
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    "output/nicu-llama-qlora/final_model",
    max_seq_length=2048,
    load_in_4bit=True
)

FastLanguageModel.for_inference(model)

# Test query
instruction = "Analyze vital signs and provide recommendations"
input_text = """
Patient: 48-hour-old term neonate
HR: 180 bpm, SpO2: 88%, RR: 72/min
Temperature: 36.1°C
Clinical: Mild retractions, dusky appearance
"""

prompt = format_nicu_prompt(instruction, input_text, "")
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.batch_decode(outputs)[0]

print(response.split("### Response:")[-1])
```

### Test with Ollama

```bash
ollama run nicu-llama
```

Then type:
```
Analyze vital signs:

Patient: 2-day-old neonate
HR: 175 bpm
SpO2: 88%
RR: 68/min
Temperature: 36.2°C
Clinical: Mild intercostal retractions
```

---

## Evaluation

### Requirements
- OpenAI API key (for GPT-4 evaluation)

### Steps

1. Set API key:
```bash
# Windows
set OPENAI_API_KEY=sk-your-key-here

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
```

2. Run evaluation:
```bash
python evaluate_model.py
```

3. View results:
```bash
# Results saved to results/
cat results/summary_*.json
```

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in train_nicu_llama.py
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
```

### Import Errors
```bash
# Update dependencies
pip install --upgrade torch transformers unsloth
```

### Model Loading Issues
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### Slow Training
- Ensure GPU is being used (check with `nvidia-smi`)
- Use smaller max_seq_length (1024 instead of 2048)
- Enable gradient checkpointing (already enabled)

---

## Next Steps

1. **Create Your Own Data**: Replace sample data with real AAP guideline scenarios
2. **Fine-tune Further**: Increase epochs or add more data
3. **Deploy**: Export to GGUF and use with LM Studio/Ollama
4. **Evaluate**: Use GPT-4 to compare against base model
5. **Contribute**: Share improved models or datasets

---

## Pro Tips

1. **Start Small**: Use sample data first to verify everything works
2. **Monitor Training**: Use wandb to track metrics (set USE_WANDB=True)
3. **Save Checkpoints**: Training saves every 50 steps automatically
4. **Test Often**: Test inference after each epoch to catch issues early
5. **Version Control**: Use git to track your experiments

---
