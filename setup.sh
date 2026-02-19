#!/bin/bash
# Setup script for NICU-GPT (Linux/Mac)
# For Windows, use setup.bat

echo "=========================================="
echo "NICU-GPT Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
    echo "ERROR: Python 3.10 or higher required"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Check CUDA
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✓ CUDA available"
else
    echo "⚠ Warning: No NVIDIA GPU detected. Training will be very slow on CPU."
    read -p "Continue anyway? (y/n): " continue_cpu
    if [ "$continue_cpu" != "y" ]; then
        echo "Setup cancelled"
        exit 1
    fi
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/evaluation
mkdir -p output
mkdir -p results
echo "✓ Directories created"
echo ""

# Create .env file from template
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "✓ .env file created"
    echo ""
    echo "⚠ IMPORTANT: Edit .env file and add your API keys"
    echo "  - OPENAI_API_KEY (for GPT-4 evaluation)"
    echo "  - WANDB_API_KEY (optional, for experiment tracking)"
else
    echo "✓ .env file already exists"
fi
echo ""

# Check disk space
echo "Checking disk space..."
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo "Available space: $available_space"
echo "Required: ~50GB for models and data"
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Generate sample data:"
echo "   python preprocess_data.py"
echo ""
echo "3. Train model:"
echo "   python train_nicu_llama.py"
echo ""
echo "4. (Optional) Evaluate with GPT-4:"
echo "   export OPENAI_API_KEY=sk-..."
echo "   python evaluate_model.py"
echo ""
echo "5. Export to GGUF:"
echo "   python export_to_gguf.py"
echo ""
echo "For detailed instructions, see:"
echo "  - README.md (complete documentation)"
echo "  - QUICKSTART.md (quick start guide)"
echo ""
echo "Happy training! 🚀"
