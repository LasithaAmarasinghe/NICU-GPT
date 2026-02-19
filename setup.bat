@echo off
REM Setup script for NICU-GPT (Windows)

echo ==========================================
echo NICU-GPT Setup Script (Windows)
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

echo Python found
echo.

REM Check CUDA
echo Checking CUDA availability...
nvidia-smi > nul 2>&1
if %errorlevel% equ 0 (
    echo GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo CUDA available
) else (
    echo WARNING: No NVIDIA GPU detected. Training will be very slow on CPU.
    set /p continue_cpu="Continue anyway? (y/n): "
    if /i not "%continue_cpu%"=="y" (
        echo Setup cancelled
        pause
        exit /b 1
    )
)
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo pip upgraded
echo.

REM Install requirements
echo Installing dependencies (this may take 5-10 minutes)...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Create directories
echo Creating project directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\evaluation" mkdir data\evaluation
if not exist "output" mkdir output
if not exist "results" mkdir results
echo Directories created
echo.

REM Create .env file from template
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.template .env
    echo .env file created
    echo.
    echo IMPORTANT: Edit .env file and add your API keys
    echo   - OPENAI_API_KEY (for GPT-4 evaluation)
    echo   - WANDB_API_KEY (optional, for experiment tracking)
) else (
    echo .env file already exists
)
echo.

REM Summary
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Generate sample data:
echo    python preprocess_data.py
echo.
echo 3. Train model:
echo    python train_nicu_llama.py
echo.
echo 4. (Optional) Evaluate with GPT-4:
echo    set OPENAI_API_KEY=sk-...
echo    python evaluate_model.py
echo.
echo 5. Export to GGUF:
echo    python export_to_gguf.py
echo.
echo For detailed instructions, see:
echo   - README.md (complete documentation)
echo   - QUICKSTART.md (quick start guide)
echo.
echo Happy training! 🚀
echo.
pause
