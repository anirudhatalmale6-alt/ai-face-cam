@echo off
echo ========================================
echo   AI Face Cam - Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Please install Python 3.10+ from python.org
    echo Make sure to check "Add to PATH" during installation
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/4] Installing PyTorch with CUDA...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo [3/4] Installing dependencies...
pip install opencv-python numpy scipy pillow pyvirtualcam onnxruntime-gpu insightface

echo [4/4] Downloading face models...
python download_models.py

echo.
echo ========================================
echo   Setup complete!
echo   Run: run.bat face_photo.png
echo ========================================
pause
