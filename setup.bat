@echo off
echo ========================================
echo   AI Face Cam v2 - Setup
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

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install opencv-python numpy pyvirtualcam onnxruntime

echo.
echo For NVIDIA GPU acceleration (much faster), also run:
echo   pip install onnxruntime-gpu
echo.

echo [3/3] AI models will download on first run.

echo.
echo ========================================
echo   Setup complete!
echo   Run: run.bat [face_photo.png]
echo   Or just: run.bat (for face gallery)
echo ========================================
pause
