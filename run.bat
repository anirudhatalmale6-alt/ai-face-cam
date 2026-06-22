@echo off
call venv\Scripts\activate.bat
python live_portrait_onnx.py %1 %2 %3 %4
