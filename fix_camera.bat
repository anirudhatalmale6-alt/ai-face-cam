@echo off
cd /d "%~dp0"
set PATH=%CD%\venv\lib\site-packages\torch\lib;%PATH%
venv\python.exe fix_camera.py
pause
