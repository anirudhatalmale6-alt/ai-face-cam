@echo off
call venv\Scripts\activate.bat
python face_cam_3d.py -i %1 %2 %3 %4
