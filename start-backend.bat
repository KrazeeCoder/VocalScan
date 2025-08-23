@echo off
echo Starting VocalScan Backend...
cd backend
call ..\\.venv\\Scripts\\activate.bat
pip install -r requirements.txt
python app\\main.py
