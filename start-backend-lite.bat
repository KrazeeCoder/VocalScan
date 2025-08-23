@echo off
echo Installing basic Flask dependencies...
pip install flask flask-cors

echo Starting VocalScan Backend (Lite Version)...
cd backend
python simple_main.py
