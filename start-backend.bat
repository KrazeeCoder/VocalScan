@echo off
setlocal

echo Creating venv if missing...
if not exist .venv ( python -m venv .venv )

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing backend requirements...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt

echo Starting backend on http://127.0.0.1:8080 ...
set FLASK_APP=backend.app.main
.venv\Scripts\python.exe -m flask run --host=0.0.0.0 --port=8080

endlocal

