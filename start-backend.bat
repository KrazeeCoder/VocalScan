@echo off
setlocal enabledelayedexpansion

rem Ensure we are in the repo root (directory of this script)
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "WIN_PY=%VENV_DIR%\Scripts\python.exe"
set "NIX_PY=%VENV_DIR%\bin\python"

rem Recreate venv if invalid/missing
if not exist "%VENV_DIR%\pyvenv.cfg" (
  if exist "%VENV_DIR%" (
    echo Detected invalid virtual environment. Removing "%VENV_DIR%"...
    rmdir /s /q "%VENV_DIR%"
  )
  echo Creating virtual environment...
  where py >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    py -3 -m venv "%VENV_DIR%"
  ) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
      python -m venv "%VENV_DIR%"
    ) else (
      echo ERROR: Python is not installed or not on PATH. Please install Python 3.10+ and re-run.
      goto :EOF
    )
  )
)

if not exist "%VENV_DIR%\pyvenv.cfg" (
  echo ERROR: Failed to create virtual environment at %VENV_DIR%. Aborting.
  goto :EOF
)

set "PYEXE="
if exist "%WIN_PY%" set "PYEXE=%WIN_PY%"
if not defined PYEXE if exist "%NIX_PY%" set "PYEXE=%NIX_PY%"
if not defined PYEXE (
  echo ERROR: Could not locate Python inside the virtual environment.
  goto :EOF
)

echo Activating virtual environment...
if exist "%VENV_DIR%\Scripts\activate.bat" (
  call "%VENV_DIR%\Scripts\activate.bat"
) else (
  echo Warning: activate.bat not found, continuing without activation.
)

echo Installing backend requirements...
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r "backend\requirements.txt"

echo Starting backend on http://127.0.0.1:8080 ...
set FLASK_APP=backend.app.main
"%PYEXE%" -m flask run --host=0.0.0.0 --port=8080

endlocal

