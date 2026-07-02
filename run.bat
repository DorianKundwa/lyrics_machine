@echo off
setlocal enabledelayedexpansion
title LyricForge Server
echo =========================================
echo       Starting LyricForge Server...
echo =========================================

:: Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please ensure the project has been set up with its dependencies.
    pause
    exit /b 1
)

:: Find an available port
echo Looking for an open port...
for /f "delims=" %%a in ('.\venv\Scripts\python.exe -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"') do set "PORT=%%a"

echo Found open port: %PORT%

:: Open the browser automatically
echo Opening browser at http://localhost:%PORT%...
start http://localhost:%PORT%

:: Start the FastAPI server using the virtual environment's python
echo Starting Uvicorn on port %PORT%...
.\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port %PORT% --reload

:: Keep window open if the server crashes or stops
pause
