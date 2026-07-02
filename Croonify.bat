@echo off
title Croonify -- AI Lyrics Sync Engine
chcp 65001 >nul 2>&1

echo.
echo ================================================
echo   Croonify  --  AI Lyrics Sync Engine
echo ================================================
echo.

:: Change to the directory containing this batch file
cd /d "%~dp0"

:: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

:: Run the launcher
python launch.py %*

:: If the server exits non-zero, pause so the user can read the error
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error. See messages above.
    pause
)
