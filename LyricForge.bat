@echo off
title LyricForge x Croonify AI
chcp 65001 >nul 2>&1

echo.
echo ======================================================
echo   LyricForge  x  Croonify AI
echo   Neural lyrics-to-video generator
echo ======================================================
echo.

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

python launch.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited with an error. See messages above.
    pause
)
