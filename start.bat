@echo off
echo ===================================================
echo Starting Embrained App
echo ===================================================

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo Select Engine Mode:
echo [1] Discrete Markov (Default)
echo [2] Continuous Action Chunking
set /p mode="Enter choice (1 or 2): "

set ENGINE_FLAG=
if "%mode%"=="2" (
    set ENGINE_FLAG=--continuous
    echo Launching Continuous Action Chunking engine...
) else (
    echo Launching Discrete Markov engine...
)

call venv\Scripts\activate.bat

echo.
echo Server starting on port 8080...
echo Please navigate to http://localhost:8080 in your default web browser.
echo.

python cognitive-engine\app.py --port 8080 --plexus %ENGINE_FLAG%
pause
