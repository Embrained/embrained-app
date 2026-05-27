@echo off
echo ===================================================
echo Embrained Setup Script (Windows)
echo ===================================================

echo Checking dependencies...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.12.
    exit /b 1
)

node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH. Please install Node.js.
    exit /b 1
)

echo.
echo Setting up Python virtual environment...
if not exist "venv" (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo.
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setting up Node.js frontend...
cd cognitive-engine\frontend
call npm install
echo Building frontend for production...
call npm run build
cd ..\..

echo.
echo ===================================================
echo Setup Complete!
echo You can now run the app using: start.bat
echo ===================================================
pause
