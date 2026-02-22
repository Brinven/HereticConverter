@echo off
cd /d "%~dp0"
echo Starting Heretic Converter setup...
echo Working directory: %CD%
echo.

if exist "venv\Scripts\activate.bat" goto :activate

echo Creating virtual environment...
python -m venv venv
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo Installing PyTorch with CUDA 12.8 (nightly — required for RTX 50-series)...
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

echo Installing dependencies...
pip install -r requirements.txt

goto :run

:activate
call venv\Scripts\activate.bat

:run
echo.
echo Launching app...
python app.py
echo.
echo App exited.
pause
