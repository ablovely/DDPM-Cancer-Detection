@echo off
echo ============================================================
echo Starting Positive Model Training (100 steps)
echo ============================================================
echo.

cd /d D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\scripts

call conda activate pytorch_latest

echo.
echo Python environment activated
echo.

python quick_train_en.py

echo.
echo ============================================================
echo Training complete
echo ============================================================
pause
