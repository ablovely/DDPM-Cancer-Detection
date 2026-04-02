@echo off
echo ============================================================
echo 开始训练阳性模型 (100步)
echo ============================================================
echo.

cd /d D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\scripts

call conda activate pytorch_latest

echo.
echo Python environment activated
echo.

python quick_train.py

echo.
echo ============================================================
echo Training complete
echo ============================================================
pause
