# Training Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting Positive Model Training (100 steps)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Change to script directory
Set-Location "D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\scripts"

# Activate conda environment
Write-Host "`nActivating conda environment..." -ForegroundColor Yellow
conda activate pytorch_latest

Write-Host "`nPython environment activated" -ForegroundColor Green
Write-Host "`nStarting training..." -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan

# Run training script
python quick_train.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Training complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# Pause to view output
Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
