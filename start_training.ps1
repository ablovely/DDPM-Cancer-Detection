# 训练脚本启动器 - 带资源监控

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "阳性模型快速训练 (100步) - 带资源监控" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 激活conda环境
Write-Host "`n激活conda环境..." -ForegroundColor Yellow
conda activate pytorch_latest

# 切换到脚本目录
Set-Location "D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\scripts"

Write-Host "环境已激活，开始训练..." -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan

# 启动Python训练脚本
python train_with_monitor.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "训练完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

# 暂停以查看输出
Write-Host "`n按任意键退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
