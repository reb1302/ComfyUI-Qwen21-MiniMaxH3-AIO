@echo off
chcp 65001 >nul
echo ============================================================================
echo 🚀 [Lightricks LTX-2.5 + Creative Lab] Cài Đặt Custom Nodes, Pip ^& Tải Model
echo ============================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_lighttrick_models.ps1" %*
pause
