@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0CHECK_ON_NEW_PC.ps1" %*
set taskCheckExit=%errorlevel%
pause
exit /b %taskCheckExit%
