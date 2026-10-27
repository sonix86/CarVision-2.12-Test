@echo off
chcp 65001 >nul
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0CHECK_ON_NEW_PC.ps1" %*
set taskCheckExit=%errorlevel%
pause
exit /b %taskCheckExit%
