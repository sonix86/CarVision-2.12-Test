@echo off
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0APPLY_UPDATE_2_3.ps1"
set "result=%errorlevel%"
pause
exit /b %result%
