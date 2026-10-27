@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RESTORE_FULL.ps1"
if errorlevel 1 echo Restore failed. Read the message above.
pause
