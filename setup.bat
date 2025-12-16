@echo off
REM This batch file executes the main PowerShell setup script.

powershell.exe -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

echo.
pause