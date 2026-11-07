@echo off
rem Watt2Bit node for RTX 4060 Laptop (Windows). Double-click after git clone.
rem Options: -Reconfigure  -CheckOnly  -SkipCheck
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\node_windows.ps1" %*
echo.
pause
