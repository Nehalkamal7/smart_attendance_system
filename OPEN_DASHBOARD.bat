@echo off
title Open Dashboard
color 0B

echo.
echo  ========================================
echo   Checking Dashboard Server Status...
echo  ========================================
echo.

:: Check if port 5000 is in use (Server is running)
netstat -ano | findstr :5000 >nul
if %errorlevel% neq 0 (
    echo  [!] Server is not running. Starting it now...
    start "Smart Attendance Server" /MIN cmd /c "title Smart Attendance Server && python dashboard.py"
    :: Wait 3 seconds for the server to fully start
    timeout /t 3 /nobreak >nul
) else (
    echo  [+] Server is already running.
)

echo.
echo  [+] Opening Dashboard in your browser...
start "" "http://127.0.0.1:5000"
