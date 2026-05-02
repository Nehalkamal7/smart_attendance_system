@echo off
title Smart Attendance System
color 0A
cls

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║        SMART ATTENDANCE SYSTEM  - Launcher               ║
echo  ║        Face Recognition  +  Real-Time Tracking           ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.
echo  [1] Start Main System (Camera + CV)
echo  [2] Start Web Dashboard only
echo  [3] Start Both (Main + Dashboard)
echo  [4] Install / Update Dependencies
echo  [5] Exit
echo.
set /p CHOICE=  Select option [1-5]: 

if "%CHOICE%"=="1" goto MAIN
if "%CHOICE%"=="2" goto DASH
if "%CHOICE%"=="3" goto BOTH
if "%CHOICE%"=="4" goto INSTALL
if "%CHOICE%"=="5" exit
goto MAIN

:INSTALL
echo.
echo  Installing dependencies...
pip install -r requirements.txt
echo.
echo  Done! Press any key to return to menu.
pause >nul
goto START

:MAIN
echo.
echo  Starting Main System...
python main.py
pause
exit

:DASH
echo.
echo  Starting Dashboard at http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
python dashboard.py
pause
exit

:BOTH
echo.
echo  Starting Dashboard in background...
start "Dashboard" cmd /c python dashboard.py
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:5000
echo  Starting Main Camera System...
python main.py
pause
exit

:START
cls
goto START
