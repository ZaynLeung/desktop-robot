@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem One-click training entry (Windows).
rem   Double-click this file, or run: run_train.bat
rem   Arguments are passed to run_train.ps1, for example:
rem     run_train.bat -Stages walk
rem     run_train.bat -Stages stand,lift -Envs 8 -Steps 2000000
rem     run_train.bat -DryRun

set "HERE=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%HERE%run_train.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" (
    echo [run_train] FAILED, exit code %RC%
    echo See code\runs\run_train_*.log
)

rem Keep the window open when double-clicked (no arguments).
if "%~1"=="" pause
exit /b %RC%
