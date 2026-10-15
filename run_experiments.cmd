@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python run_experiments.py
) else (
    py -3.12 run_experiments.py
)
if errorlevel 1 (
    echo.
    echo The run failed. Read the error above and the results folder.
    echo Press any key to close this window...
    pause >nul
    exit /b 1
)
echo.
echo Experiments completed. Results are in the results folder.
echo Press any key to close this window...
pause >nul
