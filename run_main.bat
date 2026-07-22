@echo off
REM Double-click launcher (Windows). Runs the news summarizer interactively.
cd /d "%~dp0"
python main.py %*
pause
