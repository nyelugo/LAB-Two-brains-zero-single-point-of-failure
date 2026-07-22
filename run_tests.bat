@echo off
REM Double-click launcher (Windows). Runs the unit test suite.
cd /d "%~dp0"
python -m pytest test_summarizer.py -v
pause
