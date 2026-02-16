@echo off
echo Starting Semantic Search API and Frontend in same terminal...

REM Start API in background (same terminal, no new window)
start /b cmd /c "python -m src.api"

REM Small delay to avoid race condition
timeout /t 2 /nobreak >nul

REM Start frontend in background (same terminal)
start /b cmd /c "python frontend/main.py"

echo Both processes started.
echo Press Ctrl+C to stop everything.
pause
