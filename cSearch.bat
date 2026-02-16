@echo off
echo Starting Semantic Search API...
echo API documentation will be available at http://localhost:8000/docs

REM Start backend API
start "Semantic Search API" python -m src.api

REM Start frontend
start "Frontend App" python frontend/main.py

pause
