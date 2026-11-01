@echo off
cd /d "%~dp0"
echo Starting Backend API with Python 3.10 virtual environment...
.\.venv\Scripts\uvicorn.exe app.main:app --reload --port 8000
