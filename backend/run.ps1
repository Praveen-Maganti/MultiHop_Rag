# Run Backend API with Python 3.10 virtual environment
Set-Location -Path $PSScriptRoot
Write-Host "Starting Backend API with Python 3.10 virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\uvicorn.exe" app.main:app --reload --port 8000
