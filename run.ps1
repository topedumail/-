# הפעלת הבוט.
# שימוש: .\run.ps1
# בפעם הראשונה — פותח venv ומתקין תלויות. בפעמים הבאות — רק מריץ.

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$venvPath = Join-Path $projectDir ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "יוצר סביבת Python חדשה..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "מתקין/מעדכן תלויות..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip --quiet
& $pythonExe -m pip install -r requirements.txt --quiet

if (-not (Test-Path (Join-Path $projectDir ".env"))) {
    Write-Host "" -ForegroundColor Yellow
    Write-Host "אזהרה: לא נמצא קובץ .env" -ForegroundColor Yellow
    Write-Host "צריך ליצור אותו ולשים בו את ה-OPENAI_API_KEY שלך." -ForegroundColor Yellow
    Write-Host "אפשר להעתיק מ-.env.example:" -ForegroundColor Yellow
    Write-Host "  Copy-Item .env.example .env" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "מפעיל את השרת..." -ForegroundColor Green
Write-Host "פתח/י בדפדפן: http://localhost:8000" -ForegroundColor Green
Write-Host ""

Set-Location (Join-Path $projectDir "backend")
& $pythonExe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
