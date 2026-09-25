# Start NeuroScan AI services in the current IDE terminal as background jobs
# Combined output with prefixes, colors, and clean UTF-8 formatting

# Force UTF-8 at the system level
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RootPath = $PSScriptRoot
$VenvPath = Join-Path $RootPath ".venv"

# Colors for each service
$Colors = @{
    "NeuroScan-Backend" = "Cyan"
    "NeuroScan-Frontend" = "Green"
}

# Cleanup function to stop background jobs when this script exits
function Stop-All-Jobs {
    Write-Host "`nStopping all background services..." -ForegroundColor Red
    Get-Job | Stop-Job
    Get-Job | Remove-Job -Force
    Write-Host "All jobs stopped." -ForegroundColor Green
}

# Trap Ctrl+C (SIGINT) and other termination to clean up
trap { Stop-All-Jobs; exit }

Write-Host "Cleaning up old processes..." -ForegroundColor Gray
Get-Job | Remove-Job -Force

Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray
Write-Host "Starting NeuroScan AI Services" -ForegroundColor White
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

# Check if virtual environment exists
if (-not (Test-Path $VenvPath)) {
    Write-Host "Virtual environment not found at $VenvPath" -ForegroundColor Red
    Write-Host "Please create the virtual environment first:" -ForegroundColor Yellow
    Write-Host "python -m venv '$VenvPath'" -ForegroundColor Yellow
    exit 1
}

# Check if venv Scripts exist
$VenvScripts = Join-Path $VenvPath "Scripts"
if (-not (Test-Path $VenvScripts)) {
    Write-Host "Virtual environment Scripts not found at $VenvScripts" -ForegroundColor Red
    Write-Host "Please activate the virtual environment first:" -ForegroundColor Yellow
    Write-Host "'$VenvScripts\Activate.ps1'" -ForegroundColor Yellow
    exit 1
}

# Create a development .env with fresh random secrets on first run (never committed).
$EnvFile = Join-Path $RootPath ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Host "Creating development .env with new random secrets..." -ForegroundColor Yellow
    $Secrets = & "$VenvPath\Scripts\python.exe" -c "import secrets; from cryptography.fernet import Fernet; print(secrets.token_urlsafe(48)); print(Fernet.generate_key().decode())"
    if ($LASTEXITCODE -ne 0 -or -not $Secrets) {
        Write-Host "Could not generate secrets. Install backend dependencies first: .venv\Scripts\pip install -r Backend
equirements.txt" -ForegroundColor Red
        exit 1
    }
    $EnvLines = @(
        "ENVIRONMENT=development",
        "JWT_SECRET=$($Secrets[0])",
        "DATA_ENCRYPTION_KEY=$($Secrets[1])",
        "COOKIE_SECURE=true",
        "HOST=127.0.0.1",
        "PORT=8000"
    )
    Set-Content -Path $EnvFile -Value $EnvLines -Encoding ascii
}

# Define services
$Services = @(
    @{ 
        Name = "NeuroScan-Backend"
        Path = "$RootPath\Backend"
        Kind = "backend"
    },
    @{ 
        Name = "NeuroScan-Frontend"
        Path = "$RootPath\Frontend"
        Kind = "frontend"
    }
)

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>$null
    Write-Host "Node.js version: $nodeVersion" -ForegroundColor Gray
} catch {
    Write-Host "Node.js not found. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = & "$VenvScripts\python.exe" --version 2>$null
    Write-Host "Python version: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Host "Python not found in virtual environment. Please check your venv setup." -ForegroundColor Red
    exit 1
}

# Launch each service as a background job
foreach ($Service in $Services) {
    Write-Host "Launching [$($Service.Name)]..." -ForegroundColor Gray
    
    # Check if service directory exists
    if (-not (Test-Path $Service.Path)) {
        Write-Host "Directory not found: $($Service.Path)" -ForegroundColor Red
        continue
    }
    
    Start-Job -Name $Service.Name -ScriptBlock {
        Param($Path, $Kind, $VenvScripts)
        
        # Force UTF-8 encoding INSIDE the job process
        chcp 65001 | Out-Null
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $OutputEncoding = [System.Text.Encoding]::UTF8
        
        # Ensure common dev tools use UTF-8/Colors
        $env:PYTHONIOENCODING = "utf-8"
        $env:FORCE_COLOR = "1"
        $env:NODE_ENV = "development"
        
        try {
            Set-Location $Path
            Write-Host "Changed directory to: $Path" -ForegroundColor Gray
            
            if ($Kind -eq "backend") {
                Write-Host "Checking Python dependencies..." -ForegroundColor Gray
                & "$VenvScripts\python.exe" -c "import fastapi, uvicorn, torch, nibabel, sqlmodel, jwt, pyotp, alembic, reportlab" 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
                    & "$VenvScripts\pip.exe" install -r requirements.txt
                }
                # Localhost only; the frontend dev server proxies /api to this port.
                & "$VenvScripts\python.exe" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 --log-level info 2>&1
            }
            else {
                if (-not (Test-Path "node_modules")) {
                    Write-Host "Installing Node.js dependencies from the lockfile..." -ForegroundColor Yellow
                    npm ci
                }
                npm run dev 2>&1
            }
        }
        catch {
            Write-Host "Error starting service: $_" -ForegroundColor Red
        }
    } -ArgumentList $Service.Path, $Service.Kind, $VenvScripts | Out-Null
}

# Wait a moment for services to start
Start-Sleep -Seconds 3

# Automatically open the browser for the frontend
Write-Host "`nOpening browser at http://localhost:8080..." -ForegroundColor Gray
try {
    Start-Process "http://localhost:8080"
} catch {
    Write-Host "Could not open browser automatically. Please navigate to http://localhost:8080" -ForegroundColor Yellow
}

Write-Host "`nReady! Combined logs below (Ctrl+C to quit):" -ForegroundColor White
Write-Host "Backend API: http://127.0.0.1:8000 (proxied at http://localhost:8080/api)" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:8080" -ForegroundColor Green
Write-Host "----------------------------------------------------------------------`n"

# Real-time combined output loop with prefixes
try {
    while ($true) {
        foreach ($Service in $Services) {
            $Job = Get-Job -Name $Service.Name -ErrorAction SilentlyContinue
            if ($Job) {
                $Outputs = $Job | Receive-Job -ErrorAction SilentlyContinue
                if ($Outputs) {
                    foreach ($Line in $Outputs) {
                        # Filter out messy PowerShell 'NativeCommandError' / RemoteException wrappers
                        if ($Line -is [System.Management.Automation.ErrorRecord]) {
                            $ActualMsg = $Line.Exception.Message
                            if ($ActualMsg) { $Line = $ActualMsg }
                        }
                        
                        if ($Line.ToString().Trim()) {
                            Write-Host "[$($Service.Name)] " -NoNewline -ForegroundColor $Colors[$Service.Name]
                            Write-Host $Line
                        }
                    }
                }
            }
        }
        Start-Sleep -Milliseconds 100
    }
}
finally {
    Stop-All-Jobs
}
