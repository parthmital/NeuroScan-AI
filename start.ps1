# Start NeuroScan AI services, each in its own PowerShell window for dedicated logs.
# Ctrl+C or closing this window stops every service and closes their windows.

# Force UTF-8 at the system level
chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$RootPath = $PSScriptRoot
$VenvPath = Join-Path $RootPath ".venv"

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

# Commands each service window runs. Windows stay open (-NoExit) so crash logs remain visible.
$ServiceCommands = @{
    backend = @"
`$env:PYTHONIOENCODING = 'utf-8'
Write-Host 'Checking Python dependencies...' -ForegroundColor Gray
& '$VenvScripts\python.exe' -c 'import fastapi, uvicorn, torch, nibabel, sqlmodel, jwt, pyotp, alembic, reportlab' 2>`$null
if (`$LASTEXITCODE -ne 0) {
    Write-Host 'Installing Python dependencies...' -ForegroundColor Yellow
    & '$VenvScripts\pip.exe' install -r requirements.txt
}
# Localhost only; the frontend dev server proxies /api to this port.
& '$VenvScripts\python.exe' -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 --log-level info
"@
    frontend = @"
`$env:NODE_ENV = 'development'
if (-not (Test-Path 'node_modules')) {
    Write-Host 'Installing Node.js dependencies from the lockfile...' -ForegroundColor Yellow
    npm ci
}
npm run dev
"@
}

# A kill-on-close job object: Windows terminates every process in it (and everything they
# spawn) as soon as this launcher exits, whether by Ctrl+C, closing the window, or a crash.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class ServiceJob {
    [StructLayout(LayoutKind.Sequential)]
    struct BasicLimits {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ExtendedLimits {
        public BasicLimits Basic;
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }

    const int ExtendedLimitInformation = 9;
    const uint KillOnJobClose = 0x2000;

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    static IntPtr handle = IntPtr.Zero;

    public static void Add(IntPtr process) {
        if (handle == IntPtr.Zero) {
            handle = CreateJobObject(IntPtr.Zero, null);
            var info = new ExtendedLimits();
            info.Basic.LimitFlags = KillOnJobClose;
            if (handle == IntPtr.Zero || !SetInformationJobObject(handle, ExtendedLimitInformation, ref info, (uint)Marshal.SizeOf(info)))
                throw new System.ComponentModel.Win32Exception();
        }
        if (!AssignProcessToJobObject(handle, process))
            throw new System.ComponentModel.Win32Exception();
    }

    public static void Terminate() {
        if (handle != IntPtr.Zero) TerminateJobObject(handle, 1);
    }
}
"@

function Stop-AllServices {
    Write-Host "`nStopping all services..." -ForegroundColor Red
    [ServiceJob]::Terminate()
    Write-Host "All services stopped." -ForegroundColor Green
}

try {
    # Launch each service in its own window
    foreach ($Service in $Services) {
        if (-not (Test-Path $Service.Path)) {
            Write-Host "Directory not found: $($Service.Path)" -ForegroundColor Red
            continue
        }
        Write-Host "Launching [$($Service.Name)] in a new window..." -ForegroundColor Gray

        $Script = "`$Host.UI.RawUI.WindowTitle = '$($Service.Name)'`n" +
            "chcp 65001 | Out-Null`n" +
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`n" +
            "Set-Location -LiteralPath '$($Service.Path)'`n" +
            $ServiceCommands[$Service.Kind]
        # Encoded so paths with spaces or quotes survive the command line intact.
        $Encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($Script))
        $Process = Start-Process powershell.exe -PassThru -ArgumentList @(
            "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-EncodedCommand", $Encoded
        )
        [ServiceJob]::Add($Process.Handle)
    }

    # Open the browser once the frontend answers, so it does not load a half-started dev server.
    Write-Host "Waiting for the frontend at http://localhost:8080..." -ForegroundColor Gray
    $Deadline = (Get-Date).AddSeconds(60)
    $FrontendUp = $false
    while (-not $FrontendUp -and (Get-Date) -lt $Deadline) {
        try {
            Invoke-WebRequest "http://localhost:8080" -UseBasicParsing -TimeoutSec 2 | Out-Null
            $FrontendUp = $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if ($FrontendUp) {
        Start-Process "http://localhost:8080"
    } else {
        Write-Host "Frontend did not respond within 60 seconds; check its window. Open http://localhost:8080 manually." -ForegroundColor Yellow
    }

    Write-Host "`nReady! Logs are in the service windows." -ForegroundColor White
    Write-Host "Backend API: http://127.0.0.1:8000 (proxied at http://localhost:8080/api)" -ForegroundColor Cyan
    Write-Host "Frontend: http://localhost:8080" -ForegroundColor Green
    Write-Host "Press Ctrl+C or close this window to stop everything." -ForegroundColor White

    # Ctrl+C interrupts this loop and runs the finally block.
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    Stop-AllServices
}
