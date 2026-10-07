<#
.SYNOPSIS
    Compiles the LaTeX paper projects in Materials/Research Papers using the local tectonic.exe binary.

.PARAMETER Project
    Part of a project folder name (e.g., 'Revised', 'Segmentation').
    Default: compiles all projects.
#>

param(
    [string]$Project = "*"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TectonicExe = Join-Path $ScriptDir "tectonic.exe"

if (-not (Test-Path $TectonicExe)) {
    Write-Error "tectonic.exe not found at $TectonicExe"
    exit 1
}

$PapersDir = Join-Path $ScriptDir "Research Papers"
$ProjectDirs = @(Get-ChildItem -Path $PapersDir -Directory | Where-Object { $_.Name -like "*$Project*" })

if ($ProjectDirs.Count -eq 0) {
    Write-Warning "No project directory matching '$Project' found."
    exit 0
}

foreach ($dir in $ProjectDirs) {
    $texFiles = Get-ChildItem -Path $dir.FullName -Filter "*.tex"
    if ($texFiles.Count -eq 0) {
        continue
    }

    Write-Host "`n========================================================" -ForegroundColor Cyan
    Write-Host "Compiling project: $($dir.Name)" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan

    Push-Location $dir.FullName
    try {
        # Compile main.tex first if it exists, otherwise all .tex files
        $mainTex = Join-Path $dir.FullName "main.tex"
        if (Test-Path $mainTex) {
            Write-Host "Building main.tex..." -ForegroundColor Yellow
            & $TectonicExe "main.tex"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Successfully compiled main.pdf in $($dir.Name)" -ForegroundColor Green
            } else {
                Write-Warning "Compilation failed for main.tex in $($dir.Name)"
            }
        } else {
            foreach ($tex in $texFiles) {
                Write-Host "Building $($tex.Name)..." -ForegroundColor Yellow
                & $TectonicExe $tex.Name
            }
        }
    } finally {
        Pop-Location
    }
}
