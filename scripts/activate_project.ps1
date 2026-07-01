$condaExe = "D:\Miniconda3\Scripts\conda.exe"
$projectDir = "D:\classification\data-classification-tool"

if (-not (Test-Path $condaExe)) {
    throw "Conda executable not found: $condaExe"
}

(& $condaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate data-classification
Set-Location $projectDir
Write-Host "Activated conda environment: data-classification"
Write-Host "Project directory: $projectDir"
