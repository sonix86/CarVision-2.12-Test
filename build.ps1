$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create .venv and install requirements first. See README.md.' }
& $taskPython -m PyInstaller --noconfirm --clean --onedir --console --name CarVision --collect-data torchvision --collect-data matplotlib --exclude-module pyarrow --exclude-module sklearn --hidden-import PIL._tkinter_finder launch.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$taskTarget = Join-Path $PSScriptRoot 'dist\CarVision'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'samples') -Destination $taskTarget -Recurse -Force
# Training candidates, historical weights and downloaded datasets are not runtime assets.
$taskModels = Join-Path $taskTarget 'models'
New-Item -ItemType Directory -Path $taskModels -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'models') -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $taskModels -Force }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'models\evaluation') -Destination $taskModels -Recurse -Force
$taskData = Join-Path $taskTarget 'data'
New-Item -ItemType Directory -Path (Join-Path $taskData 'images') -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data\images') -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $taskData 'images') -Recurse -Force }
Copy-Item -LiteralPath 'data\labels.jsonl' -Destination $taskData -Force
# Public training images and downloaded parquet shards are development assets, not runtime dependencies.
New-Item -ItemType Directory -Path (Join-Path $taskData 'public_prototype') -Force | Out-Null
foreach ($taskMetadata in @('provenance.json','colour_reference_review.json')) {
    $taskMetadataPath = Join-Path 'data\public_prototype' $taskMetadata
    if (Test-Path -LiteralPath $taskMetadataPath) { Copy-Item -LiteralPath $taskMetadataPath -Destination (Join-Path $taskData 'public_prototype') -Force }
}
Copy-Item -LiteralPath 'README.md','THIRD_PARTY.md','CHECKLIST.md' -Destination $taskTarget -Force
Copy-Item -LiteralPath 'requirements-lock.txt' -Destination $taskTarget -Force
New-Item -ItemType Directory -Path (Join-Path $taskTarget 'licenses') -Force | Out-Null
& $taskPython tools\collect_licenses.py $taskTarget
if ($LASTEXITCODE -ne 0) { throw 'License collection failed' }
Set-Content -LiteralPath (Join-Path $taskTarget 'Start CarVision.cmd') -Value '@echo off','cd /d "%~dp0"','start "" /b CarVision.exe gui' -Encoding ascii
Write-Output "Portable build: $taskTarget. Transfer the entire directory, not just the EXE."
