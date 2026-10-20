$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create .venv and install requirements first. See README.md.' }
& $taskPython -m PyInstaller --noconfirm --clean --onedir --console --name CarVision --collect-data torchvision --collect-data matplotlib --exclude-module pyarrow --hidden-import PIL._tkinter_finder launch.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed' }
$taskTarget = Join-Path $PSScriptRoot 'dist\CarVision'
foreach ($taskFolder in @('models','samples')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $taskFolder) -Destination $taskTarget -Recurse -Force
}
$taskData = Join-Path $taskTarget 'data'
New-Item -ItemType Directory -Path (Join-Path $taskData 'images') -Force | Out-Null
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
