param([string]$Target)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
function Hash([string]$Path) {
    $stream=[IO.File]::OpenRead($Path)
    try { $alg=[Security.Cryptography.SHA256]::Create(); try { return ([BitConverter]::ToString($alg.ComputeHash($stream))).Replace('-','').ToLowerInvariant() } finally { $alg.Dispose() } } finally { $stream.Dispose() }
}
try {
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'PATCH_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $Target) {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog=New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description='Select the existing CarVision folder containing CarVision.exe'
        if ($dialog.ShowDialog() -ne 'OK') { exit 2 }
        $Target=$dialog.SelectedPath
    }
    $root=(Resolve-Path -LiteralPath $Target).Path
    $exe=Join-Path $root 'CarVision.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw 'CarVision.exe was not found in the selected folder.' }
    if ((Hash $exe) -ne $manifest.exe_sha256) { throw 'This patch requires the supplied CarVision 2.1 EXE. No files changed.' }
    $running=Get-Process -Name CarVision -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }
    if ($running) { throw 'Close CarVision before applying the patch, then run this file again.' }
    $jobs=@()
    foreach ($entry in $manifest.files) {
        $relative=[string]$entry.path
        $destination=[IO.Path]::GetFullPath((Join-Path $root $relative))
        $source=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot ('payload/'+$relative)))
        if (-not $destination.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe destination path.' }
        if (-not $source.StartsWith((Join-Path $PSScriptRoot 'payload')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe payload path.' }
        if ((Hash $source) -ne $entry.sha256) { throw "Patch file checksum failed: $relative" }
        if ($entry.previous_sha256 -and (Hash $destination) -ne $entry.previous_sha256) { throw "Installed model differs from the expected baseline: $relative. No files changed." }
        $jobs+=@{relative=$relative;source=$source;destination=$destination;existed=(Test-Path -LiteralPath $destination)}
    }
    $backup=Join-Path $root ('patch-backups/ev22-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    foreach ($job in $jobs) {
        if ($job.existed) {
            $saved=Join-Path $backup $job.relative
            New-Item -ItemType Directory -Path (Split-Path $saved) -Force | Out-Null
            Copy-Item -LiteralPath $job.destination -Destination $saved
        }
    }
    try {
        foreach ($job in $jobs) {
            New-Item -ItemType Directory -Path (Split-Path $job.destination) -Force | Out-Null
            Copy-Item -LiteralPath $job.source -Destination $job.destination -Force
        }
        foreach ($entry in $manifest.files) {
            if ((Hash (Join-Path $root $entry.path)) -ne $entry.sha256) { throw 'Installed checksum failed.' }
        }
    } catch {
        foreach ($job in $jobs) {
            if ($job.existed) { Copy-Item -LiteralPath (Join-Path $backup $job.relative) -Destination $job.destination -Force }
            elseif (Test-Path -LiteralPath $job.destination) { Remove-Item -LiteralPath $job.destination }
        }
        throw
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'PATCH_MANIFEST.json') -Destination (Join-Path $backup 'applied-patch.json')
    Write-Host "Patch installed. Previous files saved in: $backup"
    Write-Host 'Start CarVision.exe normally. Your photos and annotations were preserved.'
    exit 0
} catch {
    Write-Host ('Patch failed: '+$_.Exception.Message)
    exit 1
}
