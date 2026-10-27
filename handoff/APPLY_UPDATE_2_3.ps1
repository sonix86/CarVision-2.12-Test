param([string]$Target)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
function Hash([string]$Path) {
    $stream=[IO.File]::OpenRead($Path)
    try { $algorithm=[Security.Cryptography.SHA256]::Create(); try { return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-','').ToLowerInvariant() } finally { $algorithm.Dispose() } } finally { $stream.Dispose() }
}
try {
    $manifest=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'UPDATE_MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $Target) {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog=New-Object Windows.Forms.FolderBrowserDialog
        $dialog.Description='Choose the existing CarVision folder containing CarVision.exe'
        if ($dialog.ShowDialog() -ne 'OK') { exit 2 }
        $Target=$dialog.SelectedPath
    }
    $root=(Resolve-Path -LiteralPath $Target).Path
    $exe=Join-Path $root 'CarVision.exe'
    if (-not (Test-Path -LiteralPath $exe)) { throw 'CarVision.exe was not found. Choose the existing program folder.' }
    $exeHash=Hash $exe
    if ($exeHash -ne $manifest.exe_sha256 -and @($manifest.previous_exe_sha256) -notcontains $exeHash) { throw 'This update requires the supplied CarVision 2.1/2.2 EXE. No files changed.' }
    if (Get-Process -Name CarVision -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe }) { throw 'Close CarVision before installing the update.' }
    foreach ($entry in $manifest.required_files) {
        $required=[IO.Path]::GetFullPath((Join-Path $root ([string]$entry.path)))
        if (-not $required.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe prerequisite path.' }
        if (-not (Test-Path -LiteralPath $required) -or (Hash $required) -ne $entry.sha256) { throw "Required unchanged file is missing or modified: $($entry.path). No files changed." }
    }
    $jobs=@();$allInstalled=$true
    foreach ($entry in $manifest.files) {
        $relative=[string]$entry.path
        $source=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot ('payload/'+$relative)))
        $destination=[IO.Path]::GetFullPath((Join-Path $root $relative))
        if (-not $destination.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe destination path.' }
        if (-not $source.StartsWith((Join-Path $PSScriptRoot 'payload')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe payload path.' }
        if ((Hash $source) -ne $entry.sha256) { throw "Damaged update payload: $relative" }
        $exists=Test-Path -LiteralPath $destination
        $before=if ($exists) { Hash $destination } else { $null }
        $allowed=@($entry.previous_sha256 | Where-Object { $_ })
        if ($before -ne $entry.sha256) {
            $allInstalled=$false
            if ($exists -and $allowed -notcontains $before) { throw "Installed file differs from the supported release: $relative. No files changed." }
            if (-not $exists -and $entry.require_existing) { throw "Required installed file is missing: $relative. No files changed." }
        }
        $jobs+=@{relative=$relative;source=$source;destination=$destination;existed=$exists;before=$before}
    }
    if ($allInstalled) { Write-Host 'Update 2.3 is already installed. Start CarVision.exe.'; exit 0 }
    $backup=Join-Path $root ('patch-backups/update23-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    foreach ($job in $jobs) {
        if ($job.existed) {
            $saved=Join-Path $backup $job.relative
            New-Item -ItemType Directory -Path (Split-Path $saved) -Force | Out-Null
            Copy-Item -LiteralPath $job.destination -Destination $saved
            if ((Hash $saved) -ne $job.before) { throw 'Backup verification failed. The installed files were not changed.' }
        }
    }
    try {
        foreach ($job in $jobs) {
            New-Item -ItemType Directory -Path (Split-Path $job.destination) -Force | Out-Null
            Copy-Item -LiteralPath $job.source -Destination $job.destination -Force
        }
        foreach ($entry in $manifest.files) {
            if ((Hash (Join-Path $root $entry.path)) -ne $entry.sha256) { throw "Installed file checksum failed: $($entry.path)" }
        }
    } catch {
        $copyError=$_;$failed=@()
        foreach ($job in $jobs) {
            try {
                if ($job.existed) { Copy-Item -LiteralPath (Join-Path $backup $job.relative) -Destination $job.destination -Force }
                elseif (Test-Path -LiteralPath $job.destination) { Remove-Item -LiteralPath $job.destination }
            } catch { $failed+=$job.relative }
        }
        if ($failed.Count) { throw "Update failed; some files need manual restoration from $backup : $($failed -join ', ')" }
        throw $copyError
    }
    Write-Host 'Update 2.3 installed successfully. Start your CarVision.exe normally.'
    Write-Host "Previous files were saved in: $backup"
    Write-Host 'Your photos, labels and saved results were preserved.'
    exit 0
} catch {
    Write-Host ('Update failed: '+$_.Exception.Message)
    exit 1
}
