param([string]$Destination = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$taskManifest = @'
{
  "repository": "https://github.com/sonix86/CarVision-2.12-Test",
  "release_tag": "v2.12-test",
  "version": "2.12 TEST",
  "original_zip": {
    "name": "CarVision-2.12-Test-Full.zip",
    "bytes": 2936560562,
    "sha256": "cd5644e7f6bdd5c046e5130e0db6c9065e802c80c4dd9b7ecdddb0fb9540995c"
  },
  "parts": [
    {
      "name": "CarVision-2.12-Test-Full.zip.001",
      "bytes": 1500000000,
      "sha256": "663bf912ed27667dec821a4ec24a665e3df40ad1e972c778748d9dc8caf24cfd"
    },
    {
      "name": "CarVision-2.12-Test-Full.zip.002",
      "bytes": 1436560562,
      "sha256": "38190006002c31688e095261c91ae08e422a7bf325d7b03de4293abc01b5a60b"
    }
  ],
  "training": {
    "name": "CarVision-2.12-Training-Crops.zip",
    "bytes": 551248619,
    "sha256": "970a1aa5812fd532355c07407865510f08577574a08b3e2db3c8ede6eaf3a458"
  },
  "weights_sha256": "c1b01c33d55e163815734ba8c37810da97ac02bba0f9d042519635815d9fac14",
  "exe_sha256": "48d81ca9214d0528553db5ec409b8fe39d686e6bf0c2a09aa45e2d91e3eea830",
  "scope": "Exact delivered portable release; supplemental frozen training crops/features, no recognition weight changes"
}
'@ | ConvertFrom-Json
function Get-ReleaseSHA([string]$LiteralPath) {
 $taskStream = [IO.File]::OpenRead($LiteralPath)
 $taskHash = [Security.Cryptography.SHA256]::Create()
 try { return [BitConverter]::ToString($taskHash.ComputeHash($taskStream)).Replace('-','').ToLowerInvariant() }
 finally { $taskStream.Dispose(); $taskHash.Dispose() }
}
foreach ($taskPart in $taskManifest.parts) {
 $taskPath = Join-Path $PSScriptRoot $taskPart.name
 if (-not (Test-Path -LiteralPath $taskPath)) { throw ('Missing part: ' + $taskPart.name) }
 if ((Get-Item -LiteralPath $taskPath).Length -ne $taskPart.bytes -or (Get-ReleaseSHA $taskPath) -ne $taskPart.sha256) { throw ('Part checksum failed: ' + $taskPart.name) }
 Write-Host ('Verified ' + $taskPart.name)
}
$taskZip = Join-Path $PSScriptRoot $taskManifest.original_zip.name
if (-not (Test-Path -LiteralPath $taskZip)) {
 $taskOutput = [IO.File]::Open($taskZip,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
 try {
  foreach ($taskPart in $taskManifest.parts) {
   $taskInput = [IO.File]::OpenRead((Join-Path $PSScriptRoot $taskPart.name))
   try { $taskInput.CopyTo($taskOutput) } finally { $taskInput.Dispose() }
  }
 } finally { $taskOutput.Dispose() }
}
if ((Get-Item -LiteralPath $taskZip).Length -ne $taskManifest.original_zip.bytes -or (Get-ReleaseSHA $taskZip) -ne $taskManifest.original_zip.sha256) { throw 'Original ZIP checksum failed' }
$taskDestination = [IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath (Join-Path $taskDestination 'CarVision')) { throw 'CarVision folder already exists. Choose a new empty destination; do not mix versions.' }
Write-Host 'Extracting the verified complete application...'
Expand-Archive -LiteralPath $taskZip -DestinationPath $taskDestination
Write-Host ('READY: ' + (Join-Path $taskDestination 'CarVision\CarVision.exe'))
Write-Host 'Launch the EXE. Python, previous patches and internet are not needed for analysis.'
