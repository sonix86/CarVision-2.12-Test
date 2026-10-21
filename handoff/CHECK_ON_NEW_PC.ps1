param([switch]$Quick)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$taskRoot = $PSScriptRoot
$taskStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$taskRun = Join-Path $taskRoot ('outputs\transfer-selfcheck\' + $taskStamp)
New-Item -ItemType Directory -Path $taskRun -Force | Out-Null
$taskExe = Join-Path $taskRoot 'CarVision.exe'
$taskResults = New-Object System.Collections.Generic.List[object]
$taskResultFile = Join-Path $taskRun 'SELF_CHECK_RESULT.json'
$taskOriginalPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
$env:PYTHONHOME = ''; $env:PYTHONPATH = ''; $env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '3'; $env:MKL_NUM_THREADS = '3'
$taskOverall = [ordered]@{ passed=$false; version='2.1'; computer=$env:COMPUTERNAME; os=[Environment]::OSVersion.VersionString; architecture=$env:PROCESSOR_ARCHITECTURE; path=$taskRoot; python_in_path=$false; quick=[bool]$Quick; started=(Get-Date).ToString('o'); scope='Engineering/portability checks on THIS computer, not recognition accuracy or an independent phone exam'; steps=$taskResults }

function Get-LocalSHA256 {
    param([string]$LiteralPath)
    $taskStream = [IO.File]::OpenRead($LiteralPath)
    $taskAlgorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($taskAlgorithm.ComputeHash($taskStream)).Replace('-','').ToLowerInvariant() }
    finally { $taskStream.Dispose(); $taskAlgorithm.Dispose() }
}

function Invoke-CarVision {
    param([string]$Name, [string[]]$Arguments, [int]$TimeoutSeconds=120, [bool]$AllowFailure=$false)
    Write-Host ("[CHECK] " + $Name)
    $taskOut = Join-Path $taskRun ($Name + '.stdout.txt')
    $taskErr = Join-Path $taskRun ($Name + '.stderr.txt')
    # Arguments are passed directly to an executable, not evaluated by a shell.
    $taskQuoted = ($Arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }) -join ' '
    $taskWatch = [Diagnostics.Stopwatch]::StartNew()
    $taskProcess = Start-Process -FilePath $taskExe -ArgumentList $taskQuoted -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput $taskOut -RedirectStandardError $taskErr -PassThru
    $null = $taskProcess.Handle
    if (-not $taskProcess.WaitForExit($TimeoutSeconds*1000)) {
        $taskProcess.Kill()
        throw ("Timeout in " + $Name + "; see " + $taskErr)
    }
    $taskProcess.WaitForExit(); $taskWatch.Stop()
    $taskCode = $taskProcess.ExitCode
    $taskResults.Add([ordered]@{ name=$Name; exit_code=$taskCode; seconds=[Math]::Round($taskWatch.Elapsed.TotalSeconds,3); stdout=$taskOut; stderr=$taskErr })
    if ($taskCode -ne 0 -and -not $AllowFailure) { throw ("Failed " + $Name + "; see " + $taskErr) }
    return [pscustomobject]@{ Code=$taskCode; Text=[IO.File]::ReadAllText($taskOut,[Text.Encoding]::UTF8) }
}

try {
    $taskOverall['checker_sha256'] = Get-LocalSHA256 -LiteralPath $PSCommandPath
    $taskOverall['executable_sha256'] = Get-LocalSHA256 -LiteralPath $taskExe
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    if (-not (Test-Path -LiteralPath $taskExe)) { throw 'CarVision.exe is missing. Extract the complete ZIP first.' }
    $taskModels = Get-Content -LiteralPath (Join-Path $taskRoot 'Development\ACTIVE_MODEL_SHA256.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskBefore = @{}
    foreach ($taskEntry in $taskModels.PSObject.Properties) {
        $taskHash = Get-LocalSHA256 -LiteralPath (Join-Path $taskRoot ('models\'+$taskEntry.Name))
        if ($taskHash -ne $taskEntry.Value) { throw ('Model checksum mismatch: '+$taskEntry.Name) }
        $taskBefore[$taskEntry.Name] = $taskHash
    }
    $taskOverall['model_sha256'] = $taskBefore
    Write-Host '[PASS] Included trained weights match the handoff manifest.'
    $null = Invoke-CarVision -Name 'help' -Arguments @('--help')
    $taskPhoto = Join-Path $taskRoot 'samples\images\03.jpg'
    $taskPrediction = Invoke-CarVision -Name 'predict' -Arguments @('predict',$taskPhoto,'--mode','fast','--output',(Join-Path $taskRun 'prediction'))
    $taskData = $taskPrediction.Text | ConvertFrom-Json
    if (@($taskData.cars).Count -lt 1) { throw 'No instances returned on the included engineering sample.' }
    foreach ($taskExtension in @('*.json','*.txt','*annotated.jpg')) {
        if (@(Get-ChildItem -LiteralPath (Join-Path $taskRun 'prediction') -Filter $taskExtension -File).Count -lt 1) { throw ('Missing export: '+$taskExtension) }
    }
    $taskMain = $taskData.cars | Sort-Object { ($_.box[2]-$_.box[0])*($_.box[3]-$_.box[1]) } -Descending | Select-Object -First 1
    $taskRoiArgs = @('predict',$taskPhoto,'--mode','fast','--region') + @($taskMain.box | ForEach-Object { [string]$_ }) + @('--single','--output',(Join-Path $taskRun 'region'))
    $taskRoi = Invoke-CarVision -Name 'region' -Arguments $taskRoiArgs
    $taskRoiData = $taskRoi.Text | ConvertFrom-Json
    if (@($taskRoiData.cars).Count -ne 1) { throw 'Single-car region returned an unexpected instance count.' }

    if (-not $Quick) {
        Write-Host '[CHECK] Isolated synthetic training fixture; production weights stay untouched.'
        Add-Type -AssemblyName System.Drawing
        $taskFixture = Join-Path $taskRun 'synthetic-engineering-fixture'
        New-Item -ItemType Directory -Path $taskFixture -Force | Out-Null
        $taskRows = New-Object System.Collections.Generic.List[string]
        for ($taskIndex=0; $taskIndex -lt 20; $taskIndex++) {
            $taskName = 'fixture-' + $taskIndex + '.png'
            $taskBitmap = New-Object Drawing.Bitmap 96,96
            $taskGraphics = [Drawing.Graphics]::FromImage($taskBitmap)
            try { $taskGraphics.Clear([Drawing.Color]::FromArgb((20+$taskIndex*10),(45+$taskIndex*5),(150-$taskIndex*4))); $taskBitmap.Save((Join-Path $taskFixture $taskName),[Drawing.Imaging.ImageFormat]::Png) }
            finally { $taskGraphics.Dispose(); $taskBitmap.Dispose() }
            $taskRow = [ordered]@{ image=$taskName; box=@(2,2,94,94); brand=$(if ($taskIndex%2) {'FixtureA'} else {'FixtureB'}); colour='blue'; source='Synthetic engineering fixture ONLY, not car photos or recognition evidence'; group=('fixture-group-'+[Math]::Floor($taskIndex/2)) }
            $taskRows.Add(($taskRow | ConvertTo-Json -Compress))
        }
        $taskLabels = Join-Path $taskFixture 'labels.jsonl'
        [IO.File]::WriteAllLines($taskLabels,$taskRows,[Text.UTF8Encoding]::new($false))
        $null = Invoke-CarVision -Name 'audit' -Arguments @('audit','--labels',$taskLabels)
        $taskCandidate = Join-Path $taskRun 'candidate-models'
        $null = Invoke-CarVision -Name 'train' -Arguments @('train','--labels',$taskLabels,'--output',$taskCandidate,'--epochs','1','--detector-epochs','1','--batch','8') -TimeoutSeconds 600
        foreach ($taskFile in @('classifier.pt','detector.pt','training.json','detector_training.json','evaluation\metrics.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $taskCandidate $taskFile))) { throw ('Training output missing: '+$taskFile) }
        }
        $null = Invoke-CarVision -Name 'evaluate' -Arguments @('evaluate','--labels',$taskLabels,'--models',$taskCandidate,'--output',(Join-Path $taskRun 'evaluation'))
    }

    foreach ($taskName in $taskBefore.Keys) {
        if ((Get-LocalSHA256 -LiteralPath (Join-Path $taskRoot ('models\'+$taskName))) -ne $taskBefore[$taskName]) { throw ('Production model unexpectedly changed: '+$taskName) }
    }
    $taskOverall.passed = $true
    Write-Host 'PASS - EXE, trained weights, prediction, region and exports work on this computer.' -ForegroundColor Green
    if (-not $Quick) { Write-Host 'PASS - isolated training and evaluation also completed; production models unchanged.' -ForegroundColor Green }
}
catch {
    $taskOverall['error'] = $_.Exception.Message
    Write-Host ('FAIL - '+$_.Exception.Message) -ForegroundColor Red
}
finally {
    $env:PATH = $taskOriginalPath
    $taskOverall['finished'] = (Get-Date).ToString('o')
    [IO.File]::WriteAllText($taskResultFile,($taskOverall | ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
    Write-Host ('Result: '+$taskResultFile)
}
if (-not $taskOverall.passed) { exit 1 }
exit 0
