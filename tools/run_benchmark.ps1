<#
.SYNOPSIS
    Run 0.6B + 1.7B realtime and batch ASR benchmarks end-to-end.

.DESCRIPTION
    For each detected Qwen3-ASR model:
      1. Launch qasr_server
      2. Run realtime streaming benchmark (2-min clip, jittered chunks at real-time pace)
      3. Run batch/offline transcription benchmark
      4. Stop the server
    Finally, generate a markdown comparison report.

    Chunk strategy for realtime simulation:
    - Chunks are random-sized between MinChunkMs..MaxChunkMs (default 200..600ms).
    - Sending is paced at real-time speed: wall-clock tracks cumulative audio sent.
    - Between sends, we poll /api/realtime/status to capture partial results.
    - This models a typical audio-streaming client that buffers variable amounts
      of PCM before flushing, with the transport running at real-time speed.
    - A right-skewed distribution (smaller median, long tail) would better model
      real clients with occasional network hiccups, but for RTF / latency
      benchmarks the uniform distribution gives sufficiently stable results.

.EXAMPLE
    .\tools\run_benchmark.ps1
    .\tools\run_benchmark.ps1 -MaxAudioSeconds 60
    .\tools\run_benchmark.ps1 -Port 8091 -Verbosity 1
#>
param(
    [string]$ServerExe      = "",
    [string]$WavPath        = "",
    [int]$MaxAudioSeconds   = 120,
    [string]$HostAddr       = "127.0.0.1",
    [int]$Port              = 8090,
    [string]$OutputDir      = "",
    [int]$ServerLoadTimeoutS = 300,
    [int]$Verbosity         = 0,
    [int]$MinChunkMs        = 200,
    [int]$MaxChunkMs        = 600
)

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version Latest

$scriptDir  = $PSScriptRoot
$projectDir = Split-Path $scriptDir -Parent
$runId      = Get-Date -Format 'yyyyMMdd-HHmmss'

# ─── Defaults ───────────────────────────────────────────────────────────────
if ([string]::IsNullOrWhiteSpace($ServerExe)) {
    $ServerExe = Join-Path $projectDir "build\windows-openblas\qasr_server.exe"
}
if ([string]::IsNullOrWhiteSpace($WavPath)) {
    $testDir = Join-Path $projectDir "testfile"
    $wavCandidates = Get-ChildItem $testDir -Filter "*.wav" -ErrorAction SilentlyContinue
    if ($wavCandidates) {
        $biggest = $wavCandidates | Sort-Object Length -Descending | Select-Object -First 1
        $WavPath = $biggest.FullName
    } else {
        throw "No .wav files found in $testDir"
    }
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $projectDir "build\benchmark_$runId"
}

# ─── Model discovery ────────────────────────────────────────────────────────
$cacheRoot = Join-Path $env:USERPROFILE ".cache\modelscope\hub\models\Qwen"
$candidateDirs = @(
    @{ Label = "Qwen3-ASR-0.6B"; Dirs = @(
        (Join-Path $cacheRoot "Qwen3-ASR-0___6B"),
        (Join-Path $cacheRoot "Qwen3-ASR-0.6B")
    )}
    @{ Label = "Qwen3-ASR-1.7B"; Dirs = @(
        (Join-Path $cacheRoot "Qwen3-ASR-1___7B"),
        (Join-Path $cacheRoot "Qwen3-ASR-1.7B")
    )}
)

$models = @()
foreach ($c in $candidateDirs) {
    $found = $null
    foreach ($d in $c.Dirs) {
        if (Test-Path $d) { $found = $d; break }
    }
    if ($null -ne $found) {
        $models += @{ Label = $c.Label; Dir = $found }
    } else {
        Write-Host "  [skip] $($c.Label) not found" -ForegroundColor DarkGray
    }
}
if ($models.Count -eq 0) { throw "No Qwen3-ASR models found under $cacheRoot" }

# ─── Validate ────────────────────────────────────────────────────────────────
if (!(Test-Path $ServerExe)) { throw "Server not found: $ServerExe" }
if (!(Test-Path $WavPath))   { throw "WAV not found: $WavPath" }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$realtimeScript = Join-Path $scriptDir "realtime_testfile_benchmark.ps1"
$offlineScript  = Join-Path $scriptDir "offline_testfile_benchmark.ps1"
if (!(Test-Path $realtimeScript)) { throw "Missing: $realtimeScript" }
if (!(Test-Path $offlineScript))  { throw "Missing: $offlineScript" }

$baseUrl = "http://${HostAddr}:${Port}"

# ─── Banner ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host ("=" * 58) -ForegroundColor Cyan
Write-Host "  Qwen3-ASR Benchmark Suite" -ForegroundColor Cyan
Write-Host ("=" * 58) -ForegroundColor Cyan
Write-Host "  Server  : $ServerExe"
Write-Host "  Audio   : $([IO.Path]::GetFileName($WavPath)) (${MaxAudioSeconds}s clip)"
Write-Host "  Chunks  : ${MinChunkMs}-${MaxChunkMs} ms (jittered, real-time paced)"
Write-Host "  Models  : $($models | ForEach-Object { $_.Label })"
Write-Host "  Output  : $OutputDir"
Write-Host ("=" * 58) -ForegroundColor Cyan
Write-Host ""

# ─── Helpers ─────────────────────────────────────────────────────────────────
function Wait-Health ([string]$Url, [int]$TimeoutS) {
    $deadline = (Get-Date).AddSeconds($TimeoutS)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-RestMethod -Method Get -Uri $Url -ErrorAction Stop
            if ($r.status -eq 'ok') { return $true }
        } catch { }
        Start-Sleep -Milliseconds 1000
    }
    return $false
}

function Stop-ServerSafe ([System.Diagnostics.Process]$P) {
    if ($null -eq $P) { return }
    if (!$P.HasExited) {
        try { Stop-Process -Id $P.Id -Force -ErrorAction SilentlyContinue } catch { }
        Start-Sleep -Seconds 2
    }
}

function Free-Port ([int]$PortNum) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $PortNum -ErrorAction SilentlyContinue
        foreach ($c in $conns) {
            if ($c.OwningProcess -gt 0) {
                Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Seconds 1
    } catch { }
}

function Start-ServerForModel ([string]$ModelDir, [string]$Label) {
    Free-Port $Port

    Write-Host "  Starting server for $Label ..." -ForegroundColor Yellow
    $argStr = "--model-dir `"$ModelDir`" --host $HostAddr --port $Port"
    if ($Verbosity -gt 0) { $argStr += " --verbosity $Verbosity" }

    $proc = Start-Process -FilePath $ServerExe -ArgumentList $argStr -PassThru
    Write-Host "  Loading model (PID $($proc.Id)) ..."

    if (!(Wait-Health "$baseUrl/health" $ServerLoadTimeoutS)) {
        if ($proc.HasExited) {
            Write-Warning "  Server exited during model load (exit=$($proc.ExitCode))"
        } else {
            Write-Warning "  Server did not become healthy within ${ServerLoadTimeoutS}s"
            Stop-ServerSafe $proc
        }
        return $null
    }
    Write-Host "  Server ready (PID $($proc.Id))" -ForegroundColor Green
    return $proc
}

# ─── Main loop ───────────────────────────────────────────────────────────────
foreach ($m in $models) {
    $label    = $m.Label
    $modelDir = $m.Dir

    Write-Host ("-" * 58) -ForegroundColor Cyan
    Write-Host "  $label" -ForegroundColor Cyan
    Write-Host ("-" * 58) -ForegroundColor Cyan

    $proc = Start-ServerForModel $modelDir $label
    if ($null -eq $proc) {
        Write-Warning "Skipping $label (server failed to start)"
        continue
    }

    # ── Realtime benchmark ──────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  >> Realtime streaming (${MaxAudioSeconds}s) ..." -ForegroundColor Yellow
    try {
        & $realtimeScript `
            -BaseUrl       $baseUrl `
            -WavPath       $WavPath `
            -ModelLabel    $label `
            -MaxAudioSeconds $MaxAudioSeconds `
            -MinChunkMs    $MinChunkMs `
            -MaxChunkMs    $MaxChunkMs `
            -OutputDir     $OutputDir
    } catch {
        Write-Warning "  Realtime FAILED: $_"
    }

    # Restart if the server crashed
    if ($proc.HasExited) {
        Write-Warning "  Server crashed during realtime (exit=$($proc.ExitCode)), restarting for batch ..."
        $proc = Start-ServerForModel $modelDir $label
        if ($null -eq $proc) {
            Write-Warning "  Cannot restart, skipping batch for $label"
            continue
        }
    }

    # ── Batch benchmark ─────────────────────────────────────────────────────
    Write-Host ""
    Write-Host "  >> Batch transcription (${MaxAudioSeconds}s) ..." -ForegroundColor Yellow
    try {
        & $offlineScript `
            -BaseUrl       $baseUrl `
            -WavPath       $WavPath `
            -ModelLabel    $label `
            -MaxAudioSeconds $MaxAudioSeconds `
            -OutputDir     $OutputDir
    } catch {
        Write-Warning "  Batch FAILED: $_"
    }

    Stop-ServerSafe $proc
    Write-Host "  Server stopped for $label`n" -ForegroundColor Green
}

# ─── Collect results & generate report ───────────────────────────────────────
Write-Host ("=" * 58) -ForegroundColor Cyan
Write-Host "  Generating Report" -ForegroundColor Cyan
Write-Host ("=" * 58) -ForegroundColor Cyan

$md = @()
$md += "# Qwen3-ASR Benchmark Report"
$md += ""
$md += "| Item | Value |"
$md += "|------|-------|"
$md += "| Date | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') |"
$md += "| Audio | ``$([IO.Path]::GetFileName($WavPath))`` (${MaxAudioSeconds}s clip) |"
$md += "| Chunk range | ${MinChunkMs}-${MaxChunkMs} ms |"
$md += "| Platform | $([Environment]::OSVersion.VersionString), $([Environment]::ProcessorCount) cores |"
$md += ""

# Realtime table
$rtFiles = Get-ChildItem $OutputDir -Filter "realtime_bench_*.json" -ErrorAction SilentlyContinue | Sort-Object Name
if ($rtFiles) {
    $md += "## Realtime Streaming"
    $md += ""
    $md += "> **Processing RTF** = lag / audio duration (comparable to batch RTF)."
    $md += "> **E2E RTF** = wall / audio (always >= 1.0, includes real-time pacing)."
    $md += ""
    $md += "| Model | Audio (s) | Wall (s) | Processing RTF | E2E RTF | 1st Stable (ms) | Lag (ms) | Stable Evts | Smooth |"
    $md += "|-------|-----------|----------|---------------|---------|-----------------|----------|-------------|--------|"

    foreach ($f in $rtFiles) {
        $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
        $aS  = [Math]::Round($j.audio_duration_ms / 1000.0, 1)
        $wS  = [Math]::Round($j.final_wall_ms / 1000.0, 1)
        $e2eRtf = $j.realtime_factor
        $procRtf = "---"
        if ($null -ne $j.final_lag_ms -and $j.audio_duration_ms -gt 0) {
            $procRtf = [Math]::Round([double]$j.final_lag_ms / [double]$j.audio_duration_ms, 3)
        }
        $fst = "---"
        if ($null -ne $j.first_stable_wall_ms) { $fst = [string]$j.first_stable_wall_ms }
        $lag = "---"
        if ($null -ne $j.final_lag_ms) { $lag = [string]$j.final_lag_ms }
        $sm  = "No"
        if ($j.smooth_stable_output) { $sm = "Yes" }
        $md += "| $($j.model_label) | $aS | $wS | $procRtf | $e2eRtf | $fst | $lag | $($j.stable_events) | $sm |"
    }
    $md += ""
}

# Batch table
$ofFiles = Get-ChildItem $OutputDir -Filter "offline_bench_*.json" -ErrorAction SilentlyContinue | Sort-Object Name
if ($ofFiles) {
    $md += "## Batch (Offline) Transcription"
    $md += ""
    $md += "| Model | Audio (s) | Wall (s) | RTF | Inference (ms) | Tokens |"
    $md += "|-------|-----------|----------|-----|----------------|--------|"

    foreach ($f in $ofFiles) {
        $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
        $aS  = [Math]::Round($j.audio_duration_ms / 1000.0, 1)
        $wS  = [Math]::Round($j.wall_ms / 1000.0, 1)
        $rtf = $j.realtime_factor
        $inf = [Math]::Round($j.reported_inference_ms, 0)
        $md += "| $($j.model_label) | $aS | $wS | $rtf | $inf | $($j.tokens) |"
    }
    $md += ""
}

$reportPath = Join-Path $OutputDir "benchmark_report.md"
($md -join "`n") | Set-Content $reportPath -Encoding UTF8

Write-Host ""
Get-Content $reportPath | ForEach-Object { Write-Host $_ }
Write-Host ""
Write-Host "Report : $reportPath" -ForegroundColor Green
Write-Host "JSON   : $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "=== Benchmark Suite Complete ===" -ForegroundColor Cyan
