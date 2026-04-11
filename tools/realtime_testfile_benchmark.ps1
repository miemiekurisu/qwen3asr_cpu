param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$WavPath,

    [Parameter(Mandatory = $true)]
    [string]$ModelLabel,

    [string]$OutputDir = "",
    [int]$Seed = 20260411,
    [int]$MinChunkMs = 280,
    [int]$MaxChunkMs = 960,
    [int]$MaxAudioSeconds = 0,
    [int]$PollIntervalMs = 250,
    [int]$ReadyTimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'

function Get-AsciiString {
    param(
        [byte[]]$Bytes,
        [int]$Offset,
        [int]$Count
    )

    return [System.Text.Encoding]::ASCII.GetString($Bytes, $Offset, $Count)
}

function Parse-WavHeader {
    param([byte[]]$Bytes)

    if ($Bytes.Length -lt 44) {
        throw "WAV too small: $($Bytes.Length) bytes"
    }
    if ((Get-AsciiString $Bytes 0 4) -ne 'RIFF' -or (Get-AsciiString $Bytes 8 4) -ne 'WAVE') {
        throw 'Only RIFF/WAVE files are supported'
    }

    $offset = 12
    $format = $null
    $dataOffset = -1
    $dataSize = 0

    while ($offset + 8 -le $Bytes.Length) {
        $chunkId = Get-AsciiString $Bytes $offset 4
        $chunkSize = [BitConverter]::ToUInt32($Bytes, $offset + 4)
        $chunkDataOffset = $offset + 8
        $nextOffset = $chunkDataOffset + [int]$chunkSize
        if (($chunkSize % 2) -ne 0) {
            $nextOffset += 1
        }

        if ($chunkId -eq 'fmt ') {
            if ($chunkDataOffset + 16 -gt $Bytes.Length) {
                throw 'Incomplete fmt chunk'
            }
            $format = [pscustomobject]@{
                AudioFormat = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset)
                Channels = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset + 2)
                SampleRate = [BitConverter]::ToUInt32($Bytes, $chunkDataOffset + 4)
                ByteRate = [BitConverter]::ToUInt32($Bytes, $chunkDataOffset + 8)
                BlockAlign = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset + 12)
                BitsPerSample = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset + 14)
            }
        } elseif ($chunkId -eq 'data') {
            $dataOffset = $chunkDataOffset
            $dataSize = [Math]::Min([int]$chunkSize, $Bytes.Length - $chunkDataOffset)
            break
        }

        if ($nextOffset -le $offset) {
            throw 'Invalid WAV chunk layout'
        }
        $offset = $nextOffset
    }

    if ($null -eq $format) {
        throw 'Missing fmt chunk'
    }
    if ($dataOffset -lt 0) {
        throw 'Missing data chunk'
    }
    if ($format.AudioFormat -ne 1) {
        throw "Only PCM WAV is supported, got audio_format=$($format.AudioFormat)"
    }
    if ($format.BitsPerSample -ne 16) {
        throw "Only 16-bit PCM WAV is supported, got bits_per_sample=$($format.BitsPerSample)"
    }
    if ($format.Channels -le 0 -or $format.BlockAlign -le 0 -or $format.SampleRate -le 0) {
        throw 'Invalid WAV format metadata'
    }

    $frameCount = [int][Math]::Floor($dataSize / $format.BlockAlign)
    return [pscustomobject]@{
        AudioFormat = $format.AudioFormat
        Channels = [int]$format.Channels
        SampleRate = [int]$format.SampleRate
        ByteRate = [int]$format.ByteRate
        BlockAlign = [int]$format.BlockAlign
        BitsPerSample = [int]$format.BitsPerSample
        DataOffset = [int]$dataOffset
        DataSize = [int]$dataSize
        FrameCount = [int]$frameCount
        DurationSeconds = [double]$frameCount / [double]$format.SampleRate
    }
}

function Clamp-Int16 {
    param([double]$Value)

    if ($Value -gt 32767.0) {
        return [int16]32767
    }
    if ($Value -lt -32768.0) {
        return [int16]-32768
    }
    return [int16][Math]::Round($Value)
}

function Convert-WavToMono16kPcm16Le {
    param(
        [byte[]]$Bytes,
        $Header
    )

    if ($Header.SampleRate -eq 16000 -and $Header.Channels -eq 1 -and $Header.BitsPerSample -eq 16 -and $Header.BlockAlign -eq 2) {
        $outBytes = New-Object byte[] $Header.DataSize
        [Array]::Copy($Bytes, $Header.DataOffset, $outBytes, 0, $Header.DataSize)
        return [pscustomobject]@{
            PcmBytes = $outBytes
            SampleRate = 16000
            FrameCount = [int]$Header.FrameCount
            DurationSeconds = [double]$Header.FrameCount / 16000.0
            SourceSampleRate = $Header.SampleRate
            SourceChannels = $Header.Channels
        }
    }

    $mono = New-Object 'double[]' $Header.FrameCount
    for ($frameIndex = 0; $frameIndex -lt $Header.FrameCount; $frameIndex++) {
        $frameBase = $Header.DataOffset + ($frameIndex * $Header.BlockAlign)
        $sum = 0.0
        for ($channelIndex = 0; $channelIndex -lt $Header.Channels; $channelIndex++) {
            $sampleOffset = $frameBase + ($channelIndex * 2)
            $sum += [double][BitConverter]::ToInt16($Bytes, $sampleOffset)
        }
        $mono[$frameIndex] = ($sum / [double]$Header.Channels) / 32768.0
    }

    $targetRate = 16000
    if ($Header.SampleRate -eq $targetRate) {
        $outBytes = New-Object byte[] ($mono.Length * 2)
        for ($index = 0; $index -lt $mono.Length; $index++) {
            $pcmValue = Clamp-Int16 ($mono[$index] * 32768.0)
            [BitConverter]::GetBytes($pcmValue).CopyTo($outBytes, $index * 2)
        }
        return [pscustomobject]@{
            PcmBytes = $outBytes
            SampleRate = $targetRate
            FrameCount = [int]$mono.Length
            DurationSeconds = [double]$mono.Length / [double]$targetRate
            SourceSampleRate = $Header.SampleRate
            SourceChannels = $Header.Channels
        }
    }

    $ratio = [double]$Header.SampleRate / [double]$targetRate
    $outFrames = [int][Math]::Max(1, [Math]::Floor(($mono.Length / $ratio)))
    $outBytes = New-Object byte[] ($outFrames * 2)
    for ($outIndex = 0; $outIndex -lt $outFrames; $outIndex++) {
        $srcPos = [double]$outIndex * $ratio
        $leftIndex = [int][Math]::Floor($srcPos)
        if ($leftIndex -ge $mono.Length) {
            $leftIndex = $mono.Length - 1
        }
        $rightIndex = [Math]::Min($leftIndex + 1, $mono.Length - 1)
        $frac = $srcPos - [double]$leftIndex
        $sample = ($mono[$leftIndex] * (1.0 - $frac)) + ($mono[$rightIndex] * $frac)
        $pcmValue = Clamp-Int16 ($sample * 32768.0)
        [BitConverter]::GetBytes($pcmValue).CopyTo($outBytes, $outIndex * 2)
    }

    return [pscustomobject]@{
        PcmBytes = $outBytes
        SampleRate = $targetRate
        FrameCount = [int]$outFrames
        DurationSeconds = [double]$outFrames / [double]$targetRate
        SourceSampleRate = $Header.SampleRate
        SourceChannels = $Header.Channels
    }
}

function Invoke-JsonGet {
    param([string]$Uri)

    return Invoke-RestMethod -Method Get -Uri $Uri
}

function Invoke-JsonPost {
    param(
        [string]$Uri,
        [byte[]]$Body
    )

    if ($null -eq $Body) {
        return Invoke-RestMethod -Method Post -Uri $Uri
    }
    return Invoke-RestMethod -Method Post -Uri $Uri -Body $Body -ContentType 'application/octet-stream'
}

function Wait-ServerReady {
    param(
        [string]$HealthUrl,
        [int]$TimeoutSeconds,
        [int]$PollMs
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-JsonGet -Uri $HealthUrl
            if ($health.status -eq 'ok') {
                return
            }
        } catch {
        }
        Start-Sleep -Milliseconds $PollMs
    }
    throw "server did not become ready within ${TimeoutSeconds}s: $HealthUrl"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'build'
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$wavBytes = [System.IO.File]::ReadAllBytes($WavPath)
$header = Parse-WavHeader -Bytes $wavBytes
$prepared = Convert-WavToMono16kPcm16Le -Bytes $wavBytes -Header $header
$baseUri = $BaseUrl.TrimEnd('/')
$totalFrames = [int]$prepared.FrameCount
if ($MaxAudioSeconds -gt 0) {
    $maxFrames = [int]($MaxAudioSeconds * 16000)
    if ($maxFrames -gt 0 -and $maxFrames -lt $totalFrames) {
        $totalFrames = $maxFrames
    }
}
$audioDurationMs = [int][Math]::Round(($totalFrames * 1000.0) / 16000.0)

Write-Host ("[{0}] waiting for server {1}" -f $ModelLabel, $baseUri)
Wait-ServerReady -HealthUrl ($baseUri + '/health') -TimeoutSeconds $ReadyTimeoutSeconds -PollMs $PollIntervalMs

$metricsBefore = Invoke-JsonGet -Uri ($baseUri + '/api/metrics')
$startResponse = Invoke-JsonPost -Uri ($baseUri + '/api/realtime/start') -Body $null
$sessionId = [string]$startResponse.session_id
if ([string]::IsNullOrWhiteSpace($sessionId)) {
    throw 'missing session_id from /api/realtime/start'
}

$random = [System.Random]::new($Seed)
$startedAt = Get-Date
$sentFrames = 0
$pcmBytes = [byte[]]$prepared.PcmBytes
$lastStable = ''
$firstStableWallMs = $null
$firstStableAudioMs = $null
$stableEvents = 0
$maxStableDeltaChars = 0

function Add-TimelineEvent {
    param(
        [string]$Phase,
        $State,
        [int]$SentFramesNow
    )

    $stableText = ''
    if ($null -ne $State.stable_text) {
        $stableText = [string]$State.stable_text
    }
    if ($stableText -eq $script:lastStable) {
        return
    }

    $elapsedMs = [int][Math]::Round(((Get-Date) - $script:startedAt).TotalMilliseconds)
    $audioSentMs = [int][Math]::Round(($SentFramesNow * 1000.0) / 16000.0)
    $deltaChars = [Math]::Max(0, $stableText.Length - $script:lastStable.Length)
    if ($stableText.Length -gt 0 -and $null -eq $script:firstStableWallMs) {
        $script:firstStableWallMs = $elapsedMs
        $script:firstStableAudioMs = $audioSentMs
    }
    if ($deltaChars -gt 0) {
        $script:stableEvents += 1
        if ($deltaChars -gt $script:maxStableDeltaChars) {
            $script:maxStableDeltaChars = $deltaChars
        }
    }

    $script:lastStable = $stableText
}

while ($sentFrames -lt $totalFrames) {
    $remainingFrames = $totalFrames - $sentFrames
    $chunkMs = $random.Next($MinChunkMs, $MaxChunkMs + 1)
    $chunkFrames = [int][Math]::Round((16000.0 * $chunkMs) / 1000.0)
    if ($chunkFrames -lt 1) {
        $chunkFrames = 1
    }
    if ($chunkFrames -gt $remainingFrames) {
        $chunkFrames = $remainingFrames
    }

    $chunkBytesLen = $chunkFrames * 2
    $chunkBytes = New-Object byte[] $chunkBytesLen
    [Array]::Copy($pcmBytes, $sentFrames * 2, $chunkBytes, 0, $chunkBytesLen)

    $chunkState = Invoke-JsonPost -Uri ($baseUri + '/api/realtime/chunk?session_id=' + [Uri]::EscapeDataString($sessionId)) -Body $chunkBytes
    $sentFrames += $chunkFrames
    Add-TimelineEvent -Phase 'chunk' -State $chunkState -SentFramesNow $sentFrames

    $targetElapsedMs = [int][Math]::Round(($sentFrames * 1000.0) / 16000.0)
    while ([int][Math]::Round(((Get-Date) - $startedAt).TotalMilliseconds) -lt $targetElapsedMs) {
        $elapsedMs = [int][Math]::Round(((Get-Date) - $startedAt).TotalMilliseconds)
        $remainingMs = $targetElapsedMs - $elapsedMs
        if ($remainingMs -le 0) {
            break
        }
        $sleepMs = [Math]::Min($PollIntervalMs, $remainingMs)
        Start-Sleep -Milliseconds $sleepMs
        $statusState = Invoke-JsonGet -Uri ($baseUri + '/api/realtime/status?session_id=' + [Uri]::EscapeDataString($sessionId))
        Add-TimelineEvent -Phase 'poll' -State $statusState -SentFramesNow $sentFrames
    }
}

$stopState = Invoke-JsonPost -Uri ($baseUri + '/api/realtime/stop?session_id=' + [Uri]::EscapeDataString($sessionId)) -Body $null
Add-TimelineEvent -Phase 'stop' -State $stopState -SentFramesNow $sentFrames
$metricsAfter = Invoke-JsonGet -Uri ($baseUri + '/api/metrics')

$finishedAt = Get-Date
$finalWallMs = [int][Math]::Round(($finishedAt - $startedAt).TotalMilliseconds)
$finalLagMs = $finalWallMs - $audioDurationMs
$firstStableLagMs = if ($null -ne $firstStableWallMs -and $null -ne $firstStableAudioMs) {
    $firstStableWallMs - $firstStableAudioMs
} else {
    $null
}
$smoothStableOutput = ($stableEvents -ge 2) -and ($null -ne $firstStableWallMs) -and ($firstStableWallMs -lt $audioDurationMs)

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$safeModel = ($ModelLabel -replace '[^A-Za-z0-9._-]', '_')
$resultBase = Join-Path $OutputDir ("realtime_bench_{0}_{1}" -f $safeModel, $timestamp)
$finalText = ''
if ($null -ne $stopState.text) {
    $finalText = [string]$stopState.text
}
$finalStableText = ''
if ($null -ne $stopState.stable_text) {
    $finalStableText = [string]$stopState.stable_text
}
$finalDisplayText = ''
if ($null -ne $stopState.display_text) {
    $finalDisplayText = [string]$stopState.display_text
}
$finalInferenceMs = 0.0
if ($null -ne $stopState.inference_ms) {
    $finalInferenceMs = [double]$stopState.inference_ms
}

$summary = [ordered]@{
    model_label = $ModelLabel
    base_url = $baseUri
    wav_path = $WavPath
    seed = $Seed
    max_audio_seconds = $MaxAudioSeconds
    chunk_range_ms = @($MinChunkMs, $MaxChunkMs)
    poll_interval_ms = $PollIntervalMs
    source_sample_rate = $prepared.SourceSampleRate
    source_channels = $prepared.SourceChannels
    prepared_sample_rate = $prepared.SampleRate
    audio_duration_ms = $audioDurationMs
    first_stable_wall_ms = $firstStableWallMs
    first_stable_audio_ms = $firstStableAudioMs
    first_stable_lag_ms = $firstStableLagMs
    stable_events = $stableEvents
    max_stable_delta_chars = $maxStableDeltaChars
    final_wall_ms = $finalWallMs
    final_lag_ms = $finalLagMs
    processing_rtf = [Math]::Round(($finalLagMs / [double]$audioDurationMs), 3)
    e2e_rtf = [Math]::Round(($finalWallMs / [double]$audioDurationMs), 3)
    realtime_factor = [Math]::Round(($finalWallMs / [double]$audioDurationMs), 3)
    smooth_stable_output = $smoothStableOutput
    final_text = $finalText
    final_stable_text = $finalStableText
    final_display_text = $finalDisplayText
    final_inference_ms = $finalInferenceMs
    metrics_before = $metricsBefore
    metrics_after = $metricsAfter
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -Path ($resultBase + '.json') -Encoding UTF8

Write-Host ("[{0}] session={1}" -f $ModelLabel, $sessionId)
Write-Host ("[{0}] audio_ms={1} first_stable_wall_ms={2} first_stable_lag_ms={3} stable_events={4}" -f
    $ModelLabel,
    $audioDurationMs,
    $(if ($null -ne $firstStableWallMs) { $firstStableWallMs } else { 'n/a' }),
    $(if ($null -ne $firstStableLagMs) { $firstStableLagMs } else { 'n/a' }),
    $stableEvents)
$procRtf = [Math]::Round(($finalLagMs / [double]$audioDurationMs), 3)
Write-Host ("[{0}] final_wall_ms={1} final_lag_ms={2} processing_rtf={3} smooth={4}" -f
    $ModelLabel,
    $finalWallMs,
    $finalLagMs,
    $procRtf,
    $smoothStableOutput)
Write-Host ("[{0}] wrote {1}.json" -f $ModelLabel, $resultBase)