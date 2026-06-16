param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$WavPath,

    [Parameter(Mandatory = $true)]
    [string]$ModelLabel,

    [string]$OutputDir = "",
    [string]$EndpointPath = "/api/transcriptions",
    [int]$MaxAudioSeconds = 0,
    [int]$ReadyTimeoutSeconds = 180,
    [int]$PollIntervalMs = 250
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

function Parse-WavDurationMs {
    param([byte[]]$Bytes)

    if ($Bytes.Length -lt 44) {
        throw "WAV too small: $($Bytes.Length) bytes"
    }
    if ((Get-AsciiString $Bytes 0 4) -ne 'RIFF' -or (Get-AsciiString $Bytes 8 4) -ne 'WAVE') {
        throw 'Only RIFF/WAVE files are supported'
    }

    $offset = 12
    $sampleRate = 0
    $blockAlign = 0
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
            $sampleRate = [BitConverter]::ToUInt32($Bytes, $chunkDataOffset + 4)
            $blockAlign = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset + 12)
        } elseif ($chunkId -eq 'data') {
            $dataSize = [Math]::Min([int]$chunkSize, $Bytes.Length - $chunkDataOffset)
            break
        }

        if ($nextOffset -le $offset) {
            break
        }
        $offset = $nextOffset
    }

    if ($sampleRate -le 0 -or $blockAlign -le 0 -or $dataSize -le 0) {
        throw 'Could not derive WAV duration'
    }
    $frameCount = [double]$dataSize / [double]$blockAlign
    return [int][Math]::Round(($frameCount * 1000.0) / [double]$sampleRate)
}

function Parse-WavLayout {
    param([byte[]]$Bytes)

    if ($Bytes.Length -lt 44) {
        throw "WAV too small: $($Bytes.Length) bytes"
    }
    if ((Get-AsciiString $Bytes 0 4) -ne 'RIFF' -or (Get-AsciiString $Bytes 8 4) -ne 'WAVE') {
        throw 'Only RIFF/WAVE files are supported'
    }

    $offset = 12
    $sampleRate = 0
    $blockAlign = 0
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
            $sampleRate = [BitConverter]::ToUInt32($Bytes, $chunkDataOffset + 4)
            $blockAlign = [BitConverter]::ToUInt16($Bytes, $chunkDataOffset + 12)
        } elseif ($chunkId -eq 'data') {
            $dataOffset = $chunkDataOffset
            $dataSize = [Math]::Min([int]$chunkSize, $Bytes.Length - $chunkDataOffset)
            break
        }

        if ($nextOffset -le $offset) {
            break
        }
        $offset = $nextOffset
    }

    if ($sampleRate -le 0 -or $blockAlign -le 0 -or $dataOffset -lt 0 -or $dataSize -le 0) {
        throw 'Could not derive WAV layout'
    }

    return [pscustomobject]@{
        SampleRate = [int]$sampleRate
        BlockAlign = [int]$blockAlign
        DataOffset = [int]$dataOffset
        DataSize = [int]$dataSize
    }
}

function Build-TrimmedWav {
    param(
        [byte[]]$Bytes,
        [int]$MaxAudioSeconds
    )

    $audioDurationMs = Parse-WavDurationMs -Bytes $Bytes
    if ($MaxAudioSeconds -le 0) {
        return [pscustomobject]@{
            Bytes = $Bytes
            AudioDurationMs = $audioDurationMs
        }
    }

    $layout = Parse-WavLayout -Bytes $Bytes
    $maxFrames = [int]($MaxAudioSeconds * $layout.SampleRate)
    if ($maxFrames -le 0) {
        return [pscustomobject]@{
            Bytes = $Bytes
            AudioDurationMs = $audioDurationMs
        }
    }

    $maxDataBytes = $maxFrames * $layout.BlockAlign
    $trimmedDataSize = [Math]::Min($layout.DataSize, $maxDataBytes)
    if ($trimmedDataSize -ge $layout.DataSize) {
        return [pscustomobject]@{
            Bytes = $Bytes
            AudioDurationMs = $audioDurationMs
        }
    }

    $outBytes = New-Object byte[] ($layout.DataOffset + $trimmedDataSize)
    [Array]::Copy($Bytes, 0, $outBytes, 0, $layout.DataOffset)
    [Array]::Copy($Bytes, $layout.DataOffset, $outBytes, $layout.DataOffset, $trimmedDataSize)
    [BitConverter]::GetBytes([uint32]($outBytes.Length - 8)).CopyTo($outBytes, 4)
    [BitConverter]::GetBytes([uint32]$trimmedDataSize).CopyTo($outBytes, $layout.DataOffset - 4)

    return [pscustomobject]@{
        Bytes = $outBytes
        AudioDurationMs = [int][Math]::Round((($trimmedDataSize / [double]$layout.BlockAlign) * 1000.0) / [double]$layout.SampleRate)
    }
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
            $health = Invoke-RestMethod -Method Get -Uri $HealthUrl
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

$baseUri = $BaseUrl.TrimEnd('/')
$endpointUri = $baseUri + $EndpointPath
$sourceBytes = [System.IO.File]::ReadAllBytes($WavPath)
$preparedInput = Build-TrimmedWav -Bytes $sourceBytes -MaxAudioSeconds $MaxAudioSeconds
$wavBytes = [byte[]]$preparedInput.Bytes
$audioDurationMs = [int]$preparedInput.AudioDurationMs

Write-Host ("[{0}] waiting for server {1}" -f $ModelLabel, $baseUri)
Wait-ServerReady -HealthUrl ($baseUri + '/health') -TimeoutSeconds $ReadyTimeoutSeconds -PollMs $PollIntervalMs

$metricsBefore = Invoke-RestMethod -Method Get -Uri ($baseUri + '/api/metrics')

$boundary = [System.Guid]::NewGuid().ToString('N')
$CRLF = "`r`n"
$headerText = "--$boundary${CRLF}Content-Disposition: form-data; name=`"file`"; filename=`"audio.wav`"${CRLF}Content-Type: audio/wav${CRLF}${CRLF}"
$footerText = "${CRLF}--${boundary}--${CRLF}"
$headerBytes = [System.Text.Encoding]::UTF8.GetBytes($headerText)
$footerBytes = [System.Text.Encoding]::UTF8.GetBytes($footerText)
$bodyBytes = New-Object byte[] ($headerBytes.Length + $wavBytes.Length + $footerBytes.Length)
[Array]::Copy($headerBytes, 0, $bodyBytes, 0, $headerBytes.Length)
[Array]::Copy($wavBytes, 0, $bodyBytes, $headerBytes.Length, $wavBytes.Length)
[Array]::Copy($footerBytes, 0, $bodyBytes, $headerBytes.Length + $wavBytes.Length, $footerBytes.Length)

$startedAt = Get-Date
$response = Invoke-WebRequest -Method Post -Uri $endpointUri `
    -Body $bodyBytes `
    -ContentType "multipart/form-data; boundary=$boundary" `
    -UseBasicParsing
$finishedAt = Get-Date

if ($response.StatusCode -ne 200) {
    throw "offline benchmark failed: HTTP $($response.StatusCode) $($response.Content)"
}

$payload = $response.Content | ConvertFrom-Json
$metricsAfter = Invoke-RestMethod -Method Get -Uri ($baseUri + '/api/metrics')
$wallMs = [int][Math]::Round(($finishedAt - $startedAt).TotalMilliseconds)
$reportedInferenceMs = 0.0
if ($null -ne $payload.inference_ms) {
    $reportedInferenceMs = [double]$payload.inference_ms
}
$reportedAudioMs = [double]$audioDurationMs
if ($null -ne $payload.audio_ms) {
    $reportedAudioMs = [double]$payload.audio_ms
}
$realtimeFactorVal = 0.0
if ($audioDurationMs -gt 0) {
    $realtimeFactorVal = [Math]::Round(($wallMs / [double]$audioDurationMs), 3)
}
$realtimeFactor = $realtimeFactorVal
$text = ''
if ($null -ne $payload.text) {
    $text = [string]$payload.text
}
$tokens = 0
if ($null -ne $payload.tokens) {
    $tokens = [int]$payload.tokens
}
$language = ''
if ($null -ne $payload.language) {
    $language = [string]$payload.language
}

$summary = [ordered]@{
    model_label = $ModelLabel
    base_url = $baseUri
    endpoint = $EndpointPath
    wav_path = $WavPath
    max_audio_seconds = $MaxAudioSeconds
    audio_duration_ms = $audioDurationMs
    wall_ms = $wallMs
    reported_inference_ms = $reportedInferenceMs
    reported_audio_ms = $reportedAudioMs
    realtime_factor = $realtimeFactor
    text_length = $text.Length
    text = $text
    tokens = $tokens
    language = $language
    metrics_before = $metricsBefore
    metrics_after = $metricsAfter
    response = $payload
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$safeModel = ($ModelLabel -replace '[^A-Za-z0-9._-]', '_')
$resultBase = Join-Path $OutputDir ("offline_bench_{0}_{1}" -f $safeModel, $timestamp)

$summary | ConvertTo-Json -Depth 8 | Set-Content -Path ($resultBase + '.json') -Encoding UTF8

Write-Host ("[{0}] wall_ms={1} inference_ms={2} audio_ms={3} rtf={4}" -f
    $ModelLabel,
    $wallMs,
    $reportedInferenceMs,
    $reportedAudioMs,
    $realtimeFactor)
Write-Host ("[{0}] wrote {1}.json" -f $ModelLabel, $resultBase)