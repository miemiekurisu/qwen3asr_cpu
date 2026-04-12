$base = "http://127.0.0.1:8082"
$wavPath = "D:\workspace\github\qwen-asr-provider\testfile\long.wav"

# Read WAV file
$bytes = [System.IO.File]::ReadAllBytes($wavPath)
Write-Host "WAV file size: $($bytes.Length) bytes"

# Parse WAV header to find data chunk
$dataOffset = 12
while ($dataOffset + 8 -le $bytes.Length) {
    $chunkId = [System.Text.Encoding]::ASCII.GetString($bytes, $dataOffset, 4)
    $chunkSize = [BitConverter]::ToUInt32($bytes, $dataOffset + 4)
    if ($chunkId -eq "data") {
        $dataOffset += 8
        break
    }
    $dataOffset += 8 + $chunkSize
}
Write-Host "PCM data offset: $dataOffset, PCM data size: $($bytes.Length - $dataOffset)"

# Step 1: Start realtime session
Write-Host "`n=== Starting realtime session ==="
$startResp = Invoke-RestMethod -Uri "$base/api/realtime/start" -Method Post -Body "" -ContentType "application/json"
$sessionId = $startResp.session_id
Write-Host "Session ID: $sessionId"

# Step 2: Upload PCM chunks (16-bit mono 16kHz)
# Send in ~2 second chunks (2 * 16000 * 2 = 64000 bytes)
$chunkSize = 320000
$pcmData = $bytes[$dataOffset..($bytes.Length - 1)]
$totalPcm = $pcmData.Length
$uploaded = 0
$chunkNum = 0

Write-Host "`n=== Uploading PCM chunks (total $totalPcm bytes) ==="
while ($uploaded -lt $totalPcm) {
    $remaining = $totalPcm - $uploaded
    $thisChunk = [Math]::Min($chunkSize, $remaining)
    $chunk = New-Object byte[] $thisChunk
    [Array]::Copy($pcmData, $uploaded, $chunk, 0, $thisChunk)
    
    try {
        $resp = Invoke-RestMethod -Uri "$base/api/realtime/chunk?session_id=$sessionId" -Method Post -Body $chunk -ContentType "application/octet-stream" -TimeoutSec 30
    } catch {
        Write-Host "Upload error at chunk $chunkNum : $_"
        break
    }
    
    $uploaded += $thisChunk
    $chunkNum++
    $pct = [math]::Round(100 * $uploaded / $totalPcm, 1)
    if ($chunkNum % 20 -eq 0) {
        Write-Host "  Uploaded $chunkNum chunks ($pct%)"
    }
}
Write-Host "Upload complete: $chunkNum chunks, $uploaded bytes"

# Step 3: Finalize (stop)
Write-Host "`n=== Finalizing session ==="
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $stopResp = Invoke-RestMethod -Uri "$base/api/realtime/stop?session_id=$sessionId" -Method Post -Body "" -TimeoutSec 600
    $sw.Stop()
    Write-Host "Finalize completed in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s"
    Write-Host "Text: $($stopResp.text)"
    Write-Host "Stable: $($stopResp.stable_text)"
} catch {
    $sw.Stop()
    Write-Host "Finalize FAILED after $([math]::Round($sw.Elapsed.TotalSeconds, 1))s: $_"
}
