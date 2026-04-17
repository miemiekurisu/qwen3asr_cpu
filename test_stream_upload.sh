#!/bin/bash
# Test streaming upload path (same as web UI) with the 57-min audio file.
# Measures repetition in output.
set -e

SERVER="http://127.0.0.1:3458"
AUDIO="/Users/kurisu/Library/CloudStorage/OneDrive-个人/文档/orange/wavs/output.wav"

echo "=== Streaming upload test ==="
echo "File: $AUDIO"
echo "Size: $(wc -c < "$AUDIO") bytes"

# Start realtime session
echo "Starting session..."
START_RESP=$(curl -s -X POST "$SERVER/api/realtime/start" \
  -H 'Content-Type: application/json' \
  -d '{"language":"auto"}')
SID=$(echo "$START_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session ID: $SID"

# Upload file as raw PCM chunks via the streaming endpoint
# The web UI reads WAV, converts to mono 16-bit 16kHz PCM, sends via /api/realtime/chunk
# For testing, we use ffmpeg to convert and pipe chunks
echo "Converting and uploading..."
TMPPCM=$(mktemp /tmp/qasr_test_XXXXXX.raw)
ffmpeg -y -i "$AUDIO" -f s16le -acodec pcm_s16le -ac 1 -ar 16000 "$TMPPCM" 2>/dev/null
PCMSIZE=$(wc -c < "$TMPPCM")
echo "PCM size: $PCMSIZE bytes"

# Upload in 64KB chunks (32K samples = 2 seconds at 16kHz)
CHUNK=65536
OFFSET=0
CHUNKS_SENT=0
while [ $OFFSET -lt $PCMSIZE ]; do
  REMAINING=$((PCMSIZE - OFFSET))
  LEN=$CHUNK
  if [ $LEN -gt $REMAINING ]; then LEN=$REMAINING; fi
  dd if="$TMPPCM" bs=1 skip=$OFFSET count=$LEN 2>/dev/null | \
    curl -s -X POST "$SERVER/api/realtime/chunk?session_id=$SID" \
      --data-binary @- -H 'Content-Type: application/octet-stream' > /dev/null
  OFFSET=$((OFFSET + LEN))
  CHUNKS_SENT=$((CHUNKS_SENT + 1))
  if [ $((CHUNKS_SENT % 100)) -eq 0 ]; then
    SECS=$((OFFSET / 32000))
    echo "  Uploaded $CHUNKS_SENT chunks (~${SECS}s audio)..."
  fi
done
echo "Upload complete: $CHUNKS_SENT chunks"
rm -f "$TMPPCM"

# Signal EOF
echo "Sending EOF..."
curl -s -X POST "$SERVER/api/realtime/eof?session_id=$SID" > /dev/null

# Poll until done
echo "Waiting for transcription to complete..."
for i in $(seq 1 1200); do
  RESP=$(curl -s "$SERVER/api/realtime/status?session_id=$SID")
  STATE=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('state',''))" 2>/dev/null)
  if [ "$STATE" = "done" ] || [ "$STATE" = "error" ]; then
    echo ""
    echo "=== RESULT (state=$STATE) ==="
    # Get full transcript
    FULL=$(curl -s -X POST "$SERVER/api/realtime/stop?session_id=$SID")
    TEXT=$(echo "$FULL" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text',''))")
    echo "$TEXT" > /tmp/qasr_stream_result.txt
    NCHARS=$(echo "$TEXT" | wc -m)
    echo "Total chars: $NCHARS"
    echo ""
    echo "--- First 500 chars ---"
    echo "$TEXT" | head -c 500
    echo ""
    echo ""
    echo "--- Last 500 chars ---"
    echo "$TEXT" | tail -c 500
    echo ""

    # Check for repetition: find any 30+ char substring that appears 3+ times
    echo ""
    echo "=== Repetition analysis ==="
    python3 -c "
import sys
text = open('/tmp/qasr_stream_result.txt').read()
print(f'Total length: {len(text)} chars')

# Check for repeated 30-char blocks
from collections import Counter
blocks = Counter()
for i in range(len(text) - 30):
    block = text[i:i+30]
    blocks[block] += 1

repeated = [(b, c) for b, c in blocks.items() if c >= 3]
repeated.sort(key=lambda x: -x[1])

if not repeated:
    print('NO repeated 30-char blocks found (GOOD)')
else:
    print(f'Found {len(repeated)} repeated 30-char blocks:')
    for block, count in repeated[:10]:
        print(f'  [{count}x] {repr(block)}')
"
    break
  fi
  # Progress indicator every 10s
  if [ $((i % 10)) -eq 0 ]; then
    PARTIAL=$(echo "$RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
st=d.get('stable_text','')
print(f'  ...processing ({len(st)} chars stable)...')
" 2>/dev/null)
    echo "$PARTIAL"
  fi
  sleep 1
done
