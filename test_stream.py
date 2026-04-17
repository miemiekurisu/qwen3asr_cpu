#!/usr/bin/env python3
"""Test streaming upload (same path as web UI) and check for repetition."""
import subprocess, time, json, os, sys, urllib.request

SERVER = "http://127.0.0.1:3458"
AUDIO = "/Users/kurisu/Library/CloudStorage/OneDrive-个人/文档/orange/wavs/output.wav"
TMPPCM = "/tmp/qasr_test.raw"

# Convert to 16kHz mono PCM
print("Converting audio...")
subprocess.run(["ffmpeg", "-y", "-i", AUDIO, "-f", "s16le", "-acodec", "pcm_s16le",
                "-ac", "1", "-ar", "16000", TMPPCM],
               capture_output=True)
pcm_size = os.path.getsize(TMPPCM)
print("PCM: %d bytes (%.1fs audio)" % (pcm_size, pcm_size / 32000))

# Start session
req = urllib.request.Request(SERVER + "/api/realtime/start", data=b'', method='POST')
resp = urllib.request.urlopen(req)
sid = json.loads(resp.read())["session_id"]
print("Session: %s" % sid)

# Upload in 64KB chunks
CHUNK = 65536
t0 = time.time()
with open(TMPPCM, "rb") as f:
    chunks_sent = 0
    while True:
        data = f.read(CHUNK)
        if not data:
            break
        req = urllib.request.Request(
            "%s/api/realtime/chunk?session_id=%s" % (SERVER, sid),
            data=data, method='POST',
            headers={"Content-Type": "application/octet-stream"})
        urllib.request.urlopen(req)
        chunks_sent += 1
        if chunks_sent % 200 == 0:
            print("  Uploaded %d chunks (~%ds)..." % (chunks_sent, chunks_sent * CHUNK // 32000))

print("Upload done: %d chunks in %.1fs" % (chunks_sent, time.time() - t0))

# EOF
req = urllib.request.Request("%s/api/realtime/eof?session_id=%s" % (SERVER, sid),
                            data=b'', method='POST')
urllib.request.urlopen(req)
print("EOF sent, waiting for transcription...")

# Poll until done
for i in range(1200):
    time.sleep(2)
    try:
        resp = urllib.request.urlopen("%s/api/realtime/status?session_id=%s" % (SERVER, sid))
        raw = resp.read()
        d = json.loads(raw.decode('utf-8', 'replace'))
        fin = d.get('finalized', False)
        dec = d.get('decoded_samples', 0)
        tot = d.get('sample_count', 0)
        st = d.get('stable_text', '')
        if i % 15 == 0:
            elapsed = time.time() - t0
            print("  [%.0fs] decoded=%d/%d stable=%dchars fin=%s" % (elapsed, dec, tot, len(st), fin))
        if fin:
            elapsed = time.time() - t0
            print("\nFINALIZED after %.1fs" % elapsed)
            # Stop and get final
            req = urllib.request.Request("%s/api/realtime/stop?session_id=%s" % (SERVER, sid),
                                       data=b'', method='POST')
            resp = urllib.request.urlopen(req)
            final = json.loads(resp.read().decode('utf-8', 'replace'))
            text = final.get('text', '') or final.get('stable_text', '')
            print("Total chars: %d" % len(text))
            print("\n--- First 400 chars ---")
            print(text[:400])
            print("\n--- Last 400 chars ---")
            print(text[-400:])

            # Repetition check
            from collections import Counter
            blocks = Counter()
            for j in range(len(text) - 30):
                blocks[text[j:j+30]] += 1
            repeated = [(b, c) for b, c in blocks.items() if c >= 3]
            repeated.sort(key=lambda x: -x[1])
            print("\n=== Repetition: %d blocks with 3+ repeats ===" % len(repeated))
            for b, c in repeated[:5]:
                print("  [%dx] %s" % (c, b))
            if not repeated:
                print("  NONE - GOOD")

            # Bad bytes
            bad = text.count('\ufffd')
            if bad:
                print("\nWARNING: %d replacement chars (bad UTF-8 bytes)" % bad)
            break
    except Exception as e:
        if i % 15 == 0:
            print("  [%ds] error: %s" % (i * 2, e))
else:
    print("TIMEOUT after 2400s")

os.remove(TMPPCM)
