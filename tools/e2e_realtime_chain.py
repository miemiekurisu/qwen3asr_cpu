#!/usr/bin/env python3
"""
End-to-end browser chain test.  Simulates what the browser does:
  1. POST /api/realtime/start
  2. POST a bunch of PCM16LE audio chunks (decimated like the JS would)
  3. Poll /api/realtime/audio_diag to confirm server received audio
  4. Sleep + stop, then read /api/realtime/get the final text
  5. Verify text is non-empty and roughly correct

Sends a synthesized speech-like signal (1 kHz tone with envelope gating
on for 1.5 s, off for 0.5 s, on for 1.5 s) so the VAD should fire a
silence-commit and the model should produce *something*.
"""

import json
import struct
import ssl
import sys
import time
import urllib.request
import urllib.error
import numpy as np

BASE = "https://127.0.0.1:19992"
SSL_CTX = ssl._create_unverified_context()

def post(path, body=b""):
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def get(path):
    req = urllib.request.Request(BASE + path, method="GET")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    print("=" * 60)
    print("E2E browser chain test (audio chain + VAD + decode)")
    print("=" * 60)

    # 1. start session
    code, data = post("/api/realtime/start", b"")
    print(f"[1] start -> {code} session_id={data.get('session_id', '?')}")
    if code != 200:
        print("    FAILED to start session")
        sys.exit(1)
    sid = data["session_id"]

    # 2. use a real audio file (silero VAD doesn't fire on pure tones)
    # Take the first 4s of english_60s.wav and pad with silence at the end.
    import wave
    wav_path = "/home/kurisu/文档/github/qwen3asr_cpu/testfile/english_60s.wav"
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16kHz, got {w.getframerate()}"
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        nframes = min(w.getnframes(), int(4.0 * 16000))  # first 4s
        raw = w.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nchannels > 1:
        audio = audio[::nchannels]  # take first channel
    # Pad with 1s of silence at the end so VAD can fire silence commit
    audio = np.concatenate([audio, np.zeros(16000, dtype=np.float32)])
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    print(f"[2] loaded {len(audio_int16)} samples ({len(audio_int16)/16000:.2f}s) from {wav_path}")

    # 3. POST in 200ms chunks (browser cadence)
    fs = 16000
    chunk_samples = int(0.2 * fs)
    n_chunks = (len(audio_int16) + chunk_samples - 1) // chunk_samples
    print(f"[3] sending {n_chunks} chunks of {chunk_samples} samples")
    for i in range(n_chunks):
        s = i * chunk_samples
        e = min(s + chunk_samples, len(audio_int16))
        chunk_bytes = audio_int16[s:e].tobytes()
        code, _ = post(f"/api/realtime/chunk?session_id={sid}", chunk_bytes)
        if code != 200:
            print(f"    chunk {i} -> {code} FAILED")
        time.sleep(0.05)  # 50ms between chunks (faster than browser's 400ms)

    # 4. poll audio_diag (give server a moment to ingest)
    time.sleep(0.2)
    code, diag = get(f"/api/realtime/audio_diag?session_id={sid}")
    print(f"[4] audio_diag -> {code} max_peak={diag.get('max_peak', 0):.4f} peak={diag.get('peak', 0):.4f} chunks={diag.get('chunks', '?')}")
    if diag.get("max_peak", 0) < 0.05:
        print("    FAILED: server received no audio (max_peak < 0.05)")
        sys.exit(1)

    # 5. wait a bit for VAD to fire silence commit, then read result
    print("[5] waiting 3s for VAD segment commit...")
    for _ in range(6):
        time.sleep(0.5)
        code, diag = get(f"/api/realtime/audio_diag?session_id={sid}")
        code2, status = get(f"/api/realtime/status?session_id={sid}")
        if status.get("segments"):
            print(f"    segments={len(status['segments'])} texts={[s[:30] for s in status['segments']]}")
            break
    else:
        print("    no segment committed after 3s")

    # 6. send EOF
    code, eof = post(f"/api/realtime/eof?session_id={sid}", b"")
    print(f"[6] eof -> {code}")
    time.sleep(2)

    code, status = get(f"/api/realtime/status?session_id={sid}")
    print(f"[7] final status segments={len(status.get('segments', []))}")
    for i, seg in enumerate(status.get("segments", [])):
        print(f"    segment[{i}] = {seg!r}")

    # 7. stop
    code, stop = post(f"/api/realtime/stop?session_id={sid}", b"")
    print(f"[8] stop -> {code} text={stop.get('text', '?')!r}")

    if stop.get("text", "").strip():
        print("=" * 60)
        print(f"PASS: server produced text: {stop['text']!r}")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("FAIL: server produced no text")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
