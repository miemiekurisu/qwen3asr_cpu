#!/usr/bin/env python3
"""Verify heartbeat during in-session silence (no EOF).

Push 3 s of audio, then 4 s of silence, then 3 s of audio.
Heartbeats should fire during the silence gap.
"""
import json
import sys
import time
import urllib.request
import wave

BASE = "http://127.0.0.1:19991"
SR = 16000
CHUNK_MS = 400
CHUNK_SAMPLES = SR * CHUNK_MS // 1000


def post(path, body=None, content_type=None):
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    wav_path = sys.argv[1]
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == SR
        audio = w.readframes(w.getnframes())
    total = len(audio) // 2
    print(f"Audio: {total/SR:.1f} s")

    sid = post("/api/realtime/start")["session_id"]
    t0 = time.time()

    # Phase 1: push 3 s of audio
    print("\n--- Phase 1: pushing 3 s audio ---")
    pushed = 0
    while pushed < 3 * SR and pushed < total:
        end = min(pushed + CHUNK_SAMPLES, total)
        chunk = audio[pushed * 2 : end * 2]
        post(f"/api/realtime/chunk?session_id={sid}", body=chunk,
             content_type="application/octet-stream")
        pushed = end
        time.sleep(CHUNK_MS / 1000)
    print(f"  pushed={pushed/SR:.1f}s, wall={time.time()-t0:.2f}s")

    # Phase 2: silence gap — 4 s of NOT pushing anything
    print("\n--- Phase 2: 4 s of silence (no pushes) ---")
    heartbeat_polls = 0
    total_polls = 0
    silence_start = time.time()
    while time.time() - silence_start < 4.0:
        snap = get(f"/api/realtime/status?session_id={sid}")
        total_polls += 1
        if snap["decoded"]:
            heartbeat_polls += 1
        time.sleep(0.3)
    print(f"  total polls during silence: {total_polls}")
    print(f"  polls with decoded=true:    {heartbeat_polls}")
    print(f"  (heartbeat fires ~once per {600}ms via wait-loop callback)")

    # Phase 3: push 3 more seconds
    print("\n--- Phase 3: pushing 3 s more audio ---")
    while pushed < 6 * SR and pushed < total:
        end = min(pushed + CHUNK_SAMPLES, total)
        chunk = audio[pushed * 2 : end * 2]
        post(f"/api/realtime/chunk?session_id={sid}", body=chunk,
             content_type="application/octet-stream")
        pushed = end
        time.sleep(CHUNK_MS / 1000)
    print(f"  pushed={pushed/SR:.1f}s, wall={time.time()-t0:.2f}s")

    # Final status
    snap = get(f"/api/realtime/status?session_id={sid}")
    print(f"\n=== Final ===")
    print(f"  Stable:    {snap.get('stable_text', '')}")
    print(f"  Partial:   {snap.get('partial_text', '')}")
    print(f"  Inference: {snap.get('inference_ms', 0):.0f} ms")

    # Stop without EOF
    post(f"/api/realtime/stop?session_id={sid}")
    print(f"\n=== Heartbeat validation ===")
    expected = 4.0 / 0.6  # ~6.6 heartbeats in 4 s
    print(f"  Expected ~{expected:.1f} heartbeats in 4 s of silence")
    print(f"  Got:      {heartbeat_polls}")
    if heartbeat_polls < 3:
        print(f"  FAIL: heartbeat not firing during silence!")
        sys.exit(1)
    else:
        print(f"  PASS: heartbeat fires during silence")


if __name__ == "__main__":
    main()
