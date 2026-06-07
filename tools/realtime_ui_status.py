#!/usr/bin/env python3
"""
Drive a real-time session the same way the UI does, and print EXACTLY
the string that appears in realtimeStatus.textContent on the page.

  realtimeStatus.textContent = `音频 ${audioDur}s / 已解码 ${decodedDur}s /
                                耗时 ${wallElapsed}s / 滞后 ${lag}s /
                                推理 ${infMs}ms / ${decodeLabel}`

The UI sends audio every 400ms and polls status every 150ms. We mimic
that exactly.  We also count how often the displayed text actually
CHANGES (a value of "0.1" difference in any field is a change).
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
SEND_INTERVAL = 0.4
POLL_INTERVAL = 0.15


def post(path, body=None, content_type=None):
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read())


def render_status(data, started_at, perf_now):
    """Mimic exactly what ui/app.js builds into realtimeStatus.textContent."""
    audioDur = data["sample_count"] / 16000
    decodedDur = data["decoded_samples"] / 16000
    wallElapsed = (perf_now - started_at) / 1000
    lag = max(0.0, wallElapsed - decodedDur)
    infMs = data.get("inference_ms", 0.0)
    decodeLabel = "已解码" if data["decoded"] else "待下轮"
    return (
        f"音频 {audioDur:.1f}s / 已解码 {decodedDur:.1f}s / "
        f"耗时 {wallElapsed:.1f}s / 滞后 {lag:.1f}s / "
        f"推理 {infMs:.0f}ms / {decodeLabel}"
    )


def main():
    wav_path = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == SR
        audio = w.readframes(w.getnframes())
    want = int(seconds * SR)
    audio = audio[: want * 2]
    total = len(audio) // 2
    print(f"Streaming {seconds:.0f}s of audio ({total} samples)")
    print("This is what realtimeStatus.textContent shows on the page:\n")

    sid = post("/api/realtime/start")["session_id"]
    started_at = time.time() * 1000  # mimic performance.now() in ms

    # 0 — initial
    snap = get(f"/api/realtime/status?session_id={sid}")
    s = render_status(snap, started_at, time.time() * 1000)
    print(f"  t=0.00s | {s}")

    pushed = 0
    chunk_n = 0
    last_text = s
    text_changes = 0
    unique_texts = {s}

    last_send = time.time()
    last_poll = time.time()
    deadline = started_at / 1000 + seconds + 0.5  # stream 0.5s past audio end

    while time.time() < deadline:
        now = time.time()
        # Send if interval elapsed
        if pushed < total and now - last_send >= SEND_INTERVAL:
            end = min(pushed + CHUNK_SAMPLES, total)
            chunk = audio[pushed * 2 : end * 2]
            post(f"/api/realtime/chunk?session_id={sid}", body=chunk,
                 content_type="application/octet-stream")
            pushed = end
            chunk_n += 1
            last_send = now

        # Poll if interval elapsed
        if now - last_poll >= POLL_INTERVAL:
            snap = get(f"/api/realtime/status?session_id={sid}")
            s = render_status(snap, started_at, time.time() * 1000)
            t = (time.time() * 1000 - started_at) / 1000
            if s != last_text:
                text_changes += 1
                unique_texts.add(s)
                marker = "*"  # mark changes
            else:
                marker = " "
            print(f"  t={t:5.2f}s {marker} {s}")
            last_text = s
            last_poll = now

        time.sleep(0.01)

    print()
    print(f"=== Display stability ===")
    print(f"  Total polls:           ~{int((time.time()*1000-started_at)/1000/POLL_INTERVAL)}")
    print(f"  Text changes:          {text_changes}")
    print(f"  Unique status strings: {len(unique_texts)}")
    snap = get(f"/api/realtime/status?session_id={sid}")
    print(f"\n=== Final stable text ===")
    print(f"  {snap.get('stable_text', '')}")


if __name__ == "__main__":
    main()
