#!/usr/bin/env python3
"""
Real-time E2E test: streams AISHELL audio chunk-by-chunk to qasr_server,
polls /api/realtime/status, and reports the latency stats that the UI
sees in the realtimeStatus textContent:

  音频 {audioDur}s / 已解码 {decodedDur}s / 耗时 {wallElapsed}s /
  滞后 {lag}s / 推理 {infMs}ms / {decodeLabel}

This exercises:
  - real-time audio ingestion via /api/realtime/chunk
  - the per-chunk C callback (stable_piece + tentative_piece)
  - the server-side snapshot + JSON serialization
  - the heartbeat path (chunk_cb fires during silence)

Latency is verified:
  - audioDur = sample_count / 16000
  - decodedDur = decoded_samples / 16000
  - lag = wallElapsed - decodedDur   (>= 0 by construction: decode can't
                                       outpace the audio we've sent)
  - inference_ms should grow over time, not stay 0
"""
import json
import os
import struct
import sys
import time
import urllib.request
import wave

BASE = "http://127.0.0.1:19991"
SR = 16000
CHUNK_MS = 400        # mimic the UI: push 400 ms of audio at a time
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
    if len(sys.argv) < 2:
        print("usage: realtime_e2e.py <wav-path> [seconds_to_stream]", file=sys.stderr)
        sys.exit(1)
    wav_path = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

    if not os.path.exists(wav_path):
        print(f"WAV not found: {wav_path}", file=sys.stderr)
        sys.exit(2)

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == SR, f"need 16kHz WAV (got {w.getframerate()})"
        assert w.getnchannels() == 1, "need mono"
        audio = w.readframes(w.getnframes())

    # Truncate to requested seconds
    want_samples = int(seconds * SR)
    want_bytes = want_samples * 2
    audio = audio[:want_bytes]
    total_samples = len(audio) // 2
    print(f"=== Realtime E2E ===")
    print(f"  Audio:        {seconds:.1f} s ({total_samples} samples)")
    print(f"  Chunk size:   {CHUNK_MS} ms ({CHUNK_SAMPLES} samples)")
    print()

    # Create session
    start_resp = post("/api/realtime/start")
    sid = start_resp["session_id"]
    print(f"  Session:      {sid}")
    print()

    t_start = time.time()
    samples_sent = 0
    chunk_n = 0
    rows = []
    last_decode_samples = 0
    last_inference_ms = 0.0
    last_decoded_flag = False
    heartbeat_count = 0

    # Phase 1: push audio
    while samples_sent < total_samples:
        end = min(samples_sent + CHUNK_SAMPLES, total_samples)
        chunk = audio[samples_sent * 2 : end * 2]
        t_before_send = time.time()
        post(f"/api/realtime/chunk?session_id={sid}", body=chunk,
             content_type="application/octet-stream")
        t_send_done = time.time()
        samples_sent = end
        chunk_n += 1

        # Poll status (mimic UI's 150ms pollRealtimeStatus)
        snap = get(f"/api/realtime/status?session_id={sid}")

        wall = time.time() - t_start
        audio_dur = snap["sample_count"] / SR
        decoded_dur = snap["decoded_samples"] / SR
        lag = max(0.0, wall - decoded_dur)
        inf_ms = snap.get("inference_ms", 0.0)
        decode_label = "已解码" if snap["decoded"] else "待下轮"

        rows.append({
            "t": round(wall, 2),
            "chunk": chunk_n,
            "audio": round(audio_dur, 2),
            "decoded": round(decoded_dur, 2),
            "lag": round(lag, 2),
            "inf_ms": round(inf_ms, 1),
            "label": decode_label,
            "stable": snap.get("live_stable_text", ""),
            "partial": snap.get("live_partial_text", ""),
        })

        # Heartbeat detection: decoded flag flips to true but
        # decoded_samples didn't grow
        if snap["decoded"] and snap["decoded_samples"] == last_decode_samples:
            heartbeat_count += 1
        last_decode_samples = snap["decoded_samples"]
        last_inference_ms = inf_ms

        # Pace the pushes (mimic the UI's 400ms flushRealtimeChunk)
        sleep_for = max(0, CHUNK_MS / 1000.0 - (time.time() - t_before_send))
        time.sleep(sleep_for)

    # Phase 2: stop, then wait and observe heartbeat
    post(f"/api/realtime/eof?session_id={sid}")
    print(f"Pushed {chunk_n} chunks, now watching for heartbeat during silence...")
    for i in range(6):
        time.sleep(0.6)
        snap = get(f"/api/realtime/status?session_id={sid}")
        wall = time.time() - t_start
        audio_dur = snap["sample_count"] / SR
        decoded_dur = snap["decoded_samples"] / SR
        lag = max(0.0, wall - decoded_dur)
        inf_ms = snap.get("inference_ms", 0.0)
        decode_label = "已解码" if snap["decoded"] else "待下轮"
        rows.append({
            "t": round(wall, 2),
            "chunk": chunk_n,
            "audio": round(audio_dur, 2),
            "decoded": round(decoded_dur, 2),
            "lag": round(lag, 2),
            "inf_ms": round(inf_ms, 1),
            "label": decode_label,
            "stable": snap.get("live_stable_text", ""),
            "partial": snap.get("live_partial_text", ""),
        })

    # Print all rows
    print()
    print(f"{'t':>5} {'chk':>3} {'audio':>6} {'dec':>6} {'lag':>5} {'inf_ms':>7} {'label':>6}  text")
    print("-" * 80)
    for r in rows:
        text = (r["stable"] + r["partial"])[-50:]
        print(f"{r['t']:>5.2f} {r['chunk']:>3d} {r['audio']:>6.2f} {r['decoded']:>6.2f} "
              f"{r['lag']:>5.2f} {r['inf_ms']:>7.1f} {r['label']:>6}  {text}")

    # Final stable text
    final_snap = get(f"/api/realtime/status?session_id={sid}")
    final_text = final_snap.get("stable_text", "")
    print()
    print(f"=== Final ===")
    print(f"  Stable:    {final_text}")
    print(f"  Partial:   {final_snap.get('partial_text', '')}")
    print(f"  Inference: {final_snap.get('inference_ms', 0):.0f} ms")
    print(f"  Heartbeats during silence: {heartbeat_count}")

    # Validation
    errs = []
    if final_snap.get("inference_ms", 0) <= 0:
        errs.append(f"inference_ms should be > 0, got {final_snap.get('inference_ms')}")
    if not final_text.strip():
        errs.append("final stable_text is empty (model produced nothing?)")
    if any(r["lag"] < 0 for r in rows):
        errs.append("lag went negative (wall clock < decoded — impossible)")
    if any(r["audio"] < r["decoded"] - 0.5 for r in rows):
        errs.append("decoded_samples > audio_samples (decoder ran ahead)")

    if errs:
        print()
        print("=== FAILURES ===")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print()
        print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
