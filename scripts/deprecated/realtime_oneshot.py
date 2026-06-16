#!/usr/bin/env python3
"""一次性送完整个长音频,测量最终延迟。

The realtime live worker decodes in 0.5 s chunks but leaves a small
"partial tail" (typically 0.5-0.8 s) that doesn't get a final decode
until EOF is signaled.  For a one-shot long audio, exit when
decoded_samples >= sample_count - 8000 (i.e. at most 0.5 s of tail
left).  That matches what the UI's "fully decoded" state would feel
like to the user.
"""
import json
import sys
import time
import urllib.request
import wave

BASE = "http://127.0.0.1:19991"
SR = 16000
TAIL_TOLERANCE_SAMPLES = 8000   # ≤ 0.5 s tail not decoded → "effectively done"


def post(path, body=None, content_type=None):
    req = urllib.request.Request(BASE + path, data=body, method="POST")
    if content_type:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=120) as resp:
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
    audio_dur = total / SR
    print(f"=== One-shot long-sentence latency ===")
    print(f"  Audio: {audio_dur:.1f} s ({total} samples, {len(audio)} bytes)")
    print()

    sid = post("/api/realtime/start")["session_id"]

    # Send all audio in ONE POST
    t_post_start = time.time()
    snap_after_post = post(f"/api/realtime/chunk?session_id={sid}",
                            body=audio, content_type="application/octet-stream")
    t_post_end = time.time()
    send_ms = (t_post_end - t_post_start) * 1000
    target = snap_after_post["sample_count"] - TAIL_TOLERANCE_SAMPLES

    print(f"  POST:        {send_ms:.0f} ms ({len(audio)} bytes)")
    print(f"  Target:      decoded_samples >= {target} ({target/SR:.1f} s)")
    print()

    # Poll every 100ms until "effectively done"
    polls = 0
    last_report = time.time()
    while True:
        snap = get(f"/api/realtime/status?session_id={sid}")
        polls += 1
        if snap["decoded_samples"] >= target:
            t_done = time.time()
            break
        if time.time() - t_post_end > 180:
            print("TIMEOUT after 180 s")
            sys.exit(2)
        if time.time() - last_report > 2.0:
            print(f"  ... poll #{polls}: decoded={snap['decoded_samples']/SR:.1f}s, "
                  f"inference={snap.get('inference_ms', 0):.0f}ms")
            last_report = time.time()
        time.sleep(0.1)

    wait_ms = (t_done - t_post_end) * 1000
    total_ms = (t_done - t_post_start) * 1000

    final = get(f"/api/realtime/status?session_id={sid}")
    final_text = final.get("stable_text", "")
    final_inf = final.get("inference_ms", 0)
    final_decoded = final["decoded_samples"]
    wall_now = time.time() - t_post_end
    lag = max(0.0, wall_now - final_decoded / SR)
    rtf = final_inf / (final_decoded / SR * 1000) if final_decoded > 0 else 0

    print()
    print(f"=== Results ===")
    print(f"  Polls:         {polls}")
    print(f"  Wait:          {wait_ms:.0f} ms (POST end → fully decoded)")
    print(f"  Total:         {total_ms:.0f} ms (POST start → fully decoded)")
    print(f"  Audio:         {audio_dur:.1f} s")
    print(f"  Decoded:       {final_decoded/SR:.1f} s (tail of "
          f"{(total-final_decoded)/SR:.2f} s not decoded in live mode)")
    print(f"  Model RTF:     {rtf:.3f} ({rtf*100:.0f}% of real-time)")
    print(f"  Inference:     {final_inf:.0f} ms (server perf_total_ms)")
    print(f"  Wall / audio:  {(wait_ms/audio_dur/10):.2f} (wait ÷ audio duration)")
    print()
    print(f"=== What the UI would show at the end ===")
    print(f"  音频 {audio_dur:.1f}s / 已解码 {final_decoded/SR:.1f}s / "
          f"耗时 {wall_now:.1f}s / 滞后 {lag:.1f}s / 推理 {final_inf:.0f}ms / "
          f"{'已解码' if final['decoded'] else '待下轮'}")
    print()
    print(f"=== Final stable text ({len(final_text)} chars) ===")
    print(f"  {final_text[:200]}{'...' if len(final_text) > 200 else ''}")


if __name__ == "__main__":
    main()
