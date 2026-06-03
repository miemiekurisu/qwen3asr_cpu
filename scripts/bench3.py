#!/usr/bin/env python3
"""
3-audio benchmark: speed (lag p50/p95/max) + quality (CER/WER).
- Speed: streaming, 200ms chunk size, measure wall - decoded lag.
- Quality: offline (qasr_cli), compare to label via CER (zh/ja) or WER (en).
"""
import json, urllib.request, wave, time, threading, subprocess, sys, os, re
from pathlib import Path

SR = 16000
SNAP = "/home/kurisu/.cache/huggingface/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
TESTFILES = Path("testfile")
BIN = "./build/linux-openblas/qasr_server"
CLI = "./build/linux-openblas/qasr_cli"

AUDIO_SETS = [
    ("aishell_zh",  "aishell_S0002_limai_108s.wav", "aishell_S0002_limai_108s.txt", "zh"),
    ("english_en", "english_60s.wav",              "english_60s.txt",              "en"),
    ("zh_en_mix",  "zh_en_mix_60s.wav",            "zh_en_mix_60s.txt",            "zh+en"),
    ("japanese_ja","japanese_60s.wav",             "japanese_60s.txt",             "ja"),
]


def http_post(url, data=None, ct=None, timeout=60):
    r = urllib.request.Request(url, data=data, method="POST")
    if ct: r.add_header("Content-Type", ct)
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def http_get(url, timeout=30):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def load_label(path):
    """Skip comment lines (#), return concatenated text."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        out.append(s)
    return " ".join(out)


def normalize_en(s):
    """WER normalize: uppercase, strip punctuation, collapse spaces."""
    s = s.upper()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_zh(s):
    """CER: remove all whitespace + ASCII punctuation, keep CJK."""
    s = re.sub(r"[\s\u3000]", "", s)
    s = re.sub(r"[A-Za-z0-9\W_]+", "", s, flags=re.UNICODE)
    return s


def cer(hyp, ref):
    """Character error rate using DP edit distance."""
    if not ref: return float("nan")
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / n


def wer(hyp, ref):
    """Word error rate."""
    h = normalize_en(hyp).split()
    r = normalize_en(ref).split()
    if not r: return float("nan")
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m] / n


def cer_or_wer(hyp, ref, lang):
    if lang in ("en", "zh+en"):
        # WER for EN
        return "WER", wer(hyp, ref)
    else:
        return "CER", cer(normalize_zh(hyp), normalize_zh(ref))


def start_server():
    p = subprocess.Popen(
        [BIN, "--model-dir", SNAP, "--port", "19991", "--host", "0.0.0.0", "--verbosity", "0"],
        stdout=open("/tmp/qasr_server.log", "w"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "QASR_MODEL_DIR": SNAP},
        preexec_fn=os.setsid,
    )
    # Wait ready
    for _ in range(20):
        try:
            if http_get("http://127.0.0.1:19991/health").get("status") == "ok":
                return p
        except: pass
        time.sleep(0.5)
    raise RuntimeError("server failed to start")


def stop_server(p):
    try: os.killpg(os.getpgid(p.pid), 9)
    except: pass
    try: p.wait(timeout=3)
    except: pass


def speed_test(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        audio = w.readframes(w.getnframes())
    sid = http_post("http://127.0.0.1:19991/api/realtime/start")["session_id"]
    CHUNK_MS = 200
    CHUNK_BYTES = CHUNK_MS * SR // 1000 * 2
    offset = [0]
    done = threading.Event()
    def feeder():
        while offset[0] + CHUNK_BYTES <= len(audio):
            b = audio[offset[0]:offset[0]+CHUNK_BYTES]
            offset[0] += CHUNK_BYTES
            tf = time.time()
            try: http_post(f"http://127.0.0.1:19991/api/realtime/chunk?session_id={sid}", b, "application/octet-stream")
            except: break
            time.sleep(max(0, CHUNK_MS/1000 - (time.time()-tf)))
        done.set()
    threading.Thread(target=feeder, daemon=True).start()
    t0 = time.time()
    samples = []
    while not (done.is_set() and time.time() - t0 > 2.5):
        try:
            s = http_get(f"http://127.0.0.1:19991/api/realtime/status?session_id={sid}")
        except: break
        wall = time.time() - t0
        dec = s["decoded_samples"] / SR
        samples.append((wall, dec, wall - dec))
        time.sleep(0.05)
    # final decode settle
    time.sleep(2)
    final = http_get(f"http://127.0.0.1:19991/api/realtime/status?session_id={sid}")
    return samples, final


def quality_test(wav_path):
    """Use qasr_cli offline transcribe, capture stdout text + inference_ms."""
    t0 = time.time()
    r = subprocess.run([CLI, "--model-dir", SNAP, "--audio", str(wav_path), "--verbosity", "0"],
                       capture_output=True, text=True, timeout=120)
    wall = time.time() - t0
    text = r.stdout.strip()
    # Parse inference_ms from last line
    inf_ms = 0
    for line in text.splitlines()[::-1]:
        if "inference_ms=" in line:
            try: inf_ms = float(line.split("inference_ms=")[1].split()[0])
            except: pass
            break
    # Return only the text portion (strip the inference_ms line)
    text_lines = [l for l in text.splitlines() if "inference_ms=" not in l]
    return "\n".join(text_lines).strip(), inf_ms, wall


def main():
    print("="*78)
    print("3-Audio Benchmark: speed (lag) + quality (CER/WER)")
    print("="*78)
    print(f"Binary: {BIN}")
    print(f"Snap:   {SNAP}")
    print()
    p = start_server()
    print("Server ready.\n")
    try:
        for name, wav, lbl, lang in AUDIO_SETS:
            wav_path = TESTFILES / wav
            lbl_path = TESTFILES / lbl
            print("="*78)
            print(f"### {name} ({lang})  wav={wav}  dur=", end="")
            with wave.open(str(wav_path), "rb") as w:
                print(f"{w.getnframes()/w.getframerate():.1f}s")
            # --- Speed test (streaming) ---
            print("  [speed] streaming 200ms chunks...")
            t0 = time.time()
            samples, final = speed_test(wav_path)
            sp_wall = time.time() - t0
            lgs = sorted(s[2] for s in samples if s[0] > 1.0)
            if lgs:
                n = len(lgs)
                p50, p75, p90, p95, mx = lgs[n//2], lgs[3*n//4], lgs[int(n*0.9)], lgs[int(n*0.95)], lgs[-1]
            else:
                p50=p75=p90=p95=mx=float("nan")
            print(f"  [speed] lag p50={p50:.2f}s  p75={p75:.2f}s  p90={p90:.2f}s  p95={p95:.2f}s  max={mx:.2f}s   (wall={sp_wall:.1f}s, total inf_ms={final['inference_ms']:.0f})")
            # --- Quality test (offline) ---
            print("  [quality] qasr_cli offline...")
            hyp, inf_ms, q_wall = quality_test(wav_path)
            ref = load_label(lbl_path)
            metric, err = cer_or_wer(hyp, ref, lang)
            print(f"  [quality] {metric}={err*100:.2f}%   (inf_ms={inf_ms:.0f}, wall={q_wall:.1f}s)")
            print(f"  [ref  ]  {ref[:100]}{'...' if len(ref)>100 else ''}")
            print(f"  [hyp  ]  {hyp[:100]}{'...' if len(hyp)>100 else ''}")
            print()
    finally:
        stop_server(p)
    print("Done.")


if __name__ == "__main__":
    main()
