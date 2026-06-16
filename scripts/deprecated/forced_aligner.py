#!/usr/bin/env python3
"""
Qwen3-ForcedAligner-0.6B sidecar process.

Reads JSON requests from stdin (one per line), runs forced alignment,
writes JSON results to stdout (one per line). Designed to be spawned
by the C++ host as a subprocess.

Protocol:
  Request  (stdin):  {"audio": "<path>", "text": "<transcript>", "language": "<lang>"}
  Response (stdout): {"ok": true, "items": [{"text": "...", "start": 0.123, "end": 0.456}, ...]}
  Error    (stdout): {"ok": false, "error": "<message>"}

Batch request:
  {"batch": [{"audio": "...", "text": "...", "language": "..."}, ...]}
  Response: {"ok": true, "results": [[{"text":...}, ...], ...]}

Special commands:
  {"cmd": "languages"} -> {"ok": true, "languages": ["chinese", ...]}
  {"cmd": "quit"}      -> process exits
"""

import json
import sys
import os
import traceback

# Force UTF-8 on stdin/stdout (Windows defaults to system code page)
if sys.platform == "win32":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

def init_model(model_path: str):
    """Load the ForcedAligner model. Returns (aligner, error_string)."""
    try:
        import torch
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
        from transformers import AutoConfig, AutoModel, AutoProcessor

        # Attempt to import the aligner wrapper
        try:
            from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner
            aligner = Qwen3ForcedAligner.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
            )
        except ImportError:
            # Fallback: manual registration + load
            AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
            AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
            AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

            from qwen_asr.inference.qwen3_forced_aligner import (
                Qwen3ForcedAligner,
                Qwen3ForceAlignProcessor,
            )
            model = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16)
            processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
            aligner_proc = Qwen3ForceAlignProcessor()
            aligner = Qwen3ForcedAligner(
                model=model, processor=processor, aligner_processor=aligner_proc
            )

        return aligner, None
    except Exception as e:
        return None, f"Failed to load model: {e}\n{traceback.format_exc()}"


def align_single(aligner, audio_path: str, text: str, language: str):
    """Run alignment for a single audio/text pair. Returns list of dicts."""
    results = aligner.align(audio=audio_path, text=text, language=language)
    # results is List[ForcedAlignResult], we want results[0]
    items = []
    for item in results[0]:
        items.append({
            "text": item.text,
            "start": item.start_time,
            "end": item.end_time,
        })
    return items


def align_batch(aligner, requests):
    """Run alignment for a batch of requests. Returns list of list of dicts."""
    audios = [r["audio"] for r in requests]
    texts = [r["text"] for r in requests]
    languages = [r["language"] for r in requests]
    results = aligner.align(audio=audios, text=texts, language=languages)
    all_items = []
    for result in results:
        items = []
        for item in result:
            items.append({
                "text": item.text,
                "start": item.start_time,
                "end": item.end_time,
            })
        all_items.append(items)
    return all_items


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3 ForcedAligner sidecar")
    parser.add_argument("--model-dir", required=True, help="Path to ForcedAligner model")
    args = parser.parse_args()

    # Send ready signal to stderr (not stdout, which is the data channel)
    sys.stderr.write(f"[aligner] loading model from {args.model_dir}\n")
    sys.stderr.flush()

    aligner, err = init_model(args.model_dir)
    if err:
        # Write error to stdout so host sees it, then exit
        sys.stdout.write(json.dumps({"ok": False, "error": err}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    # Signal ready
    sys.stderr.write("[aligner] model loaded, ready\n")
    sys.stderr.flush()
    sys.stdout.write(json.dumps({"ok": True, "status": "ready"}) + "\n")
    sys.stdout.flush()

    # Main loop: read JSON lines from stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"invalid JSON: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        try:
            # Command dispatch
            cmd = request.get("cmd")
            if cmd == "quit":
                sys.stdout.write(json.dumps({"ok": True, "status": "quit"}) + "\n")
                sys.stdout.flush()
                break
            elif cmd == "languages":
                langs = aligner.get_supported_languages()
                sys.stdout.write(json.dumps({"ok": True, "languages": langs or []}) + "\n")
                sys.stdout.flush()
                continue

            # Batch mode
            if "batch" in request:
                batch_reqs = request["batch"]
                if not isinstance(batch_reqs, list) or len(batch_reqs) == 0:
                    sys.stdout.write(json.dumps({"ok": False, "error": "batch must be non-empty list"}) + "\n")
                    sys.stdout.flush()
                    continue
                results = align_batch(aligner, batch_reqs)
                sys.stdout.write(json.dumps({"ok": True, "results": results}) + "\n")
                sys.stdout.flush()
                continue

            # Single mode
            audio = request.get("audio")
            text = request.get("text")
            language = request.get("language")
            if not audio or not text or not language:
                sys.stdout.write(json.dumps({
                    "ok": False,
                    "error": "missing required fields: audio, text, language"
                }) + "\n")
                sys.stdout.flush()
                continue

            items = align_single(aligner, audio, text, language)
            sys.stdout.write(json.dumps({"ok": True, "items": items}) + "\n")
            sys.stdout.flush()

        except Exception as e:
            sys.stdout.write(json.dumps({
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
