"""Fetch AISHELL-1 wav clips for qasr_cpu test audio.

Downloads a single speaker's tar.gz from HF, extracts a few utterances
referenced in the bundled transcript, and concats them with ffmpeg into
a single ~100s Chinese clip.

Usage:
    python3 tools/aishell_fetch.py --speaker S0002 --clips 18 --out testfile/

Default matches ``testfile/aishell_S0002_limai_108s.wav``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "AISHELL/AISHELL-1"
TRANSCRIPT = "data_aishell/transcript/aishell_transcript_v0.8.txt"


def fetch_speaker_wavs(speaker: str, work: Path) -> Path:
    tar = hf_hub_download(
        repo_id=REPO,
        filename=f"data_aishell/wav/{speaker}.tar.gz",
        repo_type="dataset",
        cache_dir=str(work / "cache"),
    )
    extract_dir = work / speaker
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", tar, "-C", str(extract_dir)],
        check=True,
    )
    return extract_dir / "train" / speaker


def fetch_transcript(work: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO,
            filename=TRANSCRIPT,
            repo_type="dataset",
            cache_dir=str(work / "cache"),
        )
    )


def collect_text(transcript: Path, ids: list[str]) -> str:
    out_lines: list[str] = []
    with transcript.open() as f:
        for line in f:
            for uid in ids:
                if line.startswith(uid + " "):
                    out_lines.append(line[len(uid) + 1:].rstrip())
                    break
    return "\n".join(out_lines) + "\n"


def concat_wavs(wav_dir: Path, ids: list[str], out_wav: Path) -> None:
    list_file = out_wav.with_suffix(".list.txt")
    with list_file.open("w") as f:
        for uid in ids:
            f.write(f"file '{wav_dir / (uid + '.wav')}'\n")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(out_wav),
        ],
        check=True,
    )
    list_file.unlink(missing_ok=True)


def next_clip_ids(transcript: Path, speaker: str, count: int) -> list[str]:
    """Pick the first N consecutive utterance IDs for this speaker."""
    prefix = f"BAC009{speaker}W"
    ids: list[str] = []
    with transcript.open() as f:
        for line in f:
            tok = line.split(" ", 1)[0]
            if tok.startswith(prefix):
                ids.append(tok)
                if len(ids) >= count:
                    break
    return ids


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--speaker", default="S0002")
    p.add_argument("--clips", type=int, default=18, help="utterance count to concat")
    p.add_argument("--out", type=Path, default=Path("testfile"))
    p.add_argument("--name", default="aishell_S0002_limai_108s")
    p.add_argument("--work", type=Path, default=Path("/tmp/aishell_setup"))
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading transcript + {args.speaker}.tar.gz …", file=sys.stderr)
    transcript = fetch_transcript(args.work)
    wav_dir = fetch_speaker_wavs(args.speaker, args.work)

    ids = next_clip_ids(transcript, args.speaker, args.clips)
    if len(ids) < args.clips:
        print(f"warning: only {len(ids)} utterances for {args.speaker}", file=sys.stderr)

    out_wav = args.out / f"{args.name}.wav"
    out_txt = args.out / f"{args.name}.txt"
    print(f"[2/4] Concatenating {len(ids)} clips → {out_wav}", file=sys.stderr)
    concat_wavs(wav_dir, ids, out_wav)

    print(f"[3/4] Writing transcript → {out_txt}", file=sys.stderr)
    body = collect_text(transcript, ids)
    header = (
        f"# AISHELL-1 sample transcript — speaker {args.speaker}\n"
        f"# Source: AISHELL-1 corpus (Apache 2.0 / non-commercial research)\n"
        f"# File: {out_wav.name}\n"
        f"# Order matches wav concatenation\n"
    )
    out_txt.write_text(header + "\n" + body)

    print("[4/4] Done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
