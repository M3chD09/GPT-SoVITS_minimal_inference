#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_models.py — Download inference models into ./pretrained_models/

Cross-platform replacement for the model-download part of install.sh / install.ps1.
Downloads components, each skipped when already present:

Base models (repo ``lj1995/GPT-SoVITS``):
  chinese-hubert-base/            <torchaudio/transformers> Wav2Vec2 feature extractor
  chinese-roberta-wwm-ext-large/  <transformers> BERT for zh text semantics
  sv/pretrained_eres2netv2w24s4ep4.ckpt   speaker-verification embedding model

Voice preset (repo ``m3chd09/ai-natsume-tts``, flattened into ./pretrained_models/):
  GPT_weights_v2ProPlus/natsume.ckpt       GPT weight (firefly V2 ProPlus finetune)
  SoVITS_weights_v2ProPlus/natsume.pth     SoVITS weight
  RefAudio/…                               reference audios for the preset

Other:
  fast_langdetect/lid.176.bin     fast_langdetect model (Meta fastText, ~126MB)

Except lid.176.bin (fetched from dl.fbaipublicfiles.com, the URL the
fast_langdetect package itself uses — download it here so its cache_dir, which
must already exist, see GPT_SoVITS/text/LangSegmenter/langsegmenter.py:12, is
pre-populated), ALL files come from HuggingFace with hf-mirror.com support.

Usage:
  python download_models.py                 # source: HF-Mirror (default; works without VPN)
  python download_models.py --source HF     # direct HuggingFace
  python download_models.py --dir pretrained_models
  python download_models.py --skip-natsume  # only base models + fast_langdetect
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(ROOT, "pretrained_models")

# Model files resolved from HF repos. Each target has an optional `repo` (default
# BASE_REPO) and `files` relative to the repo root; files land under the same
# relative path inside the output dir (no repo-name folder is created).
BASE_REPO = "lj1995/GPT-SoVITS"
NATSUME_REPO = "m3chd09/ai-natsume-tts"


def source_base(hf_source: str, repo: str) -> str:
    """Absolute resolve URL for a repo path under the chosen HF source."""
    return {
        "HF":        f"https://huggingface.co/{repo}/resolve/main",
        "HF-Mirror": f"https://hf-mirror.com/{repo}/resolve/main",
    }[hf_source]


# Meta fastText lid.176 (both sources point at the same official file)
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

TARGETS = [
    {
        "dir": "chinese-hubert-base",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    },
    {
        "dir": "chinese-roberta-wwm-ext-large",
        "files": ["config.json", "tokenizer.json", "pytorch_model.bin"],
    },
    {
        "dir": "sv",
        "files": ["pretrained_eres2netv2w24s4ep4.ckpt"],
    },
    {
        # natsume voice preset — repo root IS the pretrained_models layout, so
        # its files map 1:1 into the output dir (no wrapping folder).
        "repo": NATSUME_REPO,
        "dir": ".",
        "files": [
            "GPT_weights_v2ProPlus/natsume.ckpt",
            "SoVITS_weights_v2ProPlus/natsume.pth",
            "RefAudio/casual/ワタシ一人だけじゃなく、周りのみんなに助けてもらいながら考えて……こういうお店にしました.ogg",
            "RefAudio/romantic/ワタシだって、好きな人と一緒にいたいと思うし。こうして触れ合ってると……幸せで.ogg",
        ],
    },
]

MAX_TRIES = 25          # mirrors install.sh `wget --tries=25 --wait=5`
INITIAL_WAIT = 5
READ_TIMEOUT = 40


def log(msg: str) -> None:
    print(msg, flush=True)


def download_once(url: str, dest: str) -> None:
    """Stream a single file to dest with a progress bar. Raises on failure."""
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total:
                    pct = got * 100 // total
                    bar = "#" * (pct // 4)
                    sys.stdout.write(f"\r  {pct:3d}% [{bar:<25}] {got/1e6:6.1f}/{total/1e6:6.1f} MB")
                    sys.stdout.flush()
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    sys.stdout.write("\n")
    if os.path.exists(dest):
        os.remove(dest)
    os.rename(tmp, dest)


def fetch(url: str, dest: str) -> None:
    """Download with retry / backoff, mirroring install.sh's hardened wget."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            log(f"[INFO] Downloading {os.path.basename(dest)} ...")
            download_once(url, dest)
            log(f"[INFO] Done: {dest}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            if attempt == MAX_TRIES:
                raise RuntimeError(f"Failed after {MAX_TRIES} tries: {e}") from e
            wait = INITIAL_WAIT * attempt
            log(f"[WARNING] Attempt {attempt}/{MAX_TRIES} failed ({e}). Retrying in {wait}s ...")
            time.sleep(wait)


def file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Download GPT-SoVITS inference models into ./pretrained_models/")
    ap.add_argument("--source", choices=["HF", "HF-Mirror"], default="HF-Mirror",
                    help="Model source (default: %(default)s)")
    ap.add_argument("--dir", default=DEFAULT_OUT_DIR, help="Output directory (default: ./pretrained_models)")
    ap.add_argument("--skip-natsume", action="store_true",
                    help="Skip the m3chd09/ai-natsume-tts voice preset")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    for target in TARGETS:
        if args.skip_natsume and target.get("repo") == NATSUME_REPO:
            log("[INFO] Skipping natsume voice preset (--skip-natsume)")
            continue
        sub = os.path.join(args.dir, target["dir"])
        os.makedirs(sub, exist_ok=True)
        missing = [f for f in target["files"] if not file_ok(os.path.join(sub, f))]
        if not missing:
            log(f"[INFO] {target['dir'] or '.'}/ — all files present, skipping")
            continue
        repo = target.get("repo", BASE_REPO)
        for name in missing:
            dest = os.path.join(sub, name)
            # File paths may include subdirectories (e.g. natsume preset)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # Repo path = dir prefix + file (skip prefix when dir is ".")
            rel_path = name if target["dir"] == "." else f"{target['dir']}/{name}"
            rel = urllib.parse.quote(rel_path, safe="/")
            url = f"{source_base(args.source, repo)}/{rel}"
            fetch(url, dest)

    # fast_langdetect: package raises FileNotFoundError if the cache dir is
    # missing (fast_langdetect/infer.py), so pre-create it and pre-download.
    lmd_dir = os.path.join(args.dir, "fast_langdetect")
    os.makedirs(lmd_dir, exist_ok=True)
    lmd_file = os.path.join(lmd_dir, "lid.176.bin")
    if file_ok(lmd_file):
        log("[INFO] fast_langdetect/lid.176.bin exists, skipping")
    else:
        fetch(FASTTEXT_URL, lmd_file)

    log("[SUCCESS] All models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())