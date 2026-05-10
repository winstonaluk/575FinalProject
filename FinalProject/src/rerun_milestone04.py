"""Re-extract all Milestone 4 CNN feature caches from scratch.

Deletes stale stimulus_bank.pkl and all cnn_features_<arch>.pkl files,
rebuilds the bank from the raw .mat files, then runs all registered
backbones and writes fresh caches.

Requires:
  pip install cornet   # for CORnet-S

Usage (from FinalProject/):
    python src/rerun_milestone04.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cnn_features import (
    load_brands_stimulus_bank,
    build_and_cache_features,
)

STIMULI = ROOT / "data" / "ds004194" / "stimuli"
RESULTS = ROOT / "results"
BANK_CACHE = RESULTS / "cache" / "stimulus_bank.pkl"

ARCHS = ("resnet50", "alexnet", "vgg16", "convnext_tiny", "densenet121", "swin_t", "cornet_s")


def rebuild_bank() -> object:
    if BANK_CACHE.exists():
        print(f"Removing stale bank cache: {BANK_CACHE}")
        BANK_CACHE.unlink()

    print("Building stimulus bank from raw .mat files…")
    t0 = time.perf_counter()
    bank = load_brands_stimulus_bank(STIMULI)
    elapsed = time.perf_counter() - t0
    print(f"  {len(bank)} unique images in {elapsed:.1f}s")
    print("  per-category counts:")
    print(bank.metadata.groupby("category").size().to_string())

    BANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(BANK_CACHE, "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved → {BANK_CACHE}\n")
    return bank


def reextract(bank) -> None:
    skipped = []
    for arch in ARCHS:
        if arch == "cornet_s":
            try:
                import cornet  # noqa: F401
            except ImportError:
                print(f"[skip] cornet_s — package not installed (pip install cornet)\n")
                skipped.append(arch)
                continue

        out = RESULTS / f"cnn_features_{arch}.pkl"
        if out.exists():
            print(f"Removing stale cache: {out}")
            out.unlink()

        # VGG16 chunk cache is also stale
        if arch == "vgg16":
            chunk_dir = RESULTS / "cache" / "_vgg16_chunks"
            if chunk_dir.exists():
                for f in chunk_dir.glob("chunk_*.pkl"):
                    f.unlink()
                print(f"  cleared VGG16 chunk dir: {chunk_dir}")

        print(f"Extracting {arch}…")
        t0 = time.perf_counter()
        cache = build_and_cache_features(bank, out_path=out, arch=arch, progress=True)
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s  →  {out}")
        for ln, shape in cache.shapes().items():
            print(f"    {ln:30s} {shape}")
        print()

    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    bank = rebuild_bank()
    reextract(bank)
    print("All done. Re-run the milestone_04_cnn_features notebook to regenerate plots.")
