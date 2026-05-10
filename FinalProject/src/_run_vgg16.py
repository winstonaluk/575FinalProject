"""Standalone VGG16 driver — chunked, resumable.

Runs the forward pass in batches of 48 images, persisting per-chunk
activations to results/cache/_vgg16_chunks/ as it goes. A second pass
concatenates the chunks into the final results/cnn_features_vgg16.pkl
once all chunks exist. Designed to fit each chunk inside the sandbox's
~45 s bash timeout.

Usage:
    python src/_run_vgg16.py extract     # idempotent; resumes mid-run
    python src/_run_vgg16.py finalize    # merges chunks → final pickle
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_features import (
    FeatureCache,
    _cnn_configs,
    _imagenet_preprocess,
)

CHUNK_SIZE = 48                                    # 6 chunks total for 288 images
CHUNK_DIR = Path("results/cache/_vgg16_chunks")
FINAL_PATH = Path("results/cnn_features_vgg16.pkl")
BANK_PATH = Path("results/cache/stimulus_bank.pkl")
LAYERS = ("features.3", "features.8", "features.15",
          "features.22", "features.29", "classifier.3")


def _load_bank():
    with open(BANK_PATH, "rb") as f:
        return pickle.load(f)


def extract_one_chunk(start: int, n_total: int, bank, batch_size: int = 8,
                      force: bool = False, freshen: bool = False):
    """Run VGG16 forward pass on images[start:start+CHUNK_SIZE].

    `force=True` overwrites unconditionally.
    `freshen=True` overwrites only if the chunk's mtime is older than
    the bank pickle's mtime — useful for resuming incrementally after a
    bank rebuild without redoing already-up-to-date chunks.
    """
    import torch
    from PIL import Image

    chunk_path = CHUNK_DIR / f"chunk_{start:04d}.pkl"
    if chunk_path.exists() and not force and not freshen:
        print(f"  [skip] chunk {start} already exists")
        return chunk_path
    if chunk_path.exists() and freshen and not force:
        bank_mtime = BANK_PATH.stat().st_mtime
        if chunk_path.stat().st_mtime >= bank_mtime:
            print(f"  [skip] chunk {start} already fresher than bank")
            return chunk_path

    factory, _ = _cnn_configs()["vgg16"]
    net = factory().eval()
    activations = {ln: [] for ln in LAYERS}
    name_to_module = dict(net.named_modules())

    def make_hook(name):
        def hook(_m, _i, out):
            if out.ndim == 4:
                pooled = out.mean(dim=(2, 3))
            else:
                pooled = out.flatten(start_dim=1)
            activations[name].append(pooled.detach().cpu().numpy())
        return hook

    for ln in LAYERS:
        name_to_module[ln].register_forward_hook(make_hook(ln))

    preprocess = _imagenet_preprocess()
    end = min(start + CHUNK_SIZE, n_total)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(start, end, batch_size):
            batch = bank.images[s:min(s + batch_size, end)]
            tensors = []
            for img in batch:
                pil = Image.fromarray(np.transpose(img, (1, 2, 0)), "RGB")
                tensors.append(preprocess(pil))
            x = torch.stack(tensors, dim=0)
            _ = net(x)
    elapsed = time.perf_counter() - t0

    chunk_data = {
        "start": start,
        "end": end,
        "elapsed_s": elapsed,
        "features": {ln: np.concatenate(activations[ln], axis=0) for ln in LAYERS},
    }
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_path, "wb") as f:
        pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  chunk {start}-{end-1}: {elapsed:.1f}s → {chunk_path.name}")
    return chunk_path


def extract(force: bool = False, freshen: bool = False):
    bank = _load_bank()
    n = bank.images.shape[0]
    suffix = ""
    if force:
        suffix = " (force=True)"
    elif freshen:
        suffix = " (freshen=True; only re-do chunks older than bank)"
    print(f"VGG16 extraction: {n} images in chunks of {CHUNK_SIZE}{suffix}")
    for start in range(0, n, CHUNK_SIZE):
        extract_one_chunk(start, n, bank, force=force, freshen=freshen)


def finalize():
    bank = _load_bank()
    n = bank.images.shape[0]
    chunk_files = sorted(CHUNK_DIR.glob("chunk_*.pkl"))
    expected = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    if len(chunk_files) != expected:
        raise RuntimeError(
            f"Expected {expected} chunks, found {len(chunk_files)}. "
            "Re-run extract first."
        )

    merged: dict[str, list[np.ndarray]] = {ln: [] for ln in LAYERS}
    total_elapsed = 0.0
    for cp in chunk_files:
        with open(cp, "rb") as f:
            data = pickle.load(f)
        total_elapsed += data["elapsed_s"]
        for ln in LAYERS:
            merged[ln].append(data["features"][ln])
    features = {ln: np.concatenate(merged[ln], axis=0) for ln in LAYERS}

    cache = FeatureCache(
        arch="vgg16",
        layers=LAYERS,
        features=features,
        metadata=bank.metadata.copy(),
        n_images=n,
        image_shape=tuple(bank.images.shape[1:]),
        extraction_seconds=total_elapsed,
        extraction_device="cpu",
    )
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Finalized: {FINAL_PATH} ({FINAL_PATH.stat().st_size/1024/1024:.1f} MB), "
          f"summed extraction time {total_elapsed:.1f}s")
    for ln, sh in cache.shapes().items():
        print(f"  {ln:14s}  {sh}")


def main():
    args = sys.argv[1:] or ["extract"]
    cmd = args[0]
    force = "--force" in args[1:]
    freshen = "--freshen" in args[1:]
    if cmd == "extract":
        extract(force=force, freshen=freshen)
    elif cmd == "finalize":
        finalize()
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
