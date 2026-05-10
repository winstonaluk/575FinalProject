"""
Milestone 4 — Stimulus loading & CNN feature extraction
=======================================================

Self-contained module that:

1. Walks a set of Brands 2024 run-level stimulus .mat files (the BAIR
   v7.3 / HDF5 format), deduplicates by content hash, and returns a
   canonical (288, 3, 568, 568) uint8 array of unique displayed images
   plus per-image metadata (category, source run, etc.).

2. Forward-passes those 288 images through a pretrained torchvision
   classifier (ResNet50 by default; AlexNet and VGG16 as robustness
   checks) with forward hooks registered on the layers we care about
   (conv1, layer1–layer4, avgpool for ResNet50). Activations are
   global-average-pooled across spatial dims for conv layers and
   flattened for avgpool/fully-connected layers.

3. Caches the resulting per-layer (288, n_channels) feature matrices to
   a single pickle file under `results/cnn_features_<arch>.pkl` so
   downstream encoding-model fits can load them in <5 seconds.

The module is intentionally side-effect-free apart from the cache write
— Milestone 5/6 imports it directly. The Milestone 4 notebook just
calls these functions and adds spot-check figures + PCA panels.

Author: Winston Luk
"""

from __future__ import annotations

import hashlib
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd

# Categories follow Brands 2024 / BAIR localizer convention. Index in
# this list (1-based to match MATLAB) is the value stored in
# stimulus.cat per trial.
CATEGORY_NAMES = (
    "bodies",
    "buildings",
    "faces",
    "objects",
    "scenes",
    "scrambled",
)

# Default cohort: every p13/p14 sixcatlocdiffisi + sixcatloctemporal run
# present on disk (run-04 of p14 sixcatlocdiffisi + run-03/04 of p14
# sixcatloctemporal weren't acquired, per the Milestone 0 inventory).
DEFAULT_BRANDS_TASKS = ("sixcatlocdiffisi", "sixcatloctemporal")
DEFAULT_BRANDS_SUBJECTS = ("sub-p13", "sub-p14")


# ---------------------------------------------------------------------
# Stage 1 — stimulus loading
# ---------------------------------------------------------------------

def find_brands_run_mats(
    stimuli_dir: Path,
    subjects: Iterable[str] = DEFAULT_BRANDS_SUBJECTS,
    tasks: Iterable[str] = DEFAULT_BRANDS_TASKS,
) -> list[Path]:
    """Glob the stimuli directory for each subject × task combination."""
    paths: list[Path] = []
    for sub in subjects:
        for task in tasks:
            pattern = f"{sub}_*_task-{task}_*_run-*.mat"
            paths.extend(sorted(stimuli_dir.glob(pattern)))
    return paths


def _md5_12(buf: bytes) -> str:
    """First 12 hex chars of MD5 — collision-safe at our cohort scale."""
    return hashlib.md5(buf).hexdigest()[:12]


def _read_run_images_and_trials(
    mat_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull the displayed-images stack and trial→image mapping for one run.

    The BAIR v7.3 (HDF5) layout stores
      stimulus.images :  (n_images, 3, H, W)  — image bank
      stimulus.trialindex : (n_trials, 1)     — 1-indexed pointer
      stimulus.cat       : (n_trials, 1)      — 1-indexed category id
    """
    with h5py.File(mat_path, "r") as f:
        images = f["stimulus/images"][:]               # (N_im, 3, H, W) uint8
        trialindex = f["stimulus/trialindex"][:].squeeze().astype(int)
        cat = f["stimulus/cat"][:].squeeze().astype(int)
    return images, trialindex, cat


@dataclass
class StimulusBank:
    """Canonical 288-image set for the Brands p13/p14 cohort."""

    images: np.ndarray       # (n_unique, 3, H, W) uint8
    metadata: pd.DataFrame   # one row per unique image
    source_runs: list[Path]  # which mat files were scanned

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def H(self) -> int:
        return self.images.shape[2]

    @property
    def W(self) -> int:
        return self.images.shape[3]


def load_brands_stimulus_bank(
    stimuli_dir: Path,
    subjects: Iterable[str] = DEFAULT_BRANDS_SUBJECTS,
    tasks: Iterable[str] = DEFAULT_BRANDS_TASKS,
) -> StimulusBank:
    """Build the canonical unique-image bank from per-run mat files.

    Iterates each (subject, task, run), pulls the displayed images
    referenced by trialindex, and deduplicates by exact pixel-content
    hash. Returns 288 unique images for the standard p13+p14 cohort.
    """
    runs = find_brands_run_mats(stimuli_dir, subjects, tasks)
    if not runs:
        raise FileNotFoundError(
            f"No Brands run mats found under {stimuli_dir} for subjects "
            f"{list(subjects)} × tasks {list(tasks)}"
        )

    seen: "OrderedDict[str, dict]" = OrderedDict()
    image_stack: list[np.ndarray] = []

    for run_path in runs:
        images, trialindex, cat = _read_run_images_and_trials(run_path)
        for ti in trialindex:
            img = images[ti - 1]                       # 1-indexed → 0-indexed
            ci = cat[ti - 1]                           # cat is per-image, index by image idx
            h = _md5_12(img.tobytes())
            if h in seen:
                seen[h]["n_presentations"] += 1
                continue
            seen[h] = {
                "image_id": len(seen),                 # 0..n_unique-1
                "hash": h,
                "category_id": int(ci),                 # 1..6
                "category": CATEGORY_NAMES[int(ci) - 1],
                "first_seen_run": run_path.name,
                "first_seen_trialindex": int(ti),
                "n_presentations": 1,
            }
            image_stack.append(img)

    images_arr = np.stack(image_stack, axis=0)         # (n_unique, 3, H, W)
    metadata = pd.DataFrame(list(seen.values()))
    return StimulusBank(
        images=images_arr,
        metadata=metadata.reset_index(drop=True),
        source_runs=runs,
    )


# ---------------------------------------------------------------------
# Stage 2 — CNN feature extraction
# ---------------------------------------------------------------------

# Architecture registry: arch_name → (factory, default_layer_names).
# Layer names match keys in `dict(net.named_modules())` for the
# corresponding torchvision module.

def _cnn_configs():
    """Lazy import torchvision so this module imports cheap when unused.

    Each entry is a 2- or 3-tuple:
      (factory, layer_names)                    — feedforward
      (factory, layer_names, extra_dict)        — special handling

    extra_dict keys:
      channels_last_layers : set[str]
          Layer names whose hook output is (B, H, W, C) rather than
          (B, C, H, W). The hook will permute before global-avg-pooling.
      recurrent_times : dict[str, int]
          For recurrent architectures: maps layer name → number of times
          that layer is called per forward pass. Features are stored as
          "layer@step_N" keys so each timestep is kept separately.
    """
    from torchvision import models
    return {
        "resnet50": (
            lambda: models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            ),
            ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        ),
        "alexnet": (
            lambda: models.alexnet(
                weights=models.AlexNet_Weights.IMAGENET1K_V1
            ),
            # features.0=conv1, features.3=conv2, features.6=conv3,
            # features.8=conv4, features.10=conv5, classifier.4=fc7
            ("features.0", "features.3", "features.6",
             "features.8", "features.10", "classifier.4"),
        ),
        "vgg16": (
            lambda: models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ),
            # block-end conv outputs (post-ReLU): conv1_2, conv2_2,
            # conv3_3, conv4_3, conv5_3, then fc7
            ("features.3", "features.8", "features.15",
             "features.22", "features.29", "classifier.3"),
        ),
        # ------------------------------------------------------------------
        # Newer / biologically-motivated architectures
        # ------------------------------------------------------------------
        "convnext_tiny": (
            lambda: models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            ),
            # end-of-stage blocks (96→192→384→768 ch) + LayerNorm penultimate
            ("features.1", "features.3", "features.5", "features.7",
             "classifier.2"),
        ),
        "densenet121": (
            lambda: models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1
            ),
            ("features.denseblock1", "features.denseblock2",
             "features.denseblock3", "features.denseblock4", "classifier"),
        ),
        "swin_t": (
            lambda: models.swin_t(
                weights=models.Swin_T_Weights.IMAGENET1K_V1
            ),
            # SwinTransformerBlock stages; intermediate outputs are (B,H,W,C)
            ("features.1", "features.3", "features.5", "features.7", "head"),
            {
                "channels_last_layers": {
                    "features.1", "features.3", "features.5", "features.7",
                },
            },
        ),
        "cornet_s": (
            lambda: __import__("cornet").cornet_s(pretrained=True, map_location="cpu"),
            # output of each cortical area; V4 and IT are recurrent
            ("V1.output", "V2.output", "V4.output", "IT.output"),
            {
                "recurrent_times": {
                    "V1.output": 1,
                    "V2.output": 2,
                    "V4.output": 4,
                    "IT.output": 2,
                },
            },
        ),
    }


def _imagenet_preprocess():
    """Standard ImageNet preprocessing (Resize 256, CenterCrop 224)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_cnn_features(
    images: np.ndarray,
    arch: str = "resnet50",
    layers: Optional[tuple[str, ...]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    progress: bool = False,
    arch_extra: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Forward-pass an (N, 3, H, W) uint8 image stack through a pretrained CNN.

    Returns dict layer_name → (N, n_features) float32 array.
    Conv layers are global-average-pooled across spatial dims; everything
    else is flattened.

    For recurrent architectures (arch_extra["recurrent_times"]), keys are
    "layer@step_N" for each recurrent pass so timesteps are kept separate.
    """
    import torch
    from PIL import Image

    cfgs = _cnn_configs()
    if arch not in cfgs:
        raise ValueError(f"Unsupported arch {arch!r}. Have: {list(cfgs)}")
    entry = cfgs[arch]
    factory, defaults = entry[0], entry[1]
    extra = arch_extra if arch_extra is not None else (entry[2] if len(entry) > 2 else {})
    layers = tuple(layers) if layers is not None else defaults

    channels_last = extra.get("channels_last_layers", set())
    recurrent_times = extra.get("recurrent_times", {})

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = factory().eval().to(device)
    net = net.module if hasattr(net, "module") else net   # unwrap DataParallel

    activations: dict[str, list[np.ndarray]] = {}
    call_counter: dict[str, int] = {ln: 0 for ln in layers}

    def make_hook(name: str):
        times = recurrent_times.get(name, 1)

        def hook(module, inp, out):
            t = out
            if t.ndim == 4:
                if name in channels_last:
                    t = t.permute(0, 3, 1, 2)       # (B,H,W,C) → (B,C,H,W)
                pooled = t.mean(dim=(2, 3))          # global average pool
            elif t.ndim == 3:
                pooled = t.mean(dim=1)               # (B, seq, C) → (B, C)
            else:
                pooled = t.flatten(start_dim=1)

            arr = pooled.detach().cpu().numpy()
            if times > 1:
                step = call_counter[name] % times
                key = f"{name}@step_{step}"
            else:
                key = name
            activations.setdefault(key, []).append(arr)
            call_counter[name] += 1

        return hook

    name_to_module = dict(net.named_modules())
    for ln in layers:
        if ln not in name_to_module:
            raise KeyError(
                f"Layer {ln!r} not in {arch}. Sample modules: "
                f"{list(name_to_module)[:20]}…"
            )
        name_to_module[ln].register_forward_hook(make_hook(ln))

    preprocess = _imagenet_preprocess()

    n = images.shape[0]
    batches = range(0, n, batch_size)
    if progress:
        from tqdm import tqdm
        batches = tqdm(batches, total=-(-n // batch_size),
                       desc=arch, unit="batch", dynamic_ncols=True)

    with torch.no_grad():
        for start in batches:
            for ln in call_counter:
                call_counter[ln] = 0               # reset per-batch step index
            batch = images[start:start + batch_size]
            tensors = []
            for img in batch:
                # img is (3, H, W) uint8 in RGB. PIL expects (H, W, 3).
                pil = Image.fromarray(np.transpose(img, (1, 2, 0)), "RGB")
                tensors.append(preprocess(pil))
            x = torch.stack(tensors, dim=0).to(device)
            _ = net(x)

    return {key: np.concatenate(acts, axis=0) for key, acts in activations.items()}


# ---------------------------------------------------------------------
# Stage 3 — caching
# ---------------------------------------------------------------------

@dataclass
class FeatureCache:
    """On-disk pickle bundling features + the metadata that produced them."""

    arch: str
    layers: tuple[str, ...]
    features: dict[str, np.ndarray]   # layer_name → (N, n_features) float32
    metadata: pd.DataFrame            # parallel to features along axis 0
    n_images: int
    image_shape: tuple[int, int, int]  # (3, H, W) of the raw stim images
    extraction_seconds: float = 0.0
    extraction_device: str = "cpu"

    def shapes(self) -> dict[str, tuple]:
        return {ln: feats.shape for ln, feats in self.features.items()}


def build_and_cache_features(
    bank: StimulusBank,
    out_path: Path,
    arch: str = "resnet50",
    layers: Optional[tuple[str, ...]] = None,
    batch_size: int = 16,
    device: Optional[str] = None,
    progress: bool = False,
    arch_extra: Optional[dict] = None,
) -> FeatureCache:
    """Run extraction end-to-end and write a pickle to `out_path`."""
    import torch
    actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    feats = extract_cnn_features(
        bank.images, arch=arch, layers=layers,
        batch_size=batch_size, device=actual_device,
        progress=progress, arch_extra=arch_extra,
    )
    elapsed = time.perf_counter() - t0

    cache = FeatureCache(
        arch=arch,
        layers=tuple(feats.keys()),
        features=feats,
        metadata=bank.metadata.copy(),
        n_images=bank.images.shape[0],
        image_shape=tuple(bank.images.shape[1:]),
        extraction_seconds=elapsed,
        extraction_device=actual_device,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def load_feature_cache(path: Path) -> FeatureCache:
    """Re-hydrate a pickled FeatureCache; intended for downstream milestones."""
    with open(path, "rb") as f:
        return pickle.load(f)
