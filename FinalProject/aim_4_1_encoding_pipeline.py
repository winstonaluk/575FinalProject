"""
Aim 1 — Per-electrode CNN→Broadband-Gamma Encoding Models
=========================================================

Pipeline scaffold for fitting per-electrode encoding models on the
Brands 2024 / Groen 2022 visual ECoG dataset (OpenNeuro ds004194).

Cohort:
    Primary  — p13, p14 (Brands 2024 six-category natural-image task)
    Validate — p02, p06, p07, p10 (Groen 2022 spatiotemporal task,
               for pipeline-validation against published results)

Approach (after Kuzovkin et al. 2018):
    1. Extract per-trial broadband-gamma (50–200 Hz) responses from
       precomputed `/derivatives/ECoGBroadband/` data.
    2. Forward-pass each unique stimulus image through a pretrained
       CNN (ResNet50 or AlexNet); cache layer activations.
    3. Per electrode × per layer: ridge regression with
       leave-one-stimulus-out cross-validation.
    4. Report cross-validated R² per electrode-layer; identify
       layer-of-best-fit per electrode; visualize gradient across
       V1-V3 → VOTC → LOTC.

Author: Winston Luk
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import h5py
from joblib import Parallel, delayed

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SESSION = "ses-nyuecog01"   # BIDS session label for both p13 and p14


@dataclass
class Config:
    bids_root: Path = Path("/Users/winstonluk/Documents/NEURON/FinalProject/data/ds004194")
    derivatives: Path = field(init=False)
    broadband_dir: Path = field(init=False)
    brands_dir: Path = field(init=False)
    freesurfer_dir: Path = field(init=False)

    # Cohort selection
    primary_subjects: tuple[str, ...] = ("sub-p13", "sub-p14")
    validation_subjects: tuple[str, ...] = (
        "sub-p02", "sub-p06", "sub-p07", "sub-p10"
    )

    # Task names (BIDS task labels — confirmed on disk via Milestone 0)
    # Brands 2024 natural-image task: two variants differing in ISI sequence
    natural_image_task: str = "sixcatlocdiffisi"       # primary ISI-varying task
    natural_image_task_temporal: str = "sixcatloctemporal"  # temporal variant
    spatiotemporal_task: str = "spatialpattern"

    # Trial epoching (Brands 2024 used [-0.1, 1.2] s relative to onset)
    trial_window: tuple[float, float] = (-0.1, 1.2)
    baseline_window: tuple[float, float] = (-0.1, 0.0)
    response_window: tuple[float, float] = (0.05, 0.55)  # peri-stimulus

    # Sampling rate (post-resample; both NYU and UMCU at 512 Hz)
    fs: float = 512.0

    # CNN settings
    # cnn_arch must be a key in CNN_CONFIGS below; cnn_layers=None uses
    # that architecture's registered defaults.
    cnn_arch: str = "resnet50"   # one of: resnet50, alexnet, vgg16
    cnn_layers: Optional[tuple[str, ...]] = None
    pool_strategy: str = "global_avg"   # collapse spatial dims per channel

    # Ridge regression
    alphas: tuple[float, ...] = tuple(np.logspace(-2, 8, 21))
    cv_folds: int = 12

    # Output
    out_dir: Path = Path("FinalProject/results")

    def __post_init__(self):
        self.derivatives = self.bids_root / "derivatives"
        self.broadband_dir = self.derivatives / "ECoGBroadband"
        self.brands_dir = self.derivatives / "Brands2024TemporalAdaptationECoG"
        self.freesurfer_dir = self.derivatives / "freesurfer"
        self.out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Stage 1 — Data loading
# ---------------------------------------------------------------------

def load_subject_electrodes(cfg: Config, subject: str) -> pd.DataFrame:
    """Load per-electrode metadata + retinotopic atlas labels.

    Returns a DataFrame indexed by electrode name with columns:
        x, y, z (native T1 coords)
        wang_label_max     — max-prob Wang atlas assignment
        wang_probs         — dict of region → probability
        benson_label       — Benson anatomical-atlas assignment
        visual_group       — V1-V3 / VOTC / LOTC (per Brands 2024 rules)
    """
    # The dataset ships per-subject electrodes.tsv files in BIDS root,
    # plus atlas-matched derivatives in /derivatives/freesurfer/<sub>/
    # Adapt this path to actual layout once dataset is downloaded.
    elec_tsv = cfg.bids_root / subject / f"{subject}_electrodes.tsv"
    df = pd.read_csv(elec_tsv, sep="\t")

    # TODO: merge with atlas labels from freesurfer derivatives.
    # The Winawer lab pipeline writes a JSON sidecar with full
    # probability distributions per electrode — load that here.
    atlas_json = cfg.freesurfer_dir / subject / "atlas_matches.json"
    if atlas_json.exists():
        with open(atlas_json) as f:
            atlas = json.load(f)
        df["wang_label_max"] = df["name"].map(
            lambda e: atlas.get(e, {}).get("wang_max_label")
        )
        df["wang_probs"] = df["name"].map(
            lambda e: atlas.get(e, {}).get("wang_probs", {})
        )
        df["benson_label"] = df["name"].map(
            lambda e: atlas.get(e, {}).get("benson_label")
        )

    df["visual_group"] = df["wang_label_max"].apply(_group_assignment)
    return df


def _group_assignment(wang_label: Optional[str]) -> Optional[str]:
    """Brands 2024 grouping (Table 2 of paper)."""
    if wang_label is None:
        return None
    label = wang_label.lower()
    if any(k in label for k in ["v1", "v2", "v3v", "v3d"]):
        return "V1-V3"
    if any(k in label for k in ["hv4", "vo1", "vo2"]):
        return "VOTC"
    if any(k in label for k in
           ["to1", "to2", "lo1", "lo2", "v3a", "v3b", "ips"]):
        return "LOTC"
    return None


def load_broadband_run(
    cfg: Config, subject: str, run: int, task: str
) -> tuple[np.ndarray, dict]:
    """Load common-average-referenced broadband-gamma envelope for one run.

    Files are BrainVision format (.vhdr/.eeg/.vmrk) under the BIDS
    session subdirectory ses-nyuecog01/ieeg/.

    Returns:
        bb       — array of shape (n_channels, n_samples)
        info     — dict with channel names, fs, etc.
    """
    import mne
    ieeg_dir = cfg.broadband_dir / subject / SESSION / "ieeg"
    fname = (f"{subject}_{SESSION}_task-{task}_run-{run:02d}"
             f"_desc-broadband_ieeg.vhdr")
    fp = ieeg_dir / fname
    raw = mne.io.read_raw_brainvision(str(fp), preload=True, verbose=False)
    bb = raw.get_data()                   # (n_ch, n_samples)
    ch_names = raw.ch_names
    fs = float(raw.info["sfreq"])
    return bb, {"ch_names": ch_names, "fs": fs}


def load_events(cfg: Config, subject: str, run: int, task: str) -> pd.DataFrame:
    """Load BIDS events.tsv for a run.

    Expected columns: onset, duration, trial_type, stim_file, isi
    """
    ieeg_dir = cfg.broadband_dir / subject / SESSION / "ieeg"
    fname = (f"{subject}_{SESSION}_task-{task}_run-{run:02d}"
             f"_desc-broadband_events.tsv")
    return pd.read_csv(ieeg_dir / fname, sep="\t")


# ---------------------------------------------------------------------
# Stage 2 — Trial extraction
# ---------------------------------------------------------------------

def epoch_trials(
    bb: np.ndarray, info: dict, events: pd.DataFrame, cfg: Config
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut continuous broadband into per-trial epochs.

    Returns:
        epochs   — array (n_trials, n_channels, n_timepoints)
        ev_kept  — events DataFrame restricted to retained trials
    """
    fs = info["fs"]
    pre, post = cfg.trial_window
    n_pre = int(round(-pre * fs))
    n_post = int(round(post * fs))
    n_t = n_pre + n_post

    epochs = []
    keep_idx = []
    for i, row in events.iterrows():
        onset_sample = int(round(row["onset"] * fs))
        i0 = onset_sample - n_pre
        i1 = onset_sample + n_post
        if i0 < 0 or i1 > bb.shape[1]:
            continue
        epochs.append(bb[:, i0:i1])
        keep_idx.append(i)
    epochs = np.stack(epochs, axis=0)            # (trials, ch, t)
    ev_kept = events.loc[keep_idx].reset_index(drop=True)

    # Baseline correction → percent signal change (Brands 2024 method)
    bl0 = int(round((cfg.baseline_window[0] - pre) * fs))
    bl1 = int(round((cfg.baseline_window[1] - pre) * fs))
    baseline = epochs[:, :, bl0:bl1].mean(axis=-1, keepdims=True)
    epochs = (epochs - baseline) / baseline      # fractional change

    return epochs, ev_kept


def trial_response_summary(
    epochs: np.ndarray, cfg: Config
) -> np.ndarray:
    """Collapse each trial's broadband time course to a scalar response.

    Following Kuzovkin 2018: mean broadband-gamma power over the
    peri-stimulus window. Returns array (n_trials, n_channels).
    """
    fs = cfg.fs
    pre = cfg.trial_window[0]
    r0 = int(round((cfg.response_window[0] - pre) * fs))
    r1 = int(round((cfg.response_window[1] - pre) * fs))
    return epochs[:, :, r0:r1].mean(axis=-1)     # (trials, channels)


# ---------------------------------------------------------------------
# CNN architecture registry
# ---------------------------------------------------------------------
# Each entry: arch_name → (model_factory, default_layer_names)
# Layer names must match torch.nn.Module names as returned by
# net.named_modules() for the given architecture.

def _cnn_configs():
    from torchvision import models
    return {
        "resnet50": (
            lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
            ("conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"),
        ),
        "alexnet": (
            lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1),
            # features.0=conv1, features.3=conv2, features.6=conv3,
            # features.8=conv4, features.10=conv5, classifier.6=fc3
            ("features.0", "features.3", "features.6",
             "features.8", "features.10", "classifier.6"),
        ),
        "vgg16": (
            lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
            # block outputs: after pool1–pool5, then fc2
            ("features.4", "features.9", "features.16",
             "features.23", "features.30", "classifier.3"),
        ),
    }


# ---------------------------------------------------------------------
# Stage 3 — CNN feature extraction
# ---------------------------------------------------------------------

def extract_cnn_features(
    cfg: Config, stim_paths: list[Path]
) -> dict[str, np.ndarray]:
    """Forward-pass each stimulus through a pretrained CNN once.

    Returns dict layer_name → array (n_stimuli, n_features).
    Cached to disk so subsequent runs skip recomputation.
    """
    cache_path = cfg.out_dir / f"cnn_features_{cfg.cnn_arch}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import torch
    from torchvision import transforms
    from PIL import Image

    configs = _cnn_configs()
    if cfg.cnn_arch not in configs:
        raise ValueError(
            f"Unsupported architecture: {cfg.cnn_arch!r}. "
            f"Supported: {list(configs)}"
        )
    model_factory, default_layers = configs[cfg.cnn_arch]
    layers = cfg.cnn_layers if cfg.cnn_layers is not None else default_layers

    # Build network and forward hooks on requested layers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model_factory()
    net.eval().to(device)

    activations: dict[str, list[np.ndarray]] = {ln: [] for ln in layers}

    def make_hook(name):
        def hook(module, inp, out):
            # Global-average-pool spatial dimensions for conv layers
            if out.ndim == 4:
                pooled = out.mean(dim=(2, 3))     # (batch, channels)
            else:
                pooled = out.flatten(start_dim=1)
            activations[name].append(pooled.detach().cpu().numpy())
        return hook

    # Register hooks (module names depend on architecture; adapt as needed)
    for ln in layers:
        module = dict(net.named_modules()).get(ln)
        if module is None:
            raise KeyError(
                f"Layer {ln!r} not found in {cfg.cnn_arch}. "
                f"Available: {list(dict(net.named_modules()).keys())[:20]}…"
            )
        module.register_forward_hook(make_hook(ln))

    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for path in stim_paths:
            img = Image.open(path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            _ = net(x)

    features = {ln: np.concatenate(arrs, axis=0)
                for ln, arrs in activations.items()}

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    return features


# ---------------------------------------------------------------------
# Stage 4 — Ridge encoding model
# ---------------------------------------------------------------------

def fit_encoding_model(
    X: np.ndarray, y: np.ndarray, cfg: Config
) -> dict:
    """Per-electrode ridge regression with cross-validated alpha + R².

    Args:
        X — (n_trials, n_features) CNN-layer activations matched to trials
        y — (n_trials,) broadband response for one electrode

    Returns:
        dict with cv_r2, best_alpha, predictions
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=42)
    preds = np.zeros_like(y)
    fold_alphas = []

    for tr, te in kf.split(X):
        # Standardize features within fold (avoid leakage)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])

        # RidgeCV fits all alphas via efficient generalized CV; fast
        ridge = RidgeCV(alphas=cfg.alphas, cv=5)
        ridge.fit(X_tr, y[tr])
        preds[te] = ridge.predict(X_te)
        fold_alphas.append(ridge.alpha_)

    # Cross-validated R² (single value across all folds combined)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    cv_r2 = 1.0 - ss_res / ss_tot

    return {
        "cv_r2": float(cv_r2),
        "median_alpha": float(np.median(fold_alphas)),
        "predictions": preds,
    }


def fit_all_electrodes(
    X_per_layer: dict[str, np.ndarray],
    Y: np.ndarray,
    electrode_names: list[str],
    cfg: Config,
) -> pd.DataFrame:
    """Fit ridge encoder per electrode × per CNN layer in parallel.

    Args:
        X_per_layer — dict layer_name → (n_trials, n_features)
        Y           — (n_trials, n_electrodes) broadband responses
        electrode_names — list of electrode IDs

    Returns:
        long-format DataFrame with cv_r2 per (electrode, layer)
    """
    rows = []
    for layer_name, X in X_per_layer.items():
        # Parallelize across electrodes — ridge is CPU-bound, scales well
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(fit_encoding_model)(X, Y[:, i], cfg)
            for i in range(Y.shape[1])
        )
        for ename, res in zip(electrode_names, results):
            rows.append({
                "electrode": ename,
                "layer": layer_name,
                "cv_r2": res["cv_r2"],
                "median_alpha": res["median_alpha"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Stage 5 — Layer-of-best-fit analysis
# ---------------------------------------------------------------------

def layer_of_best_fit(results: pd.DataFrame) -> pd.DataFrame:
    """Per electrode, identify the CNN layer with highest cv R²."""
    return (
        results.sort_values("cv_r2", ascending=False)
        .drop_duplicates("electrode")
        .reset_index(drop=True)
    )


def hierarchy_gradient(
    results: pd.DataFrame, electrodes_df: pd.DataFrame
) -> pd.DataFrame:
    """Test for the predicted V1-V3 → VOTC → LOTC layer-depth gradient.

    Hypothesis (Kuzovkin 2018): early CNN layers best predict V1
    responses; later layers best predict higher-area responses.
    """
    best = layer_of_best_fit(results)
    merged = best.merge(
        electrodes_df[["name", "visual_group"]],
        left_on="electrode", right_on="name", how="left"
    )
    # Encode CNN layer as ordinal depth using the order layers appear in results
    all_layers = results["layer"].unique().tolist()
    layer_order = {ln: i for i, ln in enumerate(all_layers)}
    merged["layer_depth"] = merged["layer"].map(layer_order)
    return merged


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def run_subject(cfg: Config, subject: str, task: str) -> None:
    """Full pipeline for one subject."""
    print(f"\n=== {subject} | task: {task} ===")
    elec_df = load_subject_electrodes(cfg, subject)

    # Discover available runs
    runs = sorted(int(p.stem.split("run-")[1].split("_")[0])
                  for p in cfg.broadband_dir.glob(
                      f"{subject}/*task-{task}*_desc-broadband_ieeg.h5"))
    if not runs:
        print(f"  no runs found for task {task}; skipping")
        return

    # Load and concatenate trials across runs
    all_responses, all_events = [], []
    for run in runs:
        bb, info = load_broadband_run(cfg, subject, run, task)
        events = load_events(cfg, subject, run, task)
        epochs, ev_kept = epoch_trials(bb, info, events, cfg)
        responses = trial_response_summary(epochs, cfg)
        all_responses.append(responses)
        all_events.append(ev_kept)
    Y = np.concatenate(all_responses, axis=0)        # (trials, channels)
    events_all = pd.concat(all_events, ignore_index=True)

    # Restrict to visually responsive electrodes (per Brands 2024 selection)
    visual_mask = elec_df["visual_group"].isin(["V1-V3", "VOTC", "LOTC"])
    visual_electrodes = elec_df.loc[visual_mask, "name"].tolist()
    ch_idx = [i for i, ch in enumerate(info["ch_names"])
              if ch in visual_electrodes]
    Y = Y[:, ch_idx]
    electrode_names = [info["ch_names"][i] for i in ch_idx]

    # CNN features for each unique stim_file in events_all
    unique_stims = events_all["stim_file"].drop_duplicates().tolist()
    stim_paths = [cfg.bids_root / "stimuli" / s for s in unique_stims]
    features = extract_cnn_features(cfg, stim_paths)

    # Match trial-by-trial: each trial's stimulus → its CNN activation
    stim_to_idx = {s: i for i, s in enumerate(unique_stims)}
    trial_stim_idx = events_all["stim_file"].map(stim_to_idx).values
    X_per_layer = {ln: feats[trial_stim_idx]
                   for ln, feats in features.items()}

    # Fit encoders
    results = fit_all_electrodes(X_per_layer, Y, electrode_names, cfg)
    results.to_csv(cfg.out_dir / f"{subject}_encoding_results.csv",
                   index=False)

    # Layer-of-best-fit and hierarchy gradient
    grad = hierarchy_gradient(results, elec_df)
    grad.to_csv(cfg.out_dir / f"{subject}_hierarchy_gradient.csv",
                index=False)
    print(f"  fit complete; {len(electrode_names)} electrodes × "
          f"{len(features)} layers")


def main():
    cfg = Config()
    # Phase A — pipeline validation against Groen 2022 published results
    for subj in cfg.validation_subjects:
        run_subject(cfg, subj, cfg.spatiotemporal_task)
    # Phase B — primary Aim 1 demonstration on natural images
    for subj in cfg.primary_subjects:
        run_subject(cfg, subj, cfg.natural_image_task)


if __name__ == "__main__":
    main()