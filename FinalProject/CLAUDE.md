# CLAUDE.md

Context for the Aim 1 implementation of *Towards Representationally Informed Cortical Visual Prosthetics* (Winston Luk, neural engineering graduate project / IEEE conference paper format).

This file is the working memory for code work in this repository. Read it before generating, editing, or reviewing code.

---

## 1. Project context

### What this repository is

Implementation of **Aim 1** of a larger NIH-style proposal on cortical visual prosthetics. The proposal has three aims:

- **Aim 1** — Establish per-electrode encoding models at V1 via ECoG. Train a CNN-to-broadband-gamma encoding model so that, given a visual stimulus, we can predict the broadband response at each V1 electrode. *This is the only aim implementable with the dataset we have.*
- **Aim 2** — Record V1-stim-evoked responses at V2/V4/IT. Requires microstimulation. *Out of scope for this implementation.*
- **Aim 3a** — Compare V2/V4/IT responses to V1 stimulation via RSA. *Partially addressable via passive-viewing data; treated as stretch goal here. Subject to change.*
- **Aim 3b** — CNN feature-space decoding of V1-stim-evoked V4/IT responses. *Out of scope (requires microstimulation).*

The deliverables are a 4-page NIH-style proposal (already drafted), a course-project report, and an IEEE conference paper. The code in this repo backs all three.

### What this is *not*

- Not a full V2-V4 generative model (that's the larger research direction, separate work).
- Not a Gabor-tuning analysis at the column level — ECoG resolution doesn't support that.
- Not a microstimulation study.

---

## 2. Dataset

**OpenNeuro `ds004194`** — visual ECoG dataset (Groen, Brands, Yuasa, Petridou, Winawer; CC0 license).

- 14 patients (p01–p14), NYU + UMCU, clinical and (in some) high-density grids.
- Three task batteries on the same patients:
  - **Spatiotemporal-pattern task** (Groen 2022): grayscale curvy-line stimuli, varying duration / ISI / contrast.
  - **pRF mapping task** (Yuasa 2023): drifting bar with checkerboard pattern; gives per-electrode retinotopic position + pRF size.
  - **Six-category natural-image task** (Brands 2024): bodies / buildings / faces / objects / scenes / scrambled, varying duration and ISI. **This is the closest analog to Kuzovkin 2018 paradigm and is the primary source of data for our Aim 1.**
- Pre-computed broadband (50–200 Hz) envelopes available in `/derivatives/ECoGBroadband/`.
- Wang + Benson retinotopic atlas labels per electrode in `/derivatives/freesurfer/`.
- Reproducibility derivatives for both Groen 2022 and Brands 2024 in their respective `/derivatives/` subfolders.

### Cohort selection (rationale, not just labels)

| Cohort                                       | Subjects                     | Purpose                                                                                           | What's there                                                                      |
| -------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Primary (Aim 1 demonstration)**      | **p13, p14**           | Per-electrode CNN→broadband encoding from natural images                                         | Brands six-category task; 17 V1-V3 electrodes (p13: 3, p14: 14), ~8 VOTC, ~9 LOTC |
| **Validation (pipeline reproduction)** | **p02, p06, p07, p10** | Reproduce Groen 2022 V1 contrast/duration/ISI findings before applying pipeline to natural images | Groen spatiotemporal-pattern task; V1 contributors per Groen Table 2              |
| **Stretch / IEEE paper only**          | p11                          | Cross-task analysis (only patient with both Groen and Brands tasks)                               | HD grid, 26 included electrodes, predominantly LOTC — no V1                      |

**Why p13 + p14 specifically:** they are the only patients with both natural-image task data AND V1-V3 coverage in the Brands cohort. p11 has the densest electrodes but no V1; p12 has too few electrodes anywhere.

**Why probabilistic atlas labels matter here:** ECoG electrodes pool across mm of cortex and may straddle area boundaries. Always use the full Wang probability vector per electrode, not just the max-probability label. Bootstrap over electrode-area assignments (n=1000) for any group-level statistic. This is what Groen 2022 did; we follow the same convention.

**Coverage limitation we live with:** Wang atlas in this dataset extends to V3a/b, hV4, LO1/2, TO1/2, IPS. hV4 coverage is sparse (a few electrodes in p05/p06 only). When we say "V4" in proposal language but mean "the dataset's hV4 + adjacent regions," we use **VOTC** in actual report text following Brands 2024.

---

## 3. The spatial-resolution scoping decision

**This is the most important framing decision in the project. Don't drift from it.**

### The concern

ECoG contacts (2.3 mm diameter, 10 mm spacing) record from ~3 mm cortical patches. In V1 that's thousands of orientation columns averaged together. Column-level feature tuning is *not* recoverable at this scale. Standard clinical grids also sample sparsely across cortex — only p10 and p11 have HD grids, and neither covers V1.

### The scoping decision (Option A — adopted)

**The encoding model framework is well-posed for predicting *which CNN layer* best explains a given electrode's responses, but ill-posed for fine-grained feature-tuning interpretation of individual electrodes.**

Concretely, this means:

- ✅ **Allowed claims:** "Electrode X is best predicted by ResNet50 layer3, cv-R² = 0.42"; "layer-of-best-fit increases monotonically V1-V3 → VOTC → LOTC"; "the encoding-model framework recovers the published Groen 2022 contrast saturation in V1."
- ❌ **Claims we will not make:** "Electrode X is tuned to vertical edges"; "this electrode's receptive field is dominated by feature Y"; "we recover Gabor-like filters from encoding weights."

This matches what Kuzovkin et al. 2018 actually did with clinical-grid ECoG — no feature-level tuning claims, only layer-of-best-fit and area gradients. The framework is appropriate for our resolution.

### Where this gets called out in the report

A dedicated paragraph in the Methods Limitations section explicitly addresses spatial resolution and the population-pooling assumption. Don't bury it. It's part of the scientific framing, not a defensive footnote.

### What we deliberately deferred

The analysis we're *not* doing in this round, but is a natural next step for the IEEE paper:

- **HD-grid spatial-coherence robustness check on p10/p11.** Test whether layer-of-best-fit assignments are spatially smooth across adjacent HD-grid contacts. This would be a positive piece of evidence that the encoding model captures real cortical organization rather than electrode-by-electrode noise. Neither subject has V1 coverage, so this is a VOTC/LOTC analysis, not a V1 one.
- **pRF-aware encoding model.** Two-stage model: pRF → spatial pooling of CNN activations within pRF → broadband response. This is methodologically cleaner but a much heavier lift. Reserved for follow-up.

---

## 4. Implementation plan

### Pipeline structure

Two modules now split the work:

- **`aim_4_1_encoding_pipeline.py`** (root) — data loading, trial extraction, ridge regression. Provides `Config`, `load_broadband_run`, `load_events`, `epoch_trials`, `fit_encoding_model`. Broadband files are **BrainVision format** (`.vhdr/.eeg/.vmrk`) read via MNE; path pattern is `ECoGBroadband/sub-{p13}/ses-nyuecog01/ieeg/sub-{p13}_ses-nyuecog01_task-{task}_run-{run:02d}_desc-broadband_ieeg.vhdr`. Session label `ses-nyuecog01` applies to both p13 and p14.
- **`src/cnn_features.py`** — CNN feature extraction for all 7 architectures; `StimulusBank` and `FeatureCache` dataclasses; `build_and_cache_features`, `load_feature_cache`. Supersedes the CNN extraction stub in the pipeline module.

Five pipeline stages:

1. **Data loading** — per-subject electrode metadata + Wang atlas probabilities; load BrainVision broadband via MNE; parse BIDS events.tsv per run.
2. **Trial extraction** — epoch [-0.1, 1.2] s relative to stimulus onset; baseline-correct to fractional signal change using [-0.1, 0.0] s; reject trials with peri-stimulus peak > 3 SD outside the [0.05, 0.85] s window (Groen 2022 criterion).
3. **CNN feature extraction** — forward-pass each unique stimulus once through 7 pretrained architectures (see § 4a); global-average-pool spatial dims of conv layers; cache to disk per arch.
4. **Per-electrode ridge regression** — RidgeCV with 12-fold leave-stimulus-out outer CV; alphas in `np.logspace(-2, 8, 21)`; standardize features within fold to avoid leakage; pooled cv-R² across folds (not averaged).
5. **Layer-of-best-fit + hierarchy gradient** — Kruskal-Wallis across V1-V3 / VOTC / LOTC; pairwise Mann-Whitney with Bonferroni for adjacent groups.

### 4a. CNN architecture registry (all extracted and cached)

Seven architectures are registered in `src/cnn_features.py` and have cached feature files under `results/cnn_features_{arch}.pkl`. All produce `(288, n_features)` per layer.

| Arch | Layers cached | Notes |
|---|---|---|
| `resnet50` | conv1, layer1–4, avgpool | 6 layers; primary model for Aim 1 |
| `alexnet` | features.0,3,6,8,10, classifier.4 | 6 layers; Kuzovkin 2018 baseline |
| `vgg16` | features.3,8,15,22,29, classifier.3 | 6 layers; robustness check |
| `convnext_tiny` | features.1,3,5,7, classifier.2 | 5 layers; modern feedforward |
| `densenet121` | denseblock1–4, classifier | 5 layers; dense skip connections |
| `swin_t` | features.1,3,5,7, head | 5 layers; hierarchical vision transformer; outputs are channels-last, permuted before GAP |
| `cornet_s` | V1, V2×2, V4×4, IT×2 | 9 keys (recurrent steps stored separately as `area@step_N`); biologically motivated |

**CORnet-S notes:** V2 fires 2 times per forward pass, V4 4 times, IT 2 times; hooks track each recurrent step. Model is wrapped in `DataParallel` — always unwrap before registering hooks (`net = net.module if hasattr(net, "module") else net`). Load with `map_location="cpu"` to avoid CUDA deserialization errors on CPU-only machines.

### Phases of execution

- **Phase A:** validation cohort (p02, p06, p07, p10) on Groen spatiotemporal-pattern task. Reproduce three diagnostics from Groen 2022:

  1. V1 contrast-response function + C50
  2. Subadditive temporal summation deviation from linear prediction
  3. Repetition-suppression recovery vs. ISI

  Pipeline is "validated" when area-level summaries fall within the 68% CIs reported in the published figures.
- **Phase B:** primary cohort (p13, p14) on Brands six-category natural-image task. Per-electrode CNN encoding + layer-of-best-fit + gradient analysis.
- **Phase C (stretch):** layer-of-best-fit gradient across V1-V3 → VOTC → LOTC on p13+p14 (the bridge to Aim 3a in the proposal).

### Performance expectations

- Phase A + B together: ~15-30 min on a modern laptop, no GPU needed.
- Bottleneck is **not** model training — ridge regression is closed-form on small matrices (~250 trials × ~512 features), CPU-bound by BLAS, parallelized across electrodes with joblib.
- GPU only matters for Stage 3 (CNN forward passes), and even there it's ~2 minutes vs ~1 minute. Skip the GPU.
- Most actual time goes to BIDS parsing, stimulus-to-trial alignment, and electrode-to-area assignment debugging. Plan for that.

---

## 5. Code conventions and watch-outs

### Things to get right the first time

- **Atlas labels are probability vectors, not strings.** Anything that aggregates over electrodes within an area must bootstrap over the probability assignments. Don't shortcut with `wang_label_max == "V1"` for analysis; that's fine for plotting but not for statistics.
- **Baseline correction is to fractional signal change**, not z-score. Brands 2024 uses (x - baseline) / baseline. Z-scoring is only used during Groen 2022's electrode-selection step, not in the analyses.
- **CV folds are leave-stimulus-out**, not leave-trial-out. Each unique image (288 total in Brands) appears in the test set in exactly one fold. This prevents the model from memorizing image-specific noise that recurs across trials.
- **Standardize features within fold.** `StandardScaler` fit on training, applied to test. No leakage.
- **cv-R² is pooled across folds**, computed as `1 - sum(residuals²) / sum(deviations²)` over all out-of-fold predictions. Don't average per-fold R²; that's biased.

### Things that look right but are wrong

- **Don't use leave-one-trial-out CV.** Trials of the same image have correlated noise and the model will inflate R².
- **Don't use the max-probability atlas label for grouped statistics.** Use bootstrapped probabilistic assignment (Groen 2022 § Probabilistic electrode assignment).
- **Don't put line-noise frequencies in the broadband bands.** NYU = exclude 60/120/180 Hz; UMCU = exclude 50/100/150 Hz. The pre-computed derivatives already handle this; don't re-derive broadband from raw voltage unless you have to.
- **Don't try to interpret encoding-model weights as receptive fields.** Spatial resolution doesn't support it. See § 3 above.
- **Don't reach for a GPU.** Ridge is closed-form. Adding GPU code is a complexity tax without a speed win.

### Repository layout (current)

```
.
├── CLAUDE.md                          # this file
├── aim_4_1_encoding_pipeline.py       # data loading, trial extraction, ridge regression
├── data/                              # NOT committed; ds004194 download lives here
├── notebooks/
│   ├── milestone_04_cnn_features.ipynb        # CNN feature exploration, PCA, Fisher ratio
│   ├── milestone_04b_rerun_extraction.ipynb   # idempotent per-arch extraction runner
│   └── milestone_04c_separability_analysis.ipynb  # RSA, Isomap geodesic, scrambled check
├── results/
│   ├── cache/
│   │   └── stimulus_bank.pkl          # 288-image StimulusBank (rebuilt from .mat files)
│   ├── cnn_features_{arch}.pkl        # FeatureCache per arch (7 files)
│   └── milestone_04c_*.{csv,png}      # analysis outputs
├── src/
│   └── cnn_features.py                # CNN extraction: StimulusBank, FeatureCache, 7 arch registry
└── tests/                             # pipeline-validation diagnostics against Groen 2022
```

### Dependencies

Python 3.11+; numpy, scipy, scikit-learn, pandas, h5py, joblib, mne (BrainVision loading), torch + torchvision (CNN feature extraction), tqdm (progress bars), cornet (`pip install cornet`, for CORnet-S), matplotlib + seaborn for figures.

---

## 6. Open TODOs from the scaffold

- [x] Verify exact BIDS task labels — **confirmed (Milestone 0)**:
  - Validation (Groen): `task-spatialpattern`, `task-temporalpattern`
  - Primary (Brands): `task-sixcatlocdiffisi`, `task-sixcatloctemporal` (NOT `task-sixcatlocdiffisidur`)
  - Also present in dataset: `task-soc`, `task-sixcatlocisidiff` (not used in our pipeline)
- [x] Confirm path patterns under `/derivatives/ECoGBroadband/sub-*/` — **confirmed and corrected (Milestone 4b/c)**:
  - Files are **BrainVision format** (`.vhdr/.eeg/.vmrk`), not HDF5.
  - Full path: `ECoGBroadband/sub-{p13}/ses-nyuecog01/ieeg/sub-{p13}_ses-nyuecog01_task-{task}_run-{run:02d}_desc-broadband_ieeg.vhdr`
  - Session label `ses-nyuecog01` appears in **both** the subdirectory and the filename.
  - Load via `mne.io.read_raw_brainvision(fp, preload=True, verbose=False)` → `.get_data()` returns `(n_ch, n_samples)`.
- [ ] Confirm schema of the atlas-matches JSON sidecar from `/derivatives/freesurfer/sub-*/`. The Winawer-lab pipeline writes per-electrode probability dictionaries; check the actual key names.
- [x] Replicate the irisgroen/temporalECoG `ecog_selectElectrodes.m` reliability-based electrode selection (split-half R² > 0.22) in Python for Phase A validation. **Done in Milestone 3.**
- [x] Pull stimulus images — **done (Milestone 4)**. 288 unique images loaded from raw `.mat` files via `src/cnn_features.py:load_brands_stimulus_bank()`. Category assignment uses `cat[ti-1]` (per-image, not per-trial). Stimulus bank cached at `results/cache/stimulus_bank.pkl`.

---

## 7. Reference papers (the load-bearing ones)

- **Groen et al. 2022** — *J Neurosci.* The dataset paper for the spatiotemporal-pattern task; defines the preprocessing pipeline and the probabilistic atlas-assignment procedure we follow.
- **Brands et al. 2024** — *PLOS Comp Biol.* The six-category natural-image task paper; defines our primary stimulus set and the V1-V3 / VOTC / LOTC grouping.
- **Yuasa et al. 2023** — pRF mapping task. We don't use this task in Aim 1 but it's the source of per-electrode pRF estimates that would feed the Option C two-stage model.
- **Kuzovkin et al. 2018** — *Comm Biol.* The methodological template for CNN→broadband-gamma encoding on clinical-grid ECoG. Their result that early CNN layers preferentially predict V1 broadband, and later layers higher areas, is what Aim 1 reproduces.
- **Yamins & DiCarlo 2014/2016** — the goal-driven hierarchical CNN framework that Aim 1 sits within.
- **Dubey & Ray 2019** — ECoG spread is ~3 mm, gamma coherence drops at ≥3-4 mm. Cited in the spatial-resolution Limitations paragraph.

---

## 8. How to use this file with Claude

- This file is the source of truth for project context. If something contradicts this file, ask before changing course.
- Read § 3 (spatial-resolution scoping) before writing any code that touches encoding-model weights, feature interpretation, or claims about cortical tuning. The Option-A scoping is a deliberate decision, not an oversight.
- Read § 5 (code conventions) before writing any new analysis code; the listed watch-outs are recurring footguns in this kind of pipeline.
- When adding features, prefer expanding the existing pipeline structure (§ 4) over inventing new layers of abstraction. The pipeline is meant to be a research script that grows incrementally, not a framework.

## 9. Implementation milestones

Ordered milestones for working through the implementation after downloading `ds004194`. The **bolded** milestones (1, 3, 5) are the high-leverage ones — spend disproportionately more time there. Milestone 3 is a hard gate: if pipeline validation against Groen 2022 fails, do not proceed to the primary cohort.

### Milestone 0 — Data download and inventory (1-2 hours)

Verify what's actually on disk before any analysis.

- Download `ds004194` v3.0.0 (consider `datalad` or the OpenNeuro CLI; ~57 GB).
- Walk the directory tree and confirm subjects (p01-p14) and derivatives folders (`ECoGBroadband`, `ECoGCAR`, `ECoGPreprocessed`, `freesurfer`, `Groen2022TemporalDynamicsECoG`, `Brands2024TemporalAdaptationECoG`).
- Confirm actual BIDS task labels in events/ieeg files (the scaffold guesses these).
- For target subjects (p02, p06, p07, p10, p13, p14), inventory runs per task and broadband-derivative file sizes.

**Deliverable:** notebook cell printing a coverage matrix (subjects × tasks → run counts).
**Exit criterion:** filesystem layout known, no surprises.

### **Milestone 1 — Electrode atlas-label exploration (half a day)**

Foundation for every downstream analysis. Get this right early.

- Load relevant patient `electrodes.tsv` and freesurfer atlas-match sidecars.
- Per electrode, extract: native T1 coords, Wang max-prob label, full Wang probability vector, Benson anatomical label.
- Sanity-check against published tables:
  - Groen 2022 Table 1 — verify "Visual no." and "Matching areas" for p02, p06, p07, p10, p11.
  - Brands 2024 Table 3 — verify electrode counts per visual group for p11-p14.
- Visualize electrode positions: 3D pial-surface plot per subject (Brands 2024 Figure 2C / Supp Figure 9 style), with electrodes colored by Wang max-label and sized by area-membership probability.

**Deliverable:** PNG per subject + CSV with `(subject, electrode, x, y, z, wang_label_max, wang_probs, benson_label, visual_group)`.
**Exit criterion:** electrode counts match published tables to within ±1.

### Milestone 2 — Single-subject signal exploration (half a day)

One subject per task. Use **p10** for spatiotemporal-pattern (HD grid, dense data) and **p14** for natural-image (densest V1-V3 in Brands cohort).

- Load one run of broadband data + matching events.tsv.
- Plot:
  - Continuous broadband trace for V1 electrodes with stimulus-onset markers overlaid.
  - PSD per electrode for stimulus vs. blank periods (expect broadband elevation 50-200 Hz + alpha suppression).
  - Trial-averaged broadband response by stimulus condition for one electrode (replicate Groen 2022 Figure 2C-style).
- Verify [-0.1, 1.2] s epoching catches full response with margin.
- Check trial-rejection rates with Groen 3-SD criterion (expect ~2-3%).

**Deliverable:** notebook with 6-8 figures showing basic signal looks right.
**Exit criterion:** single-electrode V1 broadband time course visually matches canonical shape (transient ~100 ms, sustained plateau, return to baseline). If not, debug epoching/baseline before moving on.

### **Milestone 3 — Reproduce Groen 2022 Figure 4 (1 day) — HARD GATE**

Smallest meaningful pipeline-validation test against a known answer.

- For p02, p06, p07, p10: extract trial responses for contrast-varying conditions.
- Pool electrodes across subjects via probabilistic Wang assignment (bootstrap n=1000).
- Compute summed broadband 0.05-1 s per contrast level.
- Fit Naka-Rushton; extract C50.
- Plot against published figure.

**Deliverable:** your version of Groen 2022 Figure 4B (left panel — summed broadband vs. contrast).
**Exit criterion:** data points fall within/near published 68% CIs. **If this fails, do not proceed. Debug atlas-probability bootstrapping, electrode reliability filtering (split-half R² > 0.22), epoch selection, and event-condition extraction here on a known answer.**

### Milestone 4 — Stimulus loading and CNN feature extraction ✅ COMPLETE

Three notebooks implement this milestone:

- **`milestone_04_cnn_features.ipynb`** — PCA plots, Fisher discriminant ratio, and separability visualizations for all 7 archs.
- **`milestone_04b_rerun_extraction.ipynb`** — idempotent extraction runner; rebuilds `stimulus_bank.pkl` and all 7 `cnn_features_{arch}.pkl` caches. Run this if caches are stale.
- **`milestone_04c_separability_analysis.ipynb`** — three deeper analyses: (1) RSA between ECoG RDMs and CNN layer RDMs across 44 sliding 25ms windows [−100, +1000ms]; (2) Isomap geodesic Fisher ratio vs Euclidean Fisher ratio per arch/layer; (3) permutation tests for scrambled vs natural gamma in early/mid/late windows.

**Key finding from Milestone 4c:** Feedforward architectures (ResNet50, AlexNet, VGG16) show Fisher separability peaking at layer depth 1 then collapsing. CORnet-S maintains separability across its recurrent timesteps. The RSA heatmap shows when ECoG representational geometry aligns with CNN geometry — use this to select the response window for Milestone 5.

**Key bug fixed:** `stimulus.cat` in the `.mat` files is a **per-image** array, not per-trial. Index it as `cat[ti-1]` where `ti` is the trial's `trialindex`. The incorrect `zip(trialindex, cat)` pattern assigns the wrong categories.

**Exit criterion:** all 7 `cnn_features_{arch}.pkl` files present, each with 288 images and correct layer count. Cache loads in <5 seconds.

### **Milestone 5 — End-to-end encoding model on one electrode (half a day)**

Highest-value milestone. Most pipeline bugs surface here on a tractable scale.

**Context from Milestone 4c:** The RSA heatmap showed when ECoG representational geometry aligns with CNN geometry. Use that result to inform the response window for the scalar broadband summary. Default: mean broadband over [0.05, 0.55] s (`cfg.response_window`); adjust based on the RSA peak window if results are weak.

**Steps:**
- Load p13 or p14 epoch cache from `results/cache/milestone_04c_epochs_{sub}.pkl` (built by milestone_04c cell 5) to avoid re-loading broadband. Shape is `(288, n_vis_ch, 665)`.
- Select one visual electrode (prefer a V1-V3 electrode from the Wang atlas CSV).
- Collapse the time axis to a scalar per image: `mean(epochs[img_id, ch, t_mask])` over the response window → `y` of shape `(288,)`.
- Match each image to its CNN feature vector from a loaded `FeatureCache` (e.g. ResNet50).
- Fit `RidgeCV` with 12-fold leave-stimulus-out CV (see `aim_4_1_encoding_pipeline.fit_encoding_model`).
- Compute pooled cv-R² across folds.
- Plot predicted vs. actual response and per-layer cv-R² bar chart.

**Multi-arch extension (do after the single-electrode sanity check passes):** loop over all 7 architectures and all layers; build a (n_layers_total × 1) cv-R² profile for one electrode. This is a cheap precursor to the full cohort analysis.

**Deliverable:** per-layer cv-R² for one electrode across all 7 archs + scatter plot for best layer.
**Exit criterion:** cv-R² > 0 for at least one CNN layer. If negative everywhere, debug stimulus-trial alignment or response extraction here, where the full computation runs in seconds.

### Milestone 6 — Full primary cohort analysis (1 day)

Scale up to all electrodes.

- Run encoding pipeline on all visually responsive electrodes in p13 + p14 across all 6 ResNet50 layers.
- Layer-of-best-fit per electrode.
- Group analysis: V1-V3 / VOTC / LOTC layer-of-best-fit distributions.
- Statistical tests for hierarchy gradient (Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni).

**Deliverable:** figure showing layer-of-best-fit gradient across three areas + per-electrode results CSV.
**Exit criterion:** defensible answer to the central Aim 1 question — does CNN layer depth align with cortical hierarchy in this cohort? — even if "weakly" or "not significantly."

### Milestone 7 — Robustness checks (half a day)

Architecture breadth was substantially addressed in Milestone 4 (7 archs extracted and compared). Remaining robustness checks:

- Repeat hierarchy gradient with all 7 architectures (Milestone 4c already showed which layers have category structure; now test predictive power per area).
- Test alternative response-window summaries (AUC instead of mean; peak instead of mean) to rule out window artifacts. Use the RSA peak window from Milestone 4c as the biologically-motivated alternative.
- Confirm V1-V3 → VOTC → LOTC gradient direction is consistent across at least 3 of 7 architectures, and specifically across the biologically-motivated (CORnet-S) vs engineered (ResNet50, VGG16) archs.

**Deliverable:** robustness table showing hierarchy direction and peak cv-R² across all 7 architectures.
**Exit criterion:** consistent gradient across architectures, OR honest negative result that gets written up as such.

### Milestone 8 — Figure generation and report writing (1-2 days)

Convert results into report-ready figures and tables. Pairs with the Methods section already drafted (`methods_section.docx`); now write the matching Results section.
