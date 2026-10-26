# Learning Body State from Pressure Sensor Data

A small research-style project on pressure-mattress sensor data: denoise raw
readings, learn a representation with limited labels, and act on the result
with a contextual bandit. Built end-to-end on **synthetic** data (see
[Data](#step-1--data) for why), with the same rigor the real PMAP dataset
would demand: documented noise model, baselines, ablations, and an honest
failure analysis.

```
src/data.py            synthetic pressure-map generator + sensor noise model
src/models.py           shared conv backbone / decoder / projection head / CNN baseline
src/augment.py           label-preserving augmentations for contrastive pretraining
scripts/01_generate_data.py     cache all dataset splits (fixed seeds, no leakage)
scripts/02_train_denoise.py     Phase 1: denoising autoencoder
scripts/03_train_pretrain.py    Phase 2a: contrastive (SimCLR-style) pretraining
scripts/04_fewshot_eval.py      Phase 2b: few-shot posture classification, baselines, ablations
scripts/05_failure_analysis.py  Step 3: subgroup accuracy + confusion matrix
scripts/06_bandit.py            Phase 3 (stretch): contextual bandit zone-adjustment sim
scripts/07_build_report.py      renders reports/final_report.pdf from all results/figures
reports/figures/, reports/results/   all generated plots and metrics (checked in)
checkpoints/                    trained weights (checked in, ~400KB total)
```

Reproduce everything: `pip install -r requirements.txt && bash run_all.sh`
(~1 minute on CPU, no GPU needed).

## Motivation

A pressure-sensing mattress reports a 2D grid of raw taxel readings. Three
things make this a genuinely interesting ML problem rather than a plain
classification task:

1. **The raw signal is noisy.** Resistive sensor arrays drop cells, spike
   under cross-talk, and quantize coarsely. Before anything else can be
   asked of the data, it helps to know how to clean it and how much that
   cleaning is actually worth.
2. **Labels are expensive.** Nobody annotates "supine / prone / left side /
   right side" at scale for every mattress install; a real deployment might
   get 5-10 labeled examples per posture, if that. This is exactly the
   regime where self-supervised pretraining is supposed to pay off, and the
   project is set up to test whether it actually does, not just assume it.
3. **Sensing should drive action.** The end goal of a smart mattress isn't a
   label, it's a decision ("which zone do I relieve right now"), made under
   a comfort function that is never directly observable — a contextual
   bandit problem.

The three phases below map directly onto these three points.

## Step 1 — Data

**Choice: synthetic generator** (`src/data.py`). The real PMAP dataset is
human-subjects data and is request-gated; for a learning project that needs
fast iteration, a programmatic generator that preserves the structure of
the real problem is the more honest choice than stalling on data access —
and it makes every noise assumption explicit and auditable, which a
downloaded dataset would not.

**What it generates.** A 48×24 taxel grid. A body is a sum of anisotropic
Gaussian "blobs" (head, shoulders, chest, pelvis, thighs, calves) whose
position/size/intensity encode:
- **posture** (the label): `supine`, `prone`, `left_lateral`, `right_lateral`.
  Lateral postures are shifted and narrower (side-lying footprint); supine
  vs. prone is the harder pair — they differ only in the *relative*
  intensity of the chest vs. pelvis blob, not gross position.
- **body size** (nuisance, not given to the model): small / medium / large,
  plus continuous per-sample scale jitter.
- **a global placement offset** (nuisance): the body isn't perfectly
  centered on the mattress, sampled independently of posture. This turned
  out to matter a lot — see the note in [Results](#honest-surprise-raw-pixels-are-a-strong-baseline).

**The noise model** (`add_sensor_noise`, documented in code and in
`reports/results/denoise_results.json`), applied at one of three severities
(low/medium/high, mixed uniformly over the dataset):

```
noisy = quantize( clip( clean
                        + Gaussian(sigma)              # electrical noise
                        + dead-cell dropout(p_drop)     # disconnected taxel -> 0
                        + salt spikes(p_salt)           # cross-talk / spikes -> near-max
                        , 0, 1), levels )
```

sigma, p_drop, p_salt, and quantization level count all scale together
across the three severities (sigma 0.02→0.09, p_drop 1%→6%, p_salt
0.2%→1.2%, levels 64→16).

Five fixed-seed, non-overlapping splits are generated once and cached
(`scripts/01_generate_data.py`): `denoise_train/val`, `pretrain_unlabeled`,
`fewshot_pool` (to draw k-shot label sets from), `test_eval` (held out for
every evaluation in the project, including failure analysis).

![sanity check](reports/figures/00_sanity_check.png)

## Step 2 — Three phases

### Phase 1: Denoising

A small conv encoder-decoder (`Backbone` + `Decoder`, 3 stride-2 conv
blocks each way, 64-channel bottleneck) trained with MSE to map noisy →
clean. 40 epochs, 2000 train / 500 val pairs, ~14s on CPU.

| | MSE | PSNR |
|---|---|---|
| identity baseline (do nothing) | 0.00769 | 21.1 dB |
| **denoising autoencoder** | **0.00063** | **32.0 dB** |

**91.8% reduction in reconstruction MSE** vs. doing nothing. See
`reports/figures/01_denoise_qualitative.png` for side-by-side noisy /
denoised / clean grids, and `01_denoise_training_curve.png` for the
learning curve against the identity-baseline floor.

### Phase 2: Representation learning + few-shot classification

**Pretraining** (`scripts/03_train_pretrain.py`): the same `Backbone`
architecture is pretrained two different ways on *unlabeled* data, so the
only thing being compared downstream is the pretraining signal, not the
architecture:
- **contrastive**: SimCLR-style NT-Xent loss (temperature 0.3) over two
  augmented views per sample. Augmentations are noise injection, small
  translation, random dropout, and intensity scaling — deliberately
  **no horizontal flip**, because flipping would swap `left_lateral` and
  `right_lateral` and silently corrupt the label the representation is
  supposed to respect.
- **denoising**: the Phase 1 autoencoder's backbone, reused directly.

**Few-shot evaluation** (`scripts/04_fewshot_eval.py`): for k ∈ {5, 10, 20}
labeled examples per class (4 classes), averaged over 8 random label draws,
six methods are compared on the same held-out `test_eval` set:

| method | k=5 | k=10 | k=20 |
|---|---|---|---|
| Linear probe, **random-init** encoder (floor) | 44.9% | 46.4% | 47.4% |
| CNN trained from scratch (full backbone, k-shot only) | 52.4% | 54.6% | 60.9% |
| Linear probe, **denoising**-pretrained encoder | 61.8% | 65.4% | 69.2% |
| LogReg on raw noisy pixels | 65.7% | 67.8% | 72.1% |
| LogReg on **denoised** pixels | 66.6% | 69.0% | 73.5% |
| Linear probe, **contrastive**-pretrained encoder (ours) | 64.0% | **70.9%** | 72.7% |

(chance = 25%; full numbers with std in `reports/results/fewshot_summary.json`)

![few-shot accuracy](reports/figures/03_fewshot_accuracy.png)

## Step 3 — Evaluate like a researcher

**Baselines.** Logistic regression on flattened pixels, and a CNN with the
*same* backbone trained end-to-end from random init on only the k-shot
labels, are both implemented and plotted above.

**Ablations, answered directly from the table:**
- *Does pretraining help, and by how much?* Yes, decisively, against the
  only fair control (same architecture, random weights): random-init linear
  probe sits at 45-47% regardless of k, while either pretraining method
  lifts it to 62-73% — a **+17 to +26 point** jump. Pretraining is clearly
  doing something; a frozen random CNN alone is close to useless here.
- *Does denoising help downstream accuracy?* Yes, modestly and
  consistently: LogReg on denoised pixels beats LogReg on raw noisy pixels
  at every k (+0.9, +1.2, +1.4 points). Small effect size, but it never
  reverses sign across 8×3=24 trials, so it's a real if modest gain, not
  noise.
- *Does the CNN-from-scratch baseline justify its cost?* No — it is the
  worst non-floor method at every k, confirming that training a full
  conv net end-to-end on 20-80 total labeled examples overfits badly
  compared to just using those labels to fit a linear head on good frozen
  features.

#### Honest surprise: raw pixels are a strong baseline

The first version of this project used perfectly-centered bodies, and
raw-pixel logistic regression hit **99%+ accuracy with only 10 labels per
class** — beating every pretrained representation by 20+ points. That is
not a success story, it's a sign the synthetic task was too easy: with no
nuisance variation, posture differences are a handful of cleanly
linearly-separable pixels, and 1152-dimensional logistic regression with
5-20 samples per class will happily overfit a perfect linear boundary in
that regime (classic n≪d). The fix, documented in `src/data.py` and
committed in the history of this repo, was to add a **global random body
placement offset per sample**, independent of posture — i.e. the subject
isn't perfectly centered on the mattress, which is true of real sensor
data. That single change dropped raw-pixel LogReg from 99%→~70% and let
the representation-learning story actually show up in the numbers above
(contrastive pretraining wins outright at k=10, is competitive elsewhere).
**Lesson:** a synthetic benchmark that makes your proposed method look
great by a huge margin is itself a result worth being suspicious of.

Even after that fix, raw-pixel LogReg (and LogReg on denoised pixels) stay
*competitive* with the pretrained linear probes rather than clearly losing
— on a 48×24, 4-class, low-resolution problem, a 1152-dim linear boundary
is a genuinely strong baseline. We report this rather than cherry-picking a
setting that flatters the pretrained encoder.

**Failure analysis** (`scripts/05_failure_analysis.py`, contrastive probe,
k=10, single representative label draw, 800-sample held-out test set,
overall accuracy 61.4%):

![subgroup accuracy](reports/figures/04_failure_analysis_subgroups.png)

- **Body size:** small bodies are the clear weak point (51.3% vs. 66.4% /
  66.4% for medium / large). Smaller blobs produce weaker, more
  noise-fragile pressure signatures relative to the fixed sensor-noise
  magnitude — the signal-to-noise ratio is literally worse for small
  bodies, and the model has not been given any mechanism to be invariant
  to that.
- **Sensor noise level:** accuracy drops sharply at the high-noise setting
  (48.1%) vs. low/medium (64.5% / 70.9%) — expected, and a reminder that
  "works on clean sensors" and "works in the field" are different claims.
- **Posture confusion is exactly where we predicted it would be:**

![confusion matrix](reports/figures/05_confusion_matrix.png)

  supine and prone are confused with each other constantly (both directions,
  ~45-47% of each class' errors), essentially **never** confused with either
  lateral posture, while left/right lateral are confused with each other far
  less and almost never with supine/prone. This matches the generative
  design exactly: lateral vs. non-lateral is a gross, noise-robust spatial
  signal (which half of the grid lights up), while supine vs. prone is a
  subtle chest/pelvis *intensity ratio* that high sensor noise and small
  body size both directly corrupt. A real deployment should expect the same
  pattern: coarse position is easy, fine-grained orientation on the same
  spot is hard.

## Phase 3 (stretch): contextual bandit for zone adjustment

`scripts/06_bandit.py`. The mattress is split into 5 zones (head,
shoulders, lower-back/hips, thighs, calves). At each round the agent
observes per-zone mean pressure as context and picks **one** zone to
relieve. A simulated user has a hidden, fixed per-zone comfort-sensitivity
vector θ (unknown to the agent): `reward(zone z) = θ_z · pressure_z +
noise` — a disjoint linear contextual bandit.

Policies: `random`, `epsilon-greedy` (per-arm ridge regression, ε=0.1),
`LinUCB` (disjoint, Li et al. 2010, α=1.0), and an `oracle` that knows θ
exactly (defines zero regret). T=3000 rounds.

| policy | final cumulative regret | mean reward (last 500 rounds) |
|---|---|---|
| random | 321.8 | 0.139 |
| epsilon-greedy | 80.0 | 0.222 |
| **LinUCB** | **32.5** | **0.239** |
| oracle (upper bound) | 0.0 | 0.244 |

![bandit regret](reports/figures/06_bandit_regret.png)

LinUCB's context-aware confidence bound closes the gap to oracle far faster
than epsilon-greedy's undirected exploration — by round 3000 LinUCB is
averaging 98% of oracle's reward, vs. 91% for epsilon-greedy and 57% for
random. Regret is visibly sublinear (concave) for both learning policies
and linear for random, exactly as the theory predicts.

## Limitations

- **Synthetic data, not real PMAP.** The generator is a reasonable stand-in
  for the spatial structure of the problem (and the near-perfect raw-pixel
  baseline it produced before adding placement jitter is a cautionary tale
  about synthetic data hiding real difficulty) but it cannot capture real
  inter-subject anatomical variation, real sensor nonlinearity/hysteresis,
  mattress foam compression dynamics, or correlated (non-i.i.d.) noise
  across neighboring taxels — all of which would make the real problem
  harder than this one.
- **Representation learning's advantage is real but modest, not dominant.**
  On this task, a plain logistic regression on (denoised) pixels is a
  strong baseline; contrastive pretraining's clearest win is at k=10, not
  uniformly across all k. A reviewer should not walk away thinking
  self-supervised pretraining is a silver bullet here — the honest
  claim is "it reliably beats random-init features by a wide margin, and
  is competitive-to-best against simple baselines," not "it dominates."
- **The bandit's comfort function is linear and stationary.** Real human
  comfort preferences likely drift over a night's sleep and are not
  perfectly linear in zone pressure. LinUCB is the right tool for the
  linear-stationary version of the problem; a real system would need to
  handle non-stationarity (e.g. sliding-window or discounted LinUCB).
- **Small grid, 4 posture classes.** 48×24 and four coarse postures is a
  deliberately small problem to keep CPU-only iteration fast; it will not
  by itself transfer conclusions about model capacity to higher-resolution
  real sensor arrays or finer-grained posture taxonomies.
- **No real-world sensor drift or calibration across sessions was modeled**
  beyond the per-sample noise draw — a real deployment sees a sensor array
  age and drift over weeks, which is a different, slower-timescale problem
  than the per-reading noise simulated here.

## Future work

- Validate against a real (even small, consented) pressure-mat dataset to
  check whether the synthetic-data conclusions — pretraining helps
  meaningfully but doesn't dominate strong baselines, supine/prone is the
  hard pair — actually transfer.
- Try a denoising **+** contrastive joint objective (denoise as the
  pretraining task *and* a contrastive loss on the clean/denoised
  reconstruction) to see if they're complementary rather than alternatives.
- Extend the bandit's reward model to be non-stationary (user comfort
  drifting over a session) and to richer action spaces (continuous
  adjustment amount, not just zone choice).
- Scale the grid resolution and posture taxonomy once a real dataset is in
  hand, and re-run the same baseline/ablation/failure-analysis protocol
  used here unchanged — it was built to be dataset-agnostic.
