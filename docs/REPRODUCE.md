# Reproduce the TCRT Experiments

This document explains how the repository maps to the paper experiments and
what needs to be prepared on a new machine.

## 1. Required Assets

- **CLIP ViT-B/16 weights** — downloaded automatically by the `transformers`
  backend on first use (`openai/clip-vit-base-patch16`).
- **Benchmark images** — Office-31, Office-Home, VisDA-2017 and DomainNet
  (twelve C/P/R/S transfers). See `tools/prepare_data.py` for the expected
  layout and the official download sources.
- **A GPU with ≥ 24 GB memory** for the full experiments (the paper reports
  A100 40GB; TCRT peaks at ~15.1 GB on DomainNet C→R under FP16).
- For the inductive DomainNet split, the official train/test lists
  (`clipart_train.txt`, ... `sketch_test.txt`) are required:
  `python tools/prepare_data.py --domainnet-lists`.

## 2. Data-Access Protocol (must not be violated)

The paper's controlled protocol:

- **Transductive**: the unlabeled target set used for evaluation is available
  during adaptation; its labels are hidden until all training, model-selection,
  and threshold-selection decisions are frozen.
- **Inductive (DomainNet-I)**: adaptation uses the official target-train split
  only. Target-test images and labels are inaccessible to the optimizer,
  queues, anchors, donor selection, normalization statistics, stopping
  decisions, and hyperparameter selection. After adaptation all global
  parameters are frozen and target-test images are evaluated once.

The code implements this split by construction: the trainer only ever sees
`z_t` (the adaptation split); `z_t_test` is passed only to the evaluation
function. **Do not** change the trainer to read target-test features.

## 3. Config Anatomy

Each experiment config follows the same structure (example:
`configs/tcrt/office_home.yaml`):

```yaml
experiment_name: office_home
benchmark: office_home
data_root: ./data
output_dir: ./outputs/office_home
transfers: [[Art, Clipart], ...]      # directed transfers
seeds: [0, 1, 2, 3, 4]                 # five matched seeds
num_classes: 65
backend: transformers
clip_model: openai/clip-vit-base-patch16
n_ctx: 16                              # prompt context length
semantic_rank: 64                      # r0
residual_rank: 16                      # m
rho: 0.1                               # CVaR tail fraction
pi_min: 0.4  pi_max: 0.8               # annealed class-wise retained ratio
lr: 0.003  weight_decay: 0.0005
batch_size: 32  iterations_per_epoch: 1000
warmup_epochs: 10  adapt_epochs: 30    # per-benchmark adaptation length
```

Key facts:

- One epoch is defined as `iterations_per_epoch` (1000) training iterations —
  the same definition used by the paper's efficiency table. Warm-up runs for
  10 epochs; adaptation runs for 20 (Office-31), 30 (Office-Home), 40
  (VisDA-2017 and DomainNet).
- All hyperparameters live in the config and are selected with source-only
  validation; target labels never enter hyperparameter selection.
- The energy-bin boundaries are fit once from a warm-up reservoir so that
  queue entries and current samples share the same map `b(.)`.

## 4. Reproduce the Main Tables

| Paper table | Command |
|---|---|
| Table: controlled results | `python main.py benchmark --config configs/tcrt/office31.yaml` (etc. for office_home / visda / domainnet) |
| Table: inductive DomainNet | `python main.py benchmark --config configs/tcrt/domainnet_inductive.yaml` |
| Table: system ablation | vary `use_disentangle`, selection filter, and graph flags; see §5 |
| Table: certificate ablation | see §5 |
| Table: hyperparameter sensitivity | one-at-a-time `--set semantic_rank=32` etc. on Ar→Cl |
| Table: ranking | `diagnose --kind ranking` |
| Table: training-stage diagnostic | `diagnose --kind stage` (requires `model_warmup.pt`; see §6) |
| Fig: coverage–risk | `diagnose --kind coverage_risk` |
| Table: graph quality | `diagnose --kind graph` |
| Table: factorization probe | `diagnose --kind probe` |
| Table: additivity | `diagnose --kind additivity` |
| Tables: transport plausibility | `diagnose --kind transport` (Ar→Cl, then `--set transfers=[[Clipart,Real_World]]` for C→R) |
| Table: pseudo-label corruption | `--set corrupt_labels=0.1` ... (5 runs) |
| Table: graph-edge corruption | `--set corrupt_edges=0.1` ... (5 runs) |
| Table: efficiency | `count_params` + `tools/measure_efficiency.py` (A100) |
| Appendix: construction checks | `diagnose --kind properties` |

## 5. Ablations and Corruption

**System ablation (Table system_ablation).** The variants are selected by flags:

- `use_disentangle=false` disables F (the shared-private factorization);
  without F the classifier falls back to the frozen text posterior and there
  are no residual-based selection terms. (*Implemented as a separate config
  variant; see configs/tcrt/ablation/.*)
- `selection=random|confidence|certificate|none` selects the filter.
- `use_graph=false` removes the spectral term.

**Certificate ablation (Table certificate_ablation).** Each component has a
flag: `cvarr_to_mean`, `no_margin_factor`, `no_same_class`, `no_energy_match`,
`no_stopgrad`, `no_opposite_domain`.

**Corruption.** `corrupt_labels` (fraction of admitted pseudo-labels replaced
by a uniformly sampled wrong class, paired mask across methods/seeds) and
`corrupt_edges` (fraction of candidate graph edges rewired preserving degree
and endpoint type, no labels) are applied as wrappers after warm-up. Keep the
same `corrupt_seed` across methods so masks are paired.

## 6. Training-Stage Diagnostic (Table certificate_diag)

This diagnostic needs a warm-up checkpoint in addition to the final one. Run
training with `--set save_every=...` so that the checkpoint written at the end
of warm-up is saved as `model_warmup.pt` (the trainer saves a warm-up
checkpoint when `save_every` divides `warmup_epochs`; see `src/trainer.py`),
then run `diagnose --kind stage`.

## 7. Notes on Numerical Fidelity

- The vision tower is frozen and image features are cached once (`extract`).
  The text prototypes `T` are recomputed every iteration because the prompt
  context vectors are trainable.
- The semantic basis uses a differentiable QR retraction
  (`V = qf(Vbar)`); `||VV^T - V_T V_T^T||_F^2` is checked by
  `diagnose --kind properties`.
- The bipartite graph is never materialized: `L_tail` uses one or two sparse
  graph multiplications (`p ∈ {1,2}`); the appendix spectral inequality is
  verified on a small graph by explicit diagonalization.
- Paired 95% intervals use the five unrounded seedwise benchmark differences
  and Student's t with four degrees of freedom (`src/utils.py`).

## 8. Known Implementation Choices (documented deviations)

- The `max-softmax probability` score in the ranking/coverage-risk tables is
  computed from the **frozen text posterior** `q_T(z)` (same reference as the
  certificate and text margin). This is the score most comparable across
  methods under the controlled protocol.
- `AnchorBank.affinity` normalizes both semantic coordinates and anchors
  before the `tau_g`-scaled dot product (cosine affinity).
- The graph-edge corruption wrapper rewires a fraction of sample–anchor pairs
  to a random different anchor of the same predicted class, preserving degree
  and endpoint type without using labels.
- The factorization reconstruction loss uses the mean over entries (rather
  than the raw sum) for numerical stability; the `alpha/beta/xi/gamma` weights
  absorb this constant.

These choices are configurable and are recorded here so that the released
code is auditable rather than silently different from the paper.
