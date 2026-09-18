# TCRT

**Text-Certified Residual Transport for Reliable Vision–Language Unsupervised Domain Adaptation**
<img width="2144" height="936" alt="TCRT" src="https://github.com/user-attachments/assets/d8b52494-0fc2-461d-b76d-4cfd968e2971" />

TCRT is a research code release for reliability-aware vision–language UDA on a frozen CLIP ViT-B/16 backbone. It audits learned domain residuals: features are decomposed into a text-anchored semantic component and a low-rank domain residual; same-class, energy-matched residuals are transported across domains; and the change in the **frozen text posterior** under transport produces a reliability certificate that controls pseudo-label supervision, semantic-anchor updates, and a certificate-filtered bipartite graph spectral regularizer.  The current repository is a clean open-source release: it contains the full method implementation, dataset loaders, experiment configs, diagnostics, and reproduction scripts. It doesn't bundle CLIP model weights, benchmark images, training outputs, or logs — those are external assets (same policy as the reference release this layout follows).

## What Is Included

- Entry point: `main.py` (`benchmark`, `run`, `diagnose`, `count_params`, `extract`)
- Method modules: `src/model.py`, `src/factorization.py`, `src/certificate.py`,
  `src/graph.py`, `src/losses.py`, `src/trainer.py`
- Frozen-CLIP backend with trainable prompt context: `src/clip_backend.py`
- Dataset loaders + feature caching: `data.py`
- Post-hoc mechanism diagnostics: `src/diagnostics.py`
- Experiment configs: `configs/tcrt/*.yaml` (Office-31, Office-Home,
  VisDA-2017, DomainNet transductive + inductive)
- Data preparation helper, launch scripts, result aggregation:
  `tools/prepare_data.py`, `tools/run_all.sh`, `tools/analyze_results.py`
- Reproduction notes: `docs/REPRODUCE.md`
- CPU smoke test: `tests/test_smoke.py`

## What Is Not Included

- CLIP model weights (downloaded by the transformers backend on first use)
- Benchmark images (see `tools/prepare_data.py` for sources)
- Training outputs, checkpoints, logs, caches

## Repository Layout

```text
TCRT/
├── configs/tcrt/             # YAML experiment configs
├── data.py                   # dataset loaders + frozen-feature cache
├── docs/REPRODUCE.md         # how to reproduce the paper experiments
├── main.py                   # CLI entry point
├── src/
│   ├── clip_backend.py       # frozen CLIP + trainable prompt context (and mock backend)
│   ├── model.py              # factorization module, discriminators, anchor bank
│   ├── factorization.py      # batch-local alternating solver (A, E)
│   ├── certificate.py        # donor queues, transport, CVaR certificate
│   ├── graph.py              # certified bipartite graph spectral regularization
│   ├── losses.py             # hierarchical objective terms
│   ├── trainer.py            # warm-up then certificate-guided adaptation
│   ├── eval.py               # benchmark driver (transductive / inductive)
│   ├── diagnostics.py        # ranking / coverage-risk / graph / probes / ...
│   └── utils.py              # seeds, metrics, paired intervals
└── tools/
    ├── prepare_data.py       # dataset layout + DomainNet lists
    ├── run_all.sh            # run all main benchmarks
    └── analyze_results.py    # aggregate results into tables
```

## Method Components In This Code

`TCRTTrainer` in `src/trainer.py` implements Algorithm 1: a warm-up stage
(source supervision + prompt alignment + factorization + functional
disentanglement) followed by certificate-guided adaptation.

- `TextAnchoredFactorization` (`src/model.py`): the semantic basis
  `V = qf(Vbar)` (Stiefel retraction), private residual bases `B_s/B_t`, and
  the text-subspace alignment to `V_T`.
- `solve_decomposition` (`src/factorization.py`): the batch-local alternating
  solve of the coefficient matrices `A_d` and the row-sparse outlier matrices
  `E_d` (proximal update), detached before the global backward pass.
- `ResidualQueue` + `compute_certificate_batch` (`src/certificate.py`): FIFO
  donor queues, the valid-donor set (same class, energy-matched, target-donor
  margin), CVaR tail transport discrepancy `u_i`, and the certificate
  `c_i = exp(-u_i/eta) * sigmoid((m_i - kappa)/zeta)`.
- `AnchorBank` + `build_certified_graph` + `spectral_tail_loss`
  (`src/model.py`, `src/graph.py`): class-wise anchors with k-means init and
  EMA updates, the masked sample–anchor affinity, and `tr(P^T L_c^p P)/n`
  evaluated by sparse bipartite multiplications (no eigendecomposition).
- `losses.py`: every term of the hierarchical objective
  (`L_clip + λ_fac L_fac + λ_cert L_cert + λ_priv L_priv + λ_tail L_tail +
  λ_sup L_sup`), including the gradient-reversal layer for `L_leak`.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Data (see tools/prepare_data.py for layout and download sources)
python tools/prepare_data.py --validate

# 2. Pre-extract frozen CLIP features (requires the transformers weights)
python main.py extract --config configs/tcrt/office_home.yaml

# 3. Run one transfer
python main.py run --config configs/tcrt/office_home.yaml --src Art --tgt Clipart

# 4. Run a full benchmark (all transfers x five seeds)
python main.py benchmark --config configs/tcrt/office_home.yaml

# 5. Everything (five main benchmarks)
./tools/run_all.sh

# 6. Mechanism analyses on saved checkpoints (no retraining)
python main.py diagnose --config configs/tcrt/diagnostics.yaml \
    --kind ranking,coverage_risk,stage,graph,probe,additivity,transport,properties
```

## License

This repository is released under the Apache-2.0 license. The CLIP model
weights and benchmark datasets retain their own licenses.
# TRCT

