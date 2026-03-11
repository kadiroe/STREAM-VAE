# STREAM-VAE

STREAM-VAE: Dual-Path Routing for Slow and Fast Dynamics in Vehicle Telemetry Anomaly Detection by Mercedes-Benz AG

This is a clean, minimal repo you can run **immediately** to test the model and reproduce numbers.

**Authors:** Kadir-Kaan Özer, René Ebeling, Markus Enzweiler 

Paper: https://arxiv.org/abs/2511.15339
HuggingFace: https://huggingface.co/papers/2511.15339

## Folder layout

```
datasets/
  TSB-AD-M/        # put the official TSB-AD-M CSVs here (not included)
  custom/
    example_sine_spike.csv   # included toy dataset (Label column)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Quick sanity run (included toy dataset)

```bash
python scripts/run_custom.py
```

This trains on the whole series (with an internal validation split) and prints:
- anomaly scores shape + min/max
- AUC-ROC and AUC-PR (since the toy CSV includes `Label`)

## Run on TSB-AD-M (all CSV files in the folder)

1) Download TSB-AD-M from the official sources and unzip so you get many CSVs with a `Label` column.
2) Place them in: `datasets/TSB-AD-M/`
3) Run:

```bash
python scripts/run_tsb_ad_m.py --data_root datasets/TSB-AD-M --out_csv results.csv
```

### Reproducibility / seeding

Both scripts support:

- `--seed` (default **2024**)
- `--deterministic` (best-effort deterministic PyTorch mode)

Examples:

```bash
python scripts/run_custom.py --seed 2024 --deterministic
python scripts/run_tsb_ad_m.py --seed 2024 --deterministic
```

> Note: perfect determinism is not always possible across devices (CUDA/MPS) and some kernels (e.g., SDPA) may still have nondeterminism. The code enables the standard PyTorch best-effort flags.

## Metrics

If you install the **TSB-AD** pip package, the folder runner will automatically use
`TSB_AD.evaluation.metrics.get_metrics(...)` to compute the full metric suite.
Otherwise it falls back to sklearn AUC-ROC / AUC-PR.

Enable full metrics:

```bash
pip install TSB-AD
```

---


## Summary Comparison on TSB-AD Multivariate Benchmark

**Summary comparison of STREAM-VAE with representative competitors across 180 time series from 17 datasets**, evaluated using point-wise, range-based, event-based, and affiliation metrics.  
Higher is better for all metrics. We used TSB-AD-M Eval.

| Method | VUS-PR | VUS-ROC | AUC-PR | AUC-ROC | Standard-F1 | PA-F1 | Event-based-F1 | R-based-F1 | Affiliation-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **STREAM-VAE (ours)** | **0.435** | **0.777** | **0.383** | **0.757** | **0.452** | **0.535** | **0.470** | **0.415** | **0.829** |
| xLSTMAD-F (MSE) | 0.35 | 0.77 | 0.35 | 0.74 | 0.40 | 0.85 | 0.70 | 0.42 | 0.89 |
| CNN | 0.31 | 0.76 | 0.32 | 0.73 | 0.37 | 0.78 | 0.65 | 0.37 | 0.87 |
| OmniAnomaly | 0.31 | 0.69 | 0.27 | 0.65 | 0.32 | 0.55 | 0.41 | 0.37 | 0.81 |
| PCA | 0.31 | 0.74 | 0.31 | 0.70 | 0.37 | 0.79 | 0.59 | 0.29 | 0.85 |
| LSTMAD | 0.31 | 0.74 | 0.31 | 0.70 | 0.36 | 0.79 | 0.64 | 0.38 | 0.87 |
| USAD | 0.30 | 0.68 | 0.26 | 0.64 | 0.31 | 0.53 | 0.40 | 0.37 | 0.80 |
| AutoEncoder | 0.30 | 0.69 | 0.30 | 0.67 | 0.34 | 0.60 | 0.44 | 0.28 | 0.80 |
| KMeansAD | 0.29 | 0.73 | 0.25 | 0.69 | 0.31 | 0.68 | 0.49 | 0.33 | 0.82 |
| CBLOF | 0.27 | 0.70 | 0.28 | 0.67 | 0.32 | 0.65 | 0.45 | 0.31 | 0.81 |
| MCD | 0.27 | 0.69 | 0.27 | 0.65 | 0.33 | 0.46 | 0.33 | 0.20 | 0.76 |
| OCSVM | 0.26 | 0.67 | 0.23 | 0.61 | 0.28 | 0.48 | 0.41 | 0.30 | 0.80 |
| Donut | 0.26 | 0.71 | 0.20 | 0.64 | 0.28 | 0.52 | 0.36 | 0.21 | 0.81 |
| RobustPCA | 0.24 | 0.61 | 0.24 | 0.58 | 0.29 | 0.60 | 0.42 | 0.33 | 0.81 |
| FITS | 0.21 | 0.66 | 0.15 | 0.58 | 0.22 | 0.72 | 0.32 | 0.16 | 0.81 |
| EIF | 0.21 | 0.71 | 0.19 | 0.67 | 0.26 | 0.74 | 0.44 | 0.26 | 0.81 |
| COPOD | 0.20 | 0.69 | 0.20 | 0.65 | 0.27 | 0.72 | 0.41 | 0.24 | 0.80 |
| IForest | 0.20 | 0.69 | 0.19 | 0.66 | 0.26 | 0.68 | 0.41 | 0.24 | 0.80 |
| HBOS | 0.19 | 0.67 | 0.16 | 0.63 | 0.24 | 0.67 | 0.40 | 0.24 | 0.80 |
| TimesNet | 0.19 | 0.64 | 0.13 | 0.56 | 0.20 | 0.68 | 0.32 | 0.17 | 0.82 |
| KNN | 0.18 | 0.59 | 0.14 | 0.51 | 0.19 | 0.69 | 0.45 | 0.21 | 0.79 |
| TranAD | 0.18 | 0.65 | 0.14 | 0.59 | 0.21 | 0.68 | 0.40 | 0.21 | 0.79 |
| LOF | 0.14 | 0.60 | 0.10 | 0.53 | 0.15 | 0.57 | 0.32 | 0.14 | 0.76 |
| AnomalyTransformer | 0.12 | 0.57 | 0.07 | 0.52 | 0.12 | 0.53 | 0.33 | 0.14 | — |


## Acknowledgement
We thank the authors of TSB-AD for releasing the multivariate time-series anomaly detection benchmark and evaluation framework, which we use for fair and reproducible comparison. Their repo can be found at https://github.com/TheDatumOrg/TSB-AD/tree/main/TSB_AD and their paper at https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html
