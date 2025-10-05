# MCMC_Regression — NumPyro Bayesian Linear Regression for Parkinson Telemonitoring

This project fits a **Bayesian Linear Regression** with **NumPyro + NUTS** to predict motor UPDRS
from voice features (Parkinson’s Telemonitoring dataset). It outputs all plots/tables your paper needs:

**Plots**

1. Trace plots (diagnostics for α, β, σ)
2. Forest plot of coefficients with 95% CrI
3. Predicted vs True UPDRS with 95% Prediction Intervals
4. Posterior Predictive Check (observed vs. simulated)

**Tables**

1. Posterior summary (mean, sd, CrI, R-hat) → `outputs/summary.csv`
2. Performance metrics (MAE, RMSE, Pearson r) on train/test → `outputs/metrics.csv`

---

## Project layout

```
MCMC_Regression/
├─ README.md
├─ requirements.txt
├─ outputs/
│  ├─ figures/                           # auto-created (trace, forest, pred_vs_true, ppc)
│  └─ idata/                             # posterior.nc
├─ src/
│  ├─ __init__.py                        # (optional)
│  ├─ config.py                          # ← edit this file to tweak settings
│  ├─ data_loading.py
│  ├─ model_numpyro.py
│  └─ plots.py
└─ scripts/
   ├─ run_train.py
   └─ run_eval.py
```

---

## Quickstart

1) Create a Python environment and install requirements:

```bash
pip install -r requirements.txt
```

2) Decide where to load data from by editing **`src/config.py`**:

- **Use UCI directly** (no CSV needed): set `use_uciml=True` (default).  
  We fetch UCI dataset **id=189** through `ucimlrepo`:
  ```bash
  # src/config.py
  TrainConfig.use_uciml = True
  EvalConfig.use_uciml = True
  ```

- **Use a local CSV**: set `use_uciml=False` and place a file at `data/telemonitoring_parkinsons.csv`  
  If `use_uciml=True`, the dataset is fetched automatically using the `ucimlrepo` package.  
  Default target: `motor_UPDRS`.

3) Run training and evaluation (no CLI flags; everything comes from `config.py`):

```bash
python scripts/run_train.py
python scripts/run_eval.py
```

> **No CSV?** If `use_uciml=False` and no file is found, the pipeline automatically falls back to a **synthetic data
generator** that mimics
> the real columns and correlations so you still get all figures/tables for the paper.

---

## Configuration (edit `src/config.py`)

```python
from dataclasses import dataclass


@dataclass
class TrainConfig:
    use_uciml: bool = True  # True = UCI via ucimlrepo; False = CSV if present, else synthetic
    csv_path: str = "data/telemonitoring_parkinsons.csv"
    target: str = "motor_UPDRS"
    standardize: bool = True  # z-score features
    seed: int = 123
    chains: int = 4
    draws: int = 1000
    tune: int = 1000


@dataclass
class EvalConfig:
    use_uciml: bool = True
    csv_path: str = "data/telemonitoring_parkinsons.csv"
    target: str = "motor_UPDRS"
    standardize: bool = True
    seed: int = 123
    draws: int = 1000
```

> Tip: You can create `src/config_local.py` with your own `TrainConfig` / `EvalConfig` to override settings locally.
> The scripts try to import `config_local` first; if not found, they fall back to `config.py`.

---

## Artifacts

- `outputs/idata/posterior.nc` — ArviZ InferenceData (posterior)
- `outputs/figures/trace.png` — MCMC trace plots
- `outputs/figures/forest_coeffs.png` — coefficient forest plot
- `outputs/figures/pred_vs_true.png` — predicted vs. true with 95% PI
- `outputs/figures/ppc.png` — posterior predictive check
- `outputs/summary.csv` — posterior summary table
- `outputs/metrics.csv` — train/test metrics

---

## Repro & Tuning

- Set `seed` in `config.py` to make runs reproducible.
- Toggle `standardize` for z-scoring features.
- Adjust MCMC with `chains`, `draws`, `tune` for accuracy vs. speed.
