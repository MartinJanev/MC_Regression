import os
import numpy as np
import arviz as az
from src.data_loading import load_dataset, prepare_xy
from src.model_numpyro import posterior_predict
from src.plots import plot_pred_vs_true, plot_ppc, save_metrics
try:
    # optional local override
    from src.config_local import EvalConfig  # type: ignore
except Exception:
    from src.config import EvalConfig

def main():
    cfg = EvalConfig()

    if not os.path.isfile("outputs/idata/posterior.nc"):
        raise SystemExit("Run scripts/run_train.py first to create outputs/idata/posterior.nc")

    idata = az.from_netcdf("outputs/idata/posterior.nc")

    df = load_dataset(cfg.csv_path, target=cfg.target, prefer_uciml=cfg.use_uciml)
    X_tr, y_tr, X_te, y_te, features, scaler = prepare_xy(
        df, target=cfg.target, standardize=cfg.standardize, seed=cfg.seed
    )

    y_tr_s = posterior_predict(idata, X_tr, seed=cfg.seed, draws=cfg.draws)        # [S, n_tr]
    y_te_s = posterior_predict(idata, X_te, seed=cfg.seed+1, draws=cfg.draws)      # [S, n_te]

    # Plots: Pred vs True + PPC
    os.makedirs("outputs/figures", exist_ok=True)
    plot_pred_vs_true(y_te, y_te_s, "outputs/figures/pred_vs_true.png")
    plot_ppc(y_te, y_te_s, "outputs/figures/ppc.png")

    # Tables: metrics
    save_metrics(y_tr, y_tr_s, y_te, y_te_s, "outputs/metrics.csv")

    print("Evaluation complete. Figures in outputs/figures, metrics in outputs/metrics.csv")

if __name__ == "__main__":
    main()
