import os
import arviz as az
import pandas as pd
from src.data_loading import load_dataset, prepare_xy, VOICE_FEATURES
from src.model_numpyro import fit_posterior, to_inferencedata
from src.plots import plot_trace, plot_forest_coefficients, save_posterior_summary
try:
    # optional local override
    from src.config_local import TrainConfig  # type: ignore
except Exception:
    from src.config import TrainConfig

def main():
    cfg = TrainConfig()

    df = load_dataset(cfg.csv_path, target=cfg.target, prefer_uciml=cfg.use_uciml)
    X_tr, y_tr, X_te, y_te, features, scaler = prepare_xy(
        df, target=cfg.target, standardize=cfg.standardize, seed=cfg.seed
    )

    mcmc = fit_posterior(
        X_tr, y_tr,
        seed=cfg.seed, chains=cfg.chains, draws=cfg.draws, tune=cfg.tune
    )
    idata = to_inferencedata(mcmc)

    # Save posterior
    os.makedirs("../outputs1/idata", exist_ok=True)
    idata.to_netcdf("outputs1/idata/posterior.nc")

    # Plots requiring only posterior
    os.makedirs("../outputs1/figures", exist_ok=True)
    plot_trace(idata, "../outputs1/figures/trace.png")
    plot_forest_coefficients(idata, features, "../outputs1/figures/forest_coeffs.png")

    # Posterior summary
    save_posterior_summary(idata, "outputs1/summary.csv")

    print("Training complete. Posterior saved to outputs1/idata/posterior.nc")

if __name__ == "__main__":
    main()
