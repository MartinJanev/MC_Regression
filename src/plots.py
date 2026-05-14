import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

# ---------- Minimal visual theme ----------
_PALETTE = {
    "primary": "#1f77b4",  # blue
    "secondary": "#ff7f0e",  # orange
    "accent": "#2ca02c",  # green
    "muted": "#7f7f7f",  # grey
    "outline": "#4c4c4c",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def _save_both(out_path_no_ext: str):
    """Save as PNG and SVG (for print/vector)"""
    png = out_path_no_ext if out_path_no_ext.lower().endswith(".png") else out_path_no_ext + ".png"
    svg = png[:-4] + ".svg"
    plt.tight_layout()
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()


# ---------- PLOTS ----------

def plot_trace(idata, out_path: str):
    ensure_dir(out_path)
    # Use ArviZ style for consistency
    with az.style.context("arviz-whitegrid"):
        az.plot_trace(idata, compact=True)
    _save_both(out_path)


def plot_forest_coefficients(idata, feature_names, out_path: str):
    """Forest plot of β labeled by feature names (95% CrI)."""
    ensure_dir(out_path)

    # sanity: number of features must match β dimension
    n_beta = int(idata.posterior["beta"].shape[-1])
    if len(feature_names) != n_beta:
        feature_names = list(feature_names)[:n_beta]

    ax = az.plot_forest(
        idata,
        var_names=["beta"],
        combined=True,
        hdi_prob=0.95,  # <-- FIX: was credible_interval
        figsize=(8, 0.35 * n_beta + 1),
    )

    # get the axis and relabel ticks to feature names
    ax = np.atleast_1d(ax).ravel()[-1]
    ax.set_yticks(range(n_beta))
    ax.set_yticklabels(feature_names)
    ax.set_title("Posterior Coefficients (95% CrI)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_vs_true(y_true, y_pred_samples, out_path: str):
    """
    Scatter of E[y|x] vs y with vertical 95% PI errorbars,
    diagonal y=x reference, and a small metrics box.
    """
    ensure_dir(out_path)
    y_mean = y_pred_samples.mean(axis=0)
    lo = np.percentile(y_pred_samples, 2.5, axis=0)
    hi = np.percentile(y_pred_samples, 97.5, axis=0)

    # Metrics for the annotation box
    def _pearsonr(a, b):
        a = np.asarray(a);
        b = np.asarray(b)
        a = a - a.mean();
        b = b - b.mean()
        denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
        return float((a * b).sum() / denom) if denom > 0 else np.nan

    mae = float(np.mean(np.abs(y_true - y_mean)))
    rmse = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
    r = _pearsonr(y_true, y_mean)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Error bars (95% PI)
    yerr = np.vstack([y_mean - lo, hi - y_mean])
    ax.errorbar(
        y_true, y_mean, yerr=yerr,
        fmt="o", markersize=3.5, alpha=0.65,
        ecolor=_PALETTE["muted"], elinewidth=0.8, capsize=2,
        color=_PALETTE["primary"], label="Prediction (mean ±95% PI)"
    )

    # Diagonal y=x
    m = max(float(np.max(y_true)), float(np.max(y_mean)))
    n = min(float(np.min(y_true)), float(np.min(y_mean)))
    xs = np.linspace(n - 1, m + 1, 100)
    ax.plot(xs, xs, linestyle="--", linewidth=1.2, color=_PALETTE["outline"], label="y = x")

    ax.set_xlabel("Вистински UPDRS")
    ax.set_ylabel("Предвиден UPDRS (средно)")
    ax.set_title("Предвидување vs Вистинско (со 95% PI)")

    # Metrics box
    text = f"r = {r:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=_PALETTE["muted"], alpha=0.9)
    )

    ax.legend(loc="lower right", frameon=True)
    _save_both(out_path)


def plot_ppc(y_true, y_pred_samples, out_path: str):
    """
    Histogram overlay of observed vs posterior predictive samples.
    Uses transparent fill + step outline for clarity in print.
    """
    ensure_dir(out_path)
    sim = y_pred_samples.flatten()

    fig, ax = plt.subplots(figsize=(6, 4))
    # Observed (filled)
    ax.hist(
        y_true, bins=30, density=True, alpha=0.35,
        color=_PALETTE["primary"], label="Observed"
    )
    # Predictive (step)
    ax.hist(
        sim, bins=30, density=True, histtype="step",
        linewidth=1.8, color=_PALETTE["secondary"],
        label="Posterior Predictive"
    )

    ax.set_xlabel("UPDRS")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend(loc="best", frameon=True)
    _save_both(out_path)


# ---------- TABLES ----------

def save_posterior_summary(idata, out_csv: str):
    ensure_dir(out_csv)
    summ = az.summary(idata, var_names=["alpha", "beta", "sigma"], kind="stats", round_to=4)
    summ.to_csv(out_csv)


def save_metrics(y_true_tr, y_pred_tr, y_true_te, y_pred_te, out_csv: str):
    ensure_dir(out_csv)
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def pearsonr(a, b):
        a = np.asarray(a);
        b = np.asarray(b)
        a = a - a.mean();
        b = b - b.mean()
        denom = (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum()))
        return float((a * b).sum() / denom) if denom > 0 else np.nan

    yhat_tr = y_pred_tr.mean(axis=0)
    yhat_te = y_pred_te.mean(axis=0)

    metrics = pd.DataFrame([
        {
            "split": "train",
            "MAE": mean_absolute_error(y_true_tr, yhat_tr),
            "RMSE": float(np.sqrt(mean_squared_error(y_true_tr, yhat_tr))),
            "Pearson_r": pearsonr(y_true_tr, yhat_tr)
        },
        {
            "split": "test",
            "MAE": mean_absolute_error(y_true_te, yhat_te),
            "RMSE": float(np.sqrt(mean_squared_error(y_true_te, yhat_te))),
            "Pearson_r": pearsonr(y_true_te, yhat_te)
        }
    ])
    metrics.to_csv(out_csv, index=False)
