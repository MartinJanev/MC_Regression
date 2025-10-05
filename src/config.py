# src/config.py
from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Data source
    use_uciml: bool = True                 # True = fetch UCI (id=189) via ucimlrepo; False = use CSV if present, else synthetic
    csv_path: str = "data/telemonitoring_parkinsons.csv"
    target: str = "motor_UPDRS"

    # Preprocessing
    standardize: bool = True               # z-score features

    # Reproducibility
    seed: int = 123

    # NUTS / MCMC
    chains: int = 4 # number of parallel chains
    draws: int = 2000 # number of samples to draw
    tune: int = 500 # number of warmup (tuning) steps per chain

@dataclass
class EvalConfig:
    # Data source (should match train, unless you intentionally change it)
    use_uciml: bool = True
    csv_path: str = "data/telemonitoring_parkinsons.csv"
    target: str = "motor_UPDRS"

    # Preprocessing
    standardize: bool = True

    # Reproducibility
    seed: int = 123

    # Posterior predictive sampling
    draws: int = 1000
