from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional UCI import (keeps CSV path working if package isn't installed)
try:
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
except Exception:
    fetch_ucirepo = None  # type: ignore

# Columns expected in UCI Parkinson Telemonitoring
UCI_COLUMNS = [
    "subject#", "age", "sex", "test_time",
    "motor_UPDRS", "total_UPDRS",
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE"
]

VOICE_FEATURES = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "PPE"
]


def _synthetic_telemonitoring(n_patients: int = 42, n_records: int = 5800, seed: int = 123) -> DataFrame:
    rng = np.random.default_rng(seed)
    subject_ids = rng.integers(1, n_patients + 1, size=n_records)
    age = rng.normal(65, 8, size=n_records).clip(40, 85)
    sex = rng.integers(0, 2, size=n_records)
    test_time = rng.uniform(0, 180, size=n_records)
    latent = rng.normal(0, 1, size=n_records)
    feat = {}
    feat["Jitter(%)"] = 0.3 + 0.1 * latent + rng.normal(0, 0.05, size=n_records)
    feat["Jitter(Abs)"] = 0.0002 + 0.0001 * latent + rng.normal(0, 0.00005, size=n_records)
    feat["Jitter:RAP"] = 0.2 + 0.08 * latent + rng.normal(0, 0.03, size=n_records)
    feat["Jitter:PPQ5"] = 0.25 + 0.09 * latent + rng.normal(0, 0.03, size=n_records)
    feat["Jitter:DDP"] = 3 * feat["Jitter:RAP"] + rng.normal(0, 0.05, size=n_records)
    feat["Shimmer"] = 3.0 + 0.9 * latent + rng.normal(0, 0.3, size=n_records)
    feat["Shimmer(dB)"] = 0.25 + 0.07 * latent + rng.normal(0, 0.05, size=n_records)
    feat["Shimmer:APQ3"] = 1.5 + 0.45 * latent + rng.normal(0, 0.2, size=n_records)
    feat["Shimmer:APQ5"] = 1.9 + 0.5 * latent + rng.normal(0, 0.2, size=n_records)
    feat["Shimmer:APQ11"] = 2.5 + 0.6 * latent + rng.normal(0, 0.25, size=n_records)
    feat["Shimmer:DDA"] = 3 * feat["Shimmer:APQ3"] + rng.normal(0, 0.2, size=n_records)
    feat["NHR"] = 0.02 + 0.02 * latent + rng.normal(0, 0.01, size=n_records)
    feat["HNR"] = 20 - 2.0 * latent + rng.normal(0, 1.5, size=n_records)
    feat["RPDE"] = 0.45 + 0.15 * latent + rng.normal(0, 0.05, size=n_records)
    feat["DFA"] = 0.75 + 0.05 * latent + rng.normal(0, 0.02, size=n_records)
    feat["PPE"] = 0.35 + 0.2 * latent + rng.normal(0, 0.05, size=n_records)
    X = pd.DataFrame(feat)
    coef = np.array([8.0, 200.0, 1.2, 1.5, 0.6, 0.9, 2.0, 0.6, 0.7, 0.8, 0.5, 40.0, -0.8, 6.0, 12.0, 7.0])
    noise = rng.normal(0, 3.5, size=n_records)
    motor_UPDRS = 12 + X.values @ (coef / 50.0) + noise
    df = pd.DataFrame({
        "subject#": subject_ids,
        "age": age,
        "sex": sex,
        "test_time": test_time,
        "motor_UPDRS": motor_UPDRS,
        "total_UPDRS": motor_UPDRS + rng.normal(7, 3, size=n_records)
    })
    df = pd.concat([df, X], axis=1)
    return df[["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS"] + list(X.columns)]


def _load_uciml_parkinsons(target: str = "motor_UPDRS") -> pd.DataFrame:
    """
    Fetch Parkinson's Telemonitoring directly from UCI (id=189) via ucimlrepo,
    return a single DataFrame with feature + target columns. Column names are
    normalized to match our pipeline (e.g., motor_updrs -> motor_UPDRS).
    """
    if fetch_ucirepo is None:
        raise ImportError("ucimlrepo not installed")
    repo = fetch_ucirepo(id=189) # Parkinson's Telemonitoring


    X = repo.data.features
    y = repo.data.targets
    df = pd.concat([X, y], axis=1)
    df.rename(
        columns={
            "motor_updrs": "motor_UPDRS",
            "total_updrs": "total_UPDRS",
        },
        inplace=True,
    )
    return df


def load_dataset(csv_path: str, target: str = "motor_UPDRS", prefer_uciml: bool = False) -> DataFrame:
    """
    Load the dataset from (in priority):
      1) UCI via ucimlrepo if prefer_uciml=True
      2) Local CSV if present
      3) UCI via ucimlrepo as fallback (if available)
      4) Synthetic generator (last resort)
    """
    if prefer_uciml:
        try:
            return _load_uciml_parkinsons(target=target)
        except Exception:
            pass

    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if target not in df.columns:
            alt = {"motor_updrs": "motor_UPDRS", "Motor_UPDRS": "motor_UPDRS"}
            for k, v in alt.items():
                if k in df.columns:
                    df.rename(columns={k: v}, inplace=True)
        return df

    if fetch_ucirepo is not None:
        try:
            return _load_uciml_parkinsons(target=target)
        except Exception:
            pass

    return _synthetic_telemonitoring()


def prepare_xy(
        df, target: str = "motor_UPDRS", features: Optional[List[str]] = None,
        test_size: float = 0.2, seed: int = 123, standardize: bool = True
):
    if features is None:
        features = VOICE_FEATURES

    X = df[features].values.astype(float)
    y = df[target].values.astype(float)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return (X_tr, y_tr, X_te, y_te, features, scaler)
