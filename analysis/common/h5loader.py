
from pathlib import Path

import h5py
import numpy as np

def load_tci_n_eval(filename):
    with h5py.File(Path(filename), "r") as h5:
        return int(h5["tci/status"][-1]["nEval"])

def load_tci_predictions(filename):
    with h5py.File(Path(filename), "r") as h5:
        data = h5["tci/predictions"][:]

    X = np.stack(data["point"]).astype(np.float64)
    y_exact = np.asarray(data["funcvalue"], dtype=np.float64)
    y_tci = np.asarray(data["predicted"], dtype=np.float64)

    if X.ndim != 2 or X.shape[0] != y_exact.size:
        raise ValueError("Unexpected TCI prediction dimensions")

    if y_tci.shape != y_exact.shape:
        raise ValueError("TCI prediction count mismatch")

    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(y_exact))
        and np.all(np.isfinite(y_tci))
    ):
        raise ValueError("Non-finite values in TCI predictions")

    return X, y_exact, y_tci

def prediction_errors(y_exact, y_pred):
    error = np.asarray(y_pred) - np.asarray(y_exact)

    return {
        "rmse": np.sqrt(np.mean(error**2)),
        "maxDiff": np.max(np.abs(error)),
    }

