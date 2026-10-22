#!/usr/bin/env python3
"""Ishigami: TCI vs GP.

Reference:
  https://github.com/benjamin-vingren/ishigami_sensitivity

Matches the reference GP's isotropic ConstantKernel * RBF kernel, optimizer
bounds/restarts, output normalization, and SALib/Saltelli Sobol estimator.
Deviations: noiseless outputs, small fixed numerical regularization (alpha),
reproducibly seeded optimizer, no saved models, and a SciPy Sobol training design.

Requires numpy, scipy, scikit-learn, SALib.
"""

import math
import subprocess
import time
import sys
import os
import warnings

from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.exceptions import ConvergenceWarning
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze
from numba import njit

from common.parse import parse
from common.h5loader import load_tci_n_eval, load_tci_predictions, prediction_errors
from common.functions import ishigami, morris2006, legendrePoly3
from common.plot_info_lattice import read_info_lattice, plot_info_lattice

def evaluate_gp_sobol(gp, function, sobol_samples, seed, batch_size=8192):
    problem = function.problem
    dim = function.dim

    x = sobol_sample(problem, N=sobol_samples,
                     calc_second_order=True, seed=seed
                     )

    y = np.empty(len(x))
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        y[start:stop] = gp.predict(x[start:stop])

    result = sobol_analyze(
        problem, y,
        calc_second_order=True,
        num_resamples=2,
        seed=seed,
        print_to_console=False,
        parallel=False,
    )

    second = np.array([
        result["S2"][i, j]
        for i, j in function.sobol_pairs
    ])

    indices = np.r_[
        result["S1"],
        result["ST"],
        second
    ]

    if function.include_higher:
        ii, jj = np.triu_indices(dim, k=1)
        higher = (
            1.0
            - np.sum(result["S1"])
            - np.sum(result["S2"][ii, jj])
        )
        indices = np.r_[indices, higher]

    stride = 2 * dim + 2
    values = np.r_[y[0::stride], y[stride - 1::stride]]

    return values.mean(), values.var(), indices


def compare_tci_gp(args, gp_models, function):
    if args.tci_h5 is not None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, args.tci_h5)
        X_test, y_exact, y_tci = load_tci_predictions(filename)

        if X_test.shape[1] != function.dim:
            raise ValueError(f"HDF5 has {X_test.shape[1]} variables, but {function.name} expects {function.dim}")

        tci_neval = load_tci_n_eval(filename)
        tci_error = prediction_errors(y_exact, y_tci)

        print("\n=== Prediction accuracy ===")
        print(f"{'method':<10} {'nEval':>7} {'RMSE':>14} {'maxDiff':>14}")
        print(f"{'TCI':<10} {tci_neval:>7} {tci_error['rmse']:14.6e} {tci_error['maxDiff']:14.6e}")
        for n, gp in gp_models:
            y_gp = np.asarray(gp.predict(X_test)).reshape(-1)
            gp_error = prediction_errors(y_exact, y_gp)
            print(f"{'GP':<10} {n:7d} {gp_error['rmse']:>14.6e} {gp_error['maxDiff']:>14.6e}")


def sobol_names(function):
    dim = function.dim

    names = (
        [f"S{i}" for i in range(dim)]
        + [f"ST{i}" for i in range(dim)]
        + [
            f"S{i},{j}"
            for i, j in function.sobol_pairs
        ]
    )

    if function.include_higher:
        names.append("S_residual")

    return names


def run_gp(args, function):
    dim = function.dim
    nevals = args.gp_nevals

    if any(n < 2 or n & (n - 1) for n in nevals):
        raise ValueError("GP training nevals must be powers of two >= 2")

    if (args.gp_sobol_samples < 2 or args.gp_sobol_samples & (args.gp_sobol_samples - 1)):
        raise ValueError("--gp-sobol-samples must be a power of two >= 2")

    # Generate the same nested scrambled Sobol design.
    u = qmc.Sobol(d=dim, scramble=True, seed=args.gp_seed).random_base2(m=max(nevals).bit_length() - 1)

    # Transform to the physical domain of this function.
    bounds = np.asarray(function.problem["bounds"])
    x = qmc.scale(u, bounds[:, 0], bounds[:, 1])

    # Evaluate the exact function only for GP training.
    y = function.evaluate(x)

    print(f"=== {function.name} / GP (dim={dim}) ===")
    print(f"Training : scrambled Sobol | seed={args.gp_seed}")
    print(f"GP       : normalize_y=True | alpha={args.alpha:g} | restarts={args.gp_restarts}")
    print(f"SALib    : N={args.gp_sobol_samples} | predictions={(2*dim+2)*args.gp_sobol_samples}")
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxDiff':>12} {'mean':>13} {'variance':>13} {'fit(s)':>9} {'sobol est.(s)':>13}")

    models = []
    previous_kernel = None

    for n in nevals:
        if previous_kernel is None:
            kernel = ConstantKernel(1.0, (function.kernel_lower, function.kernel_upper)) * \
                RBF(length_scale=np.ones(function.dim), length_scale_bounds=(1e-5, 1e5))
        else:
            kernel = previous_kernel

        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=args.alpha,
                                      normalize_y=True,
                                      n_restarts_optimizer=args.gp_restarts,
                                      random_state=args.gp_seed,
                                      )

        start = time.perf_counter()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message=r"lbfgs failed to converge.*",
                                    category=ConvergenceWarning)
            warnings.filterwarnings("ignore",
                                    message=r"The optimal value found for dimension \d+ of parameter k2__length_scale"
                                    " is close to the specified upper bound",
                                    category=ConvergenceWarning)
            gp.fit(x[:n], y[:n])

        fit_time = time.perf_counter() - start
        previous_kernel = gp.kernel_
        models.append((n, gp))

        start = time.perf_counter()

        mu, var, indices = evaluate_gp_sobol(
            gp, function,
            args.gp_sobol_samples,
            args.gp_seed
        )

        sobol_time = time.perf_counter() - start

        # Preserve each benchmark's original RMSE convention.
        if function.exact_sobol is not None:
            selected = function.error_slice
            delta = indices[selected] - function.exact_sobol[selected]

            rmse = float(np.sqrt(np.mean(delta**2)))
            max_diff = float(np.max(np.abs(delta)))
        else:
            rmse = None
            max_diff = None

        rmse_str = f"{rmse:12.4e}" if rmse is not None else f"{'n/a':>12}"
        max_str = f"{max_diff:12.4e}" if max_diff is not None else f"{'n/a':>12}"

        print(f"{'GP':<10} {n:7d} "
              f"{rmse_str} {max_str} "
              f"{mu:13.6e} {var:13.6e} "
              f"{fit_time:9.2e} {sobol_time:13.2e}",
              flush=True
              )

    print("\nFinal GP Sobol indices:")

    exact = function.exact_sobol
    names = sobol_names(function)

    if exact is not None:
        if len(indices) != len(exact):
            raise ValueError("Sobol reference size mismatch")

        print(f"{'index':<10} {'estimated':>14} {'exact':>14} {'error':>14}")

        for name, estimated, reference in zip(names, indices, exact):
            diff = estimated - reference
            print(f"{name:<10} {estimated:14.8f} {reference:14.8f} {diff:14.3e}")

    else:
        print(f"{'index':<10} {'estimated':>14}")

        for name, estimated in zip(names, indices):
            print(f"{name:<10} {estimated:14.8f}")

    class ExactPredictor:
        def predict(self, x):
            return function.evaluate(x)

    _, _, indices_direct = evaluate_gp_sobol(
        ExactPredictor(),
        function,
        args.gp_sobol_samples,
        args.gp_seed,
    )

    selected = function.error_slice
    diff = indices_direct[selected] - function.exact_sobol[selected]

    print("Direct SALib RMSE:", np.sqrt(np.mean(diff**2)))
    print("Direct SALib maxDiff:", np.max(np.abs(diff)))

    selected = function.error_slice
    delta = indices[selected] - indices_direct[selected]

    print("GP vs direct SALib RMSE:", np.sqrt(np.mean(delta**2)))
    print("GP vs direct SALib maxDiff:", np.max(np.abs(delta)))

    return models


def run_tci(args, tci_args):
    if args.tci is None:
        return
    cmd = [
        str(args.tci.expanduser().resolve()), '--function', args.function, *tci_args,
    ]
    print(f"=== C++ TCI === [{' '.join(cmd)}]", flush=True)
    subprocess.run(cmd, check=True)
    print(flush=True)


def main():
    args, python_args, tci_args = parse()

    constructors = {"ishigami": lambda: ishigami(),
                    "morris": lambda: morris2006(dim=args.dimension, active=args.active),
                    "legendre": lambda: legendrePoly3(dim=args.dimension),
                    }

    function = constructors[args.function]()

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, args.tci_h5)
    il = read_info_lattice(filename, "tci/sil")
    fig, _ = plot_info_lattice(il, logarithmic=False, vmin=1e-5, vmax=1.0, cmap="Greens", size=280)


    run_tci(args, tci_args)
    gp_models = run_gp(args, function)

    compare_tci_gp(args, gp_models, function)
    plt.show()

    exit(0)
    




if __name__ == "__main__":
    main()
