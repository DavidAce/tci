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

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.exceptions import ConvergenceWarning
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze
from numba import njit

from common.parse import parse
from common.h5loader import load_tci_predictions, prediction_errors
from common.functions import ishigami, morris2006, legendrePoly3


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
        num_resamples=100,
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


def compare_tci_gp(args, gp):
    if args.tci_h5 is not None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, args.tci_h5)
        X_test, y_exact, y_tci = load_tci_predictions(filename)

        y_gp = np.asarray(gp.predict(X_test)).reshape(-1)
        tci_error = prediction_errors(y_exact, y_tci)
        gp_error = prediction_errors(y_exact, y_gp)

        print("\n=== Prediction accuracy ===")
        print(f"{'method':<10} {'RMSE':>14} {'maxAbs':>14}")
        print(f"{'TCI':<10} {tci_error['rmse']:>14.6e} {tci_error['maxAbs']:>14.6e}")
        print(f"{'GP':<10} {gp_error['rmse']:>14.6e} {gp_error['maxAbs']:>14.6e}")


def run_gp(args, function):
    dim = function.dim
    sizes = args.gp_sizes

    if any(n < 2 or n & (n - 1) for n in sizes):
        raise ValueError("GP training sizes must be powers of two >= 2")

    if (args.gp_sobol_samples < 2 or args.gp_sobol_samples & (args.gp_sobol_samples - 1)):
        raise ValueError("--gp-sobol-samples must be a power of two >= 2")

    # Generate the same nested scrambled Sobol design.
    u = qmc.Sobol(d=dim, scramble=True, seed=args.gp_seed).random_base2(m=max(sizes).bit_length() - 1)

    # Transform to the physical domain of this function.
    bounds = np.asarray(function.problem["bounds"])
    x = qmc.scale(u, bounds[:, 0], bounds[:, 1])

    # Evaluate the exact function only for GP training.
    y = function.evaluate(x)

    print(f"=== {function.name} / GP (dim={dim}) ===")
    print(f"Training: scrambled Sobol | seed={args.gp_seed}")
    print(f"GP: normalize_y=True | alpha={args.alpha:g} | restarts={args.gp_restarts}")
    print(f"SALib: base={args.gp_sobol_samples} | predictions={(2*dim+2)*args.gp_sobol_samples}")
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxDiff':>12} {'mean':>13} {'variance':>13} {'fit(s)':>9} {'sobol est.(s)':>13}")

    for n in sizes:
        kernel = ConstantKernel(1.0, (function.kernel_lower, function.kernel_upper)) * \
            RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))

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
            gp.fit(x[:n], y[:n])

        fit_time = time.perf_counter() - start

        start = time.perf_counter()

        mu, var, indices = evaluate_gp_sobol(
            gp, function,
            args.gp_sobol_samples,
            args.gp_seed
        )

        sobol_time = time.perf_counter() - start

        # Preserve each benchmark's original RMSE convention.
        selected = function.error_slice
        delta = (
            indices[selected] - function.EXACT[selected]
        )

        rmse = float(np.sqrt(np.mean(delta**2)))
        max_diff = float(np.max(np.abs(delta)))

        print(f"{'GP':<10} {n:7d} {rmse:12.4e} {max_diff:12.4e} "
              f"{mu:13.6e} {var:13.6e} {fit_time:9.2e} {sobol_time:13.2e}",
              flush=True)

    print("\nFinal GP Sobol indices:")
    print(f"{'index':<10} {'estimated':>14} {'exact':>14} {'absError':>12}"
          )

    for name, estimated, exact in zip(
        function.SOBOL_NAMES, indices, function.EXACT
    ):
        print(f"{name:<10} {estimated:14.8f} {exact:14.8f} {abs(estimated-exact):12.3e}")

    return gp


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

    run_tci(args, tci_args)
    gp = run_gp(args, function)

    compare_tci_gp(args, gp)


if __name__ == "__main__":
    main()
