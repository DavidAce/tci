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

import argparse
import math
import subprocess
import time
import sys

from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze
from numba import njit

A, B = 7.0, 0.1
PROBLEM = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-math.pi, math.pi]] * 3,
}
NAMES = ("S1", "S2", "S3", "ST1", "ST2", "ST3", "S1,3")
V1 = 0.5 * (1.0 + B * math.pi**4 / 5.0)**2
V2 = A**2 / 8.0
V13 = 8.0 * B**2 * math.pi**8 / 225.0
V = V1 + V2 + V13
EXACT = np.array([V1, V2, 0.0, V1 + V13, V2, V13, V13]) / V


@njit(cache=True)
def ishigami(x):
    x1, x2, x3 = np.asarray(x).T
    return np.sin(x1) * (1.0 + B * x3**4) + A * np.sin(x2)**2


def evaluate_gp_sobol(gp, base_size, seed, batch_size=8192):
    # Identical SALib estimator family and base sample size to reference script.
    # d=3 with second-order indices: 8*base_size GP predictions.
    x = sobol_sample(PROBLEM, base_size, calc_second_order=True, seed=seed)
    y = np.empty(len(x))
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        y[start:stop] = gp.predict(x[start:stop])

    result = sobol_analyze(
        PROBLEM, y, calc_second_order=True,
        num_resamples=100, seed=seed, print_to_console=False,
        parallel=False,
    )
    indices = np.r_[result["S1"], result["ST"], result["S2"][0, 2]]

    # In SALib's second-order Saltelli layout, each block contains A and B
    # at its endpoints. Use those samples to estimate mean and variance.
    stride = 2 * PROBLEM["num_vars"] + 2
    base_predictions = np.r_[y[0::stride], y[stride-1::stride]]
    return base_predictions.mean(), base_predictions.var(), indices


def run_gp(args):
    sizes = args.gp_sizes
    if any(n < 2 or n & (n - 1) for n in sizes):
        raise ValueError("All GP training sizes must be powers of two >= 2")
    if args.gp_sobol_base < 2 or args.gp_sobol_base & (args.gp_sobol_base - 1):
        raise ValueError("--gp-sobol-base must be a power of two >= 2")

    # Reuse training data across sample sizes. Reference samplers' exact
    # implementation is external, so don't claim identical coordinates.
    u = qmc.Sobol(d=3, scramble=True, seed=args.gp_seed).random_base2(
        m=max(sizes).bit_length()-1
    )
    x = qmc.scale(u, [-math.pi]*3, [math.pi]*3)
    y = ishigami(x)

    print("=== Ishigami / GP (Vingren kernel + SALib, noiseless) ===")
    print(f"Training: scrambled Sobol | seed={args.gp_seed} | noise=0")
    print("Kernel: ConstantKernel(1, [1e-5,1e5]) * RBF(1, [1e-5,1e5]) [isotropic]")
    print(f"GP: normalize_y=True | alpha={args.alpha:g} | restarts={args.gp_restarts}")
    print(f"SALib: base={args.gp_sobol_base} | predictions={8*args.gp_sobol_base} | second-order=True")
    print(f"Exact: mean={A/2:.8f}, variance={V:.8f}")
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxAbs':>12} {'mean':>13} {'variance':>13} {'fit(s)':>9}")

    for n in sizes:
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e5)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=args.alpha,          # fixed numerical diagonal regularization
            normalize_y=True,
            n_restarts_optimizer=args.gp_restarts,
            random_state=args.gp_seed,
        )
        start = time.perf_counter()
        gp.fit(x[:n], y[:n])
        fit_time = time.perf_counter() - start

        mu, var, indices = evaluate_gp_sobol(
            gp, args.gp_sobol_base, args.gp_seed
        )
        diff = indices - EXACT
        rmse = float(np.sqrt(np.mean(diff**2)))
        max_abs = float(np.max(np.abs(diff)))
        print(f"{'GP':<10} {n:7d} {rmse:12.4e} {max_abs:12.4e} "
              f"{mu:13.6e} {var:13.6e} {fit_time:9.2e}", flush=True)
        last_indices = indices

    print("\nFinal GP Sobol indices:")
    print(f"{'index':<8} {'estimated':>14} {'exact':>14} {'absError':>12}")
    for name, estimated, exact in zip(NAMES, last_indices, EXACT):
        print(f"{name:<8} {estimated:14.8f} {exact:14.8f} {abs(estimated-exact):12.3e}")
    print("\nnEval counts original Ishigami function evaluations for GP training.")
    print("GP predictions for SALib integration do not increase nEval.")


def run_tci(args, tci_args):
    if args.tci is None:
        return
    cmd = [
        str(args.tci.expanduser().resolve()), '--function', 'ishigami', *tci_args,
    ]
    print(f"=== C++ TCI === [{' '.join(cmd)}]", flush=True)
    subprocess.run(cmd, check=True)
    print(flush=True)


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        pos = argv.index("--")
        python_args = argv[:pos]
        tci_args = argv[pos+1:]
    else:
        python_args = argv
        tci_args = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tci", type=Path, help="Optional C++ TCI executable")
    parser.add_argument("--gp-seed", type=int, default=12572601)
    parser.add_argument("--gp-sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--gp-sobol-base", type=int, default=2**14)
    parser.add_argument("--gp-restarts", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1e-10,
                        help="Fixed GP numerical regularization; reference saved setup has 0.01**2")
    args = parser.parse_args(python_args)
    if tci_args and args.tci is None:
        parser.error(f"TCI arguments passed without accompanying --tci=... : {tci_args}")
    if any(s == '--function' or s.startswith('--function=') for s in tci_args):
        parser.error('C++ --function is automatically set to ishigami')


    if args.alpha < 0 or args.gp_restarts < 0:
        parser.error("alpha and gp-restarts must be nonnegative")

    run_tci(args, tci_args)
    run_gp(args)


if __name__ == "__main__":
    main()
