#!/usr/bin/env python3
"""Morris2006: C++ TCI versus Gaussian-process Sobol indices.

Reproduces source/func/Morris2006.cpp in https://github.com/DavidAce/tci.
The current C++ main.cpp constructs Morris2006(8, 5), and selects it with
--function morris. Input variables are independent Uniform(0,1).

Requires numpy, scipy, scikit-learn, numba, and SALib.
Everything after the -- separator is forwarded unchanged to the C++ executable.
"""

import argparse
import math
import shlex
import subprocess
import sys
import time
import os
import warnings

from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.exceptions import ConvergenceWarning
from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

from h5loader import load_tci_predictions, prediction_errors

def parameters(active):
    """Exact alpha and beta in source/func/Morris2006.cpp."""
    alpha = math.sqrt(12.0) - 6.0 * math.sqrt(0.1 * (active - 1))
    beta = 12.0 / math.sqrt(10.0 * (active - 1))
    return alpha, beta


@njit(cache=True)
def morris2006(x, active, alpha, beta):
    """Evaluate rows of x, shape (n_samples, dimension), without temporaries."""
    result = np.empty(x.shape[0], dtype=np.float64)
    for row in range(x.shape[0]):
        s = 0.0
        pairs = 0.0
        for i in range(active):
            pairs += s * x[row, i]
            s += x[row, i]
        result[row] = alpha * s + beta * pairs
    return result


def exact_sobol(dim, active):
    """Order: S_i, ST_i, S_ij (lexicographic i<j); residual S_ijk..."""
    variance = 1.05 * active
    first = np.zeros(dim)
    total = np.zeros(dim)
    first[:active] = 1.0 / variance
    total[:active] = 1.1 / variance

    ii, jj = np.triu_indices(dim, k=1)
    second = np.where(
        (ii < active) & (jj < active),
        1.0 / (10.0 * (active - 1) * variance),
        0.0,
    )
    residual = 1.0 - first.sum() - second.sum()
    assert abs(residual) < 1e-12
    alpha, beta = parameters(active)
    mean = alpha * (active / 2.0) + beta * active * (active - 1) / 8.0
    return mean, variance, np.r_[first, total, second, 0.0]


def evaluate_gp_sobol(gp, problem, base_size, seed, batch_size=8192):
    """SALib first-, total-, and second-order estimates from GP predictions."""
    dim = problem['num_vars']
    samples = sobol_sample(problem, base_size, calc_second_order=True, seed=seed)
    predictions = np.empty(len(samples))
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        predictions[start:stop] = gp.predict(samples[start:stop])

    sobol = sobol_analyze(
        problem, predictions, calc_second_order=True,
        num_resamples=100, seed=seed, print_to_console=False, parallel=False,
    )
    ii, jj = np.triu_indices(dim, k=1)
    first = sobol['S1']
    second = sobol['S2'][ii, jj]
    residual = 1.0 - first.sum() - second.sum()
    indices = np.r_[first, sobol['ST'], second, residual]

    # The first/last samples of each Saltelli block correspond to A and B.
    stride = 2 * dim + 2
    original = np.r_[predictions[0::stride], predictions[stride - 1::stride]]
    return original.mean(), original.var(), indices


def print_index_summary(indices, exact, dim):
    n_pairs = dim * (dim - 1) // 2
    print('\nFinal GP Sobol indices (first and total):')
    print(f"{'index':<10} {'estimated':>14} {'exact':>14} {'absError':>12}")
    for i in range(dim):
        print(f"{('S'+str(i+1)):<10} {indices[i]:14.8f} {exact[i]:14.8f} "
              f"{abs(indices[i] - exact[i]):12.3e}")
    for i in range(dim):
        k = dim + i
        print(f"{('ST'+str(i+1)):<10} {indices[k]:14.8f} {exact[k]:14.8f} "
              f"{abs(indices[k] - exact[k]):12.3e}")

    print('\nSobol index groups:')
    print(f"{'group':<10} {'count':>6} {'sum est.':>14} {'sum exact':>14} "
          f"{'absError':>12} {'RMSE':>12}")
    for label, section in (
        ('S_i', slice(0, dim)),
        ('S_ij', slice(2 * dim, 2 * dim + n_pairs)),
        ('S_ijk...', slice(-1, None)),
    ):
        estimate = indices[section]
        truth = exact[section]
        diff = estimate - truth
        print(f'{label:<10} {len(estimate):6d} {estimate.sum():14.8f} '
              f'{truth.sum():14.8f} {abs(diff.sum()):12.3e} '
              f'{np.sqrt(np.mean(diff**2)):12.3e}')


def run_gp(args):
    dim, active = args.dimension, args.active
    sizes = args.gp_sizes
    if any(n < 2 or n & (n - 1) for n in sizes):
        raise ValueError('GP training sizes must be powers of two >= 2')
    if args.gp_sobol_samples < 2 or args.gp_sobol_samples & (args.gp_sobol_samples - 1):
        raise ValueError('--gp-sobol-samples must be a power of two >= 2')

    alpha, beta = parameters(active)
    exact_mean, exact_var, exact = exact_sobol(dim, active)
    problem = {
        'num_vars': dim,
        'names': [f'x{i+1}' for i in range(dim)],
        'bounds': [[0.0, 1.0] for _ in range(dim)],
    }

    # Nested, scrambled Sobol points: each fit uses the first n samples.
    x = qmc.Sobol(d=dim, scramble=True, seed=args.gp_seed).random_base2(
        m=max(sizes).bit_length() - 1
    )
    y = morris2006(x, active, alpha, beta)

    print(f'=== Morris2006 / GP (dim={dim}, active={active}, noiseless) ===')
    print(f'Training: scrambled Sobol | seed={args.gp_seed} | noise=0')
    print('Kernel: ConstantKernel(1, [1e-5,1e5]) * RBF(1, [1e-5,1e5]) [isotropic]')
    print(f'GP: normalize_y=True | alpha={args.alpha:g} | restarts={args.gp_restarts}')
    print(f'SALib: base={args.gp_sobol_samples} | '
          f'predictions={(2*dim+2)*args.gp_sobol_samples} | second-order=True')
    print(f'Exact: mean={exact_mean:.8f}, variance={exact_var:.8f}, '
          f'sum(S1)={exact[:dim].sum():.8f}, '
          f'sum(S2)={exact[2*dim:-1].sum():.8f}, S_ijk...=0')
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxAbs':>12} "
          f"{'S_ijk...':>12} {'mean':>12} {'variance':>12} "
          f"{'fit(s)':>9} {'sobol(s)':>9}")

    last_indices = None
    for n in sizes:
        kernel = ConstantKernel(1.0, (1e-5, 1e8)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e5)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=args.alpha, normalize_y=True,
            n_restarts_optimizer=args.gp_restarts, random_state=args.gp_seed,
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"lbfgs failed to converge.*",
                                     category=ConvergenceWarning)
            gp.fit(x[:n], y[:n])

        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        mean, variance, indices = evaluate_gp_sobol(
            gp, problem, args.gp_sobol_samples, args.gp_seed
        )
        sobol_time = time.perf_counter() - t0

        # Exactly the same index set as C++ Morris2006: first, total, pairs.
        # Report third-and-higher residual separately, rather than including
        # it as an additional entry in the headline RMSE.
        delta = indices[:-1] - exact[:-1]
        rmse = float(np.sqrt(np.mean(delta**2)))
        max_abs = float(np.max(np.abs(delta)))
        print(f"{'GP':<10} {n:7d} {rmse:12.4e} {max_abs:12.4e} "
              f'{indices[-1]:12.6e} {mean:12.6e} {variance:12.6e} '
              f'{fit_time:9.2e} {sobol_time:9.2e}', flush=True)
        last_indices = indices

    print_index_summary(last_indices, exact, dim)
    print('\nRMSE and maxAbs include S_i, ST_i, and every S_ij, '
          'but not the S_ijk... residual.')
    
    if args.tci_h5 is not None:
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, args.tci_h5)
        X_test, y_exact, y_tci = load_tci_predictions(filename)

        y_gp = np.asarray(gp.predict(X_test)).reshape(-1)
        tci_error = prediction_errors(y_exact, y_tci)
        gp_error  = prediction_errors(y_exact, y_gp)

        print("\n=== Prediction accuracy ===")
        print(f"{'method':<10} {'RMSE':>14} {'maxAbs':>14}")
        print(f"{'TCI':<10} {tci_error['rmse']:>14.6e} {tci_error['maxAbs']:>14.6e}")
        print(f"{'GP':<10} {gp_error['rmse']:>14.6e} {gp_error['maxAbs']:>14.6e}")


def run_tci(args, tci_args):
    if args.tci is None:
        return
    cmd = [str(args.tci.expanduser().resolve()), '--function', 'morris', *tci_args]
    print(f'=== C++ TCI === [{shlex.join(cmd)}]', flush=True)
    subprocess.run(cmd, check=True)
    print(flush=True)


def main():
    argv = sys.argv[1:]
    if '--' in argv:
        pos = argv.index('--')
        python_args, tci_args = argv[:pos], argv[pos+1:]
    else:
        python_args, tci_args = argv, []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tci', type=Path, help='Optional C++ TCI executable')
    parser.add_argument('--dimension', type=int, default=8)
    parser.add_argument('--active', type=int, default=5)
    parser.add_argument('--gp-seed', type=int, default=12572601, help='Random seed for generating the GP training points.')
    parser.add_argument('--gp-sizes', nargs='+', type=int, default=[128, 256, 512, 1024], help='Number of function evaluations used to training the GP model (powers of two).')
    parser.add_argument('--gp-sobol-samples', type=int, default=2**14, help='Number of sample pairs used to estimate Sobol indices from the GP (powers of two).')
    parser.add_argument('--gp-restarts', type=int, default=10, help='Number of additional attempts to optimize the GP kernel parameters.')
    parser.add_argument('--alpha', type=float, default=1e-8, help='Small value added to the GP covariance diagonal for numerical stability.')
    parser.add_argument("--tci-h5", type=str, default='tci/Morris2006.h5', help="TCI HDF5 file path (relative to this .py file) containing prediction points")



    args = parser.parse_args(python_args)
    
    if not '--dimension' in tci_args:
        tci_args.append(f'--dimension={args.dimension}')
    if not '--active' in tci_args:
        tci_args.append(f'--active={args.active}')

    print(tci_args)
    args_tci, unknown = parser.parse_known_args(tci_args)

    if not 2 <= args.active <= args.dimension:
        parser.error('require 2 <= --active <= --dimension')
    if args.alpha < 0 or args.gp_restarts < 0:
        parser.error('alpha and gp-restarts must be nonnegative')
    if tci_args and args.tci is None:
        parser.error('TCI arguments supplied without --tci')
    if any(arg == '--function' or arg.startswith('--function=') for arg in tci_args):
        parser.error('C++ --function is automatically set to morris')
    if args.tci and args.dimension != args_tci.dimension:
        parser.error(f'Mismatching parameter for "--dimension": py ({args.dimension}) != C++ ({args_tci.dimension})')
    if args.tci and args.active != args_tci.active:
        parser.error(f'Mismatching parameter for "--active": py ({args.active}) != C++ ({args_tci.active})')

    run_tci(args, tci_args)
    run_gp(args)


if __name__ == '__main__':
    main()
