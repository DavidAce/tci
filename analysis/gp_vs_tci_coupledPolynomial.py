#!/usr/bin/env python3
"""CoupledPolynomial: C++ TCI versus Gaussian-process Sobol indices.

Faithfully reproduces source/func/CoupledPolynomial.cpp in DavidAce/tci.
The C++ main.cpp currently uses CoupledPolynomial(20).

Requirements: numpy, scipy, scikit-learn, SALib, numba.
Everything after -- is passed verbatim to the C++ TCI executable.
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze


def coefficients(dim):
    """Coefficients and normalization from CoupledPolynomial.cpp (0-based i,j,k)."""
    if dim < 3:
        raise ValueError('CoupledPolynomial requires at least three variables')

    a = np.zeros(dim)
    b = np.zeros((dim, dim))
    c = np.zeros((dim, dim, dim))

    for i in range(dim):
        u = i + 1.0
        a[i] = math.sin(1.7 * u + 0.3 * u * u)
        for j in range(i + 1, dim):
            v = j + 1.0
            b[i, j] = math.sin(0.83 * u * v + 0.37 * j)
            for k in range(j + 1, dim):
                w = k + 1.0
                c[i, j, k] = math.cos(0.31 * u * v * w + 0.7 * (i + j + k + 3))

    a *= math.sqrt(0.2 / np.sum(a**2))
    b *= math.sqrt(0.3 / np.sum(b**2))
    c *= math.sqrt(0.5 / np.sum(c**2))
    return a, b, c


@njit(cache=True)
def coupled_polynomial(x, a, b, c):
    """Batched function evaluation: x.shape = (n_samples, dim), x in [0,1]."""
    n, dim = x.shape
    result = np.empty(n, dtype=np.float64)
    z = np.empty(dim, dtype=np.float64)

    for row in range(n):
        for i in range(dim):
            z[i] = math.sqrt(12.0) * (x[row, i] - 0.5)

        value = 1.0
        for i in range(dim):
            value += a[i] * z[i]
            for j in range(i + 1, dim):
                value += b[i, j] * z[i] * z[j]
                for k in range(j + 1, dim):
                    value += c[i, j, k] * z[i] * z[j] * z[k]
        result[row] = value

    return result


def exact_sobol(a, b, c):
    """Vector ordered as S_i, ST_i, S_ij (i<j), S_ij..."""
    dim = len(a)
    first = a**2
    total = first.copy()
    for i in range(dim):
        for j in range(i + 1, dim):
            s2 = b[i, j] ** 2
            total[i] += s2
            total[j] += s2
            for k in range(j + 1, dim):
                s3 = c[i, j, k] ** 2
                total[i] += s3
                total[j] += s3
                total[k] += s3

    ii, jj = np.triu_indices(dim, k=1)
    second = b[ii, jj] ** 2
    higher = 1.0 - first.sum() - second.sum()
    assert np.isclose(higher, 0.5, atol=1e-12)
    return np.r_[first, total, second, higher]


def evaluate_gp_sobol(gp, problem, base_size, seed, batch_size=8192):
    """SALib Saltelli estimator: first-, total-, second-order indices."""
    dim = problem['num_vars']
    x = sobol_sample(problem, base_size, calc_second_order=True, seed=seed)
    y = np.empty(len(x))
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        y[start:stop] = gp.predict(x[start:stop])

    s = sobol_analyze(
        problem, y, calc_second_order=True,
        num_resamples=100, seed=seed,
        print_to_console=False, parallel=False,
    )
    ii, jj = np.triu_indices(dim, k=1)
    first = s['S1']
    second = s['S2'][ii, jj]
    higher = 1.0 - np.sum(first) - np.sum(second)
    indices = np.r_[first, s['ST'], second, higher]

    # Reuse the A and B samples of the Saltelli design for mean and variance.
    stride = 2 * dim + 2
    values = np.r_[y[0::stride], y[stride - 1::stride]]
    return values.mean(), values.var(), indices


def print_index_summary(indices, exact, dim):
    n_pairs = dim * (dim - 1) // 2
    print('\nFinal GP Sobol indices (first and total):')
    print(f"{'index':<10} {'estimated':>14} {'exact':>14} {'absError':>12}")
    for i in range(dim):
        print(f'S{i+1:<9} {indices[i]:14.8f} {exact[i]:14.8f} '
              f'{abs(indices[i]-exact[i]):12.3e}')
    for i in range(dim):
        k = dim + i
        print(f'ST{i+1:<8} {indices[k]:14.8f} {exact[k]:14.8f} '
              f'{abs(indices[k]-exact[k]):12.3e}')

    print('\nSobol index groups:')
    print(f"{'group':<10} {'count':>6} {'sum est.':>14} {'sum exact':>14} "
          f"{'absError':>12} {'RMSE':>12}")
    for label, section in (
        ('S_i', slice(0, dim)),
        ('S_ij', slice(2*dim, 2*dim+n_pairs)),
        ('S_ijk...', slice(-1, None)),
    ):
        estimate = indices[section]
        truth = exact[section]
        diff = estimate - truth
        print(f'{label:<10} {len(estimate):6d} {estimate.sum():14.8f} '
              f'{truth.sum():14.8f} {abs(diff.sum()):12.3e} '
              f'{np.sqrt(np.mean(diff**2)):12.3e}')


def run_gp(args):
    dim = args.dimension
    sizes = args.gp_sizes
    if any(n < 2 or n & (n-1) for n in sizes):
        raise ValueError('GP training sizes must be powers of two >= 2')
    if args.gp_sobol_base < 2 or args.gp_sobol_base & (args.gp_sobol_base-1):
        raise ValueError('--gp-sobol-base must be a power of two >= 2')

    a, b, c = coefficients(dim)
    exact = exact_sobol(a, b, c)
    problem = {
        'num_vars': dim,
        'names': [f'x{i+1}' for i in range(dim)],
        'bounds': [[0.0, 1.0]] * dim,
    }

    # Nested scrambled-Sobol designs: first n samples used for each GP fit.
    u = qmc.Sobol(d=dim, scramble=True, seed=args.gp_seed).random_base2(
        m=max(sizes).bit_length()-1
    )
    y = coupled_polynomial(u, a, b, c)

    print(f'=== CoupledPolynomial / GP (dim={dim}, noiseless) ===')
    print(f'Training: scrambled Sobol | seed={args.gp_seed} | noise=0')
    print('Kernel: ConstantKernel(1, [1e-5,1e5]) * RBF(1, [1e-5,1e5]) [isotropic]')
    print(f'GP: normalize_y=True | alpha={args.alpha:g} | restarts={args.gp_restarts}')
    print(f'SALib: base={args.gp_sobol_base} | '
          f'predictions={(2*dim+2)*args.gp_sobol_base} | second-order=True')
    print('Exact: mean=1, variance=1, sum(S1)=0.2, sum(S2)=0.3, S_>=3=0.5')
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxAbs':>12} "
          f"{'S_ij...':>12} {'mean':>12} {'variance':>12} {'fit(s)':>9} {'sobol(s)':>9}")

    for n in sizes:
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e5)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=args.alpha, normalize_y=True,
            n_restarts_optimizer=args.gp_restarts, random_state=args.gp_seed,
        )
        t0 = time.perf_counter()
        gp.fit(u[:n], y[:n])
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        mu, var, indices = evaluate_gp_sobol(
            gp, problem, args.gp_sobol_base, args.gp_seed
        )
        sobol_time = time.perf_counter() - t0
        delta = indices - exact
        rmse = float(np.sqrt(np.mean(delta**2)))
        max_abs = float(np.max(np.abs(delta)))
        print(f"{'GP':<10} {n:7d} {rmse:12.4e} {max_abs:12.4e} "
              f'{indices[-1]:12.6e} {mu:12.6e} {var:12.6e} '
              f'{fit_time:9.2e} {sobol_time:9.2e}', flush=True)

    print_index_summary(indices, exact, dim)
    print('\nRMSE and maxAbs include S_i, ST_i, S_ij, and one S_ij... residual.')
    print('nEval counts original-function GP training evaluations; '
          'SALib GP predictions do not add to nEval.')


def run_tci(args, tci_args):
    if args.tci is None:
        return
    cmd = [str(args.tci.expanduser().resolve()), '--function', 'cpoly', *tci_args]
    print(f"=== C++ TCI === [{' '.join(cmd)}]", flush=True)
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
    parser.add_argument('--gp-seed', type=int, default=12572601)
    parser.add_argument('--gp-sizes', nargs='+', type=int, default=[128, 256, 512, 1024])
    parser.add_argument('--gp-sobol-base', type=int, default=2**14)
    parser.add_argument('--gp-restarts', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1e-8)
    args = parser.parse_args(python_args)

    if args.dimension < 3:
        parser.error('--dimension must be >= 3')
    if args.alpha < 0 or args.gp_restarts < 0:
        parser.error('alpha and gp-restarts must be nonnegative')
    if tci_args and args.tci is None:
        parser.error('TCI arguments supplied without --tci')
    if any(s == '--function' or s.startswith('--function=') for s in tci_args):
        parser.error('C++ --function is automatically set to cpoly')
    if args.tci and args.dimension != 8:
        parser.error('C++ main.cpp currently uses CoupledPolynomial(20); '
                     'set --dimension 20 or change the C++ dimension')

    run_tci(args, tci_args)
    run_gp(args)


if __name__ == '__main__':
    main()
