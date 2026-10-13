#!/usr/bin/env python3
"""Ishigami Sobol indices: Gaussian process vs optional C++ TCI run.

Requires: numpy, scipy, scikit-learn.
The C++ executable, when requested, must call cli::parse(argc, argv).
"""

import argparse
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


A = 7.0
B = 0.1

EXACT_MEAN = A / 2
V1 = 0.5 * (1 + B * math.pi**4 / 5)**2
V2 = A**2 / 8
V13 = 8 * B**2 * math.pi**8 / 225
EXACT_VARIANCE = V1 + V2 + V13
NAMES = ("S1", "S2", "S3", "ST1", "ST2", "ST3", "S1,3")
EXACT = np.array([
    V1, V2, 0., V1 + V13, V2, V13, V13
]) / EXACT_VARIANCE


def ishigami(x: np.ndarray) -> np.ndarray:
    """x: shape (n,3), uniform coordinates in [-pi, pi]."""
    x1, x2, x3 = x.T
    return np.sin(x1) * (1 + B*x3**4) + A*np.sin(x2)**2


def integrate_gp(gp: GaussianProcessRegressor, n: int) -> tuple[float, float, np.ndarray]:
    """Sobol indices of GP posterior mean using an n^3 Gauss-Legendre grid.

    All integrals are expectations under independent U[-pi,pi] inputs.
    """
    t, w = np.polynomial.legendre.leggauss(n)
    x = math.pi * t
    w = w / 2  # normalized probability weights on [-pi, pi]

    xx = np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(-1, 3)
    y = np.empty(len(xx))
    for start in range(0, len(xx), 4096):
        stop = min(start + 4096, len(xx))
        y[start:stop] = gp.predict(xx[start:stop])
    f = y.reshape(n, n, n)

    mu = np.einsum("i,j,k,ijk->", w, w, w, f, optimize=True)
    second = np.einsum("i,j,k,ijk->", w, w, w, f*f, optimize=True)
    variance = second - mu*mu
    if variance <= 0:
        raise ArithmeticError("Nonpositive GP output variance")

    # m_i(x_i) = E[f | x_i].
    m1 = np.einsum("j,k,ijk->i", w, w, f, optimize=True)
    m2 = np.einsum("i,k,ijk->j", w, w, f, optimize=True)
    m3 = np.einsum("i,j,ijk->k", w, w, f, optimize=True)
    v1, v2, v3 = (
        np.dot(w, m*m) - mu*mu for m in (m1, m2, m3)
    )

    # g_i(x_{-i}) = E[f | x_{-i}].
    g1 = np.einsum("i,ijk->jk", w, f, optimize=True)
    g2 = np.einsum("j,ijk->ik", w, f, optimize=True)
    g3 = np.einsum("k,ijk->ij", w, f, optimize=True)
    st1 = (second - np.einsum("j,k,jk->", w, w, g1*g1, optimize=True)) / variance
    st2 = (second - np.einsum("i,k,ik->", w, w, g2*g2, optimize=True)) / variance
    st3 = (second - np.einsum("i,j,ij->", w, w, g3*g3, optimize=True)) / variance

    # V_{1,3} = Var(E[f|x1,x3]) - V1 - V3.
    v13 = np.einsum("i,k,ik->", w, w, g2*g2, optimize=True) - mu*mu - v1 - v3

    indices = np.array([v1/variance, v2/variance, v3/variance,
                        st1, st2, st3, v13/variance])
    return float(mu), float(variance), indices


def run_tci(args: argparse.Namespace) -> None:
    if args.tci is None:
        return
    cmd = [
        str(args.tci.expanduser().resolve()),
        "--function", "ishigami",
        "--quadrature", args.quadrature,
        "--nnodes", str(args.nnodes),
        "--bond-dim", str(args.bond_dim),
        "--max-sweeps", str(args.max_sweeps),
        "--seed", str(args.seed),
    ]
    print(f"=== C++ TCI === [{" ".join(cmd)}]", flush=True)
    subprocess.run(cmd, check=True)
    print(flush=True)


def run_gp(args: argparse.Namespace) -> None:
    sizes = args.gp_sizes
    if any(n <= 0 or n & (n - 1) for n in sizes):
        raise ValueError("GP training sizes must be positive powers of two")
    if args.gp_quad_nodes < 1:
        raise ValueError("GP quadrature nodes must be positive")

    # Nested scrambled Sobol designs: increasing nEval reuses earlier points.
    design = qmc.Sobol(d=3, scramble=True, seed=args.seed).random_base2(
        m=max(sizes).bit_length() - 1
    )
    x_all = qmc.scale(design, [-math.pi]*3, [math.pi]*3)
    y_all = ishigami(x_all)

    print("=== Ishigami / GP (ARD squared-exponential) ===")
    print(f"Inputs: independent U[-pi, pi], a={A:g}, b={B:g}")
    print(f"Training: scrambled Sobol, seed={args.seed}")
    print(f"Sobol integration: {args.gp_quad_nodes} Gauss-Legendre nodes per variable")
    print(f"Exact: mean={EXACT_MEAN:.8f}, variance={EXACT_VARIANCE:.8f}")
    print(f"{'method':<10} {'nEval':>7} {'RMSE':>12} {'maxAbs':>12} {'mean':>13} {'variance':>13} {'fit(s)':>9}")

    last = None
    for n in sizes:
        kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(
            length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-8,
            normalize_y=True, n_restarts_optimizer=3, random_state=args.seed,
        )
        tic = time.perf_counter()
        gp.fit(x_all[:n], y_all[:n])
        fit_seconds = time.perf_counter() - tic
        mean, variance, indices = integrate_gp(gp, args.gp_quad_nodes)
        errors = np.abs(indices - EXACT)
        rmse = float(np.sqrt(np.mean(errors**2)))
        max_abs = float(np.max(errors))
        print(f"{'GP':<10} {n:7d} {rmse:12.4e} {max_abs:12.4e} "
              f"{mean:13.6e} {variance:13.6e} {fit_seconds:9.2f}", flush=True)
        last = indices

    print("\nFinal GP Sobol indices:")
    print(f"{'index':<8} {'estimated':>14} {'exact':>14} {'absError':>12}")
    for name, estimated, exact in zip(NAMES, last, EXACT):
        print(f"{name:<8} {estimated:14.8f} {exact:14.8f} {abs(estimated-exact):12.3e}")
    print("\nnEval counts only evaluations of the original Ishigami function used for GP training.")
    print("GP predictions on the integration grid are not counted as function evaluations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tci", type=Path, help="Optional path to C++ TCI executable with CLI11 flags")
    parser.add_argument("--quadrature", choices=("gk15", "gl", "cc"), default="gk15")
    parser.add_argument("--nnodes", type=int, default=15, help="TCI nodes per input (GK15 always uses 15)")
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gp-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--gp-quad-nodes", type=int, default=32,
                        help="1D integration nodes for GP Sobol indices (not GP training nodes)")
    args = parser.parse_args()
    if args.tci and args.quadrature == "gk15" and args.nnodes != 15:
        parser.error("GK15 uses exactly 15 nodes")
    if args.tci and (args.nnodes < 1 or args.bond_dim < 1 or args.max_sweeps < 1):
        parser.error("TCI node count, bond dimension and half-sweeps must be positive")
    run_tci(args)
    run_gp(args)


if __name__ == "__main__":
    main()
