#!/usr/bin/env python3
"""Compare direct Sobol and GP sensitivity estimates for noiseless Ishigami.

Run: python ishigami_benchmark.py --target 1e-3 --max-sobol 131072 --max-gp 512
The saved CSV uses the same seven-index scoring convention as the xfac C++ snippet.
"""
import argparse
import csv
import math

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

SEED = 12572601
NAMES = ("S1", "S2", "S3", "ST1", "ST2", "ST3", "S13")
FIELDS = (["method", "nEval", "rmse", "maxAbs"]
          + list(NAMES) + ["err" + s for s in NAMES])
PROBLEM = {"num_vars": 3, "names": ["x1", "x2", "x3"],
           "bounds": [[-np.pi, np.pi]] * 3}


def ishigami(X):
    x1, x2, x3 = np.asarray(X).T
    return np.sin(x1) * (1.0 + 0.1 * x3**4) + 7.0 * np.sin(x2)**2


def reference():
    p4 = np.pi**4
    v1 = 0.5 * (1.0 + 0.1*p4/5.0)**2
    v2 = 7.0**2 / 8.0
    v13 = 8.0 * 0.1**2 * p4**2 / 225.0
    v = v1 + v2 + v13
    return np.array([v1/v, v2/v, 0.0, (v1+v13)/v,
                     v2/v, v13/v, v13/v])


EXACT = reference()


def record(method, n_eval, estimates):
    estimates = np.asarray(estimates, dtype=float)
    errors = np.abs(estimates - EXACT)
    result = {"method": method, "nEval": int(n_eval),
              "rmse": float(np.sqrt(np.mean(errors**2))),
              "maxAbs": float(np.max(errors))}
    result.update(zip(NAMES, estimates.tolist()))
    result.update(zip(("err" + s for s in NAMES), errors.tolist()))
    return result


def gk15_grid():
    # Exactly the abscissae and normalized weights of xfac::QuadratureGK15(-pi,pi).
    z = np.array([0.0, .2077849550078985, .4058451513773972,
                  .5860872354676911, .7415311855993945,
                  .8648644233597691, .9491079123427585,
                  .9914553711208126])
    w = np.array([.2094821410847278, .2044329400752989,
                  .1903505780647854, .1690047266392679,
                  .1406532597155259, .1047900103222502,
                  .06309209262997856, .02293532201052922])
    nodes = np.pi * np.r_[-z[:0:-1], z]
    weights = np.r_[w[:0:-1], w] / 2.0
    mesh = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1)
    return mesh.reshape(-1, 3), weights


def grid_indices(values, w):
    """Seven Sobol indices via weighted 3D quadrature, not SALib resampling."""
    F = np.asarray(values).reshape(15, 15, 15)
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    mu = np.sum(W * F)
    variance = np.sum(W * F * F) - mu**2

    def conditional_variance(keep):
        g = F
        for axis in (2, 1, 0):
            if axis not in keep:
                g = np.tensordot(g, w, axes=([axis], [0]))
        g2 = g * g
        while np.ndim(g2) != 0:
            g2 = np.tensordot(g2, w, axes=([0], [0]))
        return float(g2 - mu**2)

    vi = np.array([conditional_variance((i,)) for i in range(3)])
    vminus = np.array([conditional_variance(tuple(j for j in range(3)
                                                   if j != i)) for i in range(3)])
    v13 = conditional_variance((0, 2)) - vi[0] - vi[2]
    return np.r_[vi / variance, (variance - vminus) / variance,
                 v13 / variance]


def direct_sobol(max_evals, target, output):
    # Each Saltelli base point produces 8 Ishigami evaluations at D=3.
    X = sobol_sample.sample(PROBLEM, max_evals // 8,
                            calc_second_order=True, seed=SEED)
    Y = np.empty(max_evals)
    previous = 0
    for n in (2**k for k in range(7, int(math.log2(max_evals)) + 1)):
        # Reuse previously evaluated samples. Only count NEW calls to Ishigami.
        Y[previous:n] = ishigami(X[previous:n])
        previous = n
        si = sobol_analyze.analyze(PROBLEM, Y[:n], calc_second_order=True,
                                   num_resamples=32, seed=SEED,
                                   print_to_console=False)
        estimates = np.r_[si["S1"], si["ST"], si["S2"][0, 2]]
        row = record("Sobol", n, estimates)
        output.append(row)
        display(row)
        if row["rmse"] <= target:
            break


def gp_method(max_evals, target, gp_restarts, output):
    # Nested scrambled Sobol training points, independent of the Saltelli set.
    design = qmc.Sobol(d=3, scramble=True, seed=SEED).random_base2(
        int(math.log2(max_evals)))
    X = qmc.scale(design, [-np.pi]*3, [np.pi]*3)
    Y = np.empty(max_evals)
    previous = 0
    grid, weights = gk15_grid()
    for n in (2**k for k in range(5, int(math.log2(max_evals)) + 1)):
        Y[previous:n] = ishigami(X[previous:n])
        previous = n
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8,
                                     normalize_y=True,
                                     n_restarts_optimizer=gp_restarts,
                                     random_state=SEED)
        gp.fit(X[:n], Y[:n])
        # GP predictions are NOT expensive original-function evaluations.
        estimates = grid_indices(gp.predict(grid), weights)
        row = record("GP", n, estimates)
        output.append(row)
        display(row)
        if row["rmse"] <= target:
            break


def display(row):
    errs = " ".join(f"{row['err' + s]:9.2e}" for s in NAMES)
    print(f"{row['method']:<6} {row['nEval']:>8} {row['rmse']:>10.3e} "
          f"{row['maxAbs']:>10.3e} {errs}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=1e-2)
    parser.add_argument("--max-sobol", type=int, default=131072)
    parser.add_argument("--max-gp", type=int, default=512)
    parser.add_argument("--gp-restarts", type=int, default=1)
    parser.add_argument("--csv", default="ishigami_results.csv")
    parser.add_argument("--tci-log", help="C++ stdout containing one TCI2 CSV row")
    parser.add_argument("--plot", action="store_true", help="save two PNG comparison plots")
    args = parser.parse_args()

    for label, n, min_n in (("--max-sobol", args.max_sobol, 128),
                            ("--max-gp", args.max_gp, 32)):
        if n < min_n or (n & (n - 1)):
            parser.error(f"{label} must be a power of two >= {min_n}")
    if not (0 < args.target < 1):
        parser.error("--target must be between zero and one")

    rows = []
    print(f"Target RMSE: {args.target:g}. Columns err* are absolute index errors.")
    print(f"{'method':<6} {'nEval':>8} {'RMSE':>10} {'maxAbs':>10} "
          + " ".join(f"{'err'+s:>9}" for s in NAMES))
    direct_sobol(args.max_sobol, args.target, rows)
    gp_method(args.max_gp, args.target, args.gp_restarts, rows)

    if args.tci_log:
        with open(args.tci_log) as file:
            lines = [line.strip() for line in file if line.startswith("TCI2,")]
        if len(lines) != 1:
            parser.error("--tci-log must contain exactly one line beginning TCI2,")
        fields = lines[0].split(",")
        if len(fields) != len(FIELDS):
            parser.error(f"TCI2 row has {len(fields)} columns; expected {len(FIELDS)}")
        tci = dict(zip(FIELDS, fields))
        tci["nEval"] = int(tci["nEval"])
        for field in FIELDS[2:]:
            tci[field] = float(tci[field])
        rows.append(tci)
        display(tci)

    with open(args.csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFirst observed neval reaching each RMSE target:")
    thresholds = sorted(({args.target} | {t for t in (1e-1, 1e-2, 1e-3, 1e-4)
                                            if t >= args.target}), reverse=True)
    methods = ("Sobol", "GP") + (("TCI2",) if args.tci_log else ())
    for threshold in thresholds:
        for method in methods:
            matches = [r for r in rows if r["method"] == method
                       and r["rmse"] <= threshold]
            first = f"{matches[0]['nEval']}" if matches else "not reached"
            print(f"  {method:<6} RMSE <= {threshold:8.1e}: {first}")

    print(f"\nSaved {args.csv}. GP and TCI2 use the same GK15 integration rule.")
    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for method in methods:
            r = [row for row in rows if row["method"] == method]
            ax.loglog([x["nEval"] for x in r], [x["rmse"] for x in r],
                      marker="o", label=method)
        ax.set(xlabel="Evaluations of Ishigami function", ylabel="RMSE of seven indices")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("ishigami_rmse_vs_evals.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots()
        for method in methods:
            reached = [(t, next((r["nEval"] for r in rows
                                if r["method"] == method and r["rmse"] <= t), None))
                       for t in thresholds]
            reached = [(t, n) for t, n in reached if n is not None]
            if reached:
                ax.loglog([t for t, n in reached], [n for t, n in reached],
                          marker="o", label=method)
        ax.invert_xaxis()
        ax.set(xlabel="Target RMSE (smaller is better)",
               ylabel="First evaluated neval reaching target")
        ax.grid(True, which="both", alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("ishigami_evals_vs_target.png", dpi=180)
        plt.close(fig)
        print("Saved ishigami_rmse_vs_evals.png and ishigami_evals_vs_target.png")


if __name__ == "__main__":
    main()
