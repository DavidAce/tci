
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

SEED = 12572601
NEVALS = [128, 256, 512]
N_SURROGATE = 4096  # Sobol base sample size for GP predictions

problem = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}


def ishigami(X):
    x1, x2, x3 = X.T
    return np.sin(x1) * (1 + 0.1*x3**4) + 7*np.sin(x2)**2


def sensitivity(X, y):
    result = sobol_analyze.analyze(
        problem, y,
        calc_second_order=True,
        num_resamples=50,
        seed=SEED,
        print_to_console=False,
    )
    return np.array([
        *result["S1"], *result["ST"], result["S2"][0, 2]
    ])


def exact_indices():
    a, b = 7.0, 0.1
    v1 = 0.5 * (1 + b*np.pi**4/5)**2
    v2 = a*a / 8
    v13 = 8*b*b*np.pi**8 / 225
    v = v1 + v2 + v13
    return np.array([
        v1/v, v2/v, 0,
        (v1+v13)/v, v2/v, v13/v,
        v13/v
    ])


def train_gp(n):
    # Generate n training points in [-pi,pi]^3
    sampler = qmc.Sobol(d=3, scramble=True, seed=SEED)
    X = qmc.scale(
        sampler.random_base2(int(np.log2(n))),
        -np.pi, np.pi
    )
    y = ishigami(X)

    # Kernel from the original repository
    kernel = ConstantKernel(1.0, (1e-5, 1e5)) * \
             RBF(1.0, (1e-5, 1e5))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=SEED,
    )
    gp.fit(X, y)
    return gp


def main():
    exact = exact_indices()

    # GP predictions do not require Ishigami evaluations.
    X_sur = sobol_sample.sample(
        problem, N_SURROGATE,
        calc_second_order=True,
        seed=SEED
    )

    names = ["S1", "S2", "S3", "ST1", "ST2", "ST3", "S13"]

    print("Exact indices:")
    for name, value in zip(names, exact):
        print(f"  {name:4s}: {value:.8f}")

    print("\nComparing GP and direct Sobol:")
    print(f"{'nEval':>6} {'method':>8} {'RMSE':>12} "
          + " ".join(f"{s:>10}" for s in names))

    for n in NEVALS:
        # Method 1: Direct Sobol, using exactly n calls to f.
        # For D=3, second-order sampling costs 8*N.
        X = sobol_sample.sample(
            problem, n // 8,
            calc_second_order=True,
            seed=SEED
        )
        y = ishigami(X)
        direct = sensitivity(X, y)

        # Method 2: Train GP with exactly n evaluations.
        gp = train_gp(n)
        y_sur = gp.predict(X_sur)
        surrogate = sensitivity(X_sur, y_sur)

        for method, values in [
            ("Sobol", direct),
            ("GP", surrogate)
        ]:
            rmse = np.sqrt(np.mean((values - exact)**2))
            values_str = " ".join(f"{v:10.6f}" for v in values)
            print(f"{n:6d} {method:>8} {rmse:12.5e} "
                  + values_str)


if __name__ == "__main__":
    main()
