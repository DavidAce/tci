from numba import njit
import numpy as np


class ishigami:
    def __init__(self, a=7.0, b=0.1):
        self.A, self.B = a, b
        self.name = "Ishigami"
        self.dim = 3
        self.problem = {
            "num_vars": self.dim,
            "names": ["x0", "x1", "x2"],
            "bounds": [[-np.pi, np.pi]] * self.dim,
        }
        self.variance_x0 = 0.5 * (1 + b * np.pi**4 / 5)**2
        self.variance_x1 = a**2 / 8
        self.variance_x0_x2 = 8 * b**2 * np.pi**8 / 225
        self.V = self.variance_x0 + self.variance_x1 + self.variance_x0_x2

        self.exact_mean = a / 2
        self.exact_variance = self.V
        self.exact_sobol = np.array([
            self.variance_x0, self.variance_x1, 0.0,
            self.variance_x0 + self.variance_x0_x2, self.variance_x1,
            self.variance_x0_x2, self.variance_x0_x2
        ]) / self.V

        self.SOBOL_NAMES = (
            "S0", "S1", "S2",
            "ST0", "ST1", "ST2", "S0,2"
        )

        self.sobol_pairs = [(0, 2)]
        self.include_higher = False
        self.error_slice = slice(None)
        self.kernel_lower = 1e-5
        self.kernel_upper = 1e5

    def evaluate(self, x):
        return self._evaluate(x, self.A, self.B)

    @staticmethod
    @njit(cache=True)
    def _evaluate(x, a, b):
        x0, x1, x2 = np.asarray(x).T
        return np.sin(x0) * (1 + b * x2**4) + a * np.sin(x1)**2


class morris2006:
    '''
    Morris-Moore-McKay (2006) test function.

    A polynomial sensitivity-analysis benchmark with a configurable
    number of active and inactive inputs.

    Let dimension = active+inactive = M+p

    Reference configuration:
        Original paper: M=30, p=2,...,10
        UQTestFuns example: M=6, p=5
        Current project default: M=8, p=5
    
    For p=5:
        Variance = 5.25
        S_i  = 1/5.25   for active inputs
        ST_i = 1.1/5.25 for active inputs
        S_ij = 0.025/5.25 for active pairs
        All indices involving inactive variables are zero.

    All active variables have equal first-order Sobol contributions
    and weak pairwise interactions. There are no higher-order terms.

    This is useful for testing whether a sensitivity-analysis method
    can identify influential inputs, distinguish inactive variables,
    and recover known first- and second-order Sobol indices.

    Uses the interaction coefficient beta documented by UQTestFuns
    (with the typo correction mentioned in the link)

    References:
    [1] Morris, Moore & McKay (2006), J. Stat. Plan. Inference 136, 3203-3220.
        https://doi.org/10.1016/j.jspi.2005.01.001
    [2] https://www.sfu.ca/~ssurjano/morretal06.html
    [3] https://uqtestfuns.readthedocs.io/en/stable/test-functions/morris-m.html
    '''


    def _exact_sobol(self, dim, active, alpha, beta):
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
        mean = alpha * (active / 2.0) + beta * active * (active - 1) / 8.0
        return mean, variance, np.r_[first, total, second, 0.0]

    def __init__(self, dim=8, active=5):
        if not 2 <= active <= dim:
            raise ValueError("Require 2 <= active <= dimension")

        self.name = "Morris2006"
        self.dim = dim
        self.active = active

        self.alpha = (np.sqrt(12.0) - 6.0 * np.sqrt(0.1 * (active - 1)))
        self.beta = 12.0 / np.sqrt(10.0 * (active - 1))

        self.problem = {
            'num_vars': dim,
            'names': [f'x{i}' for i in range(dim)],
            'bounds': [[0.0, 1.0] for _ in range(dim)],
        }

        self.mean, self.variance, self.exact_sobol = self._exact_sobol(self.dim, self.active, self.alpha, self.beta)

        self.sobol_pairs = [
            (i, j)
            for i in range(self.dim)
            for j in range(i + 1, self.dim)
        ]
        self.include_higher = True
        self.error_slice = slice(None, -1)
        self.kernel_lower = 1e-5
        self.kernel_upper = 1e8

    def evaluate(self, x):
        return self._evaluate(x, self.active, self.alpha, self.beta)

    @staticmethod
    @njit(cache=True)
    def _evaluate(x, active, alpha, beta):
        result = np.empty(x.shape[0], dtype=np.float64)
        for row in range(x.shape[0]):
            s = 0.0
            pairs = 0.0
            for i in range(active):
                pairs += s * x[row, i]
                s += x[row, i]
            result[row] = alpha * s + beta * pairs
        return result





class legendrePoly3:
    '''
    Third-order Legendre polynomial.

    A custom sensitivity-analysis benchmark constructed from
    products of orthonormal, first-degree shifted Legendre polynomials
    for independent uniform inputs on [0,1].


    Includes all first-, second-, and third-order interactions.
    Coefficients are deterministically generated and normalized so
    that these interaction orders contribute 20%, 30%, and 50%
    of the output variance, respectively.

    Exact properties for any dimension d >= 3:
        Mean = 1
        Variance = 1
        First-order Sobol sum  = 0.2
        Second-order Sobol sum = 0.3
        Third-order Sobol sum  = 0.5
        Higher-order Sobol sum = 0

    The exact Sobol indices are the squared expansion coefficients.
    Useful for testing sensitivity analysis in many dimensions,
    especially the recovery of distributed pairwise and three-way
    interactions and total-order effects.

    Individual Sobol indices are squared expansion coefficients.
    The coefficients are specific to this implementation and are
    not taken from Sudret (2008).

    Sudret (2008) instead studies a distinct polynomial benchmark:
    f(x) = product_i [(1 + 3*x_i*x_i)/2]
    with d=3, mean=1, variance=91/125, and exact Sobol indices
    S_i=25/91, S_ij=5/91, S_123=1/91.

    References:
    [1] Sudret (2008), Reliab. Eng. Syst. Saf. 93, 964-979.
        https://doi.org/10.1016/j.ress.2007.04.002
    [2] PolynomialChaosReporter
        https://mooseframework.inl.gov/source/reporters/PolynomialChaosReporter.html

    '''

    def _coefficients(self, dim):
        """Coefficients and normalization from LegendrePoly3.cpp (0-based i,j,k)."""
        if dim < 3:
            raise ValueError('LegendrePoly3 requires at least three variables')

        a = np.zeros(dim)
        b = np.zeros((dim, dim))
        c = np.zeros((dim, dim, dim))

        for i in range(dim):
            u = i + 1.0
            a[i] = np.sin(1.7 * u + 0.3 * u * u)
            for j in range(i + 1, dim):
                v = j + 1.0
                b[i, j] = np.sin(0.83 * u * v + 0.37 * j)
                for k in range(j + 1, dim):
                    w = k + 1.0
                    c[i, j, k] = np.cos(0.31 * u * v * w + 0.7 * (i + j + k + 3))

        a *= np.sqrt(0.2 / np.sum(a**2))
        b *= np.sqrt(0.3 / np.sum(b**2))
        c *= np.sqrt(0.5 / np.sum(c**2))
        return a, b, c

    def _exact_sobol(self, a, b, c):
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

    def __init__(self, dim=8):
        self.name = "LegendrePoly3"
        self.dim = dim
        self.active = -1

        self.a, self.b, self.c = self._coefficients(self.dim)

        self.exact_mean = 1.0
        self.exact_variance = 1.0
        self.exact_sobol = self._exact_sobol(self.a, self.b, self.c)

        self.problem = {
            'num_vars': dim,
            'names': [f'x{i}' for i in range(dim)],
            'bounds': [[0.0, 1.0] for _ in range(dim)],
        }

        self.sobol_pairs = [
            (i, j)
            for i in range(self.dim)
            for j in range(i + 1, self.dim)
        ]
        self.include_higher = True
        self.error_slice = slice(None)
        self.kernel_lower = 1e-5
        self.kernel_upper = 1e5

    def evaluate(self, x):
        return self._evaluate(x, self.a, self.b, self.c)

    @staticmethod
    @njit(cache=True)
    def _evaluate(x, a, b, c):
        """Batched function evaluation: x.shape = (n_samples, dim), x in [0,1]."""
        n, dim = x.shape
        result = np.empty(n, dtype=np.float64)
        z = np.empty(dim, dtype=np.float64)

        for row in range(n):
            for i in range(dim):
                z[i] = np.sqrt(12.0) * (x[row, i] - 0.5)

            value = 1.0
            for i in range(dim):
                value += a[i] * z[i]
                for j in range(i + 1, dim):
                    value += b[i, j] * z[i] * z[j]
                    for k in range(j + 1, dim):
                        value += c[i, j, k] * z[i] * z[j] * z[k]
            result[row] = value

        return result
