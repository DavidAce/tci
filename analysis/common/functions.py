from numba import njit
import numpy as np

class ishigami:
    A, B = 7.0, 0.1
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    SOBOL_NAMES = ("S1", "S2", "S3", "ST1", "ST2", "ST3", "S1,3")
    V1 = 0.5 * (1.0 + B * np.pi**4 / 5.0)**2
    V2 = A**2 / 8.0
    V13 = 8.0 * B**2 * np.pi**8 / 225.0
    V = V1 + V2 + V13
    EXACT = np.array([V1, V2, 0.0, V1 + V13, V2, V13, V13]) / V

    @njit(cache=True)
    def evaluate(x):
        x1, x2, x3 = np.asarray(x).T
        return np.sin(x1) * (1.0 + B * x3**4) + A * np.sin(x2)**2



class morris2006:
    dim = 8
    problem = {
        'num_vars': dim,
        'names': [f'x{i+1}' for i in range(dim)],
        'bounds': [[0.0, 1.0] for _ in range(dim)],
    }

    @njit(cache=True)
    def morris2006(x, active, alpha, beta):
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
    
    def __coefficients(dim):
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
   
    @njit(cache=True)
    def evaluate(x, a, b, c):
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
