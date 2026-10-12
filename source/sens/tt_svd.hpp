#pragma once

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xfac/tensor/tensor_train.h>

// Normalized six-point Gauss-Legendre rule for a uniform input on [lo, hi].
inline std::pair<std::vector<double>, std::vector<double>> quadratureGL6(double lo, double hi) {
    constexpr std::array<double, 6> nodes              = {-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                                                          0.2386191860831969,  0.6612093864662645,  0.9324695142031521};
    constexpr std::array<double, 6> integrationWeights = {0.1713244923791703, 0.3607615730481386, 0.4679139345726910,
                                                          0.4679139345726910, 0.3607615730481386, 0.1713244923791703};
    std::vector<double>             x(6), w(6);
    for(std::size_t i = 0; i < 6; ++i) {
        x[i] = (lo + hi) / 2.0 + (hi - lo) / 2.0 * nodes[i];
        w[i] = integrationWeights[i] / 2.0;
    }
    return {std::move(x), std::move(w)};
}

struct TTSVDResult {
    xfac::TensorTrain<double> tt;
    double                    relativeL2Error = 0.0;
    std::size_t               nEval           = 0;
    arma::vec                 weightedValues;
};

template<class Function>
TTSVDResult makeTTSVD(Function &&f, const std::vector<std::vector<double>> &grids,
                      const std::vector<std::vector<double>> &weights, arma::uword maxBondDim) {
    const auto d = grids.size();
    if(d == 0 || weights.size() != d || maxBondDim == 0) throw std::invalid_argument("Invalid TT-SVD arguments");

    std::size_t nEval = 1;
    for(std::size_t k = 0; k < d; ++k) {
        if(grids[k].empty() || weights[k].size() != grids[k].size())
            throw std::invalid_argument("Inconsistent grid dimensions");
        nEval *= grids[k].size();
    }

    // First variable changes fastest. Include sqrt(quadrature weights),
    // so the SVD minimizes the quadrature-weighted squared error.
    arma::vec           values(nEval);
    std::vector<double> x(d);
    for(std::size_t linear = 0; linear < nEval; ++linear) {
        std::size_t n          = linear;
        double      sqrtWeight = 1.0;
        for(std::size_t k = 0; k < d; ++k) {
            const auto s = n % grids[k].size();
            n /= grids[k].size();
            x[k] = grids[k][s];
            sqrtWeight *= std::sqrt(weights[k][s]);
        }
        values(linear) = f(x) * sqrtWeight;
    }

    arma::vec originalWeightedValues = values;

    const double normSquared = arma::dot(values, values);
    if(normSquared == 0.0) throw std::invalid_argument("The function vanishes on the entire grid");

    xfac::TensorTrain<double> tt(d);
    arma::uword               leftDim          = 1;
    arma::uword               rightSize        = values.n_elem;
    double                    discardedSquared = 0.0;

    for(std::size_t k = 0; k + 1 < d; ++k) {
        const arma::uword localDim = grids[k].size();
        rightSize /= localDim;

        // Column-major layout: row index = leftBond + leftDim * physicalIndex.
        arma::mat A(values.memptr(), leftDim * localDim, rightSize, false, true);
        arma::mat U, V;
        arma::vec singularValues;
        if(!arma::svd_econ(U, singularValues, V, A, "both", "std")) throw std::runtime_error("TT-SVD failed");

        const arma::uword bond = std::min(maxBondDim, singularValues.n_elem);
        for(arma::uword i = bond; i < singularValues.n_elem; ++i)
            discardedSquared += singularValues(i) * singularValues(i);

        arma::mat left = U.head_cols(bond);
        tt.M[k]        = arma::cube(left.memptr(), leftDim, localDim, bond, true);

        arma::mat right = arma::diagmat(singularValues.head(bond)) * V.head_cols(bond).t();
        values          = arma::vectorise(right);
        leftDim         = bond;
    }

    tt.M[d - 1] = arma::cube(values.memptr(), leftDim, grids.back().size(), 1, true);

    // Restore the unweighted function values.
    for(std::size_t k = 0; k < d; ++k) {
        auto &M = tt.M[k];
        for(arma::uword s = 0; s < M.n_cols; ++s) {
            const double factor = std::sqrt(weights[k][s]);
            for(arma::uword b = 0; b < M.n_slices; ++b)
                for(arma::uword a = 0; a < M.n_rows; ++a) M(a, s, b) /= factor;
        }
    }

    return {std::move(tt), std::sqrt(discardedSquared / normSquared), nEval, std::move(originalWeightedValues)};
}
