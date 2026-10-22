#include "sobol.hpp"
#include <armadillo>
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <h5pp/details/h5ppFstr.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <xfac/tensor/tensor_train.h>

namespace sens {

    // Contract the TT with its conjugate after averaging selected variables.
    double conditional_mean_square(const xfac::TensorTrain<double> &tt, const IntegrationWeights &weights,
                                   const std::vector<bool> &integrateOut) {
        const int         dim = static_cast<int>(weights.size());
        arma::Mat<double> G(1, 1, arma::fill::ones);
        for(int k = 0; k < dim; ++k) {
            const auto &M = tt.M[k];
            if(integrateOut[k]) {
                arma::Mat<double> B(M.n_rows, M.n_slices, arma::fill::zeros);
                for(arma::uword s = 0; s < M.n_cols; ++s) B += weights[k][s] * arma::Mat<double>(M.col(s));
                G = B.t() * G * B;
            } else {
                arma::Mat<double> next(M.n_slices, M.n_slices, arma::fill::zeros);
                for(arma::uword s = 0; s < M.n_cols; ++s) {
                    arma::Mat<double> A(M.col(s));
                    next += weights[k][s] * (A.t() * G * A);
                }
                G = std::move(next);
            }
        }
        return G(0, 0);
    }

    double maximumSobolChange(const SobolResult &current, const SobolResult &previous) {
        if(current.sobolIndices.size() != previous.sobolIndices.size()) return std::numeric_limits<double>::quiet_NaN();
        double delta = 0;
        for(std::size_t i = 0; i < current.sobolIndices.size(); ++i)
            delta = std::max(delta, std::abs(current.sobolIndices[i].estimated - previous.sobolIndices[i].estimated));
        return delta;
    }

    bool hasExactSobolIndices(const SobolReference &reference, int dim) {
        return reference.first.size() == static_cast<std::size_t>(dim) &&
               reference.total.size() == static_cast<std::size_t>(dim);
    }

    void appendSobolIndex(std::vector<SobolIndex> &indices, const std::string &name, double estimated, double exact,
                          bool hasExact, double &maxDiff) {
        const double error = hasExact ? std::abs(estimated - exact) : std::numeric_limits<double>::quiet_NaN();
        indices.push_back({h5pp::fstr_t<32>(name), estimated, exact, error});
        if(hasExact) maxDiff = std::max(maxDiff, error);
    }

    SobolResult calculateSobolIndices(const xfac::TensorTrain<double> &tt, const IntegrationWeights &weights,
                                      const Statistics &statistics, const SobolReference &reference) {
        const int    dim      = statistics.dim;
        const bool   hasExact = hasExactSobolIndices(reference, dim);
        const double nan      = std::numeric_limits<double>::quiet_NaN();
        if(!(statistics.variance > 0.0)) throw std::runtime_error("Non-positive output variance");
        std::vector<SobolIndex> indices;
        std::vector<double>     first(dim), total(dim), V1(dim);
        double                  maxDiff = hasExact ? 0.0 : nan;
        for(int i = 0; i < dim; ++i) {
            std::vector<bool> mask(dim, true);
            mask[i]  = false;
            V1[i]    = conditional_mean_square(tt, weights, mask) - statistics.mean2;
            first[i] = V1[i] / statistics.variance;
            std::fill(mask.begin(), mask.end(), false);
            mask[i]  = true;
            total[i] = (statistics.m2 - conditional_mean_square(tt, weights, mask)) / statistics.variance;
        }
        for(int i = 0; i < dim; ++i)
            appendSobolIndex(indices, fmt::format("S{}", i), first[i], hasExact ? reference.first[i] : nan,
                             hasExact, maxDiff);
        for(int i = 0; i < dim; ++i)
            appendSobolIndex(indices, fmt::format("ST{}", i), total[i], hasExact ? reference.total[i] : nan,
                             hasExact, maxDiff);
        auto pairs = reference.second;
        if(!hasExact) {
            pairs.clear();
            for(int i = 0; i < dim; ++i)
                for(int j = i + 1; j < dim; ++j) pairs.push_back({i, j, nan});
        }
        for(const auto &pair : pairs) {
            std::vector<bool> mask(dim, true);
            mask[pair.i]     = false;
            mask[pair.j]     = false;
            const double vij = conditional_mean_square(tt, weights, mask) - statistics.mean2 - V1[pair.i] - V1[pair.j];
            appendSobolIndex(indices, fmt::format("S{},{}", pair.i, pair.j), vij / statistics.variance,
                             pair.value, hasExact, maxDiff);
        }
        double rmse = nan;
        if(hasExact && !indices.empty()) {
            double error2 = 0;
            for(const auto &index : indices) error2 += index.error * index.error;
            rmse = std::sqrt(error2 / indices.size());
        }
        return SobolResult{.rmse = rmse, .maxDiff = maxDiff, .sobolIndices = std::move(indices)};
    }

}