#include "sens.hpp"
#include "func/Base.hpp"
#include "util.h"
#include <fmt/core.h>
#include <string>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {
    using namespace util;
    template<typename TensorTrain>
    SensitivityResult calculateSobol(const TensorTrain &tt, const std::vector<std::vector<double>> &weights,
                                     const SobolReference &reference) {
        const int dim = static_cast<int>(weights.size());

        // Integrate selected variables before squaring.
        // The remaining variables are integrated after squaring.
        auto conditional_mean_square = [&](const std::vector<bool> &integrateOut) -> double {
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
        };

        SensitivityResult result;

        result.mean = tt.sum(weights);

        const double mean2 = result.mean * result.mean;
        const double m2    = conditional_mean_square(std::vector<bool>(dim, false));

        result.variance = m2 - mean2;

        if(!(result.variance > 0.0)) throw std::runtime_error("Non-positive output variance");

        std::vector<double> first(dim), total(dim), V1(dim);

        for(int i = 0; i < dim; ++i) {
            // First-order: retain only variable i.
            std::vector<bool> mask(dim, true);
            mask[i] = false;

            V1[i]    = conditional_mean_square(mask) - mean2;
            first[i] = V1[i] / result.variance;

            // Total-order: integrate out only variable i.
            std::fill(mask.begin(), mask.end(), false);
            mask[i] = true;

            total[i] = (m2 - conditional_mean_square(mask)) / result.variance;
        }

        auto addIndex = [&](std::string name, double estimated, double exact) {
            const double error = std::abs(estimated - exact);
            result.indices.push_back({std::move(name), estimated, exact, error});
            result.maxAbs = std::max(result.maxAbs, error);
        };

        for(int i = 0; i < dim; ++i) addIndex(fmt::format("S{}", i + 1), first[i], reference.first[i]);

        for(int i = 0; i < dim; ++i) addIndex(fmt::format("ST{}", i + 1), total[i], reference.total[i]);

        // Selected second-order indices from the reference.
        for(const auto &pair : reference.second) {
            std::vector<bool> mask(dim, true);
            mask[pair.i] = false;
            mask[pair.j] = false;

            const double Vij = conditional_mean_square(mask) - mean2 - V1[pair.i] - V1[pair.j];

            addIndex(fmt::format("S{},{}", pair.i + 1, pair.j + 1), Vij / result.variance, pair.value);
        }

        double sum2 = 0.0;
        for(const auto &index : result.indices) sum2 += index.error * index.error;

        result.rmse = std::sqrt(sum2 / result.indices.size());

        return result;
    }

    void runSensitivity(const BaseFunction &benchmark, const std::vector<double> &targets, int bondDim, int maxSweeps) {
        const int  dim       = benchmark.dimension();
        const auto reference = benchmark.exactSobol();

        // Independent uniform input distributions.
        std::vector<std::vector<double>> grids(dim);
        std::vector<std::vector<double>> weights(dim);

        for(int k = 0; k < dim; ++k) {
            auto [lo, hi] = benchmark.domain(k);
            auto [xi, wi] = xfac::grid::QuadratureGK15(lo, hi);

            for(auto &w : wi) w /= (hi - lo);

            grids[k]   = std::move(xi);
            weights[k] = std::move(wi);
        }

        xfac::TensorCI2Param param;
        param.bondDim = bondDim;
        param.reltol  = 1e-12;
        param.fullPiv = false;
        param.pivot1  = benchmark.initialPivot();

        auto f = [&benchmark](std::vector<double> x) { return benchmark.evaluate(x); };

        xfac::CTensorCI2<double, double> ci(f, grids, param);

        std::vector<std::size_t> firstEval(targets.size(), 0);
        std::vector<bool>        reached(targets.size(), false);

        auto allReached = [&]() { return std::all_of(reached.begin(), reached.end(), [](bool r) { return r; }); };

        fmt::print("\n=== {} (dim={}) ===\n", benchmark.name(), dim);

        fmt::print("{:<11} {:>6} {:>8} {:>12} {:>12}\n", "method", "sweeps", "nEval", "RMSE", "maxAbs");

        SensitivityResult lastResult;

        auto report = [&](int sweeps) {
            lastResult = calculateSobol(ci.tt, weights, reference);

            const auto nEval = ci.f.nEval();

            fmt::print("{:<11} {:>6} {:>8} {:>12.4e} {:>12.4e}\n", "TCI2", sweeps, nEval, lastResult.rmse,
                       lastResult.maxAbs);

            for(std::size_t i = 0; i < targets.size(); ++i) {
                if(!reached[i] && lastResult.rmse <= targets[i]) {
                    reached[i]   = true;
                    firstEval[i] = nEval;
                }
            }
        };

        int sweeps = 0;
        report(sweeps); // Initial TT, before explicit sweeps.

        while(!allReached() && sweeps < maxSweeps) {
            ci.iterate();
            report(++sweeps);
        }

        fmt::print("\n{:>12} {:>12}\n", "Target RMSE", "nEval");

        for(std::size_t i = 0; i < targets.size(); ++i) {
            if(reached[i])
                fmt::print("{:>12.1e} {:>12}\n", targets[i], firstEval[i]);
            else
                fmt::print("{:>12.1e} {:>12}\n", targets[i], "not reached");
        }

        fmt::print("\nFinal bond dimensions: ");
        for(int k = 0; k < dim - 1; ++k) fmt::print("{} ", ci.tt.M[k].n_slices);

        fmt::print("\nMean: {:.12f}, Variance: {:.12f}\n", lastResult.mean, lastResult.variance);

        fmt::print("\n{:<8} {:>14} {:>14} {:>12}\n", "index", "estimated", "exact", "absError");

        for(const auto &index : lastResult.indices) {
            fmt::print("{:<8} {:>14.8f} {:>14.8f} {:>12.3e}\n", index.name, index.estimated, index.exact, index.error);
        }
    }
}