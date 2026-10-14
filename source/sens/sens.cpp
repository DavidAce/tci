#include "sens.hpp"
#include "func/Base.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "tt_svd.hpp"
#include "util.h"
#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {
    using namespace util;
    template<typename TensorTrain>
    SensitivityResult calculateSobol(const TensorTrain &tt, const std::vector<std::vector<double>> &weights,
                                     const SobolReference &sobolRef) {
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
        const bool hasExact = sobolRef.first.size() == static_cast<std::size_t>(dim) &&
                              sobolRef.total.size() == static_cast<std::size_t>(dim);

        constexpr double nan      = std::numeric_limits<double>::quiet_NaN();
        auto             addIndex = [&](std::string name, double estimated, double exact) {
            const double error = hasExact ? std::abs(estimated - exact) : nan;
            result.indices.push_back({std::move(name), estimated, exact, error});
            if(hasExact) result.maxAbs = std::max(result.maxAbs, error);
        };

        for(int i = 0; i < dim; ++i) addIndex(fmt::format("S{}", i + 1), first[i], hasExact ? sobolRef.first[i] : nan);
        for(int i = 0; i < dim; ++i) addIndex(fmt::format("ST{}", i + 1), total[i], hasExact ? sobolRef.total[i] : nan);

        auto pairs   = sobolRef.second;
        auto triples = sobolRef.third;

        if(!hasExact) {
            for(int i = 0; i < dim; ++i) {
                for(int j = i + 1; j < dim; ++j) {
                    pairs.push_back({i, j, nan});
                    for(int k = j + 1; k < dim; ++k) triples.push_back({i, j, k, nan});
                }
            }
        }

        std::map<std::pair<int, int>, double> V2;

        auto secondVariance = [&](int i, int j) -> double {
            const auto key = std::pair{std::min(i, j), std::max(i, j)};

            if(auto it = V2.find(key); it != V2.end()) return it->second;

            std::vector<bool> mask(dim, true);
            mask[i] = false;
            mask[j] = false;

            const double Vij = conditional_mean_square(mask) - mean2 - V1[i] - V1[j];

            V2.emplace(key, Vij);
            return Vij;
        };

        // Second-order indices.
        for(const auto &pair : pairs) {
            const double Vij = secondVariance(pair.i, pair.j);

            addIndex(fmt::format("S{},{}", pair.i + 1, pair.j + 1), Vij / result.variance, pair.value);
        }

        // Third-order indices.
        for(const auto &triple : triples) {
            const int i = triple.i;
            const int j = triple.j;
            const int k = triple.k;

            std::vector<bool> mask(dim, true);
            mask[i] = false;
            mask[j] = false;
            mask[k] = false;

            const double Vijk = conditional_mean_square(mask) - mean2 - V1[i] - V1[j] - V1[k] - secondVariance(i, j) -
                                secondVariance(i, k) - secondVariance(j, k);

            addIndex(fmt::format("S{},{},{}", i + 1, j + 1, k + 1), Vijk / result.variance, triple.value);
        }

        if(hasExact) {
            double sum2 = 0.0;
            for(const auto &index : result.indices) sum2 += index.error * index.error;
            result.rmse = std::sqrt(sum2 / result.indices.size());
        } else {
            result.rmse = nan;
        }
        return result;
    }
    template<typename Function>
    std::vector<int> randomInitialPivot(Function &&function, const std::vector<std::vector<double>> &grids) {
        if(grids.empty()) throw std::invalid_argument("Empty quadrature grids");
        int                 dim   = static_cast<int>(grids.size());
        int                 trial = 0;
        std::vector<int>    pivot1(dim, 0);
        std::vector<double> initial_state(dim, 0);
        while(true) {
            for(int k = 0; k < dim; k++) {
                if(grids[k].empty()) throw std::invalid_argument("Empty quadrature grid");
                pivot1[k]        = rnd::uniform_integer_box<int>(0, grids[k].size() - 1);
                initial_state[k] = grids[k][pivot1[k]];
            }
            double initial_value = std::abs(function.evaluate(initial_state));
            if(std::isfinite(initial_value) && initial_value != 0.0) {
                fmt::print("param.pivot1: {} | value {:.8e}\n", pivot1, initial_value);
                break;
            }
            if(++trial >= 20) throw std::runtime_error("Failed to find a nonzero initial pivot within 20 trials");
        }
        return pivot1;
    }

    template<typename Function, typename TCI>
    void runSensitivityAnalysis_TTSVD(Function &&function, TCI &&ci, const std::vector<std::vector<double>> &grids,
                                      const std::vector<std::vector<double>> &weights, arma::uword maxBondDim) {
        const int  dim        = function.dimension();
        const auto exactSobol = function.exactSobolIndices();
        auto   svd = makeTTSVD([&function](const auto &x) { return function.evaluate(x); }, grids, weights, maxBondDim);
        double error2 = 0.0;
        double norm2  = arma::dot(svd.weightedValues, svd.weightedValues);

        std::vector<int> indices(dim);

        for(std::size_t linear = 0; linear < svd.weightedValues.n_elem; ++linear) {
            std::size_t n          = linear;
            double      sqrtWeight = 1.0;

            for(int k = 0; k < dim; ++k) {
                const auto s = n % grids[k].size();
                n /= grids[k].size();

                indices[k] = static_cast<int>(s);
                sqrtWeight *= std::sqrt(weights[k][s]);
            }

            const double difference = svd.weightedValues(linear) - sqrtWeight * ci.tt.eval(indices);

            error2 += difference * difference;
        }

        const double tciError = std::sqrt(error2 / norm2);

        fmt::print("Relative RMS error: TCI={:.6e}, TT-SVD={:.6e}\n", tciError, svd.relativeL2Error);

        auto svdSobol = calculateSobol(svd.tt, weights, exactSobol);

        fmt::print("TT-SVD: nEval={}, relativeL2Error={:.6e}\n", svd.nEval, svd.relativeL2Error);

        fmt::print("Mean: {:.8e}, Variance: {:.8e}\n", svdSobol.mean, svdSobol.variance);
    }

    void runSensitivityAnalysis(const BaseFunction &function, const std::vector<double> &relTolTargets, int bondDim,
                                int maxSweeps, const Quadrature &makeQuadrature) {
        const int  dim        = function.dimension();
        const auto exactSobol = function.exactSobolIndices();
        const bool hasExact   = exactSobol.first.size() == static_cast<std::size_t>(dim) &&
                                exactSobol.total.size() == static_cast<std::size_t>(dim);

        // Independent uniform input distributions.
        std::vector<std::vector<double>> grids(dim);
        std::vector<std::vector<double>> weights(dim);

        for(int k = 0; k < dim; ++k) {
            auto [lo, hi] = function.domain(k);
            auto rule     = makeQuadrature(lo, hi);
            grids[k]      = std::move(rule.nodes);
            weights[k]    = std::move(rule.weights);
        }

        xfac::TensorCI2Param param;
        param.bondDim = bondDim;
        param.reltol  = 1e-12;
        param.fullPiv = false;
        param.pivot1  = randomInitialPivot(function, grids);

        auto f = [&function](const std::vector<double> &x) -> double { return function.evaluate(x); };
        xfac::CTensorCI2<double, double> ci(f, grids, param);

        std::vector<std::size_t> firstEval(relTolTargets.size(), 0);
        std::vector<bool>        reached(relTolTargets.size(), false);

        auto allReached = [&]() { return std::all_of(reached.begin(), reached.end(), [](bool r) { return r; }); };

        fmt::print("\n=== {} (dim={}) ===\n", function.name(), dim);

        if(hasExact) {
            fmt::print("{:<11} {:>6} {:>8} {:>12} {:>12}\n", "method", "sweeps", "nEval", "RMSE", "maxAbs");
        } else {
            fmt::print("{:<11} {:>6} {:>8} {:>14} {:>14} {:>12}\n", "method", "sweeps", "nEval", "mean", "variance",
                       "maxDeltaS");
        }

        auto report = [&](int sweeps, SensitivityResult &result) {
            auto previous = std::move(result);
            result        = calculateSobol(ci.tt, weights, exactSobol);

            const auto nEval = ci.f.nEval();

            if(hasExact) {
                fmt::print("{:<11} {:>6} {:>8} {:>12.4e} {:>12.4e}\n", "TCI2", sweeps, nEval, result.rmse,
                           result.maxAbs);

                for(std::size_t i = 0; i < relTolTargets.size(); ++i) {
                    if(!reached[i] && result.rmse <= relTolTargets[i]) {
                        reached[i]   = true;
                        firstEval[i] = nEval;
                    }
                }
            } else {
                std::string delta = "-";

                if(previous.indices.size() == result.indices.size()) {
                    double maxDeltaS = 0.0;

                    for(std::size_t i = 0; i < result.indices.size(); ++i) {
                        maxDeltaS =
                            std::max(maxDeltaS, std::abs(result.indices[i].estimated - previous.indices[i].estimated));
                    }

                    delta = fmt::format("{:.3e}", maxDeltaS);
                }

                fmt::print("{:<11} {:>6} {:>8} {:>14.6e} {:>14.6e} {:>12}\n", "TCI2", sweeps, nEval, result.mean,
                           result.variance, delta);
            }
        };

        constexpr double changeTol          = 1e-12;
        constexpr int    minSweepsUnchanged = 1;

        auto isUnchanged = [](double x, double y) {
            return std::abs(x - y) <= changeTol * std::max({1.0, std::abs(x), std::abs(y)});
        };

        auto bondDimensions = [&]() {
            std::vector<arma::uword> dimensions;
            for(int k = 0; k < dim - 1; ++k) dimensions.push_back(ci.tt.M[k].n_slices);
            return dimensions;
        };

        auto lastResult      = SensitivityResult();
        int  sweepsUnchanged = 0;
        int  sweeps          = 0;
        report(sweeps, lastResult); // Initial TT, before explicit sweeps.

        auto        previousResult   = lastResult;
        auto        previousBondDims = bondDimensions();
        std::string stopMsg          = fmt::format("maximum number of sweeps reached ({})", maxSweeps);

        auto stableResult = lastResult;
        auto stableNEval  = ci.f.nEval();
        auto stableSweeps = sweeps;

        while(sweeps < maxSweeps) {
            ci.iterate();
            report(++sweeps, lastResult);

            const auto bondDims       = bondDimensions();
            bool       isSameMean     = isUnchanged(lastResult.mean, previousResult.mean);
            bool       isSameVariance = isUnchanged(lastResult.variance, previousResult.variance);
            bool       isSame         = isSameMean && isSameVariance && bondDims == previousBondDims;

            if(ci.isDone()) {
                for(std::size_t i = 0; i < lastResult.indices.size(); ++i) {
                    isSame &= isUnchanged(lastResult.indices[i].estimated, previousResult.indices[i].estimated);
                }

                sweepsUnchanged = isSame ? sweepsUnchanged + 1 : 0;

                if(!isSame) {
                    stableResult = lastResult;
                    stableNEval  = ci.f.nEval();
                    stableSweeps = sweeps;
                }

                if(sweepsUnchanged >= minSweepsUnchanged) {
                    stopMsg = fmt::format("no change for {} consecutive sweep(s)", sweepsUnchanged);
                    break;
                }

            } else {
                sweepsUnchanged = 0;
            }

            previousResult   = lastResult;
            previousBondDims = bondDims;
        }

        // Print results

        fmt::print("Stopped: {}\n", stopMsg);

        if(hasExact) {
            fmt::print("\n{:>12} {:>12}\n", "Target RMSE", "nEval");
            for(std::size_t i = 0; i < relTolTargets.size(); ++i) {
                std::string nEvalMsg = reached[i] ? fmt::format("{}", firstEval[i]) : "not reached";
                fmt::print("{:>12.1e} {:>12}\n", relTolTargets[i], nEvalMsg);
            }
        }

        fmt::print("\nFinal bond dimensions: ");
        for(int k = 0; k < dim - 1; ++k) fmt::print("{} ", ci.tt.M[k].n_slices);

        fmt::print("\nMean: {:.8e}, Variance: {:.8e}\n", stableResult.mean, stableResult.variance);

        if(hasExact) {
            fmt::print("\n{:<8} {:>14} {:>14} {:>12}\n", "index", "estimated", "exact", "absError");
        } else {
            fmt::print("\n{:<8} {:>14}\n", "index", "estimated");
        }

        const std::size_t nPrinted =
            hasExact && stableResult.indices.size() <= 10 ? stableResult.indices.size() : 2 * dim;

        for(std::size_t i = 0; i < nPrinted; ++i) {
            const auto &index = stableResult.indices.at(i);

            if(hasExact) {
                fmt::print("{:<8} {:>14.8f} {:>14.8f} {:>12.3e}\n", index.name, index.estimated, index.exact,
                           index.error);
            } else {
                fmt::print("{:<8} {:>14.8f}\n", index.name, index.estimated);
            }
        }

        auto printSum = [&](const char *name, std::size_t offset, std::size_t extent) -> double {
            if(extent == 0) return 0.0;

            double estimated = 0.0;
            double exact     = 0.0;
            double error2    = 0.0;

            for(std::size_t i = offset; i < offset + extent; ++i) {
                const auto &index = stableResult.indices.at(i);

                estimated += index.estimated;

                if(hasExact) {
                    exact += index.exact;
                    error2 += index.error * index.error;
                }
            }

            if(hasExact) {
                fmt::print("{:<12} {:>6} {:>14.8f} {:>14.8f} {:>12.3e} {:>12.3e}\n", name, extent, estimated, exact,
                           std::abs(estimated - exact), std::sqrt(error2 / extent));
            } else {
                fmt::print("{:<12} {:>6} {:>14.8f}\n", name, extent, estimated);
            }

            return estimated;
        };

        if(hasExact) {
            fmt::print("\n{:<12} {:>6} {:>14} {:>14} {:>12} {:>12}\n", "order", "count", "sum est.", "sum exact",
                       "absError", "RMSE");
        } else {
            fmt::print("\n{:<12} {:>6} {:>14}\n", "order", "count", "sum est.");
        }

        const std::size_t nFirst  = exactSobol.first.size();
        const std::size_t nSecond = hasExact ? exactSobol.second.size() : static_cast<std::size_t>(dim) * (dim - 1) / 2;

        const std::size_t nThird =
            hasExact ? exactSobol.third.size() : static_cast<std::size_t>(dim) * (dim - 1) * (dim - 2) / 6;

        const std::size_t nThirdAndUp = stableResult.indices.size() - nFirst - nSecond;
        const std::size_t nFourAndUp  = stableResult.indices.size() - nFirst - nSecond - nThird;

        const std::size_t offsetSecond = 2 * dim;
        const std::size_t offsetThird  = offsetSecond + nSecond;

        const double sumFirst       = printSum("S_i", 0, dim);
        const double sumSecond      = printSum("S_ij", offsetSecond, nSecond);
        const double sumThird       = printSum("S_ijk", offsetThird, nThird);
        const double sumThirdAndUp  = 1.0 - sumFirst - sumSecond;
        const double sumFourthAndUp = 1.0 - sumFirst - sumSecond - sumThird;
        fmt::print("{:<12} {:>6} {:>14.8f}\n", "S_ijk...", nThirdAndUp, sumThirdAndUp);
        fmt::print("{:<12} {:>6} {:>14.8f}\n", "S_ijkl...", nFourAndUp, sumFourthAndUp);

        if(hasExact) {
            fmt::print("\n=== Final TCI accuracy ===\n");
            fmt::print("{:<10} {:>8} {:>12} {:>12} {:>14} {:>14}\n", "method", "nEval", "RMSE", "maxAbs", "mean",
                       "variance");

            fmt::print("{:<10} {:>8} {:>12.4e} {:>12.4e} {:>14.6e} {:>14.6e}\n", "TCI2", stableNEval, stableResult.rmse,
                       stableResult.maxAbs, stableResult.mean, stableResult.variance);
        }

        // runSensitivityAnalysis_TTSVD(function, ci, grids, weights, bondDim);
    }
}