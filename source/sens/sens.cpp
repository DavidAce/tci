#include "sens.hpp"
#include "func/Base.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "tt_svd.hpp"
#include "util.h"
#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <h5pp/details/h5ppEnums.h>
#include <h5pp/details/h5ppVarr.h>
#include <h5pp/h5pp.h>
#include <H5Tpublic.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {
    using namespace util;

    // Integrate selected variables before squaring.
    // The remaining variables are integrated after squaring.
    template<typename TensorTrain>
    double conditional_mean_square(const TensorTrain &tt, const std::vector<std::vector<double>> &weights,
                                   const std::vector<bool> &integrateOut) {
        const int dim = static_cast<int>(weights.size());

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

    template<typename TensorTrain>
    Statistics calculateStatstics(const TensorTrain &tt, const std::vector<std::vector<double>> &weights) {
        Statistics fs;
        fs.dim      = static_cast<int>(weights.size());
        fs.mean     = tt.sum(weights);
        fs.mean2    = fs.mean * fs.mean;
        fs.m2       = conditional_mean_square(tt, weights, std::vector<bool>(fs.dim, false));
        fs.variance = fs.m2 - fs.mean2;
        return fs;
    }

    template<typename TensorTrain>
    SobolResult calculateSobol(const TensorTrain &tt, const std::vector<std::vector<double>> &weights,
                               const Statistics &fs, const SobolReference &sobolRef) {
        std::vector<SobolIndex> indices;
        indices.reserve(sobolRef.first.size() + sobolRef.second.size() + sobolRef.third.size());

        constexpr double nan = std::numeric_limits<double>::quiet_NaN();

        assert(fs.dim == static_cast<int>(weights.size()));
        const int    dim      = fs.dim;
        const double mean     = fs.mean;
        const double mean2    = fs.mean2;
        const double m2       = fs.m2;
        const double variance = fs.variance;

        double maxDiff = nan;
        double rmse    = nan;

        if(!(variance > 0.0)) throw std::runtime_error("Non-positive output variance");

        std::vector<double> first(dim), total(dim), V1(dim);

        for(int i = 0; i < dim; ++i) {
            // First-order: retain only variable i.
            std::vector<bool> mask(dim, true);
            mask[i] = false;

            V1[i]    = conditional_mean_square(tt, weights, mask) - mean2;
            first[i] = V1[i] / variance;

            // Total-order: integrate out only variable i.
            std::fill(mask.begin(), mask.end(), false);
            mask[i] = true;

            total[i] = (m2 - conditional_mean_square(tt, weights, mask)) / variance;
        }
        const bool hasExact = sobolRef.first.size() == static_cast<std::size_t>(dim) &&
                              sobolRef.total.size() == static_cast<std::size_t>(dim);

        auto addIndex = [&](std::vector<SobolIndex> &indices, std::string name, double estimated, double exact) {
            const double error = hasExact ? std::abs(estimated - exact) : nan;
            indices.push_back({h5pp::fstr_t<32>(name), estimated, exact, error});
            if(hasExact) maxDiff = std::max(maxDiff, error);
        };

        {
            for(int i = 0; i < dim; ++i)
                addIndex(indices, fmt::format("S{}", i + 1), first[i], hasExact ? sobolRef.first[i] : nan);
            for(int i = 0; i < dim; ++i)
                addIndex(indices, fmt::format("ST{}", i + 1), total[i], hasExact ? sobolRef.total[i] : nan);
        }
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

            const double Vij = conditional_mean_square(tt, weights, mask) - mean2 - V1[i] - V1[j];

            V2.emplace(key, Vij);
            return Vij;
        };

        // Second-order indices.
        for(const auto &pair : pairs) {
            const double Vij = secondVariance(pair.i, pair.j);
            addIndex(indices, fmt::format("S{},{}", pair.i + 1, pair.j + 1), Vij / variance, pair.value);
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

            const double Vijk = conditional_mean_square(tt, weights, mask) - mean2 - V1[i] - V1[j] - V1[k] -
                                secondVariance(i, j) - secondVariance(i, k) - secondVariance(j, k);
            addIndex(indices, fmt::format("S{},{},{}", i + 1, j + 1, k + 1), Vijk / variance, triple.value);
        }

        if(hasExact) {
            double sum2 = 0.0;
            for(const auto &index : indices) sum2 += index.error * index.error;
            rmse = std::sqrt(sum2 / indices.size());
        }
        return SobolResult{.rmse = rmse, .maxDiff = maxDiff, .sobolIndices = std::move(indices)};
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

    struct h5XfacParam {
        int    bondDim   = 30;    ///< the max bond dimension of the tensor train
        double reltol    = 1e-12; ///< CI will stop when pivotError < reltol*max(pivotError)
        bool   fullPiv   = false; ///< whether to search in the full Pi matrix to find a new pivot
        int    nRookIter = 3;     ///< number of rook pivoting iterations, used when fullPiv=false
        bool   useCachedFunction =
            true; ///< whether the internal caching should be used to avoid function call to the same point.
    };

    void writeParam(h5pp::File &file, xfac::TensorCI2Param &p) {
        auto                  h5param = h5XfacParam{.bondDim           = p.bondDim,
                                                    .reltol            = p.reltol,
                                                    .fullPiv           = p.fullPiv,
                                                    .nRookIter         = p.nRookIter,
                                                    .useCachedFunction = p.useCachedFunction};
        static h5pp::hid::h5t H5_PARAM_TYPE;
        if(!H5_PARAM_TYPE.valid()) {
            // Register the compound type
            H5_PARAM_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(h5param));
            H5Tinsert(H5_PARAM_TYPE, "bondDim", HOFFSET(h5XfacParam, bondDim), H5T_NATIVE_INT);
            H5Tinsert(H5_PARAM_TYPE, "reltol", HOFFSET(h5XfacParam, reltol), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_PARAM_TYPE, "fullPiv", HOFFSET(h5XfacParam, fullPiv), H5T_NATIVE_HBOOL);
            H5Tinsert(H5_PARAM_TYPE, "nRookIter", HOFFSET(h5XfacParam, nRookIter), H5T_NATIVE_INT);
            H5Tinsert(H5_PARAM_TYPE, "useCachedFunction", HOFFSET(h5XfacParam, useCachedFunction), H5T_NATIVE_HBOOL);
            file.createTable(H5_PARAM_TYPE, "xfac/param", "xfac parameters");
        }

        // Create an empty table
        file.appendTableRecords(h5param, "xfac/param");
    }

    struct Status {
        int              sweeps             = 0;
        int              sweepsUnchanged    = 0;
        int              sweepLastProgress  = 0;
        int              minSweepsUnchanged = 1;
        double           changeTol          = 1e-12;
        int              nEval              = 0;
        Statistics       statistics         = {};
        SobolResult      sobolResult        = {};
        std::vector<int> bondDims;
        std::string      stopMsg = {};
    };

    void appendStatus(h5pp::File &file, const Status &status) {
        struct H5Status {
            int               sweeps;
            int               sweepsUnchanged;
            int               sweepLastProgress;
            int               minSweepsUnchanged;
            double            changeTol;
            int               nEval;
            double            stat_mean;
            double            stat_variance;
            double            sobol_rmse;
            double            sobol_maxDiff;
            h5pp::varr_t<int> bondDims;
            h5pp::vstr_t      stopMsg;
            using vlen_type = h5pp::varr_t<int>;
        };

        auto h5status = H5Status{
            .sweeps             = status.sweeps,
            .sweepsUnchanged    = status.sweepsUnchanged,
            .sweepLastProgress  = status.sweepLastProgress,
            .minSweepsUnchanged = status.minSweepsUnchanged,
            .changeTol          = status.changeTol,
            .nEval              = status.nEval,
            .stat_mean          = status.statistics.mean,
            .stat_variance      = status.statistics.variance,
            .sobol_rmse         = status.sobolResult.rmse,
            .sobol_maxDiff      = status.sobolResult.maxDiff,
            .bondDims           = status.bondDims,
            .stopMsg            = status.stopMsg.c_str(),
        };

        static h5pp::hid::h5t H5_STATUS_TYPE;
        std::string           statusTablePath = "tci/status";

        if(!H5_STATUS_TYPE.valid()) {
            // Register the compound type
            H5_STATUS_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(H5Status));
            H5Tinsert(H5_STATUS_TYPE, "sweeps", HOFFSET(H5Status, sweeps), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "sweepsUnchanged", HOFFSET(H5Status, sweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "sweepLastProgress", HOFFSET(H5Status, sweepLastProgress), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "minSweepsUnchanged", HOFFSET(H5Status, minSweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "changeTol", HOFFSET(H5Status, changeTol), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "nEval", HOFFSET(H5Status, nEval), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "stat_mean", HOFFSET(H5Status, stat_mean), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "stat_variance", HOFFSET(H5Status, stat_variance), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "sobol_rmse", HOFFSET(H5Status, sobol_rmse), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "sobol_maxDiff", HOFFSET(H5Status, sobol_maxDiff), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "bondDims", HOFFSET(H5Status, bondDims), h5pp::varr_t<int>::get_h5type());
            H5Tinsert(H5_STATUS_TYPE, "stopMsg", HOFFSET(H5Status, stopMsg), h5pp::vstr_t::get_h5type());
        }
        if(!file.linkExists(statusTablePath)) { file.createTable(H5_STATUS_TYPE, statusTablePath, "Status Log"); }
        file.appendTableRecords(h5status, statusTablePath);

        static h5pp::hid::h5t H5_INDEX_TYPE;
        std::string           sobolTablePath = fmt::format("tci/sweep_{}/sobol", status.sweeps);
        if(!H5_INDEX_TYPE.valid()) {
            H5_INDEX_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(SobolIndex));
            H5Tinsert(H5_INDEX_TYPE, "name", HOFFSET(SobolIndex, name), h5pp::fstr_t<32>::get_h5type());
            H5Tinsert(H5_INDEX_TYPE, "estimated", HOFFSET(SobolIndex, estimated), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_INDEX_TYPE, "exact", HOFFSET(SobolIndex, exact), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_INDEX_TYPE, "error", HOFFSET(SobolIndex, error), H5T_NATIVE_DOUBLE);
        }
        if(!file.linkExists(sobolTablePath)) { file.createTable(H5_INDEX_TYPE, sobolTablePath, "Sobol Indices"); }
        file.writeTableRecords(status.sobolResult.sobolIndices, sobolTablePath);
    }
    void writeStatusLog(h5pp::File &file, const std::vector<Status> &statusLog) {
        for(auto &status : statusLog) appendStatus(file, status);
    }

    void runSensitivityAnalysis(const BaseFunction &function, const std::vector<double> &relTolTargets, int bondDim,
                                int maxSweeps, const Quadrature &makeQuadrature) {
        const int  dim        = function.dimension();
        const auto exactSobol = function.exactSobolIndices();
        const bool hasExact   = exactSobol.first.size() == static_cast<std::size_t>(dim) &&
                                exactSobol.total.size() == static_cast<std::size_t>(dim);

        auto h5file = h5pp::File(fmt::format("{}.h5", function.name()), h5pp::FileAccess::REPLACE);

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

        writeParam(h5file, param);

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

        auto report = [&](int sweeps, SobolResult &result) {
            auto previous   = std::move(result);
            auto statistics = calculateStatstics(ci.tt, weights);
            result          = calculateSobol(ci.tt, weights, statistics, exactSobol);

            const auto nEval = ci.f.nEval();

            if(hasExact) {
                fmt::print("{:<11} {:>6} {:>8} {:>12.4e} {:>12.4e}\n", "TCI2", sweeps, nEval, result.rmse,
                           result.maxDiff);

                for(std::size_t i = 0; i < relTolTargets.size(); ++i) {
                    if(!reached[i] && result.rmse <= relTolTargets[i]) {
                        reached[i]   = true;
                        firstEval[i] = nEval;
                    }
                }
            } else {
                std::string delta = "-";

                if(previous.sobolIndices.size() == result.sobolIndices.size()) {
                    double maxDeltaS = 0.0;

                    for(std::size_t i = 0; i < result.sobolIndices.size(); ++i) {
                        maxDeltaS = std::max(
                            maxDeltaS, std::abs(result.sobolIndices[i].estimated - previous.sobolIndices[i].estimated));
                    }

                    delta = fmt::format("{:.3e}", maxDeltaS);
                }

                fmt::print("{:<11} {:>6} {:>8} {:>14.6e} {:>14.6e} {:>12}\n", "TCI2", sweeps, nEval, statistics.mean,
                           statistics.variance, delta);
            }
        };

        auto isUnchanged = [](double x, double y, double changeTol) {
            return std::abs(x - y) <= changeTol * std::max({1.0, std::abs(x), std::abs(y)});
        };

        auto bondDimensions = [&]() {
            std::vector<int> dimensions;
            for(int k = 0; k < dim - 1; ++k) dimensions.push_back(static_cast<int>(ci.tt.M[k].n_slices));
            return dimensions;
        };

        auto statusLog            = std::vector<Status>(1);
        statusLog.back().nEval    = ci.f.nEval();
        statusLog.back().bondDims = bondDimensions();
        report(statusLog.back().sweeps, statusLog.back().sobolResult); // Report initial TT, before explicit sweeps.

        while(true) {
            assert(!statusLog.empty());
            statusLog.push_back(statusLog.back()); // Should be safe
            const auto &sPrev = statusLog.rbegin()[1];
            auto       &sLast = statusLog.back();

            ci.iterate();
            report(++sLast.sweeps, sLast.sobolResult);

            sLast.bondDims = bondDimensions();
            sLast.nEval    = ci.f.nEval();

            bool isSameMean     = isUnchanged(sLast.statistics.mean, sPrev.statistics.mean, sLast.changeTol);
            bool isSameVariance = isUnchanged(sLast.statistics.variance, sPrev.statistics.variance, sLast.changeTol);
            bool isSame         = isSameMean && isSameVariance && sLast.bondDims == sPrev.bondDims;

            if(ci.isDone()) {
                for(std::size_t i = 0; i < sLast.sobolResult.sobolIndices.size(); ++i) {
                    isSame &= isUnchanged(sLast.sobolResult.sobolIndices[i].estimated,
                                          sPrev.sobolResult.sobolIndices[i].estimated, sLast.changeTol);
                }

                sLast.sweepsUnchanged = isSame ? sLast.sweepsUnchanged + 1 : 0;

                if(!isSame) { sLast.sweepLastProgress = sLast.sweeps; }

                if(sLast.sweepsUnchanged >= sLast.minSweepsUnchanged) {
                    sLast.stopMsg = fmt::format("no change for {} consecutive sweep(s)", sLast.sweepsUnchanged);
                    break;
                }

            } else {
                sLast.sweepsUnchanged   = 0;
                sLast.sweepLastProgress = sLast.sweeps;
            }
            if(sLast.sweeps >= maxSweeps) {
                sLast.stopMsg = fmt::format("maximum number of sweeps reached ({})", maxSweeps);
            }
        }
        writeStatusLog(h5file, statusLog);

        auto        sweepLastProgress = statusLog.back().sweepLastProgress;
        const auto &sStable           = statusLog.at(sweepLastProgress);
        const auto &stableSobol       = sStable.sobolResult;
        const auto &stableStatistics  = sStable.statistics;
        auto        stableSweeps      = sStable.sweeps;
        auto        stableNEval       = sStable.nEval;

        // Print results
        auto &lastshot = statusLog.back();
        fmt::print("Stopped: {}\n", lastshot.stopMsg);

        if(hasExact) {
            fmt::print("\n{:>12} {:>12}\n", "Target RMSE", "nEval");
            for(std::size_t i = 0; i < relTolTargets.size(); ++i) {
                std::string nEvalMsg = reached[i] ? fmt::format("{}", firstEval[i]) : "not reached";
                fmt::print("{:>12.1e} {:>12}\n", relTolTargets[i], nEvalMsg);
            }
        }

        fmt::print("\nFinal bond dimensions: ");
        for(int k = 0; k < dim - 1; ++k) fmt::print("{} ", ci.tt.M[k].n_slices);

        fmt::print("\nMean: {:.8e}, Variance: {:.8e}\n", stableStatistics.mean, stableStatistics.variance);

        if(hasExact) {
            fmt::print("\n{:<8} {:>14} {:>14} {:>12}\n", "index", "estimated", "exact", "absError");
        } else {
            fmt::print("\n{:<8} {:>14}\n", "index", "estimated");
        }

        const std::size_t nPrinted =
            hasExact && stableSobol.sobolIndices.size() <= 10 ? stableSobol.sobolIndices.size() : 2 * dim;

        for(std::size_t i = 0; i < nPrinted; ++i) {
            const auto &index = stableSobol.sobolIndices.at(i);

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
                const auto &index = stableSobol.sobolIndices.at(i);

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

        const std::size_t nThirdAndUp = stableSobol.sobolIndices.size() - nFirst - nSecond;
        const std::size_t nFourAndUp  = stableSobol.sobolIndices.size() - nFirst - nSecond - nThird;

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

            fmt::print("{:<10} {:>8} {:>12.4e} {:>12.4e} {:>14.6e} {:>14.6e}\n", "TCI2", stableNEval, stableSobol.rmse,
                       stableSobol.maxDiff, stableStatistics.mean, stableStatistics.variance);
        }

        // runSensitivityAnalysis_TTSVD(function, ci, grids, weights, bondDim);
    }
}