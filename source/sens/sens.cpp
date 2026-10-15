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

        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        auto addIndex = [&](std::vector<IndexResult> &indices, std::string name, double estimated, double exact) {
            const double error = hasExact ? std::abs(estimated - exact) : nan;
            indices.push_back({h5pp::vstr_t(name), estimated, exact, error});
            if(hasExact) result.maxAbs = std::max(result.maxAbs, error);
        };

        {
            std::vector<IndexResult> indices = result.indices;
            for(int i = 0; i < dim; ++i)
                addIndex(indices, fmt::format("S{}", i + 1), first[i], hasExact ? sobolRef.first[i] : nan);
            for(int i = 0; i < dim; ++i)
                addIndex(indices, fmt::format("ST{}", i + 1), total[i], hasExact ? sobolRef.total[i] : nan);
            result.indices = std::move(indices);
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

            const double Vij = conditional_mean_square(mask) - mean2 - V1[i] - V1[j];

            V2.emplace(key, Vij);
            return Vij;
        };

        // Second-order indices.
        {
            std::vector<IndexResult> indices = result.indices;
            for(const auto &pair : pairs) {
                const double Vij = secondVariance(pair.i, pair.j);

                addIndex(indices, fmt::format("S{},{}", pair.i + 1, pair.j + 1), Vij / result.variance, pair.value);
            }
            result.indices = std::move(indices);
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
            std::vector<IndexResult> indices = result.indices;
            addIndex(indices, fmt::format("S{},{},{}", i + 1, j + 1, k + 1), Vijk / result.variance, triple.value);
            result.indices = indices;
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
        int               sweeps             = 0;
        int               sweepsUnchanged    = 0;
        int               sweepLastProgress  = 0;
        int               minSweepsUnchanged = 1;
        double            changeTol          = 1e-12;
        int               nEval              = 0;
        SensitivityResult sensRes            = {};
        h5pp::varr_t<int> bondDims;
        h5pp::vstr_t      stopMsg = {};
    };

    // struct IndexResult {
    //     std::string name;
    //     double      estimated;
    //     double      exact;
    //     double      error;
    // };

    // struct SensitivityResult {
    //     double                   mean     = 0.0;
    //     double                   variance = 0.0;
    //     double                   rmse     = 0.0;
    //     double                   maxAbs   = 0.0;
    //     std::vector<IndexResult> indices;
    // };

    void appendStatus(h5pp::File &file, const Status &status) {
        static h5pp::hid::h5t H5_INDEX_TYPE;
        static h5pp::hid::h5t H5_SENSRES_TYPE;
        static h5pp::hid::h5t H5_STATUS_TYPE;
        if(!H5_INDEX_TYPE.valid()) {
            H5_INDEX_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(IndexResult));
            H5Tinsert(H5_INDEX_TYPE, "name", HOFFSET(IndexResult, name), h5pp::vstr_t::get_h5type());
            H5Tinsert(H5_INDEX_TYPE, "estimated", HOFFSET(IndexResult, estimated), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_INDEX_TYPE, "exact", HOFFSET(IndexResult, exact), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_INDEX_TYPE, "error", HOFFSET(IndexResult, error), H5T_NATIVE_DOUBLE);
        }

        if(!H5_SENSRES_TYPE.valid()) {
            h5pp::hid::h5t H5_INDEXVARR_TYPE = H5Tvlen_create(H5_INDEX_TYPE);
            H5_SENSRES_TYPE                  = H5Tcreate(H5T_COMPOUND, sizeof(SensitivityResult));
            H5Tinsert(H5_SENSRES_TYPE, "mean", HOFFSET(SensitivityResult, mean), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_SENSRES_TYPE, "variance", HOFFSET(SensitivityResult, variance), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_SENSRES_TYPE, "rmse", HOFFSET(SensitivityResult, rmse), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_SENSRES_TYPE, "maxAbs", HOFFSET(SensitivityResult, maxAbs), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_SENSRES_TYPE, "indices", HOFFSET(SensitivityResult, indices), H5_INDEXVARR_TYPE);
        }

        if(!H5_STATUS_TYPE.valid()) {
            // Register the compound type
            H5_STATUS_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(Status));
            H5Tinsert(H5_STATUS_TYPE, "sweeps", HOFFSET(Status, sweeps), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "sweepsUnchanged", HOFFSET(Status, sweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "sweepLastProgress", HOFFSET(Status, sweepLastProgress), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "minSweepsUnchanged", HOFFSET(Status, minSweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "changeTol", HOFFSET(Status, changeTol), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_STATUS_TYPE, "nEval", HOFFSET(Status, nEval), H5T_NATIVE_INT);
            H5Tinsert(H5_STATUS_TYPE, "sensRes", HOFFSET(Status, sensRes), H5_SENSRES_TYPE);
            H5Tinsert(H5_STATUS_TYPE, "bondDims", HOFFSET(Status, bondDims), h5pp::varr_t<int>::get_h5type());
            H5Tinsert(H5_STATUS_TYPE, "stopMsg", HOFFSET(Status, stopMsg), h5pp::vstr_t::get_h5type());
            file.createTable(H5_STATUS_TYPE, "tci/statusLog", "Status Log");
        }

        file.appendTableRecords(status, "tci/statusLog");
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

        auto isUnchanged = [](double x, double y, double changeTol) {
            return std::abs(x - y) <= changeTol * std::max({1.0, std::abs(x), std::abs(y)});
        };

        auto bondDimensions = [&]() {
            std::vector<int> dimensions;
            for(int k = 0; k < dim - 1; ++k) dimensions.push_back(static_cast<int>(ci.tt.M[k].n_slices));
            return dimensions;
        };

        auto statusLog = std::vector<Status>();

        {
            // The initial state
            statusLog.emplace_back();
            auto &statusInit = statusLog.back();
            report(statusInit.sweeps, statusInit.sensRes); // Initial TT, before explicit sweeps.
        }

        while(true) {
            assert(!statusLog.empty());
            statusLog.push_back(statusLog.back()); // Should be safe
            const auto &sPrev = statusLog.rbegin()[1];
            auto       &sLast = statusLog.back();

            ci.iterate();
            report(++sLast.sweeps, sLast.sensRes);

            sLast.bondDims = bondDimensions();
            sLast.nEval    = ci.f.nEval();

            bool isSameMean     = isUnchanged(sLast.sensRes.mean, sPrev.sensRes.mean, sLast.changeTol);
            bool isSameVariance = isUnchanged(sLast.sensRes.variance, sPrev.sensRes.variance, sLast.changeTol);
            bool isSame         = isSameMean && isSameVariance && sLast.bondDims == sPrev.bondDims;

            if(ci.isDone()) {
                for(std::size_t i = 0; i < sLast.sensRes.indices.size(); ++i) {
                    isSame &= isUnchanged(sLast.sensRes.indices[i].estimated, sPrev.sensRes.indices[i].estimated,
                                          sLast.changeTol);
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
        const auto &stableshot        = statusLog.at(sweepLastProgress);
        auto        stableResult      = stableshot.sensRes;
        auto        stableNEval       = stableshot.nEval;
        auto        stableSweeps      = stableshot.sweeps;

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