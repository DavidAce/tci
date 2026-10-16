#include "sens.hpp"
#include "cfg/cfg.hpp"
#include "func/Base.hpp"
#include "interp/interp.hpp"
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
    struct Predictions {
        using vlen_type                = h5pp::varr_t<double>;
        double               predicted = 0.0;
        double               funcvalue = 0.0;
        double               error     = 0.0;
        h5pp::varr_t<double> point;
    };

    struct h5XfacParam {
        using vlen_type = h5pp::varr_t<int>;

        int               bondDim; ///< the max bond dimension of the tensor train
        double            reltol;  ///< CI will stop when pivotError < reltol*max(pivotError)
        h5pp::varr_t<int> pivot1;
        bool              fullPiv;   ///< whether to search in the full Pi matrix to find a new pivot
        int               nRookIter; ///< number of rook pivoting iterations, used when fullPiv=false
        bool useCachedFunction;      ///< whether the internal caching should be used to avoid function call to the same
        ///< point.
    };

    struct h5TCIParam {
        using vlen_type = h5pp::varr_t<double>;
        int                  dim;
        h5pp::vstr_t         name;
        h5pp::varr_t<double> reltols;
    };

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

    std::vector<double> randomOffGridPoint(const BaseFunction                     &function,
                                           const std::vector<std::vector<double>> &grids) {
        const int dim = function.dimension();
        if(grids.size() != static_cast<std::size_t>(dim)) throw std::invalid_argument("Grid dimension mismatch");

        auto pointOutOfBounds = [](double xk, double lo, double hi) -> bool { return xk <= lo || xk >= hi; };
        auto pointIsOnTheGrid = [](double xk, const std::vector<double> &grid) -> bool {
            return std::find(grid.begin(), grid.end(), xk) != grid.end();
        };
        constexpr double    nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> x(dim, nan);
        for(int k = 0; k < dim; ++k) {
            const auto [lo, hi] = function.domain(k);
            if(!(lo < hi)) throw std::invalid_argument("Invalid domain");
            while(std::isnan(x[k]) || pointOutOfBounds(x[k], lo, hi) || pointIsOnTheGrid(x[k], grids[k])) {
                x[k] = rnd::uniform_double_box(lo, hi);
            }
        }

        return x;
    }
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

        const bool hasExact = sobolRef.first.size() == static_cast<std::size_t>(dim) &&
                              sobolRef.total.size() == static_cast<std::size_t>(dim);

        double maxDiff = hasExact ? 0.0 : nan;
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

    template<typename Function, typename TensorTrain>
    std::vector<Predictions> calculatePredictions(Function &&function, const TensorTrain &tt,
                                                  const std::vector<std::vector<double>> &grids) {
        std::vector<Predictions> predictions;
        predictions.reserve(cfg::npredictions);
        auto   ctt       = interp::interpolateTT(tt, grids);
        double maxError  = 0.0;
        double sumError2 = 0.0;
        fmt::print("=== Model predictions ===\n");
        for(int i = 0; i < cfg::npredictions; ++i) {
            auto point     = randomOffGridPoint(function, grids);
            auto predicted = ctt.eval(point);
            auto funcvalue = function.evaluate(point);
            auto error     = std::abs(predicted - funcvalue);
            maxError       = std::max(maxError, error);
            sumError2 += error * error;
            predictions.emplace_back(predicted, funcvalue, error, point);
            // fmt::print("prediction {:<4}: {:<15.8e} funcvalue: {:<15.8e} error: {:<15.8e}\n", i, predicted,
            // funcvalue,
            //    error);
        }
        double rmse = std::sqrt(sumError2 / predictions.size());
        fmt::print("maxError: {:<15.8e} | rmse {:<15.8e}\n", maxError, rmse);
        return predictions;
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

    void writeParam(h5pp::File &file, const xfac::TensorCI2Param &p, const BaseFunction &f,
                    const std::vector<double> &relTolTargets, const Quadrature &makeQuadrature) {
        {
            auto h5Xfac = h5XfacParam{.bondDim           = p.bondDim,
                                      .reltol            = p.reltol,
                                      .pivot1            = p.pivot1,
                                      .fullPiv           = p.fullPiv,
                                      .nRookIter         = p.nRookIter,
                                      .useCachedFunction = p.useCachedFunction

            };

            static h5pp::hid::h5t H5_XFAC_PARAM_TYPE;
            std::string           xfacParamTablePath = "param/xfac";
            if(!H5_XFAC_PARAM_TYPE.valid()) {
                // Register the compound type
                H5_XFAC_PARAM_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(h5Xfac));
                H5Tinsert(H5_XFAC_PARAM_TYPE, "bondDim", HOFFSET(h5XfacParam, bondDim), H5T_NATIVE_INT);
                H5Tinsert(H5_XFAC_PARAM_TYPE, "reltol", HOFFSET(h5XfacParam, reltol), H5T_NATIVE_DOUBLE);
                H5Tinsert(H5_XFAC_PARAM_TYPE, "pivot1", HOFFSET(h5XfacParam, pivot1), h5pp::varr_t<int>::get_h5type());
                H5Tinsert(H5_XFAC_PARAM_TYPE, "fullPiv", HOFFSET(h5XfacParam, fullPiv), H5T_NATIVE_HBOOL);
                H5Tinsert(H5_XFAC_PARAM_TYPE, "nRookIter", HOFFSET(h5XfacParam, nRookIter), H5T_NATIVE_INT);
                H5Tinsert(H5_XFAC_PARAM_TYPE, "useCachedFunction", HOFFSET(h5XfacParam, useCachedFunction),
                          H5T_NATIVE_HBOOL);
            }
            if(!file.linkExists(xfacParamTablePath)) {
                file.createTable(H5_XFAC_PARAM_TYPE, xfacParamTablePath, "xfac parameters");
            }
            file.appendTableRecords(h5Xfac, xfacParamTablePath);
        }
        {
            auto h5TCI = h5TCIParam{
                .dim     = f.dimension(),
                .name    = f.name(),
                .reltols = relTolTargets,
            };
            static h5pp::hid::h5t H5_TCI_PARAM_TYPE;
            std::string           tciParamTablePath = "param/tci";
            if(!H5_TCI_PARAM_TYPE.valid()) {
                // Register the compound type
                H5_TCI_PARAM_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(h5TCIParam));
                H5Tinsert(H5_TCI_PARAM_TYPE, "dim", HOFFSET(h5TCIParam, dim), H5T_NATIVE_INT);
                H5Tinsert(H5_TCI_PARAM_TYPE, "name", HOFFSET(h5TCIParam, name), h5pp::vstr_t::get_h5type());
                H5Tinsert(H5_TCI_PARAM_TYPE, "reltols", HOFFSET(h5TCIParam, reltols),
                          h5pp::varr_t<double>::get_h5type());
            }
            if(!file.linkExists(tciParamTablePath)) {
                file.createTable(H5_TCI_PARAM_TYPE, tciParamTablePath, "TCI parameters");
            }
            file.appendTableRecords(h5TCI, tciParamTablePath);
        }
    }

    void writePredictions(h5pp::File &file, const std::vector<Predictions> &predictions) {
        if(predictions.empty()) return;
        static h5pp::hid::h5t H5_PREDICTION_TYPE;
        std::string           PredictionTablePath = "tci/predictions";
        if(!H5_PREDICTION_TYPE.valid()) {
            // Register the compound type
            H5_PREDICTION_TYPE = H5Tcreate(H5T_COMPOUND, sizeof(Predictions));
            H5Tinsert(H5_PREDICTION_TYPE, "predicted", HOFFSET(Predictions, predicted), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_PREDICTION_TYPE, "funcvalue", HOFFSET(Predictions, funcvalue), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_PREDICTION_TYPE, "error", HOFFSET(Predictions, error), H5T_NATIVE_DOUBLE);
            H5Tinsert(H5_PREDICTION_TYPE, "point", HOFFSET(Predictions, point), h5pp::varr_t<double>::get_h5type());
        }
        if(!file.linkExists(PredictionTablePath)) {
            file.createTable(H5_PREDICTION_TYPE, PredictionTablePath, "Function value predictions");
        }
        file.writeTableRecords(predictions, PredictionTablePath);
    }

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
        auto       filepath   = fmt::format("{}/{}.h5", cfg::outdir, function.name());
        auto       h5file     = h5pp::File(filepath, h5pp::FileAccess::REPLACE);

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
        param.bondDim           = bondDim;
        param.reltol            = 1e-12;
        param.fullPiv           = false;
        param.useCachedFunction = true;
        param.nRookIter         = 3;
        param.pivot1            = randomInitialPivot(function, grids);

        writeParam(h5file, param, function, relTolTargets, makeQuadrature);

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

        auto report = [&](Status &status) {
            auto previous      = std::move(status.sobolResult);
            status.statistics  = calculateStatstics(ci.tt, weights);
            status.sobolResult = calculateSobol(ci.tt, weights, status.statistics, exactSobol);

            const auto nEval = ci.f.nEval();

            if(hasExact) {
                fmt::print("{:<11} {:>6} {:>8} {:>12.4e} {:>12.4e}\n", "TCI2", status.sweeps, nEval,
                           status.sobolResult.rmse, status.sobolResult.maxDiff);

                for(std::size_t i = 0; i < relTolTargets.size(); ++i) {
                    if(!reached[i] && status.sobolResult.rmse <= relTolTargets[i]) {
                        reached[i]   = true;
                        firstEval[i] = nEval;
                    }
                }
            } else {
                std::string delta = "-";

                if(previous.sobolIndices.size() == status.sobolResult.sobolIndices.size()) {
                    double maxDeltaS = 0.0;

                    for(std::size_t i = 0; i < status.sobolResult.sobolIndices.size(); ++i) {
                        maxDeltaS = std::max(maxDeltaS, std::abs(status.sobolResult.sobolIndices[i].estimated -
                                                                 previous.sobolIndices[i].estimated));
                    }

                    delta = fmt::format("{:.3e}", maxDeltaS);
                }

                fmt::print("{:<11} {:>6} {:>8} {:>14.6e} {:>14.6e} {:>12}\n", "TCI2", status.sweeps, nEval,
                           status.statistics.mean, status.statistics.variance, delta);
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
        report(statusLog.back()); // Report initial TT, before explicit sweeps.

        while(statusLog.back().sweeps < maxSweeps) {
            assert(!statusLog.empty());
            statusLog.push_back(statusLog.back()); // Should be safe
            const auto &sPrev = statusLog.rbegin()[1];
            auto       &sLast = statusLog.back();

            ci.iterate();
            sLast.sweeps++;
            report(sLast);

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
                break;
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

        fmt::print("\nFinal bond dimensions: {} ", sStable.bondDims);
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

        const std::size_t nFirst  = dim;
        const std::size_t nSecond = hasExact ? exactSobol.second.size() : static_cast<std::size_t>(dim) * (dim - 1) / 2;

        const std::size_t nThird =
            hasExact ? exactSobol.third.size() : static_cast<std::size_t>(dim) * (dim - 1) * (dim - 2) / 6;

        const std::size_t offsetSecond = 2 * dim;
        const std::size_t offsetThird  = offsetSecond + nSecond;

        const double sumFirst       = printSum("S_i", 0, dim);
        const double sumSecond      = printSum("S_ij", offsetSecond, nSecond);
        const double sumThird       = printSum("S_ijk", offsetThird, nThird);
        const double sumThirdAndUp  = 1.0 - sumFirst - sumSecond;
        const double sumFourthAndUp = 1.0 - sumFirst - sumSecond - sumThird;
        fmt::print("{:<12} {:>6} {:>14.8f}\n", "S_ijk...", "-", sumThirdAndUp);
        fmt::print("{:<12} {:>6} {:>14.8f}\n", "S_ijkl...", "-", sumFourthAndUp);

        if(hasExact) {
            fmt::print("\n=== Final TCI accuracy ===\n");
            fmt::print("{:<10} {:>8} {:>12} {:>12} {:>14} {:>14}\n", "method", "nEval", "RMSE", "maxAbs", "mean",
                       "variance");

            fmt::print("{:<10} {:>8} {:>12.4e} {:>12.4e} {:>14.6e} {:>14.6e}\n", "TCI2", stableNEval, stableSobol.rmse,
                       stableSobol.maxDiff, stableStatistics.mean, stableStatistics.variance);
        }

        // runSensitivityAnalysis_TTSVD(function, ci, grids, weights, bondDim);

        auto predictions = calculatePredictions(function, ci.tt, grids);
        writePredictions(h5file, predictions);
    }
}