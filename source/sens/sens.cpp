#include "sens.hpp"
#include "accuracy.hpp"
#include "cfg/cfg.hpp"
#include "func/BaseFunction.hpp"
#include "grid.hpp"
#include "hdf5.hpp"
#include "info/sil.hpp"
#include "interp/interp.hpp"
#include "print.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "sobol.hpp"
#include "util.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <h5pp/h5pp.h>
#include <H5Tpublic.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {
    using namespace util;

    struct AnalysisSettings {
        static constexpr int    nGridProbes         = 128;
        static constexpr int    nControlPoints      = 16;
        static constexpr int    maxRecoveryAttempts = 3;
        static constexpr double probeTolerance      = 1e-10;
    };

    std::vector<int> selectInitialPivot(const BaseFunction &function, const Grids &grids) {
        if(grids.empty()) throw std::invalid_argument("Empty quadrature grids");
        const int           dim = static_cast<int>(grids.size());
        std::vector<int>    pivot(dim);
        std::vector<double> point(dim);
        for(int trial = 0; trial < 20; ++trial) {
            for(int k = 0; k < dim; ++k) {
                if(grids[k].empty()) throw std::invalid_argument("Empty quadrature grid");
                pivot[k] = rnd::uniform_integer_box<int>(0, static_cast<int>(grids[k].size()) - 1);
                point[k] = grids[k][pivot[k]];
            }
            const double value = std::abs(function.evaluate(point));
            if(std::isfinite(value) && value != 0.0) {
                fmt::print("param.pivot1: {} | value {:.8e}\n", pivot, value);
                return pivot;
            }
        }
        throw std::runtime_error("Failed to find a nonzero initial pivot within 20 trials");
    }

    xfac::TensorCI2Param makeTCIParameters(const BaseFunction &function, const Grids &grids, int bondDim) {
        xfac::TensorCI2Param param;
        param.bondDim           = bondDim;
        param.reltol            = 1e-12;
        param.fullPiv           = false;
        param.useCachedFunction = true;
        param.nRookIter         = 3;
        param.pivot1            = selectInitialPivot(function, grids);
        return param;
    }

    template<typename TT>
    Statistics calculateMoments(const TT &tt, const Grids &weights) {
        Statistics result;
        result.dim      = static_cast<int>(weights.size());
        result.mean     = tt.sum(weights);
        result.mean2    = result.mean * result.mean;
        result.m2       = conditional_mean_square(tt, weights, std::vector<bool>(result.dim, false));
        result.variance = result.m2 - result.mean2;
        return result;
    }

    std::vector<int> getTTBondDimensions(const TensorCI &ci) {
        std::vector<int> dimensions;
        for(std::size_t k = 0; k + 1 < ci.tt.M.size(); ++k) dimensions.push_back(static_cast<int>(ci.tt.M[k].n_slices));
        return dimensions;
    }

    bool approximatelyEqual(double a, double b, double tolerance) {
        return std::isfinite(a) && std::isfinite(b) &&
               std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
    }

    bool approximationUnchanged(const Status &current, const Status &previous) {
        if(!approximatelyEqual(current.statistics.mean, previous.statistics.mean, current.changeTol) ||
           !approximatelyEqual(current.statistics.variance, previous.statistics.variance, current.changeTol) ||
           current.bondDims != previous.bondDims ||
           current.sobolResult.sobolIndices.size() != previous.sobolResult.sobolIndices.size())
            return false;
        for(std::size_t i = 0; i < current.sobolResult.sobolIndices.size(); ++i)
            if(!approximatelyEqual(current.sobolResult.sobolIndices[i].estimated,
                                   previous.sobolResult.sobolIndices[i].estimated, current.changeTol))
                return false;
        return true;
    }

    void recordReachedAccuracyTargets(TargetProgress &progress, const std::vector<double> &targets,
                                      const Status &status) {
        if(!std::isfinite(status.sobolResult.rmse)) return;
        for(std::size_t i = 0; i < targets.size(); ++i)
            if(progress.firstEval[i] == 0 && status.sobolResult.rmse <= targets[i])
                progress.firstEval[i] = static_cast<std::size_t>(status.nEval);
    }

    void updateSweepStatus(Status &status, const TensorCI &ci, const Grids &weights, const SobolReference &reference,
                           const std::vector<double> &targets, TargetProgress &progress) {
        status.sweeps      = ci.cIter - 1; // xfac constructs the TT with one initial half-sweep.
        status.nEval       = static_cast<int>(ci.f.nEval());
        status.bondDims    = getTTBondDimensions(ci);
        status.statistics  = calculateMoments(ci.tt, weights);
        status.sobolResult = calculateSobolIndices(ci.tt, weights, status.statistics, reference);
        if(hasExactSobolIndices(reference, status.statistics.dim)) {
            recordReachedAccuracyTargets(progress, targets, status);
        }
    }

    // Return true only when both the requested accuracy and independent grid accuracy are satisfactory.
    bool hasConverged(Status &status, const TensorCI &ci, const Grids &grids, const ConvergenceData &control,
                      bool hasExact, const GridSampleAccuracy &probes) {
        if(!ci.isDone()) return false;
        const double rmse = hasExact ? status.sobolResult.rmse : offGridPredictionRMSE(ci, grids, control);
        if(!std::isfinite(rmse) || rmse > cfg::convTol || probes.relRmse > AnalysisSettings::probeTolerance)
            return false;
        status.stopMsg = fmt::format("Converged: target RMSE {:.3e}, grid RMSE {:.3e}", rmse, probes.relRmse);
        return true;
    }

    // Return true when the current TT has stopped changing and recovery should be considered.
    bool hasStagnated(Status &current, const Status &previous, const TensorCI &ci) {
        if(!ci.isDone()) {
            current.sweepsUnchanged = 0;
            return false;
        }
        current.sweepsUnchanged = approximationUnchanged(current, previous) ? previous.sweepsUnchanged + 1 : 0;
        return current.sweepsUnchanged >= current.minSweepsUnchanged;
    }

    // A recovery attempt is permitted only if an inaccurate, unused grid pivot is available.
    bool insertRecoveryPivot(TensorCI &ci, Status &status, const GridSampleAccuracy &accuracy, RecoveryState &recovery,
                             int maxSweeps) {
        if(accuracy.relRmse <= AnalysisSettings::probeTolerance) {
            status.stopMsg = "Saturated: grid-point error small, but target RMSE not reached";
            return false;
        }
        if(recovery.attempts >= AnalysisSettings::maxRecoveryAttempts) {
            status.stopMsg = "Saturated: maximum pivot recovery attempts reached";
            return false;
        }
        if(accuracy.proposedPivot.empty()) {
            status.stopMsg = "Saturated: no unused grid-probe pivot available";
            return false;
        }
        // addPivotsAllBonds and makeCanonical each advance xfac's internal half-sweep counter.
        constexpr int recoverySweeps = 4;
        if(status.sweeps + recoverySweeps > maxSweeps) {
            status.stopMsg = "Maximum half-sweeps reached before pivot recovery";
            return false;
        }
        const auto &pivot = accuracy.proposedPivot;
        fmt::print("\n=== Pivot recovery {} ===\nInserting global pivot: {}\n", recovery.attempts + 1, pivot);
        recovery.insertedPivots.insert(pivot);
        ci.addPivotsAllBonds({pivot});
        ci.makeCanonical();
        ++recovery.attempts;
        status.sweepsUnchanged = 0;
        return true;
    }

    void runTCISweeps(TensorCI &ci, const Grids &grids, const Grids &weights, const SobolReference &reference,
                      const ConvergenceData &control, const std::vector<util::FunctionSample> &gridProbes,
                      const std::vector<double> &targets, int maxSweeps, std::vector<Status> &statusLog,
                      TargetProgress &progress) {
        const bool    hasExact = hasExactSobolIndices(reference, static_cast<int>(grids.size()));
        RecoveryState recovery;
        statusLog.emplace_back();
        updateSweepStatus(statusLog.back(), ci, weights, reference, targets, progress);
        printSweepStatus(statusLog.back(), nullptr, reference);
        while(statusLog.back().sweeps < maxSweeps) {
            const Status previous = statusLog.back();
            ci.iterate();
            statusLog.push_back(previous);
            Status &current = statusLog.back();
            updateSweepStatus(current, ci, weights, reference, targets, progress);
            printSweepStatus(current, &previous, reference);

            auto accuracy = measureGridApproximationError(ci, gridProbes, recovery.insertedPivots);
            // printGridProbeAccuracy(accuracy);
            if(hasConverged(current, ci, grids, control, hasExact, accuracy)) break;
            if(hasStagnated(current, previous, ci)) {
                fmt::print("\n=== TCI saturation detected ===\n");
                if(!insertRecoveryPivot(ci, current, accuracy, recovery, maxSweeps)) break;
                // Record the post-recovery TT as a separate status rather than overwrite a sweep.
                const Status beforeRecovery = current;
                statusLog.push_back(beforeRecovery);
                Status &recovered = statusLog.back();
                updateSweepStatus(recovered, ci, weights, reference, targets, progress);
                printSweepStatus(recovered, &beforeRecovery, reference);
                const auto recoveredAccuracy = measureGridApproximationError(ci, gridProbes, recovery.insertedPivots);
                // printGridProbeAccuracy(recoveredAccuracy);
                fmt::print("Bond dimensions after recovery: {}\n\n", recovered.bondDims);
                if(hasConverged(recovered, ci, grids, control, hasExact, recoveredAccuracy)) break;
            }
        }
        if(statusLog.back().stopMsg.empty())
            statusLog.back().stopMsg = fmt::format("MaxSweeps: maximum half-sweeps reached ({})", maxSweeps);
    }

    std::vector<Predictions> calculatePredictions(const BaseFunction &function, const TensorCI &ci,
                                                  const Grids &grids) {
        std::vector<Predictions> predictions;
        if(cfg::npredictions <= 0) return predictions;
        predictions.reserve(cfg::npredictions);
        const auto ctt      = interp::interpolateTT(ci.tt, grids);
        double     maxError = 0, sumError2 = 0;
        fmt::print("=== Model predictions ===\n");
        for(int i = 0; i < cfg::npredictions; ++i) {
            const auto   point     = randomOffGridPoint(function, grids);
            const double predicted = ctt.eval(point);
            const double actual    = function.evaluate(point);
            const double error     = std::abs(predicted - actual);
            maxError               = std::max(maxError, error);
            sumError2 += error * error;
            predictions.push_back({predicted, actual, error, point});
        }
        fmt::print("maxError: {:<15.8e} | rmse {:<15.8e}\n", maxError, std::sqrt(sumError2 / predictions.size()));
        return predictions;
    }

    void runSensitivityAnalysis(const BaseFunction &function, const std::vector<double> &relTolTargets, int bondDim,
                                int maxSweeps, const Quadrature &makeQuadrature) {
        const auto reference = function.exactSobolIndices();
        auto h5file = h5pp::File(fmt::format("{}/{}.h5", cfg::outdir, function.name()), h5pp::FileAccess::REPLACE);
        auto [grids, weights] = makeIntegrationGrids(function, makeQuadrature);
        auto param            = makeTCIParameters(function, grids, bondDim);
        writeSimulationParameters(h5file, param, function, relTolTargets);

        FunctionEvaluator evaluate{function};
        TensorCI          ci(evaluate, grids, param);

        const auto     gridProbes = sampleGridReferencePoints(function, grids, AnalysisSettings::nGridProbes);
        const auto     control    = hasExactSobolIndices(reference, function.dimension())
                                        ? ConvergenceData{}
                                        : sampleOffGridControlPoints(function, grids, AnalysisSettings::nControlPoints);
        TargetProgress progress(relTolTargets.size());
        std::vector<Status> statusLog;
        printSweepHeader(function, hasExactSobolIndices(reference, function.dimension()));
        runTCISweeps(ci, grids, weights, reference, control, gridProbes, relTolTargets, maxSweeps, statusLog, progress);

        writeStatusLog(h5file, statusLog);
        printFinalResults(statusLog.back(), reference, relTolTargets, progress, function.dimension(),
                          gridProbes.size());
        writePredictions(h5file, calculatePredictions(function, ci, grids));
        sil::writeSIL(h5file, sil::calculateSIL(ci.tt, weights));
    }
}
