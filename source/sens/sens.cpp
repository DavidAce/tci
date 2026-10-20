#include "sens.hpp"
#include "cfg/cfg.hpp"
#include "file.hpp"
#include "func/BaseFunction.hpp"
#include "grid.hpp"
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
    using TensorCI = xfac::CTensorCI2<double, double>;

    struct AnalysisSettings {
        static constexpr int    nGridProbes         = 128;
        static constexpr int    nControlPoints      = 16;
        static constexpr int    maxRecoveryAttempts = 3;
        static constexpr double probeTolerance      = 1e-10;
    };

    std::vector<int> randomInitialPivot(const BaseFunction &function, const Grids &grids) {
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
        param.pivot1            = randomInitialPivot(function, grids);
        return param;
    }

    template<typename TT>
    Statistics calculateStatistics(const TT &tt, const Grids &weights) {
        Statistics result;
        result.dim      = static_cast<int>(weights.size());
        result.mean     = tt.sum(weights);
        result.mean2    = result.mean * result.mean;
        result.m2       = conditional_mean_square(tt, weights, std::vector<bool>(result.dim, false));
        result.variance = result.m2 - result.mean2;
        return result;
    }

    std::vector<int> bondDimensions(const TensorCI &ci) {
        std::vector<int> dimensions;
        for(std::size_t k = 0; k + 1 < ci.tt.M.size(); ++k) dimensions.push_back(static_cast<int>(ci.tt.M[k].n_slices));
        return dimensions;
    }

    ProbeAccuracy assessGridProbes(const TensorCI &ci, const std::vector<GridProbe> &probes,
                                   const std::set<std::vector<int>> &insertedPivots) {
        ProbeAccuracy accuracy;
        if(probes.empty()) return accuracy;
        double error2 = 0, exact2 = 0, largestUnusedError = -1;
        for(const auto &probe : probes) {
            const double error = ci.tt.eval(probe.indices) - probe.exact;
            if(!std::isfinite(error)) throw std::runtime_error("Non-finite grid-probe error");
            const double absError = std::abs(error);
            error2 += error * error;
            exact2 += probe.exact * probe.exact;
            accuracy.maxAbsError = std::max(accuracy.maxAbsError, absError);
            if(!insertedPivots.contains(probe.indices) && absError > largestUnusedError) {
                largestUnusedError     = absError;
                accuracy.proposedPivot = probe.indices;
            }
        }
        accuracy.rmse    = std::sqrt(error2 / probes.size());
        accuracy.relRmse = exact2 > 0 ? std::sqrt(error2 / exact2) : accuracy.rmse;
        return accuracy;
    }

    void printGridProbeAccuracy(const ProbeAccuracy &accuracy) {
        fmt::print("Grid probes: RMSE={:.4e}, relRMSE={:.4e}, maxAbs={:.4e}\n", accuracy.rmse, accuracy.relRmse,
                   accuracy.maxAbsError);
    }

    ConvergenceData sampleOffGridControlPoints(const BaseFunction &function, const Grids &grids, int count) {
        ConvergenceData data;
        data.points.reserve(count);
        data.values.reserve(count);
        for(int i = 0; i < count; ++i) {
            auto point = randomOffGridPoint(function, grids);
            data.values.push_back(function.evaluate(point));
            data.points.push_back(std::move(point));
        }
        return data;
    }

    double offGridPredictionRMSE(const TensorCI &ci, const Grids &grids, const ConvergenceData &control) {
        if(control.points.empty()) return std::numeric_limits<double>::quiet_NaN();
        if(control.points.size() != control.values.size()) throw std::runtime_error("Control-point size mismatch");
        const auto ctt    = interp::interpolateTT(ci.tt, grids);
        double     error2 = 0;
        for(std::size_t i = 0; i < control.points.size(); ++i) {
            const double error = ctt.eval(control.points[i]) - control.values[i];
            error2 += error * error;
        }
        return std::sqrt(error2 / control.points.size());
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

    double maximumSobolChange(const SobolResult &current, const SobolResult &previous) {
        if(current.sobolIndices.size() != previous.sobolIndices.size()) return std::numeric_limits<double>::quiet_NaN();
        double delta = 0;
        for(std::size_t i = 0; i < current.sobolIndices.size(); ++i)
            delta = std::max(delta, std::abs(current.sobolIndices[i].estimated - previous.sobolIndices[i].estimated));
        return delta;
    }

    void printSweepHeader(const BaseFunction &function, bool hasExact) {
        fmt::print("\n=== {} (dim={}) ===\n", function.name(), function.dimension());
        if(hasExact)
            fmt::print("{:<11} {:>6} {:>8} {:>12} {:>12}\n", "method", "sweeps", "nEval", "RMSE", "maxDiff");
        else
            fmt::print("{:<11} {:>6} {:>8} {:>14} {:>14} {:>12}\n", "method", "sweeps", "nEval", "mean", "variance",
                       "maxDeltaS");
    }

    void recordTargetEvaluations(TargetProgress &progress, const std::vector<double> &targets, const Status &status) {
        if(!std::isfinite(status.sobolResult.rmse)) return;
        for(std::size_t i = 0; i < targets.size(); ++i)
            if(progress.firstEval[i] == 0 && status.sobolResult.rmse <= targets[i])
                progress.firstEval[i] = static_cast<std::size_t>(status.nEval);
    }

    void updateAndPrintStatus(Status &status, const Status *previous, const TensorCI &ci, const Grids &weights,
                              const SobolReference &reference, const std::vector<double> &targets,
                              TargetProgress &progress) {
        status.sweeps      = ci.cIter - 1; // xfac constructs the TT with one initial half-sweep.
        status.nEval       = ci.f.nEval();
        status.bondDims    = bondDimensions(ci);
        status.statistics  = calculateStatistics(ci.tt, weights);
        status.sobolResult = calculateSobol(ci.tt, weights, status.statistics, reference);
        if(hasExactSobolIndices(reference, status.statistics.dim)) {
            fmt::print("{:<11} {:>6} {:>8} {:>12.4e} {:>12.4e}\n", "TCI2", status.sweeps, status.nEval,
                       status.sobolResult.rmse, status.sobolResult.maxDiff);
            recordTargetEvaluations(progress, targets, status);
        } else {
            const double      delta     = previous ? maximumSobolChange(status.sobolResult, previous->sobolResult)
                                                   : std::numeric_limits<double>::quiet_NaN();
            const std::string deltaText = std::isfinite(delta) ? fmt::format("{:.3e}", delta) : "-";
            fmt::print("{:<11} {:>6} {:>8} {:>14.6e} {:>14.6e} {:>12}\n", "TCI2", status.sweeps, status.nEval,
                       status.statistics.mean, status.statistics.variance, deltaText);
        }
    }

    // Return true only when both the requested accuracy and independent grid accuracy are satisfactory.
    bool convergenceReached(Status &status, const TensorCI &ci, const Grids &grids, const ConvergenceData &control,
                            bool hasExact, const ProbeAccuracy &probes) {
        if(!ci.isDone()) return false;
        const double rmse = hasExact ? status.sobolResult.rmse : offGridPredictionRMSE(ci, grids, control);
        if(!std::isfinite(rmse) || rmse > cfg::convTol || probes.relRmse > AnalysisSettings::probeTolerance)
            return false;
        status.stopMsg = fmt::format("Converged: target RMSE {:.3e}, grid RMSE {:.3e}", rmse, probes.relRmse);
        return true;
    }

    // Return true when the current TT has stopped changing and recovery should be considered.
    bool saturationDetected(Status &current, const Status &previous, const TensorCI &ci) {
        if(!ci.isDone()) {
            current.sweepsUnchanged = 0;
            return false;
        }
        current.sweepsUnchanged = approximationUnchanged(current, previous) ? previous.sweepsUnchanged + 1 : 0;
        return current.sweepsUnchanged >= current.minSweepsUnchanged;
    }

    // A recovery attempt is permitted only if an inaccurate, unused grid pivot is available.
    bool recoverFromSaturation(TensorCI &ci, Status &status, const ProbeAccuracy &accuracy, RecoveryState &recovery,
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
                      const ConvergenceData &control, const std::vector<GridProbe> &gridProbes,
                      const std::vector<double> &targets, int maxSweeps, std::vector<Status> &statusLog,
                      TargetProgress &progress) {
        const bool    hasExact = hasExactSobolIndices(reference, static_cast<int>(grids.size()));
        RecoveryState recovery;
        statusLog.emplace_back();
        updateAndPrintStatus(statusLog.back(), nullptr, ci, weights, reference, targets, progress);
        while(statusLog.back().sweeps < maxSweeps) {
            const Status previous = statusLog.back();
            ci.iterate();
            statusLog.push_back(previous);
            Status &current = statusLog.back();
            updateAndPrintStatus(current, &previous, ci, weights, reference, targets, progress);
            auto accuracy = assessGridProbes(ci, gridProbes, recovery.insertedPivots);
            // printGridProbeAccuracy(accuracy);
            if(convergenceReached(current, ci, grids, control, hasExact, accuracy)) break;
            if(saturationDetected(current, previous, ci)) {
                fmt::print("\n=== TCI saturation detected ===\n");
                if(!recoverFromSaturation(ci, current, accuracy, recovery, maxSweeps)) break;
                // Record the post-recovery TT as a separate status rather than overwrite a sweep.
                const Status beforeRecovery = current;
                statusLog.push_back(beforeRecovery);
                Status &recovered = statusLog.back();
                updateAndPrintStatus(recovered, &beforeRecovery, ci, weights, reference, targets, progress);
                const auto recoveredAccuracy = assessGridProbes(ci, gridProbes, recovery.insertedPivots);
                // printGridProbeAccuracy(recoveredAccuracy);
                fmt::print("Bond dimensions after recovery: {}\n\n", recovered.bondDims);
                if(convergenceReached(recovered, ci, grids, control, hasExact, recoveredAccuracy)) break;
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
        writeParam(h5file, param, function, relTolTargets);

        FunctionEvaluator evaluate{function};
        TensorCI          ci(evaluate, grids, param);

        const auto     gridProbes = sampleGridProbes(function, grids, AnalysisSettings::nGridProbes);
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
