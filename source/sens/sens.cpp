#include "sens.hpp"
#include "accuracy.hpp"
#include "cfg/cfg.hpp"
#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "grid.hpp"
#include "h5t.hpp"
#include "hdf5.hpp"
#include "info/sil.hpp"
#include "interp/interp.hpp"
#include "io/environment.hpp"
#include "print.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "samples.hpp"
#include "sobol.hpp"
#include "status.hpp"
#include "types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <h5pp/h5pp.h>
#include <H5Tpublic.h>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {

    struct AnalysisSettings {
        static constexpr int    nNodeSamples        = 128;
        static constexpr int    maxRecoveryAttempts = 3;
        static constexpr double nodeSampleTolerance = 1e-10;
    };

    struct FunctionEvaluator {
        const BaseFunction &function;
        double              operator()(const std::vector<double> &point) const { return function.evaluate(point); }
    };

    std::vector<int> selectInitialPivot(const BaseFunction &function, const Grid &grid, rnd::Rng &rng) {
        if(grid.empty()) throw std::invalid_argument("Empty quadrature grids");
        const int           dim = static_cast<int>(grid.size());
        std::vector<int>    pivot(dim);
        std::vector<double> point(dim);
        for(int trial = 0; trial < 20; ++trial) {
            for(int k = 0; k < dim; ++k) {
                if(grid[k].empty()) throw std::invalid_argument("Empty quadrature grid");
                pivot[k] = rnd::uniform_integer_box<int>(0, static_cast<int>(grid[k].size()) - 1, rng);
                point[k] = grid[k][pivot[k]];
            }
            const double value = std::abs(function.evaluate(point));
            if(std::isfinite(value) && value != 0.0) {
                fmt::print("param.pivot1: {} | value {:.8e}\n", pivot, value);
                return pivot;
            }
        }
        throw std::runtime_error("Failed to find a nonzero initial pivot within 20 trials");
    }

    xfac::TensorCI2Param makeTCIParameters(const BaseFunction &function, const Grid &grid, int bondDim, rnd::Rng &rng) {
        xfac::TensorCI2Param param;
        param.bondDim           = bondDim;
        param.reltol            = 1e-12;
        param.fullPiv           = false;
        param.useCachedFunction = true;
        param.nRookIter         = 3;
        param.pivot1            = selectInitialPivot(function, grid, rng);
        return param;
    }

    template<typename TT>
    Statistics calculateMoments(const TT &tt, const IntegrationWeights &weights) {
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

    void updateSweepStatus(Status &status, const TensorCI &ci, const Discretization &discretization,
                           const IntegrationWeights &weights, const FunctionSamples &samples,
                           const SobolReference &reference, const std::vector<double> &targets,
                           TargetProgress &progress) {
        status.sweeps                 = ci.cIter - 1;
        status.nEval                  = static_cast<int>(ci.f.nEval());
        status.bondDims               = getTTBondDimensions(ci);
        status.statistics             = calculateMoments(ci.tt, weights);
        status.sobolResult            = calculateSobolIndices(ci.tt, weights, status.statistics, reference);
        const auto predictionError    = measurePredictionError(ci, discretization, samples);
        status.predictionRMSE         = predictionError.rmse;
        status.predictionRelativeRMSE = predictionError.relativeRMSE;
        if(hasExactSobolIndices(reference, status.statistics.dim)) {
            recordReachedAccuracyTargets(progress, targets, status);
        }
    }

    // Return true only when both the requested accuracy and independent grid accuracy are satisfactory.
    bool hasConverged(Status &status, const TensorCI &ci, const NodeSampleError &nodeError) {
        if(!ci.isDone()) return false;
        const bool predictionsAccurate =
            std::isfinite(status.predictionRelativeRMSE) && status.predictionRelativeRMSE <= cfg::convTol;
        const bool nodeSamplesAccurate =
            std::isfinite(nodeError.relativeRMSE) && nodeError.relativeRMSE <= AnalysisSettings::nodeSampleTolerance;
        if(!predictionsAccurate || !nodeSamplesAccurate) return false;
        status.stopMsg = fmt::format("Converged: prediction relRMSE {:.3e}, node relRMSE {:.3e}",
                                     status.predictionRelativeRMSE, nodeError.relativeRMSE);
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
    bool insertRecoveryPivot(TensorCI &ci, Status &status, const NodeSampleError &accuracy, RecoveryState &recovery,
                             int maxSweeps) {
        if(accuracy.relativeRMSE <= AnalysisSettings::nodeSampleTolerance) {
            status.stopMsg = "Saturated: grid-point error small, but target RMSE not reached";
            return false;
        }
        if(recovery.attempts >= AnalysisSettings::maxRecoveryAttempts) {
            status.stopMsg = "Saturated: maximum pivot recovery attempts reached";
            return false;
        }
        if(accuracy.suggestedPoint.empty()) {
            status.stopMsg = "Saturated: no unused node sample available";
            return false;
        }
        // addPivotsAllBonds and makeCanonical each advance xfac's internal half-sweep counter.
        constexpr int recoverySweeps = 4;
        if(status.sweeps + recoverySweeps > maxSweeps) {
            status.stopMsg = "Maximum half-sweeps reached before pivot recovery";
            return false;
        }
        const auto &pivot = accuracy.suggestedPoint;
        fmt::print("\n=== Pivot recovery {} ===\nInserting global pivot: {}\n", recovery.attempts + 1, pivot);
        recovery.insertedPoints.insert(pivot);
        ci.addPivotsAllBonds({pivot});
        ci.makeCanonical();
        ++recovery.attempts;
        status.sweepsUnchanged = 0;
        return true;
    }

    void runTCISweeps(TensorCI &ci, const Discretization &discretization, const IntegrationWeights &weights,
                      const SobolReference &reference, const FunctionSamples &funcSamples,
                      const NodeSamples &nodeSamples, const std::vector<double> &targets, int maxSweeps,
                      std::vector<Status> &statusLog, TargetProgress &progress) {
        const bool    hasExact = hasExactSobolIndices(reference, static_cast<int>(discretization.nodes.size()));
        RecoveryState recovery;
        statusLog.emplace_back();
        updateSweepStatus(statusLog.back(), ci, discretization, weights, funcSamples, reference, targets, progress);
        printSweepStatus(statusLog.back(), reference);
        while(statusLog.back().sweeps < maxSweeps) {
            const Status previous = statusLog.back();
            ci.iterate();
            statusLog.push_back(previous);
            Status &current = statusLog.back();
            updateSweepStatus(current, ci, discretization, weights, funcSamples, reference, targets, progress);
            printSweepStatus(current, reference);

            const auto nodeError = measureNodeSampleError(ci, nodeSamples, recovery.insertedPoints);
            if(hasConverged(current, ci, nodeError)) break;
            if(hasStagnated(current, previous, ci)) {
                fmt::print("\n=== TCI saturation detected ===\n");
                if(!insertRecoveryPivot(ci, current, nodeError, recovery, maxSweeps)) break;
                // Record the post-recovery TT as a separate status rather than overwrite a sweep.
                const Status beforeRecovery = current;
                statusLog.push_back(beforeRecovery);
                Status &recovered = statusLog.back();
                updateSweepStatus(recovered, ci, discretization, weights, funcSamples, reference, targets, progress);
                printSweepStatus(recovered, reference);
                const auto recoveredAccuracy = measureNodeSampleError(ci, nodeSamples, recovery.insertedPoints);
                fmt::print("Bond dimensions after recovery: {}\n\n", recovered.bondDims);
                if(hasConverged(recovered, ci, recoveredAccuracy)) break;
            }
        }
        if(statusLog.back().stopMsg.empty())
            statusLog.back().stopMsg = fmt::format("MaxSweeps: maximum half-sweeps reached ({})", maxSweeps);
    }

    std::vector<PredictionRecord> calculatePredictions(const TensorCI &ci, const Discretization &discretization,
                                                       const FunctionSamples &samples) {
        const auto interpolatedTT = interp::interpolateTT(ci.tt, discretization);

        std::vector<PredictionRecord> predictions;
        predictions.reserve(samples.size());

        for(const auto &sample : samples) {
            const double predicted = interpolatedTT.eval(sample.point);

            predictions.push_back({predicted, sample.value, std::abs(predicted - sample.value), sample.point});
        }

        return predictions;
    }

    void runSensitivityAnalysis(const BaseFunction &function, const std::vector<double> &relTolTargets, int bondDim,
                                int maxSweeps, const QuadratureRuleFactory &makeRule,
                                const std::optional<cfg::InputConfiguration> &inputCfg, rnd::RandomEngines &rngs) {
        auto h5file =
            h5pp::File(fmt::format("{}/{}.{}.h5", cfg::outdir, function.name(), cfg::seed), h5pp::FileAccess::REPLACE);
        ::env::writeEnvironmentMetadata(h5file);
        h5file.writeDataset(rngs.seeds.init, "param/rng/master_seed");
        h5file.writeDataset(rngs.seeds.pivot, "param/rng/pivot_seed");
        h5file.writeDataset(rngs.seeds.nodeSamples, "param/rng/node_sample_seed");
        h5file.writeDataset(rngs.seeds.predictions, "param/rng/prediction_seed");
        h5file.writeDataset(rngs.seeds.xfac, "param/rng/xfac_seed");
        h5file.writeDataset(std::string("splitmix64-tags-v1"), "param/rng/seed_derivation");
        const auto discretization = inputCfg.has_value()
                                        ? discretizeInputs(function, inputCfg.value(), cfg::permutation)
                                        : discretizeInputs(function, makeRule);
        // Benchmark Sobol references assume continuous uniform inputs on the original domains.
        // Changed domains or discrete input probabilities generally give different Sobol indices.
        const bool allContinuous      = std::all_of(discretization.kinds.begin(), discretization.kinds.end(),
                                                    [](InputKind kind) { return kind == InputKind::Continuous; });
        const bool hasMatchingDomains = allContinuous && domainsMatch(function.domains(), discretization.domains);
        const auto reference          = hasMatchingDomains ? function.exactSobolIndices() : SobolReference{};
        if(!hasMatchingDomains)
            fmt::print("Exact Sobol references disabled: input domains or distributions differ from benchmark.\n");
        const auto &nodes   = discretization.nodes;
        const auto &weights = discretization.integrationWeights;
        if(inputCfg) {
            std::vector<std::size_t> nodeCounts;
            nodeCounts.reserve(nodes.size());
            for(const auto &axis : nodes) nodeCounts.push_back(axis.size());
            fmt::print("TOML discretization nodes per parameter: [{}]\n", fmt::join(nodeCounts, ", "));
        }
        const auto param = makeTCIParameters(function, nodes, bondDim, rngs.pivot);

        writeSimulationParameters(h5file, param, function, relTolTargets);
        writeInputMetadata(h5file, discretization, inputCfg, cfg::permutation, cfg::quadratureRuleName);

        FunctionEvaluator evaluate{function};
        TensorCI          ci(evaluate, nodes, param);

        // Samples used to identify missing information on the node grid.
        const auto nodeSamples = sampleNodePoints(function, nodes, AnalysisSettings::nNodeSamples, rngs.nodeSamples);
        // Independent samples used only to measure prediction accuracy.
        const auto functionSamples = sampleFunctionValues(function, discretization, cfg::nsamples, rngs.predictions);

        TargetProgress      progress(relTolTargets.size());
        std::vector<Status> statusLog;

        printSweepHeader(function);
        runTCISweeps(ci, discretization, weights, reference, functionSamples, nodeSamples, relTolTargets, maxSweeps,
                     statusLog, progress);

        writeStatusLog(h5file, statusLog);
        printFinalResults(ci, statusLog.back(), reference, functionSamples, nodeSamples, function.dimension());

        const auto predictions = calculatePredictions(ci, discretization, functionSamples);
        printPredictions(predictions);
        writePredictions(h5file, predictions);
        sil::writeSIL(h5file, sil::calculateSIL(ci.tt, weights));
    }

}
