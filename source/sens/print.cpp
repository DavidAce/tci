#include "print.hpp"
#include "accuracy.hpp"
#include "func/BaseFunction.hpp"
#include "h5t.hpp"
#include "sobol.hpp"
#include "status.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <xfac/tensor/tensor_ci_2.h>
namespace sens {

    void printSweepHeader(const BaseFunction &function) {
        fmt::print("\n=== {} (dim={}) ===\n", function.name(), function.dimension());

        fmt::print("{:<8} {:>7} {:>9} {:>13} {:>13} {:>13}\n", "method", "sweeps", "nEval", "predRMSE", "predRelRMSE",
                   "sobolRMSE");
    }

    void printSweepStatus(const Status &status, const SobolReference &reference) {
        const bool        hasExact  = hasExactSobolIndices(reference, status.statistics.dim);
        const std::string sobolRMSE = hasExact ? fmt::format("{:.4e}", status.sobolResult.rmse) : "-";
        fmt::print("{:<8} {:>7} {:>9} {:>13.4e} {:>13.4e} {:>13}\n", "TCI2", status.sweeps, status.nEval,
                   status.predictionRMSE, status.predictionRelativeRMSE, sobolRMSE);
    }

    void printSobolIndices(const SobolResult &result, int dim, bool hasExact) {
        if(hasExact)
            fmt::print("\n{:<8} {:>14} {:>14} {:>12}\n", "index", "estimated", "exact", "absError");
        else
            fmt::print("\n{:<8} {:>14}\n", "index", "estimated");
        const std::size_t count = hasExact && result.sobolIndices.size() <= 10
                                      ? result.sobolIndices.size()
                                      : std::min(result.sobolIndices.size(), static_cast<std::size_t>(2 * dim));
        for(std::size_t i = 0; i < count; ++i) {
            const auto &index = result.sobolIndices[i];
            if(hasExact)
                fmt::print("{:<8} {:>14.8f} {:>14.8f} {:>12.3e}\n", index.name, index.estimated, index.exact,
                           index.error);
            else
                fmt::print("{:<8} {:>14.8f}\n", index.name, index.estimated);
        }
    }

    double printSobolGroup(const SobolResult &result, const char *name, std::size_t offset, std::size_t count,
                           bool hasExact) {
        if(count == 0) return 0;
        double estimated = 0, exact = 0, error2 = 0;
        for(std::size_t i = offset; i < offset + count; ++i) {
            const auto &index = result.sobolIndices.at(i);
            estimated += index.estimated;
            if(hasExact) {
                exact += index.exact;
                error2 += index.error * index.error;
            }
        }
        if(hasExact)
            fmt::print("{:<12} {:>6} {:>14.8f} {:>14.8f} {:>12.3e} {:>12.3e}\n", name, count, estimated, exact,
                       std::abs(estimated - exact), std::sqrt(error2 / count));
        else
            fmt::print("{:<12} {:>6} {:>14.8f}\n", name, count, estimated);
        return estimated;
    }

    void printSobolSummary(const SobolResult &result, const SobolReference &reference, int dim, bool hasExact) {
        if(hasExact)
            fmt::print("\n{:<12} {:>6} {:>14} {:>14} {:>12} {:>12}\n", "order", "count", "sum est.", "sum exact",
                       "absError", "RMSE");
        else
            fmt::print("\n{:<12} {:>6} {:>14}\n", "order", "count", "sum est.");
        const std::size_t nSecond  = hasExact ? reference.second.size() : static_cast<std::size_t>(dim) * (dim - 1) / 2;
        const double      sumFirst = printSobolGroup(result, "S_i", 0, dim, hasExact);
        const double      sumSecond = printSobolGroup(result, "S_ij", 2 * dim, nSecond, hasExact);
        fmt::print("{:<12} {:>6} {:>14.8f}\n", "S_residual", "-", 1.0 - sumFirst - sumSecond);
    }

    void printPredictions(const std::vector<PredictionRecord> &predictions, bool printIndividualSamples) {
        if(predictions.empty()) return;

        double sumSquaredError  = 0.0;
        double sumSquaredValues = 0.0;
        double maxAbsoluteError = 0.0;

        for(const auto &prediction : predictions) {
            const double error = prediction.predicted - prediction.funcvalue;

            sumSquaredError += error * error;
            sumSquaredValues += prediction.funcvalue * prediction.funcvalue;
            maxAbsoluteError = std::max(maxAbsoluteError, std::abs(error));
        }
        const double rmse         = std::sqrt(sumSquaredError / predictions.size());
        const double relativeRMSE = sumSquaredValues > 0.0 ? std::sqrt(sumSquaredError / sumSquaredValues)
                                                           : std::numeric_limits<double>::quiet_NaN();

        fmt::print("\n=== Function sample predictions ===\n");
        fmt::print("Number of samples : {}\n", predictions.size());
        fmt::print("RMSE              :        {:.8e}\n", rmse);
        fmt::print("Relative RMSE     : {:.8e}\n", relativeRMSE);
        fmt::print("Maximum error     : {:.8e}\n", maxAbsoluteError);

        if(!printIndividualSamples) return;

        fmt::print("\n{:>6} {:>16} {:>16} {:>16}\n", "sample", "function value", "TT prediction", "absError");

        for(std::size_t i = 0; i < predictions.size(); ++i) {
            const auto &prediction = predictions[i];
            fmt::print("{:>6} {:>16.8e} {:>16.8e} {:>16.8e}\n", i, prediction.funcvalue, prediction.predicted,
                       prediction.error);
        }
    }

    void printFinalResults(const xfac::TensorCI2<double> &ci, const Status &finalStatus,
                           const SobolReference &reference, const FunctionSamples &functionSamples,
                           const NodeSamples &nodeSamples, int dim) {
        const bool hasExact = hasExactSobolIndices(reference, dim);
        fmt::print("{}\n", finalStatus.stopMsg);

        fmt::print("\nFinal bond dimensions: {}\nMean: {:.8e}, Variance: {:.8e}\n", finalStatus.bondDims,
                   finalStatus.statistics.mean, finalStatus.statistics.variance);
        printSobolIndices(finalStatus.sobolResult, dim, hasExact);
        printSobolSummary(finalStatus.sobolResult, reference, dim, hasExact);
        if(hasExact) {
            fmt::print("\n=== Final TCI accuracy ===\n");
            fmt::print("{:<10} {:>8} {:>12} {:>12} {:>14} {:>14}\n", "method", "nEval", "RMSE", "maxDiff", "mean",
                       "variance");
            fmt::print("{:<10} {:>8} {:>12.4e} {:>12.4e} {:>14.6e} {:>14.6e}\n", "TCI2", finalStatus.nEval,
                       finalStatus.sobolResult.rmse, finalStatus.sobolResult.maxDiff, finalStatus.statistics.mean,
                       finalStatus.statistics.variance);
        }

        const std::size_t nTCI             = ci.f.nEval();
        const std::size_t nNodeSamples     = nodeSamples.size();
        const std::size_t nFunctionSamples = functionSamples.size();

        fmt::print("\nFunction evaluations:\n"
                   "  TCI and recovery: {}\n"
                   "  Node samples: {}\n"
                   "  Independent function samples: {}\n"
                   "  Subtotal: {}\n",
                   nTCI, nNodeSamples, nFunctionSamples, nTCI + nNodeSamples + nFunctionSamples);
    }
    void printNodeSampleError(const NodeSampleError &accuracy) {
        fmt::print("Grid samples: RMSE={:.4e}, relRMSE={:.4e}, maxAbs={:.4e}\n", accuracy.rmse, accuracy.relativeRMSE,
                   accuracy.maxAbsoluteError);
    }
}