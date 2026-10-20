#include "print.hpp"
#include "sobol.hpp"
#include "util.hpp"

namespace sens {
    using namespace util;

    void printSweepHeader(const BaseFunction &function, bool hasExact) {
        fmt::print("\n=== {} (dim={}) ===\n", function.name(), function.dimension());
        if(hasExact)
            fmt::print("{:<11} {:>6} {:>8} {:>12} {:>12}\n", "method", "sweeps", "nEval", "RMSE", "maxDiff");
        else
            fmt::print("{:<11} {:>6} {:>8} {:>14} {:>14} {:>12}\n", "method", "sweeps", "nEval", "mean", "variance",
                       "maxDeltaS");
    }

    void printTargetEvaluations(const std::vector<double> &targets, const TargetProgress &progress) {
        fmt::print("\n{:>12} {:>12}\n", "Target RMSE", "nEval");
        for(std::size_t i = 0; i < targets.size(); ++i) {
            const auto value = progress.firstEval[i] == 0 ? "not reached" : fmt::format("{}", progress.firstEval[i]);
            fmt::print("{:>12.1e} {:>12}\n", targets[i], value);
        }
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

    void printFinalResults(const Status &finalStatus, const SobolReference &reference,
                           const std::vector<double> &targets, const TargetProgress &progress, int dim,
                           std::size_t nGridProbes) {
        const bool hasExact = hasExactSobolIndices(reference, dim);
        fmt::print("{}\n", finalStatus.stopMsg);
        if(hasExact) printTargetEvaluations(targets, progress);
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
        fmt::print("\nFunction evaluations:\n"
                   "  TCI and recovery: {}\n"
                   "  Independent grid probes: {}\n"
                   "  Combined subtotal: {}\n",
                   finalStatus.nEval, nGridProbes, finalStatus.nEval + nGridProbes);
    }

}