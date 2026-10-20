#pragma once
#include <vector>

class BaseFunction;

namespace sens {
    struct SobolReference;
    struct SobolResult;

    namespace util {
        struct Status;
        struct TargetProgress;
    }
    using namespace util;

    void   printSweepHeader(const BaseFunction &function, bool hasExact);
    void   printTargetEvaluations(const std::vector<double> &targets, const TargetProgress &progress);
    void   printSobolIndices(const SobolResult &result, int dim, bool hasExact);
    double printSobolGroup(const SobolResult &result, const char *name, std::size_t offset, std::size_t count,
                           bool hasExact);
    void   printSobolSummary(const SobolResult &result, const SobolReference &reference, int dim, bool hasExact);
    void   printFinalResults(const Status &finalStatus, const SobolReference &reference,
                             const std::vector<double> &targets, const TargetProgress &progress, int dim,
                             std::size_t nGridProbes);

}