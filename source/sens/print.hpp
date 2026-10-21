#pragma once
#include <vector>

class BaseFunction;

namespace sens {
    struct SobolReference;
    struct SobolResult;

    namespace util {
        struct Status;
        struct TargetProgress;
        struct ProbeAccuracy;
    }

    void   printSweepHeader(const BaseFunction &function, bool hasExact);
    void   printSweepStatus(util::Status &status, const util::Status *previous, const SobolReference &reference);
    void   printTargetEvaluations(const std::vector<double> &targets, const util::TargetProgress &progress);
    void   printSobolIndices(const SobolResult &result, int dim, bool hasExact);
    double printSobolGroup(const SobolResult &result, const char *name, std::size_t offset, std::size_t count,
                           bool hasExact);
    void   printSobolSummary(const SobolResult &result, const SobolReference &reference, int dim, bool hasExact);
    void   printFinalResults(const util::Status &finalStatus, const SobolReference &reference,
                             const std::vector<double> &targets, const util::TargetProgress &progress, int dim,
                             std::size_t nGridProbes);

    void printGridProbeAccuracy(const util::ProbeAccuracy &accuracy);

}