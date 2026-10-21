#pragma once
#include "samples.hpp"
#include "sens/h5t.hpp"
#include <vector>

class BaseFunction;
namespace xfac {
    template<class T1>
    class TensorCI2;
}
namespace sens {
    struct SobolReference;
    struct SobolResult;
    struct Status;
    struct TargetProgress;
    struct NodeSampleError;
    struct PredictionRecord;

    void   printSweepHeader(const BaseFunction &function);
    void   printSweepStatus(const Status &status, const SobolReference &reference);
    void   printSobolIndices(const SobolResult &result, int dim, bool hasExact);
    double printSobolGroup(const SobolResult &result, const char *name, std::size_t offset, std::size_t count,
                           bool hasExact);
    void   printSobolSummary(const SobolResult &result, const SobolReference &reference, int dim, bool hasExact);
    void   printPredictions(const std::vector<PredictionRecord> &predictions, bool printIndividualSamples = false);
    void   printFinalResults(const xfac::TensorCI2<double> &ci, const Status &finalStatus,
                             const SobolReference &reference, const FunctionSamples &functionSamples,
                             const NodeSamples &nodeSamples, int dim);
    void   printNodeSampleError(const NodeSampleError &accuracy);

}