#pragma once

#include "samples.hpp"
#include "types.hpp"
#include <limits>
#include <set>

class BaseFunction;

namespace xfac {
    template<class T1, class T2>
    class CTensorCI2;
}

namespace sens {
    struct Discretization;

    using TensorCI = xfac::CTensorCI2<double, double>;

    struct NodeSampleError {
        double rmse             = 0.0;
        double relativeRMSE     = 0.0;
        double maxAbsoluteError = 0.0;

        NodeIndices suggestedPoint;
    };

    struct PredictionError {
        double rmse             = std::numeric_limits<double>::quiet_NaN();
        double relativeRMSE     = std::numeric_limits<double>::quiet_NaN();
        double maxAbsoluteError = std::numeric_limits<double>::quiet_NaN();
    };

    FunctionSamples sampleFunctionValues(const BaseFunction &function, const Discretization &discretization, int count);
    NodeSampleError measureNodeSampleError(const TensorCI &ci, const NodeSamples &samples,
                                           const std::set<NodeIndices> &insertedPoints);
    PredictionError measurePredictionError(const TensorCI &ci, const Grid &grid, const FunctionSamples &samples);

}
