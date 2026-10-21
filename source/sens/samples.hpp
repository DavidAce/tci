#pragma once

#include "types.hpp"
#include <vector>

namespace sens {

    // One directly evaluated function value at a continuous point.
    struct FunctionSample {
        Point  point;
        double value = 0.0;
    };

    // One directly evaluated function value at a point on the grid.
    struct NodeSample {
        NodeIndices indices;
        double      value = 0.0;
    };

    using FunctionSamples = std::vector<FunctionSample>;
    using NodeSamples     = std::vector<NodeSample>;

}
