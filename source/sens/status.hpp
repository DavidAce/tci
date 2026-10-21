#pragma once

#include "sobol.hpp"
#include "types.hpp"
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace sens {

    struct Status {
        int sweeps             = 0;
        int sweepsUnchanged    = 0;
        int minSweepsUnchanged = 3;

        double changeTol = 1e-12;
        int    nEval     = 0;

        Statistics  statistics;
        SobolResult sobolResult;

        double predictionRMSE         = std::numeric_limits<double>::quiet_NaN();
        double predictionRelativeRMSE = std::numeric_limits<double>::quiet_NaN();

        std::vector<int> bondDims;
        std::string      stopMsg;
    };

    struct TargetProgress {
        std::vector<std::size_t> firstEval;

        explicit TargetProgress(std::size_t count) : firstEval(count, 0) {}
    };

    struct RecoveryState {
        std::set<NodeIndices> insertedPoints;
        int                   attempts = 0;
    };

}
