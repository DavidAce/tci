#pragma once

#include <vector>

class BaseFunction;

namespace sens {
    // ------------------------------------------------------------
    // TCI2 benchmark runner
    // ------------------------------------------------------------
    void runSensitivityAnalysis(const BaseFunction &benchmark, const std::vector<double> &targets, int bondDim,
                                int maxSweeps);
}