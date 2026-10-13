#pragma once

#include "quad/quadrature.hpp"
#include <functional>
#include <vector>

class BaseFunction;

namespace sens {
    // ------------------------------------------------------------
    // TCI2 benchmark runner
    // ------------------------------------------------------------
    using Quadrature = std::function<quadrature::Rule(double lo, double hi)>;
    void runSensitivityAnalysis(const BaseFunction &benchmark, const std::vector<double> &targets, int bondDim,
                                int maxSweeps, const Quadrature &makeQuadrature);
}