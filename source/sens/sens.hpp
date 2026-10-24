#pragma once

#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include <optional>
#include <vector>

class BaseFunction;
namespace cfg {
    struct InputConfiguration;
}

namespace sens {
    // ------------------------------------------------------------
    // TCI2 benchmark runner
    // ------------------------------------------------------------
    void runSensitivityAnalysis(const BaseFunction &benchmark, const std::vector<double> &targets, int bondDim,
                                int maxSweeps, const QuadratureRuleFactory &makeQuadrature,
                                const std::optional<cfg::InputConfiguration> &configuredInputs,
                                rnd::RandomEngines                           &rngs);
}
