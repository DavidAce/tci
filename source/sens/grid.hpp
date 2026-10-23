
#pragma once

#include "quad/quadrature.hpp"
#include "samples.hpp"
#include "types.hpp"
#include <vector>

class BaseFunction;
namespace cfg {
    struct InputConfiguration;
}

namespace sens {
    struct Discretization {
        Grid               nodes;
        IntegrationWeights integrationWeights;
    };

    [[nodiscard]] Discretization discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule);
    [[nodiscard]] Discretization discretizeInputs(const BaseFunction            &function,
                                                  const cfg::InputConfiguration &configuredInputs,
                                                  const std::vector<int>        &permutation);
    [[nodiscard]] Point          sampleContinuousPoint(const BaseFunction &function, const sens::Grid &grid);
    [[nodiscard]] NodeSamples    sampleNodePoints(const BaseFunction &function, const sens::Grid &grid, int count);
}