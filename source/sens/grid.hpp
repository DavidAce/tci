
#pragma once

#include "quad/quadrature.hpp"
#include "samples.hpp"
#include "types.hpp"
#include <vector>

class BaseFunction;

namespace sens {
    struct Discretization {
        Grid               nodes;
        IntegrationWeights integrationWeights;
    };

    Discretization discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule);
    Point          sampleContinuousPoint(const BaseFunction &function, const sens::Grid &grid);
    NodeSamples    sampleNodePoints(const BaseFunction &function, const sens::Grid &grid, int count);
}