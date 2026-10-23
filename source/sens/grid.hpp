
#pragma once

#include "quad/quadrature.hpp"
#include "samples.hpp"
#include "types.hpp"
#include <utility>
#include <vector>

class BaseFunction;
namespace cfg {
    struct InputConfiguration;
}

namespace sens {
    using Domain  = std::pair<double, double>;
    using Domains = std::vector<Domain>;
    struct Discretization {
        Grid               nodes;
        Domains            domains; // TT-axis order, independent uniform inputs
        IntegrationWeights integrationWeights;
    };

    [[nodiscard]] Discretization discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule);
    [[nodiscard]] Discretization discretizeInputs(const BaseFunction            &function,
                                                  const cfg::InputConfiguration &configuredInputs,
                                                  const std::vector<int>        &permutation);
    [[nodiscard]] bool           domainsMatch(const Domains &domains1, const Domains &domains2);
    [[nodiscard]] Point          sampleContinuousPoint(const Discretization &discretization);
    [[nodiscard]] NodeSamples    sampleNodePoints(const BaseFunction &function, const sens::Grid &grid, int count);
}