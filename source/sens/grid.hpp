
#pragma once

#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "samples.hpp"
#include "types.hpp"
#include <string>
#include <utility>
#include <vector>

class BaseFunction;
namespace cfg {
    struct InputConfiguration;
}

namespace sens {
    using Domain  = std::pair<double, double>;
    using Domains = std::vector<Domain>;
    enum class InputKind { Continuous, Discrete };

    struct Discretization {
        Grid                   nodes;
        Domains                domains;            // Bounds for continuous axes; discrete min/max otherwise
        IntegrationWeights     integrationWeights; // Quadrature weights or discrete probabilities
        std::vector<InputKind> kinds;              // TT-axis order
    };

    [[nodiscard]] Discretization discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule);
    [[nodiscard]] Discretization discretizeInputs(const BaseFunction            &function,
                                                  const cfg::InputConfiguration &configuredInputs,
                                                  const std::vector<int>        &permutation);
    [[nodiscard]] bool           domainsMatch(const Domains &domains1, const Domains &domains2);
    [[nodiscard]] Point          sampleInputPoint(const Discretization &discretization, rnd::Rng &rng);
    [[nodiscard]] NodeSamples    sampleNodePoints(const BaseFunction &function, const sens::Grid &grid, int count,
                                                  rnd::Rng &rng);
}
