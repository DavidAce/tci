#include "cfg/cfg.hpp"
#include "cfg/inputs.hpp"
#include "grid.hpp"
#include "hdf5.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <h5pp/h5pp.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sens {

    void writeInputMetadata(h5pp::File &file, const Discretization &discretization,
                            const std::optional<cfg::InputConfiguration> &configuration,
                            const std::vector<int> &permutation, std::string_view fallbackQuadrature) {
        const auto dim = discretization.nodes.size();
        if(dim == 0 || discretization.integrationWeights.size() != dim || discretization.domains.size() != dim ||
           discretization.kinds.size() != dim)
            throw std::invalid_argument("Incomplete input discretization");
        if(configuration &&
           (configuration->dimension != static_cast<int>(dim) || configuration->parameters.size() != dim))
            throw std::invalid_argument("Input configuration dimension mismatch");
        if(!permutation.empty() && permutation.size() != dim)
            throw std::invalid_argument("Input permutation dimension mismatch");

        std::vector<int> permutedIndices(dim);
        std::vector<int> originalIndices(dim, -1);
        for(std::size_t oidx = 0; oidx < dim; ++oidx) { // oidx: original parameter index (before permutation)
            const int axis = permutation.empty() ? static_cast<int>(oidx) : permutation[oidx];
            if(axis < 0 || static_cast<std::size_t>(axis) >= dim || originalIndices[axis] != -1)
                throw std::invalid_argument(fmt::format("Failed to invert permutation: {}", permutation));
            permutedIndices[oidx] = static_cast<int>(axis);
            originalIndices[axis] = static_cast<int>(oidx);
        }

        // permutedIndices[i] is the TT axis carrying original input parameter x[i].
        // originalIndices[k] is the original index of the input parameter represented by TT at axis k.
        std::string prfx = "param/inputs";
        file.writeDataset(static_cast<int>(dim), prfx + "/dimension");
        file.writeDataset(permutedIndices, prfx + "/permutation");
        file.writeDataset(originalIndices, prfx + "/permuted_index_order");
        file.writeDataset(configuration ? "TOML" : "CLI", prfx + "/source");

        for(std::size_t axis = 0; axis < dim; ++axis) {
            const auto  oidx    = static_cast<std::size_t>(originalIndices[axis]);
            const auto  prefix  = fmt::format("{}/x{}", prfx, axis);
            const auto &nodes   = discretization.nodes[axis];
            const auto &weights = discretization.integrationWeights[axis];
            if(nodes.empty() || nodes.size() != weights.size())
                throw std::invalid_argument(
                    fmt::format("Invalid nodes/weights for axis {} with original index {}", axis, oidx));

            const auto axis_name  = configuration ? configuration->parameters[oidx].name : fmt::format("x{}", oidx);
            int        node_count = static_cast<int>(nodes.size());
            file.writeDataset(axis_name, prefix + "/name");
            file.writeDataset(static_cast<int>(oidx), prefix + "/original_index");
            file.writeDataset(nodes, prefix + "/discretization/nodes");
            file.writeDataset(weights, prefix + "/discretization/weights");
            file.writeDataset(node_count, prefix + "/discretization/node_count");

            switch(discretization.kinds[axis]) {
                case sens::InputKind::Continuous: {
                    const auto [lo, hi] = discretization.domains[axis];
                    auto quadratureRule = std::string(fallbackQuadrature);
                    if(configuration.has_value()) {
                        const auto &specification = configuration->parameters[oidx].specification;
                        const auto *continuous    = std::get_if<cfg::ContinuousParameter>(&specification);
                        if(!continuous)
                            throw std::invalid_argument(
                                fmt::format("Input {} with original index {} should be continuous", axis_name, oidx));
                        const auto *generated = std::get_if<cfg::GeneratedNodes>(&continuous->nodes);
                        if(!generated)
                            throw std::invalid_argument(
                                fmt::format("Input {}  with original index {} has unsupported node specification",
                                            axis_name, oidx));
                        quadratureRule = generated->quadratureRule;
                        if(generated->count != static_cast<int>(nodes.size()))
                            throw std::invalid_argument(
                                fmt::format("Input {} node count disagrees with discretization", axis_name));
                    }

                    file.writeDataset("continuous", prefix + "/kind");
                    file.writeDataset(std::array<double, 2>{lo, hi}, prefix + "/domain");
                    file.writeDataset(quadratureRule, prefix + "/discretization/quadrature_rule");
                    break;
                }
                case sens::InputKind::Discrete: {
                    if(!configuration) throw std::invalid_argument("Discrete input has no resolved configuration");
                    const auto *discrete =
                        std::get_if<cfg::DiscreteParameter>(&configuration->parameters[oidx].specification);
                    if(!discrete || discrete->values.size() != nodes.size())
                        throw std::invalid_argument(
                            fmt::format("Input {} has inconsistent discrete values", axis_name));
                    for(std::size_t j = 0; j < nodes.size(); ++j)
                        if(static_cast<double>(discrete->values[j]) != nodes[j])
                            throw std::invalid_argument(
                                fmt::format("Input {} discrete values disagree with nodes", axis_name));
                    // Store the original integers, not their double representations.
                    file.writeDataset("discrete", prefix + "/kind");
                    file.writeDataset(discrete->values, prefix + "/values");
                    break;
                }
                default:
                    throw std::invalid_argument(
                        fmt::format("Unknown input type for axis {} with original index", axis, oidx));
            }
        }
    }

} // namespace sens
