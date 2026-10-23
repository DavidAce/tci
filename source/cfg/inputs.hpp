#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace cfg {

    enum class TableType { Single, Group };

    // Description of input parameters, independent of the numerical TT/grid.
    struct GeneratedNodes {
        std::string quadratureRule;
        int         count = 0;
    };

    struct ExplicitNodes {
        std::vector<double> values;
        std::vector<double> integrationWeights;
    };

    using NodeSpecification = std::variant<GeneratedNodes, ExplicitNodes>;

    struct ContinuousParameter {
        std::array<double, 2> domain{};
        NodeSpecification     nodes;
    };

    struct DiscreteParameter {
        std::vector<std::int64_t> values;
    };

    using ParameterSpecification = std::variant<ContinuousParameter, DiscreteParameter>;

    struct InputParameter {
        std::string            name; // x0, x1, ... refer to original input identities.
        ParameterSpecification specification;
    };

    struct InputConfiguration {
        std::string                 functionName;
        int                         dimension;
        std::vector<InputParameter> parameters;
    };

    // Parse TOML and resolve defaults and nonoverlapping parameter-group overrides.
    // No discretization, BaseFunction access, or TCI evaluation occurs here.
    [[nodiscard]] InputConfiguration readInputConfiguration(const std::filesystem::path &path);

} // namespace cfg
