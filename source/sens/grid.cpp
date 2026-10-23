#include "grid.hpp"
#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <fmt/format.h>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

sens::Discretization sens::discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule) {
    const int dim = function.dimension();

    Discretization result;
    result.nodes.resize(dim);
    result.integrationWeights.resize(dim);

    for(int k = 0; k < dim; ++k) {
        const auto [lo, hi] = function.domain(k);
        auto rule           = makeRule(lo, hi);

        result.nodes[k]              = std::move(rule.nodes);
        result.integrationWeights[k] = std::move(rule.weights);
    }

    return result;
}

sens::Discretization sens::discretizeInputs(const BaseFunction &function,
                                                    const cfg::InputConfiguration &configuredInputs,
                                                    const std::vector<int> &permutation) {
    const int dim = function.dimension();
    if(configuredInputs.dimension != dim || configuredInputs.parameters.size() != static_cast<std::size_t>(dim))
        throw std::invalid_argument("TOML dimension does not match the function dimension");
    if(!permutation.empty() && permutation.size() != static_cast<std::size_t>(dim))
        throw std::invalid_argument("Permutation size does not match the function dimension");

    Discretization result;
    result.nodes.resize(dim);
    result.integrationWeights.resize(dim);
    std::vector<bool> assigned(dim, false);

    // Configured inputs are named in original parameter order. The permutation
    // maps original parameter indices to positions in the TT.
    for(int original = 0; original < dim; ++original) {
        const int k = permutation.empty() ? original : permutation[original];
        if(k < 0 || k >= dim || assigned[k]) throw std::invalid_argument("Invalid input permutation");
        assigned[k] = true;

        const auto &input = configuredInputs.parameters[original];
        const auto *continuous = std::get_if<cfg::ContinuousParameter>(&input.specification);
        if(!continuous) throw std::invalid_argument(fmt::format("Input {} is not continuous", input.name));
        const auto *generated = std::get_if<cfg::GeneratedNodes>(&continuous->nodes);
        if(!generated) throw std::invalid_argument(fmt::format("Input {} has no supported node generator", input.name));

        // Until the next increment, continuous sampling still uses
        // BaseFunction::domain(). Refuse incompatible TOML domains instead
        // of silently computing Sobol moments for another input measure.
        const auto [functionLo, functionHi] = function.domain(k);
        const auto [lo, hi] = continuous->domain;
        const auto sameBound = [](double a, double b) {
            return std::isfinite(a) && std::isfinite(b) &&
                   std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
        };
        if(!sameBound(lo, functionLo) || !sameBound(hi, functionHi))
            throw std::invalid_argument(fmt::format(
                "Input {} domain [{}, {}] differs from the function domain [{}, {}]; "
                "custom domains are not supported in this increment",
                input.name, lo, hi, functionLo, functionHi));

        quadrature::Rule rule;
        if(generated->rule == "gk15")
            rule = quadrature::gaussKronrod15(lo, hi);
        else if(generated->rule == "gl")
            rule = quadrature::gaussLegendre(generated->count, lo, hi);
        else if(generated->rule == "cc")
            rule = quadrature::clenshawCurtis(generated->count, lo, hi);
        else
            throw std::invalid_argument(fmt::format("Unknown quadrature rule '{}'", generated->rule));

        result.nodes[k] = std::move(rule.nodes);
        result.integrationWeights[k] = std::move(rule.weights);
    }
    return result;
}

std::vector<double> sens::sampleContinuousPoint(const BaseFunction &function, const sens::Grid &grid) {
    static thread_local std::mt19937_64 rng(987654321);

    const int dim = function.dimension();
    if(grid.size() != static_cast<std::size_t>(dim)) throw std::invalid_argument("Grid dimension mismatch");

    std::vector<double> point(dim);

    for(int k = 0; k < dim; ++k) {
        const auto [lo, hi] = function.domain(k);
        if(!(lo < hi)) throw std::invalid_argument("Invalid domain");

        std::uniform_real_distribution<double> distribution(lo, hi);

        bool accepted = false;
        while(!accepted) {
            const double x = distribution(rng);

            accepted = x > lo && x < hi && std::find(grid[k].begin(), grid[k].end(), x) == grid[k].end();

            if(accepted) point[k] = x;
        }
    }
    return point;
}

sens::NodeSamples sens::sampleNodePoints(const BaseFunction &function, const Grid &nodes, int count) {
    std::mt19937_64 rng(123456789);

    NodeSamples samples;
    samples.reserve(count);

    for(int n = 0; n < count; ++n) {
        NodeSample sample;
        sample.indices.resize(nodes.size());

        Point point(nodes.size());

        for(std::size_t k = 0; k < nodes.size(); ++k) {
            std::uniform_int_distribution<int> distribution(0, static_cast<int>(nodes[k].size()) - 1);

            const int index = distribution(rng);

            sample.indices[k] = index;
            point[k]          = nodes[k][index];
        }

        sample.value = function.evaluate(point);
        samples.push_back(std::move(sample));
    }

    return samples;
}