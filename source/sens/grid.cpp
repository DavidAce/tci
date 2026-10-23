#include "grid.hpp"
#include "cfg/cfg.hpp"
#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fmt/format.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

sens::Discretization sens::discretizeInputs(const BaseFunction &function, const QuadratureRuleFactory &makeRule) {
    const int dim = function.dimension();

    Discretization result;
    result.nodes.resize(dim);
    result.integrationWeights.resize(dim);
    result.domains.resize(dim);
    result.kinds.resize(dim, InputKind::Continuous);

    for(int k = 0; k < dim; ++k) {
        result.domains[k]   = function.domain(k);
        const auto [lo, hi] = result.domains[k];
        auto rule           = makeRule(lo, hi);

        result.nodes[k]              = std::move(rule.nodes);
        result.integrationWeights[k] = std::move(rule.weights);
    }

    return result;
}

sens::Discretization sens::discretizeInputs(const BaseFunction            &function,
                                            const cfg::InputConfiguration &configuredInputs,
                                            const std::vector<int>        &permutation) {
    const int dim = function.dimension();
    if(configuredInputs.dimension != dim || configuredInputs.parameters.size() != static_cast<std::size_t>(dim))
        throw std::invalid_argument("TOML dimension does not match the function dimension");
    if(!permutation.empty() && permutation.size() != static_cast<std::size_t>(dim))
        throw std::invalid_argument("Permutation size does not match the function dimension");

    Discretization result;
    result.nodes.resize(dim);
    result.integrationWeights.resize(dim);
    result.domains.resize(dim);
    result.kinds.resize(dim, InputKind::Continuous);
    std::vector<bool> assigned(dim, false);

    // Configured inputs are named in original parameter order. The permutation
    // maps original parameter indices to positions in the TT.
    for(int original = 0; original < dim; ++original) {
        const int k = permutation.empty() ? original : permutation[original];
        if(k < 0 || k >= dim || assigned[k]) throw std::invalid_argument("Invalid input permutation");
        assigned[k] = true;

        const auto &input = configuredInputs.parameters[original];
        if(const auto *integer = std::get_if<cfg::DiscreteParameter>(&input.specification)) {
            result.kinds[k] = InputKind::Discrete;
            for(const auto value : integer->values) result.nodes[k].push_back(static_cast<double>(value));
            result.integrationWeights[k].assign(integer->values.size(), 1.0 / integer->values.size());
            const auto [minimum, maximum] = std::minmax_element(result.nodes[k].begin(), result.nodes[k].end());
            result.domains[k]             = {*minimum, *maximum};
            continue;
        }

        const auto *continuous = std::get_if<cfg::ContinuousParameter>(&input.specification);
        if(!continuous) throw std::invalid_argument(fmt::format("Unsupported input {}", input.name));
        const auto *generated = std::get_if<cfg::GeneratedNodes>(&continuous->nodes);
        if(!generated) throw std::invalid_argument(fmt::format("Input {} has no supported node generator", input.name));

        // TOML defines the input measure; the generated quadrature weights
        // approximate expectations under a uniform distribution on this domain.
        const auto [lo, hi] = continuous->domain;
        result.domains[k]   = {lo, hi};

        quadrature::Rule rule;
        if(generated->quadratureRule == "gk15")
            rule = quadrature::gaussKronrod15(lo, hi);
        else if(generated->quadratureRule == "gl")
            rule = quadrature::gaussLegendre(generated->count, lo, hi);
        else if(generated->quadratureRule == "cc")
            rule = quadrature::clenshawCurtis(generated->count, lo, hi);
        else
            throw std::invalid_argument(fmt::format("Unknown quadrature rule '{}'", generated->quadratureRule));

        result.nodes[k]              = std::move(rule.nodes);
        result.integrationWeights[k] = std::move(rule.weights);
    }
    return result;
}

bool sens::domainsMatch(const Domains &domains1, const Domains &domains2) {
    if(domains1.size() != domains2.size()) return false;
    const auto dim = static_cast<std::size_t>(domains1.size());
    for(std::size_t k = 0; k < dim; ++k) {
        const auto [a1, b1]  = domains1[k];
        const auto [a2, b2]  = domains2[k];
        const auto sameBound = [](double lhs, double rhs) {
            return std::isfinite(lhs) && std::isfinite(rhs) &&
                   std::abs(lhs - rhs) <= cfg::convTol * std::max({1.0, std::abs(lhs), std::abs(rhs)});
        };
        if(!sameBound(a1, a2) || !sameBound(b1, b2)) return false;
    }
    return true;
}

sens::Point sens::sampleInputPoint(const Discretization &discretization) {
    static thread_local std::mt19937_64 rng(987654321);

    const auto &grid    = discretization.nodes;
    const auto &domains = discretization.domains;
    if(grid.empty() || domains.size() != grid.size() || discretization.kinds.size() != grid.size() ||
       discretization.integrationWeights.size() != grid.size())
        throw std::invalid_argument("Sampling input dimensions do not match");

    Point point(grid.size());
    for(std::size_t k = 0; k < grid.size(); ++k) {
        if(discretization.kinds[k] != InputKind::Continuous) {
            const auto &weights = discretization.integrationWeights[k];
            if(weights.size() != grid[k].size() || weights.empty())
                throw std::invalid_argument("Invalid discrete sampling weights");
            std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
            point[k] = grid[k][distribution(rng)];
            continue;
        }
        const auto [lo, hi] = domains[k];
        if(!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
            throw std::invalid_argument("Invalid continuous input domain");

        std::uniform_real_distribution<double> distribution(lo, hi);
        bool                                   accepted = false;
        while(!accepted) {
            const double x = distribution(rng);
            accepted       = x > lo && x < hi && std::find(grid[k].begin(), grid[k].end(), x) == grid[k].end();
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