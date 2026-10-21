#include "grid.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include <algorithm>
#include <cstddef>
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

std::vector<double> sens::sampleContinuousPoint(const BaseFunction &function, const sens::Grid &grid) {
    const int dim = function.dimension();
    if(grid.size() != static_cast<std::size_t>(dim)) throw std::invalid_argument("Grid dimension mismatch");
    std::vector<double> point(dim);
    for(int k = 0; k < dim; ++k) {
        const auto [lo, hi] = function.domain(k);
        if(!(lo < hi)) throw std::invalid_argument("Invalid domain");
        do {
            point[k] = rnd::uniform_double_box(lo, hi);
        } while(point[k] <= lo || point[k] >= hi ||
                std::find(grid[k].begin(), grid[k].end(), point[k]) != grid[k].end());
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