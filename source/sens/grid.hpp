
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "util.hpp"
#include <random>
#include <vector>
namespace sens {
    using namespace util;
    using Grids = std::vector<std::vector<double>>;

    std::pair<Grids, Grids> makeIntegrationGrids(const BaseFunction &function, const Quadrature &makeQuadrature) {
        const int dim = function.dimension();
        Grids     grids(dim), weights(dim);
        for(int k = 0; k < dim; ++k) {
            auto [lo, hi] = function.domain(k);
            auto rule     = makeQuadrature(lo, hi);
            grids[k]      = std::move(rule.nodes);
            weights[k]    = std::move(rule.weights);
        }
        return {std::move(grids), std::move(weights)};
    }

    std::vector<double> randomOffGridPoint(const BaseFunction &function, const Grids &grids) {
        const int dim = function.dimension();
        if(grids.size() != static_cast<std::size_t>(dim)) throw std::invalid_argument("Grid dimension mismatch");
        std::vector<double> point(dim);
        for(int k = 0; k < dim; ++k) {
            const auto [lo, hi] = function.domain(k);
            if(!(lo < hi)) throw std::invalid_argument("Invalid domain");
            do {
                point[k] = rnd::uniform_double_box(lo, hi);
            } while(point[k] <= lo || point[k] >= hi ||
                    std::find(grids[k].begin(), grids[k].end(), point[k]) != grids[k].end());
        }
        return point;
    }

    std::vector<GridProbe> sampleGridProbes(const BaseFunction &function, const Grids &grids, int count) {
        std::mt19937_64        rng(123456789); // Separate from the RNG used by TCI.
        std::vector<GridProbe> probes;
        probes.reserve(count);
        for(int n = 0; n < count; ++n) {
            GridProbe probe;
            probe.indices.resize(grids.size());
            std::vector<double> point(grids.size());
            for(std::size_t k = 0; k < grids.size(); ++k) {
                std::uniform_int_distribution<int> distribution(0, static_cast<int>(grids[k].size()) - 1);
                probe.indices[k] = distribution(rng);
                point[k]         = grids[k][probe.indices[k]];
            }
            probe.exact = function.evaluate(point);
            probes.push_back(std::move(probe));
        }
        return probes;
    }

}