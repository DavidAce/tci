
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "util.hpp"
#include <vector>
namespace sens {
    using namespace util;
    using Grids = std::vector<std::vector<double>>;

    std::pair<Grids, Grids> makeIntegrationGrids(const BaseFunction &function, const Quadrature &makeQuadrature);
    std::vector<double>     randomOffGridPoint(const BaseFunction &function, const Grids &grids);
    std::vector<GridProbe>  sampleGridProbes(const BaseFunction &function, const Grids &grids, int count);
}