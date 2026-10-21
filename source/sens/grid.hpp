
#pragma once

#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "util.hpp"
#include <vector>
namespace sens {
    using Grids = std::vector<std::vector<double>>;

    std::pair<Grids, Grids>      makeIntegrationGrids(const BaseFunction &function, const Quadrature &makeQuadrature);
    std::vector<double>          randomOffGridPoint(const BaseFunction &function, const Grids &grids);
    std::vector<util::GridProbe> sampleGridReferencePoints(const BaseFunction &function, const Grids &grids, int count);
}