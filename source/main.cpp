#include "cfg/cfg.hpp"
#include "cli/cli.hpp"
#include "func/Ishigami.hpp"
#include "func/LegendreANOVA3.hpp"
#include "func/Morris2006.hpp"
#include "func/NonlinearPolynomial.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "sens/sens.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include <vector>
#include <xfac/grid.h>
#include <xfac/tensor/tensor_ci.h>
#include <xfac/tensor/tensor_ci_2.h>

int main(int argc, char **argv) {
    cli::parse(argc, argv);
    rnd::seed(cfg::seed);

    using namespace cfg;
    auto makeQuadrature = [&](double lo, double hi) {
        if(cfg::quadratureNameHas("gk15")) return quadrature::gaussKronrod15(lo, hi);
        if(cfg::quadratureNameHas("gl")) return quadrature::gaussLegendre(nnodes, lo, hi);
        if(cfg::quadratureNameHas("cc")) return quadrature::clenshawCurtis(nnodes, lo, hi);
        throw std::invalid_argument(fmt::format("Unhandled quadrature name: {}", cfg::quadratureName));
    };

    if(cfg::functionNameHas("ishigami")) {
        Ishigami function;
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature);
    }
    if(cfg::functionNameHas("morris")) {
        Morris2006 function(dimension, active);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature);
    }
    if(cfg::functionNameHas("cpoly")) {
        LegendreANOVA3 function(dimension);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature);
    }
    if(cfg::functionNameHas("npoly")) {
        NonlinearPolynomial function(16, 0.1);
        sens::runSensitivityAnalysis(function, {}, bondDim, maxSweeps, makeQuadrature);
    }
}
