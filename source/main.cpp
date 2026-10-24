#include "cfg/cfg.hpp"
#include "cli/cli.hpp"
#include "func/Ishigami.hpp"
#include "func/LegendrePoly3.hpp"
#include "func/Morris2006.hpp"
#include "func/NonlinearPolynomial.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "sens/sens.hpp"
#include <chrono>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include <vector>
#include <xfac/grid.h>
#include <xfac/tensor/tensor_ci.h>
#include <xfac/tensor/tensor_ci_2.h>

int main(int argc, char **argv) {
    const auto         inputConfiguration = cli::parse(argc, argv);
    rnd::RandomEngines rngs(cfg::seed);
    std::srand(rngs.seeds.xfac);

    fmt::print("RNG seeds: init={} pivot={} nodeSamples={} predictions={} xfac={}\n", rngs.seeds.init, rngs.seeds.pivot,
               rngs.seeds.nodeSamples, rngs.seeds.predictions, rngs.seeds.xfac);

    using namespace cfg;
    using namespace sens;

    auto makeQuadrature = [&](double lo, double hi) {
        if(cfg::quadratureRuleNameHas("gk15")) return quadrature::gaussKronrod15(lo, hi);
        if(cfg::quadratureRuleNameHas("gl")) return quadrature::gaussLegendre(nnodes, lo, hi);
        if(cfg::quadratureRuleNameHas("cc")) return quadrature::clenshawCurtis(nnodes, lo, hi);
        throw std::invalid_argument(fmt::format("Unhandled quadrature name: {}", cfg::quadratureRuleName));
    };

    auto t_start = std::chrono::steady_clock::now();
    if(cfg::functionNameHas("ishigami")) {
        Ishigami function(cfg::permutation);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature, inputConfiguration, rngs);
    }
    if(cfg::functionNameHas("morris")) {
        Morris2006 function(dimension, active, cfg::permutation);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature, inputConfiguration, rngs);
    }
    if(cfg::functionNameHas("legendre")) {
        LegendrePoly3 function(dimension, cfg::permutation);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature, inputConfiguration, rngs);
    }
    if(cfg::functionNameHas("npoly")) {
        NonlinearPolynomial function(dimension, 0.1, cfg::permutation);
        sens::runSensitivityAnalysis(function, {}, bondDim, maxSweeps, makeQuadrature, inputConfiguration, rngs);
    }
    auto t_end     = std::chrono::steady_clock::now();
    auto t_elapsed = t_end - t_start;
    fmt::print("\nTotal time elapsed: {:.4e} s\n", std::chrono::duration<double>(t_elapsed).count());
}
