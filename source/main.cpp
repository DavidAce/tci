#include "cfg/cfg.hpp"
#include "func/CoupledPolynomial.hpp"
#include "func/Ishigami.hpp"
#include "func/Morris2006.hpp"
#include "func/NonlinearPolynomial.hpp"
#include "quad/quadrature.hpp"
#include "rnd/rnd.hpp"
#include "sens/sens.hpp"
#include <cmath>
#include <complex>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include <vector>
#include <xfac/grid.h>
#include <xfac/tensor/tensor_ci.h>
#include <xfac/tensor/tensor_ci_2.h>

auto oscillatory_testfuncton(const std::vector<double> &xs) -> std::complex<double> {
    double x = 0, y = 0, c = 0;
    for(auto xi : xs) {
        c++;
        x += c * xi;
        y += xi * xi / c;
    }
    double arg = 1.0 + (x + 2 * y + x * y) * M_PI;
    return std::complex<double>{1 + x + cos(arg), x * x + 0.5 * sin(arg)};
}

void ci1() {
    int dim       = 5;
    auto [xi, wi] = xfac::grid::QuadratureGK15(0, 1);
    auto ci       = xfac::CTensorCI1<std::complex<double>, double>(oscillatory_testfuncton, std::vector(dim, xi));
    fmt::print("{:<6} {:<6} {:<12} {:<}", "iter", "nEval", "pivotError", "integral(f)\n");
    for(int i = 1; i <= 120; i++) {
        ci.iterate();
        if(i % 10 == 0) {
            auto ttsum = ci.get_TensorTrain().sum(std::vector(dim, wi));
            fmt::print("{:<6} {:<6} {:<12.5e} {:>.16f}{:<+.16f}i\n", i, ci.f.nEval(), ci.pivotError.back(),
                       std::real(ttsum), std::imag(ttsum));
        }
    }
    fmt::print("Bond dimensions:\n");
    for(const auto &p : ci.P) { fmt::print("{}, ", p.n_rows); }
    fmt::print("\n");
    fmt::print("Max absolute tensor error: {:.5e}\n", ci.trueError());
}

void ci2() {
    xfac::TensorCI2Param param;
    param.bondDim = 300;
    param.reltol  = 1e-5;

    int dim       = 5;
    auto [xi, wi] = xfac::grid::QuadratureGK15(0, 1);
    auto ci = xfac::CTensorCI2<std::complex<double>, double>(oscillatory_testfuncton, std::vector(dim, xi), param);
    fmt::print("{:<6} {:<8} {:<12} {}\n", "iter", "nEval", "pivotError", "integral(f)");
    for(int i = 1; i <= 120; ++i) {
        ci.iterate();
        if(i % 1 == 0 || ci.isDone()) {
            auto ttsum = ci.tt.sum(std::vector(dim, wi));
            fmt::print("{:<6} {:<8} {:<12.5e} {:>.16f}{:<+.16f}i\n", i, ci.f.nEval(), ci.pivotError.back(),
                       std::real(ttsum), std::imag(ttsum));
        }
        if(ci.isDone()) break;
    }

    fmt::print("Bond dimensions:\n");
    for(const auto &M : ci.tt.M) fmt::print("{}, ", M.n_slices);
    fmt::print("\n");

    fmt::print("Max absolute tensor error: {:.5e}\n", ci.trueError());
}

int main(int argc, char **argv) {
    // ci1();
    // ci2();
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
        Morris2006 function(8, 5);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature);
    }
    if(cfg::functionNameHas("cpoly")) {
        CoupledPolynomial function(20);
        sens::runSensitivityAnalysis(function, targets, bondDim, maxSweeps, makeQuadrature);
    }
    if(cfg::functionNameHas("npoly")) {
        NonlinearPolynomial function(16, 0.1);
        sens::runSensitivityAnalysis(function, {}, bondDim, maxSweeps, makeQuadrature);
    }
}
