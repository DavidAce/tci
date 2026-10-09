#include "func/Ishigami.hpp"
#include "func/Morris2006.hpp"
#include "sens/sens.hpp"
#include <cmath>
#include <complex>
#include <fmt/core.h>
#include <fmt/format.h>
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


int main() {
    // ci1();
    // ci2();
    const std::vector<double> targets{1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};

    Ishigami   ishigami;
    Morris2006 morris(8, 5);

    sens::runSensitivity(ishigami, targets, 15, 30 );
    sens::runSensitivity(morris, targets, 15, 30);
}
