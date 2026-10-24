#include "Ishigami.hpp"
#include "sens/sobol.hpp"
#include <cmath>
#include <numbers>

using namespace sens;
static constexpr double pi = std::numbers::pi;

Ishigami::Ishigami(std::vector<int> permutation) : PermutedFunction(3, std::move(permutation)) {}

std::string_view Ishigami::name() const { return "Ishigami"; }

std::pair<double, double> Ishigami::domainOriginal(int) const { return {-pi, pi}; }

std::vector<int> Ishigami::initialPivotOriginal() const { return {5, 9, 5}; }

double Ishigami::evaluateOriginal(const std::vector<double> &x) const {
    return std::sin(x[0]) * (1.0 + b * std::pow(x[2], 4)) + a * std::pow(std::sin(x[1]), 2);
}

SobolReference Ishigami::exactSobolIndicesOriginal() const {
    const double p4 = std::pow(pi, 4);
    const double p8 = p4 * p4;

    const double variance_x0    = 0.5 * std::pow(1 + b * p4 / 5, 2);
    const double variance_x1    = a * a / 8;
    const double variance_x0_x2 = 8 * b * b * p8 / 225;
    const double v              = variance_x0 + variance_x1 + variance_x0_x2;

    return {.total  = {(variance_x0 + variance_x0_x2) / v, variance_x1 / v, variance_x0_x2 / v},
            .first  = {variance_x0 / v, variance_x1 / v, 0.0},
            .second = {{0, 2, variance_x0_x2 / v}}};
}
