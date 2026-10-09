#include "Ishigami.hpp"
#include <cmath>
#include <numbers>

using namespace sens::util;
static constexpr double pi = std::numbers::pi;

std::string_view Ishigami::name() const { return "Ishigami"; }
int              Ishigami::dimension() const { return 3; }

std::pair<double, double> Ishigami::domain(int) const { return {-pi, pi}; }

std::vector<int> Ishigami::initialPivot() const { return {5, 9, 5}; }

double Ishigami::evaluate(const std::vector<double> &x) const {
    return std::sin(x[0]) * (1.0 + b * std::pow(x[2], 4)) + a * std::pow(std::sin(x[1]), 2);
}

SobolReference Ishigami::exactSobol() const {
    const double p4 = std::pow(pi, 4);
    const double p8 = p4 * p4;

    const double v1  = 0.5 * std::pow(1 + b * p4 / 5, 2);
    const double v2  = a * a / 8;
    const double v13 = 8 * b * b * p8 / 225;
    const double v   = v1 + v2 + v13;

    return {.first = {v1 / v, v2 / v, 0.0}, .total = {(v1 + v13) / v, v2 / v, v13 / v}, .second = {{0, 2, v13 / v}}};
}
