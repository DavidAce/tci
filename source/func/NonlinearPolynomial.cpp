#include "func/NonlinearPolynomial.hpp"
#include "LegendreANOVA3.hpp"
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

NonlinearPolynomial::NonlinearPolynomial(int dimension, double lambda) : polynomial_(dimension), lambda_(lambda) {
    if(!std::isfinite(lambda_) || lambda_ <= 0.0) throw std::invalid_argument("lambda must be positive and finite");
}

std::string_view          NonlinearPolynomial::name() const { return "NonlinearPolynomial"; }
int                       NonlinearPolynomial::dimension() const { return polynomial_.dimension(); }
std::pair<double, double> NonlinearPolynomial::domain(int i) const { return polynomial_.domain(i); }
std::vector<int>          NonlinearPolynomial::initialPivot() const { return polynomial_.initialPivot(); }

double NonlinearPolynomial::evaluate(const std::vector<double> &x) const {
    return std::tanh(lambda_ * (polynomial_.evaluate(x) - 1.0));
}

sens::util::SobolReference NonlinearPolynomial::exactSobolIndices() const { return {}; }
