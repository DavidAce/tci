#include "func/NonlinearPolynomial.hpp"
#include "LegendrePoly3.hpp"
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

NonlinearPolynomial::NonlinearPolynomial(int dimension, double lambda, std::vector<int> permutation)
    : PermutedFunction(dimension, {}), polynomial_(dimension, std::move(permutation)), lambda_(lambda) {
    if(!std::isfinite(lambda_) || lambda_ <= 0.0) throw std::invalid_argument("lambda must be positive and finite");
}

std::string_view          NonlinearPolynomial::name() const { return "NonlinearPolynomial"; }
std::pair<double, double> NonlinearPolynomial::domainOriginal(int i) const { return polynomial_.domain(i); }
std::vector<int>          NonlinearPolynomial::initialPivotOriginal() const { return polynomial_.initialPivot(); }

double NonlinearPolynomial::evaluateOriginal(const std::vector<double> &x) const {
    return std::tanh(lambda_ * (polynomial_.evaluate(x) - 1.0));
}

sens::SobolReference NonlinearPolynomial::exactSobolIndicesOriginal() const { return {}; }
