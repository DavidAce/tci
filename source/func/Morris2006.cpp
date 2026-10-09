#include "Morris2006.hpp"
#include "sens/util.h"
#include <cmath>
#include <stdexcept>

using namespace sens::util;

Morris2006::Morris2006(int dimension, int active) : dim_(dimension), p_(active) {
    if(p_ < 2 || p_ > dim_) throw std::invalid_argument("Require 2 <= active <= dimension");
}

std::string_view Morris2006::name() const { return "Morris2006"; }
int              Morris2006::dimension() const { return dim_; }

std::pair<double, double> Morris2006::domain(int) const { return {0.0, 1.0}; }

double Morris2006::evaluate(const std::vector<double> &x) const {
    const double alpha = std::sqrt(12.0) - 6 * std::sqrt(0.1 * (p_ - 1));

    const double beta = 12.0 / std::sqrt(10.0 * (p_ - 1));

    double sum   = 0.0;
    double pairs = 0.0;

    for(int i = 0; i < p_; ++i) {
        pairs += sum * x[i];
        sum += x[i];
    }

    return alpha * sum + beta * pairs;
}

SobolReference Morris2006::exactSobol() const {
    SobolReference ref;
    ref.first.resize(dim_, 0.0);
    ref.total.resize(dim_, 0.0);

    const double variance     = 1.05 * p_;
    const double pairVariance = 1.0 / (10.0 * (p_ - 1));

    for(int i = 0; i < p_; ++i) {
        ref.first[i] = 1.0 / variance;
        ref.total[i] = 1.1 / variance;
    }

    for(int i = 0; i < dim_; ++i)
        for(int j = i + 1; j < dim_; ++j)
            ref.second.push_back({i, j, (i < p_ && j < p_) ? pairVariance / variance : 0.0});

    return ref;
}

std::vector<int> Morris2006::initialPivot() const {
    std::vector<int> pivot(dim_, 7);
    for(int i = 0; i < p_; ++i) { pivot[i] = (2 + 3 * i) % 15; }
    return pivot;
};
