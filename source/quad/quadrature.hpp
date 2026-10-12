#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

namespace quadrature {

    // Nodes are sorted in ascending order on [lo, hi].
    // Weights approximate E[f(X)] for X ~ Uniform(lo, hi):
    //     E[f(X)] ~= sum_i weights[i] * f(nodes[i]).
    // Thus sum(weights) = 1 (up to floating-point roundoff).
    struct Rule {
        std::vector<double> nodes;
        std::vector<double> weights;
    };

    // Fixed 15-point Gauss-Kronrod rule (7-point Gauss subset).
    Rule gaussKronrod15(double lo, double hi);

    // n-point Gauss-Legendre rule (n >= 1).
    Rule gaussLegendre(int n, double lo, double hi);

    // n-point Clenshaw-Curtis rule (n >= 1). For nested refinement,
    // use n = 2^m + 1, e.g. 5, 9, 17, 33, ...
    Rule clenshawCurtis(int n, double lo, double hi);

    namespace detail {

        inline void checkInterval(double lo, double hi) {
            if(!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
                throw std::invalid_argument("quadrature: require finite lo < hi");
        }

        inline void checkCount(int n) {
            if(n < 1) throw std::invalid_argument("quadrature: node count must be >= 1");
        }

        inline double mapNode(double x, double lo, double hi) {
            // x in [-1, 1] -> [lo, hi].
            return lo + 0.5 * (hi - lo) * (1.0 + x);
        }

    }

}
