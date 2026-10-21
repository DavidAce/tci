#pragma once

#include <functional>
#include <vector>

namespace sens {
    namespace quadrature {
        struct Rule;
    }

    using QuadratureRuleFactory = std::function<quadrature::Rule(double lo, double hi)>;
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
        [[nodiscard]] Rule gaussKronrod15(double lo, double hi);

        // n-point Gauss-Legendre rule (n >= 1).
        [[nodiscard]] Rule gaussLegendre(int n, double lo, double hi);

        // n-point Clenshaw-Curtis rule (n >= 1). For nested refinement,
        // use n = 2^m + 1, e.g. 5, 9, 17, 33, ...
        [[nodiscard]] Rule clenshawCurtis(int n, double lo, double hi);

        namespace detail {

            void                 checkInterval(double lo, double hi);
            void                 checkCount(int n);
            [[nodiscard]] double mapNode(double x, double lo, double hi);

        }

    }

}