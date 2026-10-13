#include "quadrature.hpp"
#include <cmath>
#include <numbers>

namespace quadrature {

    Rule clenshawCurtis(int n, double lo, double hi) {
        detail::checkCount(n);
        detail::checkInterval(lo, hi);

        if(n == 1) return {{detail::mapNode(0.0, lo, hi)}, {1.0}};

        const int N = n - 1;
        Rule      rule;
        rule.nodes.resize(n);
        rule.weights.resize(n);

        // Standard cosine-spaced Clenshaw-Curtis nodes and weights on [-1,1].
        // O(n^2) construction; negligible next to a TCI calculation.
        // Store in ascending order rather than the conventional descending order.
        const double endWeight =
            (N % 2 == 0) ? 1.0 / (static_cast<double>(N) * N - 1.0) : 1.0 / (static_cast<double>(N) * N);

        for(int j = 0; j <= N; ++j) {
            const double theta          = std::numbers::pi * j / N;
            const double x              = std::cos(theta);
            double       integralWeight = endWeight;

            if(j != 0 && j != N) {
                double v = 1.0;
                if(N % 2 == 0) {
                    for(int k = 1; k < N / 2; ++k) v -= 2.0 * std::cos(2.0 * k * theta) / (4.0 * k * k - 1.0);
                    v -= std::cos(N * theta) / (static_cast<double>(N) * N - 1.0);
                } else {
                    for(int k = 1; k <= (N - 1) / 2; ++k) v -= 2.0 * std::cos(2.0 * k * theta) / (4.0 * k * k - 1.0);
                }
                integralWeight = 2.0 * v / N;
            }

            const int dst     = N - j;
            rule.nodes[dst]   = detail::mapNode(x, lo, hi);
            rule.weights[dst] = 0.5 * integralWeight;
        }

        return rule;
    }

}
