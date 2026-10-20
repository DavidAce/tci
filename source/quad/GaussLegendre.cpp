#include "quadrature.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace sens::quadrature {
    namespace {

        struct PolynomialAndDerivative {
            double p;
            double dp;
        };

        PolynomialAndDerivative legendre(int n, double x) {
            double p0 = 1.0;
            double p1 = x;
            if(n == 1) return {p1, 1.0};

            for(int k = 2; k <= n; ++k) {
                const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0              = p1;
                p1              = p2;
            }
            const double dp = n * (x * p1 - p0) / (x * x - 1.0);
            return {p1, dp};
        }

    } // namespace

    Rule gaussLegendre(int n, double lo, double hi) {
        detail::checkCount(n);
        detail::checkInterval(lo, hi);

        Rule rule;
        rule.nodes.resize(n);
        rule.weights.resize(n);

        // Roots occur in +/- pairs. Find the positive root in each pair by Newton's method.
        const int pairs = n / 2;
        for(int j = 0; j < pairs; ++j) {
            double                  z = std::cos(std::numbers::pi * (j + 0.75) / (n + 0.5));
            PolynomialAndDerivative pd{};
            bool                    converged = false;
            for(int iteration = 0; iteration < 50; ++iteration) {
                pd                = legendre(n, z);
                const double step = pd.p / pd.dp;
                z -= step;
                if(std::abs(step) <= 8.0e-16) {
                    converged = true;
                    break;
                }
            }
            if(!converged) throw std::runtime_error("quadrature: Gauss-Legendre root did not converge");

            pd                      = legendre(n, z);
            const double weight     = 1.0 / ((1.0 - z * z) * pd.dp * pd.dp);
            rule.nodes[j]           = detail::mapNode(-z, lo, hi);
            rule.nodes[n - 1 - j]   = detail::mapNode(z, lo, hi);
            rule.weights[j]         = weight;
            rule.weights[n - 1 - j] = weight;
        }

        if(n % 2 == 1) {
            const int  mid    = n / 2;
            const auto pd     = legendre(n, 0.0);
            rule.nodes[mid]   = detail::mapNode(0.0, lo, hi);
            rule.weights[mid] = 1.0 / (pd.dp * pd.dp);
        }

        return rule;
    }
}
