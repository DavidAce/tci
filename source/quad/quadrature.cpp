#include "quadrature.hpp"
#include <cmath>
#include <stdexcept>

namespace sens::quadrature::detail {

    void checkInterval(double lo, double hi) {
        if(!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
            throw std::invalid_argument("quadrature: require finite lo < hi");
    }

    void checkCount(int n) {
        if(n < 1) throw std::invalid_argument("quadrature: node count must be >= 1");
    }

    double mapNode(double x, double lo, double hi) {
        // x in [-1, 1] -> [lo, hi].
        return lo + 0.5 * (hi - lo) * (1.0 + x);
    }

}