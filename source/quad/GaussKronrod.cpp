#include "quadrature.hpp"

#include <array>

namespace quadrature {

Rule gaussKronrod15(double lo, double hi) {
    detail::checkInterval(lo, hi);

    // Positive Gauss-Kronrod abscissas in descending order, then zero.
    // Standard GK(7,15) constants (QUADPACK / Gauss-Kronrod tables).
    constexpr std::array<double, 8> x = {
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.0
    };
    constexpr std::array<double, 8> w = {
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714
    };

    Rule rule;
    rule.nodes.reserve(15);
    rule.weights.reserve(15);

    // Ascending physical coordinates; normalized expectation weights.
    for (int j = 0; j < 7; ++j) {
        rule.nodes.push_back(detail::mapNode(-x[j], lo, hi));
        rule.weights.push_back(0.5 * w[j]);
    }
    rule.nodes.push_back(detail::mapNode(0.0, lo, hi));
    rule.weights.push_back(0.5 * w[7]);
    for (int j = 6; j >= 0; --j) {
        rule.nodes.push_back(detail::mapNode(x[j], lo, hi));
        rule.weights.push_back(0.5 * w[j]);
    }
    return rule;
}

} // namespace quadrature
