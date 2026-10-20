#include "sobol.hpp"
#include <armadillo>
#include <h5pp/details/h5ppFstr.h>
#include <vector>

namespace sens {
    using Grids = std::vector<std::vector<double>>;

    bool hasExactSobolIndices(const SobolReference &reference, int dim) {
        return reference.first.size() == static_cast<std::size_t>(dim) &&
               reference.total.size() == static_cast<std::size_t>(dim);
    }

    void appendSobolIndex(std::vector<SobolIndex> &indices, const std::string &name, double estimated, double exact,
                          bool hasExact, double &maxDiff) {
        const double error = hasExact ? std::abs(estimated - exact) : std::numeric_limits<double>::quiet_NaN();
        indices.push_back({h5pp::fstr_t<32>(name), estimated, exact, error});
        if(hasExact) maxDiff = std::max(maxDiff, error);
    }

}