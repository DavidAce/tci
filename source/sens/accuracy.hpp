#pragma once
#include "sens/util.hpp"
#include <set>
#include <vector>

struct BaseFunction;
namespace xfac {
    template<class T1, class T2>
    class CTensorCI2;
}
using TensorCI = xfac::CTensorCI2<double, double>;

namespace sens {

    namespace util {
        struct ConvergenceData;
        struct ProbeAccuracy;
        struct GridProbe;
    }
    using Grids = std::vector<std::vector<double>>;

    util::ConvergenceData sampleOffGridControlPoints(const BaseFunction &function, const Grids &grids, int count);

    util::ProbeAccuracy measureGridApproximationError(const TensorCI &ci, const std::vector<util::GridProbe> &probes,
                                         const std::set<std::vector<int>> &insertedPivots);

    double offGridPredictionRMSE(const TensorCI &ci, const Grids &grids, const util::ConvergenceData &control);
}