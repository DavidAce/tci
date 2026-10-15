#pragma once

#include "sens/util.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sens {

    struct SweepRecord {
        int                        halfSweeps  = 0;
        std::uint64_t              nEval       = 0;
        double                     mean        = 0.0;
        double                     variance    = 0.0;
        double                     sobolRMSE   = 0.0;
        double                     sobolMaxAbs = 0.0;
        std::vector<std::uint64_t> bondDimensions;
    };

    // A complete set of independent continuous-domain test points.
    // x is row-major, so x[n * dimension + i] is variable i in test point n.
    struct OffGridSamples {
        std::size_t         dimension = 0;
        std::vector<double> x;
        std::vector<double> yExact;
        std::vector<double> yTCI;
    };

    struct RunResults {
        std::string                   functionName;
        int                           dimension = 0;
        std::vector<SweepRecord>      history;
        util::SensitivityResult       firstStableSobol;
        util::SensitivityResult       finalSobol;
        std::uint64_t                 firstStableNEval      = 0;
        std::uint64_t                 finalNEval            = 0;
        int                           firstStableHalfSweeps = 0;
        int                           finalHalfSweeps       = 0;
        std::string                   stopReason;
        std::optional<OffGridSamples> offGrid;
    };

} // namespace sens
