#pragma once
#include <string>
#include <vector>

namespace sens::util {
    struct PairReference {
        int    i, j; // Zero-based variable indices
        double value;
    };

    struct SobolReference {
        std::vector<double>        first;
        std::vector<double>        total;
        std::vector<PairReference> second;
    };

    struct IndexResult {
        std::string name;
        double      estimated;
        double      exact;
        double      error;
    };

    struct SensitivityResult {
        double                   mean     = 0.0;
        double                   variance = 0.0;
        double                   rmse     = 0.0;
        double                   maxAbs   = 0.0;
        std::vector<IndexResult> indices;
    };
}