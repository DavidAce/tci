#pragma once
#include <h5pp/details/h5ppFstr.h>
#include <string>
#include <vector>
namespace sens::util {
    struct PairReference {
        int    i, j; // Zero-based variable indices
        double value;
    };

    struct TripleReference {
        int    i, j, k; // Zero-based variable indices
        double value;
    };

    struct SobolReference {
        std::vector<double>          total;
        std::vector<double>          first;
        std::vector<PairReference>   second;
        std::vector<TripleReference> third;
    };

    struct SobolIndex {
        h5pp::fstr_t<32> name;
        double           estimated;
        double           exact;
        double           error;
    };
    struct Statistics {
        int    dim      = 0;
        double mean     = 0.0;
        double mean2    = 0.0;
        double m2       = 0.0;
        double variance = 0.0;
    };
    struct SobolResult {
        double                  rmse    = 0.0;
        double                  maxDiff = 0.0;
        std::vector<SobolIndex> sobolIndices;
    };
}