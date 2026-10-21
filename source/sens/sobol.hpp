#pragma once

#include <h5pp/details/h5ppFstr.h>
#include <vector>

namespace xfac {
    template<typename T>
    class TensorTrain;
}
namespace sens {
    using Grids = std::vector<std::vector<double>>;
    struct Statistics {
        int    dim      = 0;
        double mean     = 0.0;
        double mean2    = 0.0;
        double m2       = 0.0;
        double variance = 0.0;
    };

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

    struct SobolResult {
        double                  rmse    = 0.0;
        double                  maxDiff = 0.0;
        std::vector<SobolIndex> sobolIndices;
    };

    // Contract the TT with its conjugate after averaging selected variables.
    double conditional_mean_square(const xfac::TensorTrain<double> &tt, const Grids &weights,
                                   const std::vector<bool> &integrateOut);

    double maximumSobolChange(const SobolResult &current, const SobolResult &previous);

    bool hasExactSobolIndices(const SobolReference &reference, int dim);
    void appendSobolIndex(std::vector<SobolIndex> &indices, const std::string &name, double estimated, double exact,
                          bool hasExact, double &maxDiff);

    SobolResult calculateSobolIndices(const xfac::TensorTrain<double> &tt, const Grids &weights,
                                      const Statistics &statistics, const SobolReference &reference);

}