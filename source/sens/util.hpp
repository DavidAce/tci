#pragma once
#include "func/BaseFunction.hpp"
#include "sobol.hpp"
#include <h5pp/details/h5ppFstr.h>
#include <h5pp/details/h5ppVarr.h>
#include <set>
#include <string>
#include <vector>

namespace sens::util {


    struct Status {
        int              sweeps             = 0;
        int              sweepsUnchanged    = 0;
        int              minSweepsUnchanged = 3;
        double           changeTol          = 1e-12;
        int              nEval              = 0;
        Statistics       statistics         = {};
        SobolResult      sobolResult        = {};
        std::vector<int> bondDims;
        std::string      stopMsg;
    };
    struct GridProbe {
        std::vector<int> indices;
        double           exact = 0;
    };
    struct ProbeAccuracy {
        double           rmse        = 0;
        double           relRmse     = 0;
        double           maxAbsError = 0;
        std::vector<int> proposedPivot;
    };
    struct Predictions {
        using vlen_type                = h5pp::varr_t<double>;
        double               predicted = 0;
        double               funcvalue = 0;
        double               error     = 0;
        h5pp::varr_t<double> point;
    };

    struct FunctionEvaluator {
        const BaseFunction &function;
        double              operator()(const std::vector<double> &point) const { return function.evaluate(point); }
    };
    struct TargetProgress {
        std::vector<std::size_t> firstEval;
        explicit TargetProgress(std::size_t n) : firstEval(n, 0) {}
    };
    struct ConvergenceData {
        std::vector<std::vector<double>> points;
        std::vector<double>              values;
    };
    struct RecoveryState {
        std::set<std::vector<int>> insertedPivots;
        int                        attempts = 0;
    };

}