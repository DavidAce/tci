#include "accuracy.hpp"
#include "grid.hpp"
#include "interp/interp.hpp"
#include "samples.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <xfac/tensor/tensor_ci_2.h>

sens::PredictionError sens::measurePredictionError(const TensorCI &ci, const Grid &grid,
                                                   const sens::FunctionSamples &samples) {
    if(samples.empty()) throw std::invalid_argument("No function samples available");

    const auto interpolatedTT = interp::interpolateTT(ci.tt, grid);

    double sumSquaredError  = 0.0;
    double sumSquaredValues = 0.0;
    double maxAbsoluteError = 0.0;

    for(const auto &sample : samples) {
        const double predicted = interpolatedTT.eval(sample.point);
        const double error     = predicted - sample.value;

        if(!std::isfinite(error)) throw std::runtime_error("Non-finite prediction error");

        sumSquaredError += error * error;
        sumSquaredValues += sample.value * sample.value;
        maxAbsoluteError = std::max(maxAbsoluteError, std::abs(error));
    }

    sens::PredictionError result;
    result.rmse             = std::sqrt(sumSquaredError / samples.size());
    result.relativeRMSE     = sumSquaredValues > 0.0 ? std::sqrt(sumSquaredError / sumSquaredValues)
                                                     : std::numeric_limits<double>::quiet_NaN();
    result.maxAbsoluteError = maxAbsoluteError;

    return result;
}

sens::FunctionSamples sens::sampleFunctionValues(const BaseFunction &function, const Grid &grid, int count) {
    if(count <= 0) throw std::invalid_argument("Sample count must be positive");

    sens::FunctionSamples samples;
    samples.reserve(count);

    for(int n = 0; n < count; ++n) {
        auto point = sens::sampleContinuousPoint(function, grid);

        samples.push_back({.point = point, .value = function.evaluate(point)});
    }

    return samples;
}

sens::NodeSampleError sens::measureNodeSampleError(const TensorCI &ci, const NodeSamples &samples,
                                                   const std::set<std::vector<int>> &insertedPivots) {
    NodeSampleError accuracy;
    if(samples.empty()) return accuracy;
    double error2 = 0, exact2 = 0, largestUnusedError = -1;
    for(const auto &probe : samples) {
        const double error = ci.tt.eval(probe.indices) - probe.value;
        if(!std::isfinite(error)) throw std::runtime_error("Non-finite grid-probe error");
        const double absError = std::abs(error);
        error2 += error * error;
        exact2 += probe.value * probe.value;
        accuracy.maxAbsoluteError = std::max(accuracy.maxAbsoluteError, absError);
        if(!insertedPivots.contains(probe.indices) && absError > largestUnusedError) {
            largestUnusedError      = absError;
            accuracy.suggestedPoint = probe.indices;
        }
    }
    accuracy.rmse         = std::sqrt(error2 / samples.size());
    accuracy.relativeRMSE = exact2 > 0 ? std::sqrt(error2 / exact2) : accuracy.rmse;
    return accuracy;
}