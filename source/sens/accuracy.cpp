#include "accuracy.hpp"
#include "func/BaseFunction.hpp"
#include "grid.hpp"
#include "interp/interp.hpp"
#include "util.hpp"
#include <xfac/tensor/tensor_ci_2.h>

using TensorCI        = xfac::CTensorCI2<double, double>;
using Grids           = sens::Grids;
using GridProbe       = sens::util::FunctionSample;
using ConvergenceData = sens::util::ConvergenceData;
using ProbeAccuracy   = sens::util::GridSampleAccuracy;

ConvergenceData sens::sampleOffGridControlPoints(const BaseFunction &function, const Grids &grids, int count) {
    util::ConvergenceData data;
    data.points.reserve(count);
    data.values.reserve(count);
    for(int i = 0; i < count; ++i) {
        auto point = randomOffGridPoint(function, grids);
        data.values.push_back(function.evaluate(point));
        data.points.push_back(std::move(point));
    }
    return data;
}
ProbeAccuracy sens::measureGridApproximationError(const TensorCI &ci, const std::vector<GridProbe> &probes,
                               const std::set<std::vector<int>> &insertedPivots) {
    ProbeAccuracy accuracy;
    if(probes.empty()) return accuracy;
    double error2 = 0, exact2 = 0, largestUnusedError = -1;
    for(const auto &probe : probes) {
        const double error = ci.tt.eval(probe.indices) - probe.exact;
        if(!std::isfinite(error)) throw std::runtime_error("Non-finite grid-probe error");
        const double absError = std::abs(error);
        error2 += error * error;
        exact2 += probe.exact * probe.exact;
        accuracy.maxAbsError = std::max(accuracy.maxAbsError, absError);
        if(!insertedPivots.contains(probe.indices) && absError > largestUnusedError) {
            largestUnusedError     = absError;
            accuracy.proposedPivot = probe.indices;
        }
    }
    accuracy.rmse    = std::sqrt(error2 / probes.size());
    accuracy.relRmse = exact2 > 0 ? std::sqrt(error2 / exact2) : accuracy.rmse;
    return accuracy;
}

double sens::offGridPredictionRMSE(const TensorCI &ci, const Grids &grids, const ConvergenceData &control) {
    if(control.points.empty()) return std::numeric_limits<double>::quiet_NaN();
    if(control.points.size() != control.values.size()) throw std::runtime_error("Control-point size mismatch");
    const auto ctt    = interp::interpolateTT(ci.tt, grids);
    double     error2 = 0;
    for(std::size_t i = 0; i < control.points.size(); ++i) {
        const double error = ctt.eval(control.points[i]) - control.values[i];
        error2 += error * error;
    }
    return std::sqrt(error2 / control.points.size());
}