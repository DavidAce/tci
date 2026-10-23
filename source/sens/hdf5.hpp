#pragma once

#include <optional>
#include <string_view>
#include <vector>

class BaseFunction;
namespace cfg {
    struct InputConfiguration;
}

namespace h5pp {
    class File;
}
namespace xfac {
    struct TensorCI2Param;
}

namespace sens {
    struct PredictionRecord;
    struct Status;
    struct Discretization;

    void writeSimulationParameters(h5pp::File &file, const xfac::TensorCI2Param &p, const BaseFunction &f,
                                   const std::vector<double> &targets);
    // Store resolved inputs in TT-axis order and both permutation directions.
    void writeInputMetadata(h5pp::File &file, const Discretization &discretization,
                            const std::optional<cfg::InputConfiguration> &configuration,
                            const std::vector<int> &permutation, std::string_view fallbackQuadrature);
    void writePredictions(h5pp::File &file, const std::vector<PredictionRecord> &predictions);
    void appendStatus(h5pp::File &file, const Status &status);
    void writeStatusLog(h5pp::File &file, const std::vector<Status> &statuses);
}