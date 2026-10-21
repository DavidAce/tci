#pragma once

#include <vector>

class BaseFunction;

namespace h5pp {
    class File;
}
namespace xfac {
    struct TensorCI2Param;
}

namespace sens {
    struct PredictionRecord;
    struct Status;

    void writeSimulationParameters(h5pp::File &file, const xfac::TensorCI2Param &p, const BaseFunction &f,
                                   const std::vector<double> &targets);
    void writePredictions(h5pp::File &file, const std::vector<PredictionRecord> &predictions);
    void appendStatus(h5pp::File &file, const Status &status);
    void writeStatusLog(h5pp::File &file, const std::vector<Status> &statuses);
}