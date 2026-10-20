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

    namespace util {
        struct Status;
        struct Predictions;
    }
    using namespace util;

    void writeParam(h5pp::File &file, const xfac::TensorCI2Param &p, const BaseFunction &f,
                    const std::vector<double> &targets);
    void writePredictions(h5pp::File &file, const std::vector<Predictions> &predictions);
    void appendStatus(h5pp::File &file, const Status &status);
    void writeStatusLog(h5pp::File &file, const std::vector<Status> &statuses);
}