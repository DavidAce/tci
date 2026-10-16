#include <xfac/tensor/tensor_train.h>

namespace interp {

    using Grid  = std::vector<double>;
    using Grids = std::vector<Grid>;

    xfac::CTensorTrain<double, double> interpolateTT(const xfac::TensorTrain<double> &tt, const Grids &grids);

}