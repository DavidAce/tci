#pragma once
#include "sens/types.hpp"
#include <xfac/tensor/tensor_train.h>
namespace sens {
    struct Discretization;
}
namespace interp {

    // Discrete axes require an exact node match; they are never interpolated.
    xfac::CTensorTrain<double, double> interpolateTT(const xfac::TensorTrain<double> &tt,
                                                     const sens::Discretization      &discretization);

}