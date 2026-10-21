#pragma once
#include "sens/types.hpp"
#include <xfac/tensor/tensor_train.h>
namespace interp {

    xfac::CTensorTrain<double, double> interpolateTT(const xfac::TensorTrain<double> &tt, const sens::Grid &grid);

}