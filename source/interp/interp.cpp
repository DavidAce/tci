#include "interp.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <xfac/tensor/tensor_train.h>

namespace {
    using namespace interp;
    Grid barycentricWeights(const Grid &nodes) {
        Grid w(nodes.size(), 1.0);

        for(std::size_t j = 0; j < nodes.size(); ++j) {
            for(std::size_t m = 0; m < nodes.size(); ++m) {
                if(j != m) w[j] /= nodes[j] - nodes[m];
            }
        }

        // Only relative weights matter.
        double scale = 0.0;
        for(double v : w) scale = std::max(scale, std::abs(v));

        for(double &v : w) v /= scale;

        return w;
    }
}

xfac::CTensorTrain<double, double> interp::interpolateTT(const xfac::TensorTrain<double> &tt, const Grids &grids) {
    if(tt.M.size() != grids.size()) throw std::invalid_argument("Grid dimension mismatch");

    xfac::CTensorTrain<double, double> result;
    result.M.reserve(tt.M.size());

    for(std::size_t k = 0; k < tt.M.size(); ++k) {
        auto core  = tt.M[k];
        auto nodes = grids[k];

        if(nodes.size() != core.n_cols) throw std::invalid_argument("Grid size mismatch");

        auto w = barycentricWeights(nodes);

        result.M.emplace_back(
            [core = std::move(core), nodes = std::move(nodes), w = std::move(w)](double x) -> arma::Mat<double> {
                // Exact recovery at a grid node.
                for(std::size_t j = 0; j < nodes.size(); ++j)
                    if(x == nodes[j]) return arma::Mat<double>(core.col(j));

                Grid   coeff(nodes.size());
                double denom = 0.0;

                for(std::size_t j = 0; j < nodes.size(); ++j) {
                    coeff[j] = w[j] / (x - nodes[j]);
                    denom += coeff[j];
                }

                arma::Mat<double> A(core.n_rows, core.n_slices, arma::fill::zeros);

                for(std::size_t j = 0; j < nodes.size(); ++j) A += (coeff[j] / denom) * arma::Mat<double>(core.col(j));

                return A;
            });
    }

    return result;
}