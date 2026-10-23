#include "interp.hpp"
#include "sens/grid.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <xfac/tensor/tensor_train.h>

namespace {
    using namespace interp;
    std::vector<double> calculateBarycentricWeights(const sens::Nodes &nodes) {
        sens::Weights w(nodes.size(), 1.0);

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

namespace {
    xfac::CTensorTrain<double, double> makeInterpolatedTT(const xfac::TensorTrain<double> &tt, const sens::Grid &grid,
                                                          const std::vector<sens::InputKind> *kinds) {
        if(tt.M.size() != grid.size()) throw std::invalid_argument("Grid dimension mismatch");
        if(kinds && kinds->size() != grid.size())
            throw std::invalid_argument("Input kinds and grid dimension mismatch");

        xfac::CTensorTrain<double, double> result;
        result.M.reserve(tt.M.size());

        for(std::size_t k = 0; k < tt.M.size(); ++k) {
            auto core  = tt.M[k];
            auto nodes = grid[k];

            if(nodes.size() != core.n_cols) throw std::invalid_argument("Grid size mismatch");

            if(kinds && (*kinds)[k] != sens::InputKind::Continuous) {
                result.M.emplace_back(
                    [core = std::move(core), nodes = std::move(nodes)](double x) -> arma::Mat<double> {
                        const auto position = std::find(nodes.begin(), nodes.end(), x);
                        if(position == nodes.end())
                            throw std::invalid_argument("Discrete TT evaluation requires a configured input value");
                        return arma::Mat<double>(core.col(static_cast<arma::uword>(position - nodes.begin())));
                    });
                continue;
            }
            const auto interpolationWeights = calculateBarycentricWeights(nodes);

            result.M.emplace_back([core = std::move(core), nodes = std::move(nodes),
                                   w = std::move(interpolationWeights)](double x) -> arma::Mat<double> {
                // Exact recovery at a grid node.
                for(std::size_t j = 0; j < nodes.size(); ++j)
                    if(x == nodes[j]) return arma::Mat<double>(core.col(j));

                std::vector<double> coeff(nodes.size());
                double              denom = 0.0;

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
}

xfac::CTensorTrain<double, double> interp::interpolateTT(const xfac::TensorTrain<double> &tt,
                                                         const sens::Discretization      &discretization) {
    return makeInterpolatedTT(tt, discretization.nodes, &discretization.kinds);
}
