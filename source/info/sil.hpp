#pragma once

// Sobol information lattice (SIL) of a real-valued tensor train.
// Independent inputs are represented by normalized discrete quadrature weights.
//
// SIL(scale, position) = sum S_u over nonempty subsets u satisfying
//   min(u) = position, max(u) = position + scale.
// Valid entries satisfy scale + position < d; other entries are zero.
//
// xfac stores each MPS tensor in an Armadillo cube with dimensions
// (left bond, physical index, right bond). Eigen::TensorMap accesses it
// without copying. All index summations use Eigen::Tensor::contract.

#include <array>
#include <cmath>
#include <cstddef>
#include <Eigen/Dense>
#include <h5pp/h5pp.h>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <xfac/tensor/tensor_train.h>

namespace sens::sil {
    namespace detail {
        using Index      = Eigen::Index;
        using Pair       = Eigen::IndexPair<Index>;
        using Tensor0    = Eigen::Tensor<double, 0>;
        using Tensor1    = Eigen::Tensor<double, 1>;
        using Tensor2    = Eigen::Tensor<double, 2>;
        using Tensor3    = Eigen::Tensor<double, 3>;
        using TensorMap3 = Eigen::TensorMap<const Tensor3>;

        // All summed tensor indices are contracted here. A different backend
        // can provide the same operations without changing calculateSIL().
        struct EigenContractions {
            // Integrate the physical index: mpsIntegrated[l,r] = sum_s p[s] mps[l,s,r].
            [[nodiscard]] static Tensor2 contract_mps_weights(const TensorMap3 &mps, const Tensor1 &probabilities) {
                constexpr std::array<Pair, 1> pairs{{Pair{1, 0}}};
                Tensor2                       mpsIntegrated(mps.dimension(0), mps.dimension(2));
                mpsIntegrated = mps.contract(probabilities, pairs);
                return mpsIntegrated;
            }

            // Extend the single-layer, weighted MPS contraction from the left.
            [[nodiscard]] static Tensor1 contract_wmpsL(const Tensor1 &wmpsL, const Tensor2 &mpsIntegrated) {
                constexpr std::array<Pair, 1> pairs{{Pair{0, 0}}};
                Tensor1                       next(mpsIntegrated.dimension(1));
                next = wmpsL.contract(mpsIntegrated, pairs);
                return next;
            }

            // Extend the single-layer, weighted MPS contraction from the right.
            [[nodiscard]] static Tensor1 contract_wmpsR(const Tensor2 &mpsIntegrated, const Tensor1 &wmpsR) {
                constexpr std::array<Pair, 1> pairs{{Pair{1, 0}}};
                Tensor1                       next(mpsIntegrated.dimension(0));
                next = mpsIntegrated.contract(wmpsR, pairs);
                return next;
            }

            // Construct the initial two-bond-index MPS Gram tensor.
            [[nodiscard]] static Tensor2 contract_wmpsL_wmpsL(const Tensor1 &wmpsL) {
                constexpr std::array<Pair, 0> noPairs{};
                Tensor2                       mpsGramL(wmpsL.dimension(0), wmpsL.dimension(0));
                mpsGramL = wmpsL.contract(wmpsL, noPairs);
                return mpsGramL;
            }

            // Extend the two-layer MPS contraction, summing the physical index:
            // next[r,r'] = sum_{l,l',s} mpsGramL[l,l'] mps[l,s,r] mps[l',s,r'].
            // The physical quadrature weights are already incorporated in mps.
            [[nodiscard]] static Tensor2 contract_mpsGramL_mps(const Tensor2 &mpsGramL, const Tensor3 &mps) {
                constexpr std::array<Pair, 1> first{{Pair{0, 0}}};
                constexpr std::array<Pair, 2> second{{Pair{0, 0}, Pair{1, 1}}};
                Tensor3                       tmp(mpsGramL.dimension(1), mps.dimension(1), mps.dimension(2));
                tmp = mpsGramL.contract(mps, first);
                Tensor2 next(mps.dimension(2), mps.dimension(2));
                next = tmp.contract(mps, second);
                return next;
            }

            // Contract the remaining two bond indices with the right MPS vector.
            [[nodiscard]] static double contract_mpsGramL_wmpsR(const Tensor2 &mpsGramL, const Tensor1 &wmpsR) {
                constexpr std::array<Pair, 1> first{{Pair{1, 0}}};
                constexpr std::array<Pair, 1> second{{Pair{0, 0}}};
                Tensor1                       tmp(mpsGramL.dimension(0));
                tmp = mpsGramL.contract(wmpsR, first);
                Tensor0 result;
                result = wmpsR.contract(tmp, second);
                return result();
            }
        };
    } // namespace detail

    // Computes every SIL node without enumerating Sobol subsets.
    // With d variables, m quadrature points per variable and typical MPS bond
    // dimension chi: O(d^2*m*chi^3) arithmetic, O(d*m*chi^2 + d^2) memory.
    template<typename Backend = detail::EigenContractions>
    Eigen::MatrixXd calculateSIL(const xfac::TensorTrain<double> &tt, const std::vector<std::vector<double>> &weights,
                                 const Backend &backend = {}) {
        const std::size_t d = tt.M.size();
        if(d == 0 || weights.size() != d) throw std::invalid_argument("calculateSIL: empty MPS or mismatched weights");
        if(tt.M.front().n_rows != 1 || tt.M.back().n_slices != 1)
            throw std::invalid_argument("calculateSIL: expected open-boundary MPS");

        // The original MPS tensor M_k is integrated over the physical index
        // or weighted by sqrt(p_k). Its fluctuations are M_k - E[M_k].
        std::vector<detail::Tensor2> mpsIntegrated(d);
        std::vector<detail::Tensor3> mpsWeighted(d);
        std::vector<detail::Tensor3> mpsFluctuations(d);

        for(std::size_t k = 0; k < d; ++k) {
            const auto &cube = tt.M[k];
            if(k > 0 && tt.M[k - 1].n_slices != cube.n_rows)
                throw std::invalid_argument("calculateSIL: incompatible MPS bond dimensions");

            const auto nleft  = static_cast<detail::Index>(cube.n_rows);
            const auto nphys  = static_cast<detail::Index>(cube.n_cols);
            const auto nright = static_cast<detail::Index>(cube.n_slices);
            if(nleft == 0 || nphys == 0 || nright == 0 || weights[k].size() != static_cast<std::size_t>(nphys))
                throw std::invalid_argument("calculateSIL: invalid MPS dimensions or quadrature weights");

            const Eigen::Map<const Eigen::ArrayXd> inputWeights(weights[k].data(), nphys);
            if(!inputWeights.allFinite() || (inputWeights < 0.0).any())
                throw std::invalid_argument("calculateSIL: weights must be finite and nonnegative");
            const double weightSum = inputWeights.sum();
            if(!(weightSum > 0.0) || !std::isfinite(weightSum))
                throw std::invalid_argument("calculateSIL: invalid quadrature weight sum");

            const detail::TensorMap3                      mps(cube.memptr(), nleft, nphys, nright);
            const Eigen::TensorMap<const detail::Tensor1> inputWeightTensor(weights[k].data(), nphys);
            detail::Tensor1                               probabilities(nphys);
            probabilities = inputWeightTensor / weightSum;

            detail::Tensor1 sqrtProbabilities(nphys);
            sqrtProbabilities = probabilities.sqrt();
            mpsIntegrated[k]  = backend.contract_mps_weights(mps, probabilities);

            const Eigen::array<detail::Index, 3> weightShape{1, nphys, 1};
            const Eigen::array<detail::Index, 3> weightCopies{nleft, 1, nright};
            const Eigen::array<detail::Index, 3> meanShape{nleft, 1, nright};
            const Eigen::array<detail::Index, 3> meanCopies{1, nphys, 1};
            const auto sqrtWeightsPerSlice   = sqrtProbabilities.reshape(weightShape).broadcast(weightCopies);
            const auto integratedMpsPerSlice = mpsIntegrated[k].reshape(meanShape).broadcast(meanCopies);

            mpsWeighted[k].resize(nleft, nphys, nright);
            mpsFluctuations[k].resize(nleft, nphys, nright);
            mpsWeighted[k]     = mps * sqrtWeightsPerSlice;
            mpsFluctuations[k] = (mps - integratedMpsPerSlice) * sqrtWeightsPerSlice;
        }

        // One-index partial contractions of the MPS with probability weights.
        // wmpsL[k] integrates sites [0,k); wmpsR[k] integrates sites [k,d).
        std::vector<detail::Tensor1> wmpsL(d + 1), wmpsR(d + 1);
        wmpsL[0]    = detail::Tensor1(1);
        wmpsL[0](0) = 1.0;
        for(std::size_t k = 0; k < d; ++k) wmpsL[k + 1] = backend.contract_wmpsL(wmpsL[k], mpsIntegrated[k]);

        wmpsR[d]    = detail::Tensor1(1);
        wmpsR[d](0) = 1.0;
        for(std::size_t k = d; k-- > 0;) wmpsR[k] = backend.contract_wmpsR(mpsIntegrated[k], wmpsR[k + 1]);

        Eigen::MatrixXd lattice  = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));
        long double     variance = 0.0L;

        for(std::size_t i = 0; i < d; ++i) {
            // Fluctuations at i select Sobol components involving variable i.
            auto mpsGramL = backend.contract_mpsGramL_mps(backend.contract_wmpsL_wmpsL(wmpsL[i]), mpsFluctuations[i]);

            const double first                       = backend.contract_mpsGramL_wmpsR(mpsGramL, wmpsR[i + 1]);
            lattice(0, static_cast<Eigen::Index>(i)) = first;
            variance += first;

            for(std::size_t j = i + 1; j < d; ++j) {
                // Fluctuations at j select components whose last participating
                // variable is j; interior sites allow any subset of variables.
                const auto   mpsGramEndpoint = backend.contract_mpsGramL_mps(mpsGramL, mpsFluctuations[j]);
                const double node            = backend.contract_mpsGramL_wmpsR(mpsGramEndpoint, wmpsR[j + 1]);
                lattice(static_cast<Eigen::Index>(j - i), static_cast<Eigen::Index>(i)) = node;
                variance += node;

                if(j + 1 < d) mpsGramL = backend.contract_mpsGramL_mps(mpsGramL, mpsWeighted[j]);
            }
        }

        if(!(variance > 0.0L) || !std::isfinite(variance))
            throw std::runtime_error("calculateSIL: zero or nonfinite output variance");
        lattice *= 1.0 / variance;
        return lattice;
    }

    // HDF5 matrix: row = scale, column = starting position.
    inline void writeSIL(h5pp::File &file, const Eigen::MatrixXd &lattice, const std::string &path = "tci/sil") {
        if(lattice.rows() == 0 || lattice.rows() != lattice.cols())
            throw std::invalid_argument("writeSIL: expected nonempty square matrix");
        file.writeDataset(lattice, path);
    }
}
