#include "lbit.hpp"
#include "cfg/inputs.hpp"
#include "rnd/rnd.hpp"
#include "sens/sobol.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <variant>

namespace {
    // Development settings: no per-realization variance normalization.
    constexpr double sigma1 = 1.0;
    constexpr double sigma2 = 1.0;
    constexpr double eta    = 2.0;
}

Lbit::Lbit(int dimension, std::uint64_t seed, double xi, std::vector<int> permutation,
           const std::optional<cfg::InputConfiguration> &inputs)
    : PermutedFunction(dimension, std::move(permutation)), seed_(seed), xi_(xi), domains_(dimension, {0.0, 1.0}),
      centers_(dimension), inverseStd_(dimension), fields_(dimension),
      pairs_(Eigen::MatrixXd::Zero(dimension, dimension)) {
    if(!std::isfinite(xi_) || xi_ <= 0.0) throw std::invalid_argument("lbit: xi must be finite and positive");
    if(inputs && (inputs->dimension != dim_ || inputs->parameters.size() != static_cast<std::size_t>(dim_)))
        throw std::invalid_argument("lbit: input dimension mismatch");

    // Configuration uses original input order. Continuous inputs are uniform
    // on their domain; the current discrete-input measure is uniform over values.
    for(int i = 0; i < dim_; ++i) {
        if(inputs) {
            const auto &spec = inputs->parameters[i].specification;
            if(const auto *discrete = std::get_if<cfg::DiscreteParameter>(&spec)) {
                if(discrete->values.empty()) throw std::invalid_argument("lbit: empty discrete input");
                const auto &values  = discrete->values;
                const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
                domains_[i]         = {static_cast<double>(*lo), static_cast<double>(*hi)};
                double mean         = 0.0;
                for(auto value : values) mean += static_cast<double>(value) / values.size();
                double variance = 0.0;
                for(auto value : values) {
                    const double delta = static_cast<double>(value) - mean;
                    variance += delta * delta / values.size();
                }
                centers_[i]    = mean;
                inverseStd_[i] = variance > 0.0 ? 1.0 / std::sqrt(variance) : 0.0;
                continue;
            }
            const auto &continuous = std::get<cfg::ContinuousParameter>(spec);
            domains_[i]            = {continuous.domain[0], continuous.domain[1]};
        }
        const auto [lo, hi] = domains_[i];
        if(!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
            throw std::invalid_argument("lbit: invalid continuous domain");
        centers_[i]    = lo + 0.5 * (hi - lo);
        inverseStd_[i] = std::sqrt(12.0) / (hi - lo);
    }

    // Local RNG, used only during construction. No TCI or sampling RNG is consumed.
    rnd::Rng                         rng(seed_);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    for(int i = 0; i < dim_; ++i) fields_[i] = sigma1 * gaussian(rng);

    // Each crossed bond contributes exp(-1/xi) * exp(-eta*|h_r-h_{r+1}|).
    Eigen::VectorXd bondDecay(dim_ - 1);
    for(int r = 0; r + 1 < dim_; ++r) bondDecay[r] = std::exp(-1.0 / xi_ - eta * std::abs(fields_[r] - fields_[r + 1]));

    for(int i = 0; i < dim_; ++i) {
        double decay = 1.0;
        for(int j = i + 1; j < dim_; ++j) {
            decay *= bondDecay[j - 1];
            pairs_(i, j) = sigma2 * gaussian(rng) * decay;
        }
    }
}

std::pair<double, double> Lbit::domainOriginal(int i) const { return domains_.at(i); }

double Lbit::evaluateOriginal(const std::vector<double> &x) const {
    if(x.size() != static_cast<std::size_t>(dim_)) throw std::invalid_argument("lbit: point dimension mismatch");
    const Eigen::Map<const Eigen::VectorXd> point(x.data(), dim_);
    const Eigen::VectorXd                   phi = (point - centers_).cwiseProduct(inverseStd_);
    return fields_.dot(phi) + phi.dot(pairs_ * phi);
}

sens::SobolReference Lbit::exactSobolIndicesOriginal() const {
    // For independent centered unit-variance phi_i, V_i=h_i^2 and V_ij=J_ij^2.
    // A singleton discrete input has phi_i=0 and contributes no variance.
    sens::SobolReference ref;
    ref.first.resize(dim_, 0.0);
    ref.total.resize(dim_, 0.0);
    double variance = 0.0;
    for(int i = 0; i < dim_; ++i) {
        const double vi = inverseStd_[i] == 0.0 ? 0.0 : fields_[i] * fields_[i];
        ref.first[i]    = vi;
        ref.total[i] += vi;
        variance += vi;
        for(int j = i + 1; j < dim_; ++j) {
            const double vij = inverseStd_[i] == 0.0 || inverseStd_[j] == 0.0 ? 0.0 : pairs_(i, j) * pairs_(i, j);
            ref.second.push_back({i, j, vij});
            ref.total[i] += vij;
            ref.total[j] += vij;
            variance += vij;
        }
    }
    if(!(variance > 0.0)) return {};
    for(auto &v : ref.first) v /= variance;
    for(auto &v : ref.total) v /= variance;
    for(auto &pair : ref.second) pair.value /= variance;
    return ref;
}
