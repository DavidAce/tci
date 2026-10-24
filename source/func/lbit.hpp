#pragma once

#include "PermutedFunction.hpp"
#include <cstdint>
#include <Eigen/Core>
#include <optional>
#include <string_view>
#include <vector>

namespace cfg {
    struct InputConfiguration;
}

// Seed-fixed onsite and two-body interactions at every index distance.
// Coefficients and locality are defined in original input order; the inherited
// permutation only changes the order presented to TCI.
class Lbit final : public PermutedFunction {
    private:
    std::uint64_t   seed_;
    double          xi_;
    Domains         domains_;
    Eigen::VectorXd centers_, inverseStd_, fields_;
    Eigen::MatrixXd pairs_; // Strict upper triangle; each pair is counted once.

    public:
    Lbit(int dimension, std::uint64_t seed, double xi, std::vector<int> permutation,
         const std::optional<cfg::InputConfiguration> &inputs);

    [[nodiscard]] std::string_view name() const override { return "lbit"; }
    [[nodiscard]] std::uint64_t    seed() const { return seed_; }
    [[nodiscard]] double           xi() const { return xi_; }

    protected:
    [[nodiscard]] std::pair<double, double> domainOriginal(int i) const override;
    [[nodiscard]] double                    evaluateOriginal(const std::vector<double> &x) const override;
    [[nodiscard]] sens::SobolReference      exactSobolIndicesOriginal() const override;
};
