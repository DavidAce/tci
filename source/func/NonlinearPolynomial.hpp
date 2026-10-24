#pragma once

#include "LegendrePoly3.hpp"
#include <string_view>
#include <utility>
#include <vector>

class NonlinearPolynomial final : public PermutedFunction {
    public:
    explicit NonlinearPolynomial(int dimension, double lambda, std::vector<int> permutation = {});
    std::string_view          name() const override;
    std::pair<double, double> domainOriginal(int i) const override;
    std::vector<int>          initialPivotOriginal() const override;
    double                    evaluateOriginal(const std::vector<double> &x) const override;
    sens::SobolReference      exactSobolIndicesOriginal(/*unknown*/) const override;

    private:
    LegendrePoly3 polynomial_;
    double        lambda_;
};
