#pragma once
#include "Base.hpp"

class CoupledPolynomial final : public BaseFunction {
public:
    explicit CoupledPolynomial(int dimension = 8);

    std::string_view name() const override;
    int dimension() const override;
    std::pair<double, double> domain(int) const override;
    std::vector<int> initialPivot() const override;
    double evaluate(const std::vector<double>& x) const override;
    sens::util::SobolReference exactSobolIndices() const override;

private:
    int dim_;
    std::vector<double> a_, b_, c_;

    double b(int i, int j) const { return b_[i * dim_ + j]; }
    double c(int i, int j, int k) const { return c_[(i * dim_ + j) * dim_ + k]; }
};
