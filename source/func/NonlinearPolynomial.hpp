#pragma once

#include "CoupledPolynomial.hpp"
#include <string_view>
#include <utility>
#include <vector>

class NonlinearPolynomial final : public BaseFunction {
    public:
    explicit NonlinearPolynomial(int dimension, double lambda);
    std::string_view           name() const override;
    int                        dimension() const override;
    std::pair<double, double>  domain(int i) const override;
    std::vector<int>           initialPivot() const override;
    double                     evaluate(const std::vector<double> &x) const override;
    sens::util::SobolReference exactSobolIndices(/*unknown*/) const override;

    private:
    CoupledPolynomial polynomial_;
    double            lambda_;
};
