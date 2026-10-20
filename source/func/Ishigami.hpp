#pragma once
#include "PermutedFunction.hpp"
#include "sens/util.h"
// ------------------------------------------------------------
// Ishigami
// ------------------------------------------------------------

class Ishigami final : public PermutedFunction {
    public:
    static constexpr double a = 7.0;
    static constexpr double b = 0.1;

    explicit Ishigami(std::vector<int> permutation = {});

    std::string_view name() const override;

    protected:
    std::pair<double, double>  domainOriginal(int) const override;
    std::vector<int>           initialPivotOriginal() const override;
    double                     evaluateOriginal(const std::vector<double> &x) const override;
    sens::util::SobolReference exactSobolIndicesOriginal() const override;
};
