#pragma once
#include "Base.hpp"
#include "sens/util.h"
// ------------------------------------------------------------
// Ishigami
// ------------------------------------------------------------

class Ishigami final : public BaseFunction {
    public:
    static constexpr double a  = 7.0;
    static constexpr double b  = 0.1;

    std::string_view           name() const override;
    int                        dimension() const override;
    std::pair<double, double>  domain(int) const override;
    std::vector<int>           initialPivot() const override;
    double                     evaluate(const std::vector<double> &x) const override;
    sens::util::SobolReference exactSobol() const override;
};
