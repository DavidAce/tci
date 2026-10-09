#pragma once
#include "Base.hpp"
#include "sens/util.h"
#include <string_view>
#include <vector>

class Morris2006 final : public BaseFunction {
    private:
    int dim_;
    int p_;

    public:
    Morris2006(int dimension, int active);

    std::string_view           name() const override;
    int                        dimension() const override;
    std::pair<double, double>  domain(int) const override;
    double                     evaluate(const std::vector<double> &x) const override;
    sens::util::SobolReference exactSobol() const override;
    std::vector<int>           initialPivot() const override;
};
