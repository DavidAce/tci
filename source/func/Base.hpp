#pragma once
#include "sens/util.h"
#include <string_view>
#include <utility>
#include <vector>

class BaseFunction {
    public:
    virtual ~BaseFunction()                                                         = default;
    virtual std::string_view           name() const                                 = 0;
    virtual int                        dimension() const                            = 0;
    virtual double                     evaluate(const std::vector<double> &x) const = 0;
    virtual std::pair<double, double>  domain(int i) const                          = 0;
    virtual sens::util::SobolReference exactSobol() const                           = 0;
    virtual std::vector<int>           initialPivot() const { return std::vector<int>(dimension(), 4); }
};