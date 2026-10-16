#pragma once
#include <string>
#include <vector>
namespace cfg {
    inline std::string         functionName   = "ishigami";
    inline std::string         quadratureName = "gk15";
    inline std::string         outdir         = "analysis/tci";
    inline int                 nnodes         = 15;
    inline int                 bondDim        = 8;
    inline int                 maxSweeps      = 20;
    inline long                seed           = 42;
    inline int                 dimension      = 8;
    inline int                 active         = 5;
    inline int                 npredictions   = 0;
    inline std::vector<double> targets = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
    inline bool functionNameHas(std::string_view key) { return functionName.find(key) != std::string::npos; };
    inline bool quadratureNameHas(std::string_view key) { return quadratureName.find(key) != std::string::npos; };
}