#pragma once
#include <string>
#include <vector>
namespace cfg {
    inline std::string         functionName       = "";
    inline std::string         quadratureRuleName = "gk15";
    inline std::string         outdir             = "analysis/tci";
    inline std::string         input              = "";
    inline int                 nnodes             = 15;
    inline int                 dimension          = 8;
    inline int                 active             = 5;
    inline int                 bondDim            = 8;
    inline long                seed               = 42;
    inline int                 maxSweeps          = 20;
    inline double              convTol            = 1e-12; /*!< Convergence tolerance */
    inline int                 nsamples           = 128;
    inline std::vector<double> targets = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
    inline std::vector<int>    permutation = {};
    inline bool functionNameHas(std::string_view key) { return functionName.find(key) != std::string::npos; };
    inline bool quadratureRuleNameHas(std::string_view key) {
        return quadratureRuleName.find(key) != std::string::npos;
    };
}