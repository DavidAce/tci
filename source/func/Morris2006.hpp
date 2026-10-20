#pragma once
#include "PermutedFunction.hpp"
#include "sens/util.h"
#include <string_view>
#include <vector>

/*!
 * Morris-Moore-McKay (2006) test function.
 *
 * A polynomial sensitivity-analysis benchmark with a configurable
 * number of active and inactive inputs.
 *
 * Let dimension = active+inactive = M+p
 *
 * Reference configuration:
 *   Original paper: M=30, p=2,...,10
 *   UQTestFuns example: M=6, p=5
 *   Current project default: M=8, p=5
 *
 * For p=5:
 *   Variance = 5.25
 *   S_i  = 1/5.25   for active inputs
 *   ST_i = 1.1/5.25 for active inputs
 *   S_ij = 0.025/5.25 for active pairs
 *   All indices involving inactive variables are zero.

 * All active variables have equal first-order Sobol contributions
 * and weak pairwise interactions. There are no higher-order terms.
 *
 * This is useful for testing whether a sensitivity-analysis method
 * can identify influential inputs, distinguish inactive variables,
 * and recover known first- and second-order Sobol indices.
 *
 * Uses the interaction coefficient beta documented by UQTestFuns
 * (with the typo correction mentioned in the link)
 *
 * References:
 * [1] Morris, Moore & McKay (2006), J. Stat. Plan. Inference 136, 3203-3220.
 *     https://doi.org/10.1016/j.jspi.2005.01.001
 * [2] https://www.sfu.ca/~ssurjano/morretal06.html
 * [3] https://uqtestfuns.readthedocs.io/en/stable/test-functions/morris-m.html
 */
class Morris2006 final : public PermutedFunction {
    private:
    int dim_;
    int p_;

    public:
    Morris2006(int dimension, int active, std::vector<int> permutation);

    std::string_view name() const override;

    protected:
    std::pair<double, double>  domainOriginal(int) const override;
    double                     evaluateOriginal(const std::vector<double> &x) const override;
    sens::util::SobolReference exactSobolIndicesOriginal() const override;
    std::vector<int>           initialPivotOriginal() const override;
};
