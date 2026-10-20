#pragma once

#include "BaseFunction.hpp"
#include <utility>
#include <vector>

class PermutedFunction : public BaseFunction {
    protected:
    int dim_;

    private:
    /// Maps original indices to permuted indices.
    /// permutation_[i] is the new position of original parameter i.
    std::vector<int> permutation_;

    /// Maps permuted indices back to original indices.
    /// inversePermutation_[permutation_[i]] == i.
    std::vector<int> inversePermutation_;

    bool isIdentity_ = true;

    protected:
    explicit PermutedFunction(int dim, std::vector<int> permutation);

    // Derived classes implement these in original coordinates.
    virtual double                    evaluateOriginal(const std::vector<double> &x) const = 0;
    virtual std::pair<double, double> domainOriginal(int i) const                          = 0;
    virtual sens::SobolReference      exactSobolIndicesOriginal() const                    = 0;
    virtual std::vector<int>          initialPivotOriginal() const;

    public:
    int                       dimension() const final;
    double                    evaluate(const std::vector<double> &x) const final;
    std::pair<double, double> domain(int k) const final;
    std::vector<int>          initialPivot() const final;
    sens::SobolReference      exactSobolIndices() const final;
    const std::vector<int>   &permutation() const;
};
