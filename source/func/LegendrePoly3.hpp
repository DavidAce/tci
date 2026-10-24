#pragma once
#include "PermutedFunction.hpp"

/*!
 * Third-order Legendre polynomial.
 *
 * A custom sensitivity-analysis benchmark constructed from
 * products of orthonormal, first-degree shifted Legendre polynomials
 * for independent uniform inputs on [0,1].
 *
 *
 * Includes all first-, second-, and third-order interactions.
 * Coefficients are deterministically generated and normalized so
 * that these interaction orders contribute 20%, 30%, and 50%
 * of the output variance, respectively.
 *
 * Exact properties for any dimension d >= 3:
 *   Mean = 1
 *   Variance = 1
 *   First-order Sobol sum  = 0.2
 *   Second-order Sobol sum = 0.3
 *   Third-order Sobol sum  = 0.5
 *   Higher-order Sobol sum = 0
 *
 * The exact Sobol indices are the squared expansion coefficients.
 * Useful for testing sensitivity analysis in many dimensions,
 * especially the recovery of distributed pairwise and three-way
 * interactions and total-order effects.
 *
 * Individual Sobol indices are squared expansion coefficients.
 * The coefficients are specific to this implementation and are
 * not taken from Sudret (2008).
 *
 * Sudret (2008) instead studies a distinct polynomial benchmark:
 *   f(x) = product_i [(1 + 3*x_i*x_i)/2]
 * with d=3, mean=1, variance=91/125, and exact Sobol indices
 * S_i=25/91, S_ij=5/91, S_123=1/91.
 *
 * References:
 * [1] Sudret (2008), Reliab. Eng. Syst. Saf. 93, 964-979.
 *     https://doi.org/10.1016/j.ress.2007.04.002
 * [2] PolynomialChaosReporter
 *     https://mooseframework.inl.gov/source/reporters/PolynomialChaosReporter.html
 */
class LegendrePoly3 final : public PermutedFunction {
    public:
    explicit LegendrePoly3(int dimension, std::vector<int> permutation = {});
    std::string_view name() const override;

    protected:
    std::pair<double, double> domainOriginal(int) const override;
    std::vector<int>          initialPivotOriginal() const override;
    double                    evaluateOriginal(const std::vector<double> &x) const override;
    sens::SobolReference      exactSobolIndicesOriginal() const override;

    private:
    std::vector<double> a_, b_, c_;

    double b(int i, int j) const;
    double c(int i, int j, int k) const;
};
