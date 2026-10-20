#include "PermutedFunction.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

PermutedFunction::PermutedFunction(int dim, std::vector<int> permutation)
    : dim_(dim), permutation_(std::move(permutation)), inversePermutation_(dim) {
    if(dim <= 0) throw std::invalid_argument("Invalid dimension");

    if(permutation_.empty()) {
        permutation_.resize(dim);
        std::iota(permutation_.begin(), permutation_.end(), 0);
    }

    if(permutation_.size() != static_cast<std::size_t>(dim)) throw std::invalid_argument("Permutation size mismatch");

    std::vector<bool> seen(dim, false);

    for(int i = 0; i < dim; ++i) {
        const int k = permutation_[i];
        if(k < 0 || k >= dim || seen[k]) throw std::invalid_argument("Invalid permutation");
        seen[k]                = true;
        inversePermutation_[k] = i;
        if(k != i) isIdentity_ = false;
    }
}

std::vector<int> PermutedFunction::initialPivotOriginal() const { return std::vector<int>(dim_, 4); }

int PermutedFunction::dimension() const { return dim_; }

double PermutedFunction::evaluate(const std::vector<double> &x) const {
    if(isIdentity_) return evaluateOriginal(x);
    static thread_local std::vector<double> originalX;
    if(originalX.size() != permutation_.size()) originalX.resize(permutation_.size());
    for(int i = 0; i < dim_; ++i) originalX[i] = x[permutation_[i]];
    return evaluateOriginal(originalX);
}

std::pair<double, double> PermutedFunction::domain(int k) const { return domainOriginal(inversePermutation_[k]); }

std::vector<int> PermutedFunction::initialPivot() const {
    auto original = initialPivotOriginal();
    if(original.size() != permutation_.size()) throw std::runtime_error("Initial pivot size mismatch");
    std::vector<int> result(dim_);
    for(int i = 0; i < dim_; ++i) result[permutation_[i]] = original[i];
    return result;
}

sens::SobolReference PermutedFunction::exactSobolIndices() const {
    auto ref = exactSobolIndicesOriginal();

    auto reorder = [&](std::vector<double> &values) {
        if(values.empty()) return;
        if(values.size() != permutation_.size()) throw std::runtime_error("Sobol reference size mismatch");
        auto original = values;
        for(int i = 0; i < dim_; ++i) values[permutation_[i]] = original[i];
    };

    reorder(ref.first);
    reorder(ref.total);

    for(auto &pair : ref.second) {
        pair.i = permutation_.at(pair.i);
        pair.j = permutation_.at(pair.j);
        if(pair.i > pair.j) std::swap(pair.i, pair.j);
    }

    for(auto &triple : ref.third) {
        std::array<int, 3> indices{permutation_.at(triple.i), permutation_.at(triple.j), permutation_.at(triple.k)};
        std::sort(indices.begin(), indices.end());
        triple.i = indices[0];
        triple.j = indices[1];
        triple.k = indices[2];
    }

    std::sort(ref.second.begin(), ref.second.end(),
              [](const auto &a, const auto &b) { return std::tie(a.i, a.j) < std::tie(b.i, b.j); });
    std::sort(ref.third.begin(), ref.third.end(),
              [](const auto &a, const auto &b) { return std::tie(a.i, a.j, a.k) < std::tie(b.i, b.j, b.k); });

    return ref;
}

const std::vector<int> &PermutedFunction::permutation() const { return permutation_; }
