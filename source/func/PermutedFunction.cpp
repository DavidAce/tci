
#pragma once

#include "BaseFunction.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

class PermutedFunction final : public BaseFunction {
    const BaseFunction &base_;

    std::vector<int> original_order_; /*!< Permuted TCI index -> original function index */
    std::vector<int> permuted_order;  /*!< Original function index -> Permuted TCI index */

    public:
    PermutedFunction(const BaseFunction &base, std::vector<int> order)
        : base_(base), original_order_(std::move(order)), permuted_order(base.dimension()) {
        const int dim = base_.dimension();

        // Empty permutation means identity.
        if(original_order_.empty()) {
            original_order_.resize(dim);
            std::iota(original_order_.begin(), original_order_.end(), 0);
        }

        if(original_order_.size() != static_cast<std::size_t>(dim))
            throw std::invalid_argument("Permutation size does not match function dimension");

        std::vector<bool> seen(dim, false);

        for(int k = 0; k < dim; ++k) {
            const int i = original_order_[k];

            if(i < 0 || i >= dim || seen[i])
                throw std::invalid_argument("Invalid permutation: duplicate or out-of-range index");

            seen[i]           = true;
            permuted_order[i] = k;
        }
    }

    std::string_view name() const override { return base_.name(); }

    int dimension() const override { return base_.dimension(); }

    double evaluate(const std::vector<double> &x) const override {
        // Reuse memory between calls, independently for each thread.
        static thread_local std::vector<double> originalX;
        if(originalX.size() != original_order_.size()) originalX.resize(original_order_.size());
        for(std::size_t k = 0; k < original_order_.size(); ++k) originalX[original_order_[k]] = x[k];
        return base_.evaluate(originalX);
    }

    std::pair<double, double> domain(int k) const override { return base_.domain(original_order_[k]); }

    std::vector<int> initialPivot() const override {
        const auto original = base_.initialPivot();
        if(original.size() != original_order_.size()) throw std::runtime_error("Invalid initial pivot dimension");
        std::vector<int> result(original_order_.size());
        for(std::size_t k = 0; k < original_order_.size(); ++k) result[k] = original[original_order_[k]];
        return result;
    }

    sens::util::SobolReference exactSobolIndices() const override {
        auto ref = base_.exactSobolIndices();

        // Reorder a vector indexed by individual variables.
        auto reorder = [&](std::vector<double> &values) {
            if(values.empty()) return;
            if(values.size() != original_order_.size()) throw std::runtime_error("Invalid Sobol reference dimension");
            auto original = values;
            for(std::size_t k = 0; k < original_order_.size(); ++k) { values[k] = original[original_order_[k]]; }
        };

        reorder(ref.first);
        reorder(ref.total);

        // Transform every pair from original indices to TCI indices.
        for(auto &pair : ref.second) {
            pair.i = permuted_order.at(pair.i);
            pair.j = permuted_order.at(pair.j);

            if(pair.i > pair.j) std::swap(pair.i, pair.j);
        }

        // Transform every triple, then restore ascending index order.
        for(auto &triple : ref.third) {
            std::array<int, 3> indices{permuted_order.at(triple.i), permuted_order.at(triple.j),
                                       permuted_order.at(triple.k)};

            std::sort(indices.begin(), indices.end());

            triple.i = indices[0];
            triple.j = indices[1];
            triple.k = indices[2];
        }

        // Restore lexicographic ordering of the reference lists.
        std::sort(ref.second.begin(), ref.second.end(),
                  [](const auto &a, const auto &b) { return std::tie(a.i, a.j) < std::tie(b.i, b.j); });

        std::sort(ref.third.begin(), ref.third.end(),
                  [](const auto &a, const auto &b) { return std::tie(a.i, a.j, a.k) < std::tie(b.i, b.j, b.k); });

        return ref;
    }

    const std::vector<int> &order() const { return original_order_; }
};
