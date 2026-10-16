#include "LegendrePoly3.hpp"
#include <cmath>
#include <stdexcept>

LegendrePoly3::LegendrePoly3(int dimension)
    : dim_(dimension), a_(dimension), b_(dimension * dimension, 0.0),
      c_(dimension * dimension * dimension, 0.0) {
    if(dim_ < 3) throw std::invalid_argument("Need at least three variables");

    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    for(int i = 0; i < dim_; ++i) {
        const double u = i + 1.0;
        a_[i] = std::sin(1.7 * u + 0.3 * u * u);
        sum1 += a_[i] * a_[i];

        for(int j = i + 1; j < dim_; ++j) {
            const double v = j + 1.0;
            b_[i * dim_ + j] = std::sin(0.83 * u * v + 0.37 * j);
            sum2 += b(i, j) * b(i, j);

            for(int k = j + 1; k < dim_; ++k) {
                const double w = k + 1.0;
                const auto index = (i * dim_ + j) * dim_ + k;
                c_[index] = std::cos(0.31 * u * v * w + 0.7 * (i + j + k + 3));
                sum3 += c(i, j, k) * c(i, j, k);
            }
        }
    }

    const double scale1 = std::sqrt(0.2 / sum1);
    const double scale2 = std::sqrt(0.3 / sum2);
    const double scale3 = std::sqrt(0.5 / sum3);

    for(auto& a : a_) a *= scale1;
    for(auto& b : b_) b *= scale2;
    for(auto& c : c_) c *= scale3;
}

std::string_view LegendrePoly3::name() const { return "LegendrePoly3"; }
int LegendrePoly3::dimension() const { return dim_; }
std::pair<double, double> LegendrePoly3::domain(int) const { return {0.0, 1.0}; }

std::vector<int> LegendrePoly3::initialPivot() const {
    std::vector<int> pivot(dim_);
    for(int i = 0; i < dim_; ++i) pivot[i] = (2 + 3 * i) % 15;
    return pivot;
}

double LegendrePoly3::evaluate(const std::vector<double>& x) const {
    std::vector<double> z(dim_);
    for(int i = 0; i < dim_; ++i) z[i] = std::sqrt(12.0) * (x[i] - 0.5);

    double result = 0.0;

    for(int i = 0; i < dim_; ++i) {
        result += a_[i] * z[i];

        for(int j = i + 1; j < dim_; ++j) {
            result += b(i, j) * z[i] * z[j];

            for(int k = j + 1; k < dim_; ++k)
                result += c(i, j, k) * z[i] * z[j] * z[k];
        }
    }

    return result;
}

sens::util::SobolReference LegendrePoly3::exactSobolIndices() const {
    sens::util::SobolReference ref;
    ref.first.resize(dim_);
    ref.total.resize(dim_);

    for(int i = 0; i < dim_; ++i) {
        ref.first[i] = a_[i] * a_[i];
        ref.total[i] += ref.first[i];

        for(int j = i + 1; j < dim_; ++j) {
            const double Vij = b(i, j) * b(i, j);
            ref.second.push_back({i, j, Vij});
            ref.total[i] += Vij;
            ref.total[j] += Vij;

            for(int k = j + 1; k < dim_; ++k) {
                const double Vijk = c(i, j, k) * c(i, j, k);
                ref.third.push_back({i, j, k, Vijk});
                ref.total[i] += Vijk;
                ref.total[j] += Vijk;
                ref.total[k] += Vijk;
            }
        }
    }
    return ref;
}
