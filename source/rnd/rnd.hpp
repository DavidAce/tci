#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <pcg_random.hpp>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace rnd {
    using fp64 = double;
    using cx64 = std::complex<fp64>;

    namespace sfinae {
        template<typename T>
        using Real = decltype(std::real(std::declval<std::remove_cvref_t<T>>()));

        template<typename T>
        concept is_native_float =
            std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, long double>;

        template<typename T>
        concept is_extended_float = std::is_floating_point_v<T> && !is_native_float<T>;
    }

    using Rng = pcg64;

    struct Seeds {
        std::uint64_t init;
        std::uint64_t pivot;
        std::uint64_t nodeSamples;
        std::uint64_t predictions;
        unsigned      xfac;
    };

    struct RandomEngines {
        Seeds seeds{};
        Rng   pivot;
        Rng   nodeSamples;
        Rng   predictions;
        explicit RandomEngines(std::uint64_t seed);
    };

    enum class dist { uniform, normal, lognormal };
    constexpr std::string_view enum2sv(const dist &d);
    constexpr dist             sv2enum(std::string_view d);

    // Random functions
    extern int                  uniform_integer_01(Rng &rng);
    extern double               uniform_double_01(Rng &rng);
    extern double               uniform_double_box(double min, double max, Rng &rng);
    extern double               uniform_double_box(double halfwidth, Rng &rng);
    extern std::complex<double> uniform_complex_in_unit_circle(Rng &rng);
    extern std::complex<double> uniform_complex_on_unit_circle(Rng &rng);
    extern std::complex<double> uniform_complex_box(double real_min, double real_max, double imag_min, double imag_max,
                                                    Rng &rng);
    extern std::complex<double> uniform_complex_slice(double radius_max, double angle_min, double angle_max, Rng &rng);
    extern std::vector<int>     random_with_replacement(const std::vector<int> &indata, Rng &rng);
    extern std::vector<int>     random_with_replacement(const std::vector<int> &indata, size_t num_choose, Rng &rng);
    extern double gaussian_truncated(double lowerLimit, double upperLimit, double mean, double std, Rng &rng);
    template<typename T>
    T uniform_integer_box(T min, T max, Rng &rng);

    template<typename out_t>
    out_t uniform(out_t min, out_t max, Rng &rng);
    template<typename out_t>
    out_t normal(out_t mean, out_t std, Rng &rng);
    template<typename out_t>
    out_t log_normal(out_t mean, out_t std, Rng &rng);

    template<typename real_t = double>
    auto uniform_complex_in_circle(real_t radius, Rng &rng) {
        const auto r     = std::sqrt(uniform(real_t{0}, real_t{1}, rng)) * radius;
        const auto angle = uniform(real_t{0}, 2 * std::numbers::pi_v<real_t>, rng);
        return std::polar(r, angle);
    }
    template<typename real_t = double>
    auto uniform_complex_on_circle(real_t radius, Rng &rng) {
        return std::polar(radius, uniform(real_t{0}, 2 * std::numbers::pi_v<real_t>, rng));
    }
    template<typename real_t = double>
    auto uniform_complex_in_square(real_t real_min, real_t real_max, real_t imag_min, real_t imag_max, Rng &rng) {
        const auto re = uniform(real_min, real_max, rng);
        const auto im = uniform(imag_min, imag_max, rng);
        return std::complex(re, im);
    }

    template<typename T>
    std::vector<T> uniform_unit_n_sphere(size_t n, Rng &rng);

    template<typename T>
    void shuffle(T &list, Rng &rng);

    template<typename out_t = double>
    out_t random(dist d, out_t mean, out_t width, Rng &rng);
    template<typename out_t = double>
    std::vector<out_t> random(dist d, out_t mean, out_t width, size_t num, Rng &rng);
    template<typename out_t = double>
    std::vector<out_t> random(dist d, out_t mean, out_t width, const std::vector<out_t> &weights, Rng &rng);

    template<typename out_t = double>
    out_t random(std::string_view distribution, out_t mean, out_t width, Rng &rng);
    template<typename out_t = double>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, size_t num, Rng &rng);

    template<typename out_t = double>
    std::vector<out_t> random(std::string_view distribution, out_t mean, out_t width, const std::vector<out_t> &weights,
                              Rng &rng);

}
