#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "sens/accuracy.hpp"
#include "sens/grid.hpp"
#include "sens/sobol.hpp"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    void require(bool condition, const char *message) {
        if(!condition) throw std::runtime_error(message);
    }

    class TestFunction final : public BaseFunction {
        public:
        std::string_view name() const override { return "test"; }
        int              dimension() const override { return 3; }
        double evaluate(const std::vector<double> &x) const override { return x.at(0) + 2.0 * x.at(1) + 3.0 * x.at(2); }
        std::pair<double, double> domain(int) const override { return {0.0, 1.0}; }
        sens::SobolReference      exactSobolIndices() const override { return {}; }
    };

    void checkSamples(const sens::Discretization &discretization, const sens::FunctionSamples &samples) {
        for(const auto &sample : samples) {
            require(sample.point.size() == 3, "sample dimension mismatch");
            for(std::size_t k = 0; k < 3; ++k) {
                const auto [lo, hi] = discretization.domains[k];
                const auto x        = sample.point[k];
                require(x > lo && x < hi, "sample outside configured input domain");
            }
        }
    }
} // namespace

int main(int argc, char **argv) {
    rnd::Rng predictionRng(123456789);
    require(argc == 2, "usage: test-discretization <path-to-toml>");
    const auto             config = cfg::readInputConfiguration(argv[1]);
    TestFunction           function;
    const std::vector<int> permutation{2, 0, 1}; // Original input -> TT position.
    const auto             discretization = sens::discretizeInputs(function, config, permutation);

    require(discretization.nodes.size() == 3, "wrong number of nodes axes");
    require(discretization.integrationWeights.size() == 3, "wrong number of weight axes");
    require(discretization.domains.size() == 3, "wrong number of sampling domains");

    // TOML original x0 -> TT axis 2, original x1 -> TT axis 0, original x2 -> TT axis 1.
    require(discretization.domains[0] == std::pair<double, double>(0.2, 0.4), "axis 0 domain mismatch");
    require(discretization.domains[1] == std::pair<double, double>(7.0, 8.0), "axis 1 domain mismatch");
    require(discretization.domains[2] == std::pair<double, double>(-5.0, -4.0), "axis 2 domain mismatch");

    for(std::size_t k = 0; k < 3; ++k) {
        const auto &nodes   = discretization.nodes[k];
        const auto &weights = discretization.integrationWeights[k];
        require(nodes.size() == 5 && weights.size() == 5, "wrong quadrature node count");
        double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);
        require(std::abs(totalWeight - 1.0) < 1e-12, "quadrature weights do not sum to one");
        double mean = 0.0;
        for(std::size_t j = 0; j < nodes.size(); ++j) mean += weights[j] * nodes[j];
        const auto [lo, hi] = discretization.domains[k];
        require(std::abs(mean - 0.5 * (lo + hi)) < 1e-12, "quadrature mean differs from uniform midpoint");
    }

    require(!sens::domainsMatch(function.domains(), discretization.domains),
            "changed TOML domain must invalidate reference measure");

    // TOML domains matching the benchmark remain compatible even with a permutation.
    auto matchingConfig = config;
    for(auto &input : matchingConfig.parameters)
        std::get<cfg::ContinuousParameter>(input.specification).domain = {0.0, 1.0};
    const auto matching = sens::discretizeInputs(function, matchingConfig, permutation);
    require(sens::domainsMatch(function.domains(), matching.domains),
            "reference domains should match after permutation");

    // Verify that the actual off-grid function sampler uses configured domains.
    const auto samples = sens::sampleFunctionValues(function, discretization, 200, predictionRng);
    require(samples.size() == 200, "wrong number of function samples");
    checkSamples(discretization, samples);
    for(const auto &sample : samples)
        require(sample.value == function.evaluate(sample.point), "incorrect function sample value");

    // The legacy path must retain BaseFunction domains and valid benchmark references.
    const auto gl5    = [](double lo, double hi) { return sens::quadrature::gaussLegendre(5, lo, hi); };
    const auto legacy = sens::discretizeInputs(function, gl5);
    require(sens::domainsMatch(function.domains(), legacy.domains), "legacy domains should match reference domains");
    for(const auto &[lo, hi] : legacy.domains) require(lo == 0.0 && hi == 1.0, "legacy domain changed unexpectedly");
    checkSamples(legacy, sens::sampleFunctionValues(function, legacy, 50, predictionRng));
}
