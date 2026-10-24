#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "sens/accuracy.hpp"
#include "sens/grid.hpp"
#include "sens/sobol.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
    void require(bool condition, const char *message) {
        if(!condition) throw std::runtime_error(message);
    }
    bool near(double lhs, double rhs, double tolerance = 1e-12) { return std::abs(lhs - rhs) <= tolerance; }

    class TestFunction final : public BaseFunction {
        public:
        std::string_view name() const override { return "mixed"; }
        int              dimension() const override { return 3; }
        double evaluate(const std::vector<double> &x) const override { return x.at(0) + 2.0 * x.at(1) + 3.0 * x.at(2); }
        std::pair<double, double> domain(int) const override { return {-1.0, 1.0}; }
        sens::SobolReference      exactSobolIndices() const override { return {}; }
    };
}

int main(int argc, char **argv) {
    rnd::Rng predictionRng(123456789);
    require(argc == 2, "usage: test-discrete-inputs <path-to-toml>");
    const auto config = cfg::readInputConfiguration(argv[1]);
    require(config.dimension == 3, "wrong dimension");
    require(std::holds_alternative<cfg::ContinuousParameter>(config.parameters[0].specification), "x0 not continuous");
    const auto &integer = std::get<cfg::DiscreteParameter>(config.parameters[1].specification);
    require(integer.values == std::vector<std::int64_t>({-2, 0, 4}), "wrong integer values");

    TestFunction           function;
    const std::vector<int> permutation{2, 0, 1}; // Original x0,x1,x2 -> TT axes 2,0,1.
    const auto             d = sens::discretizeInputs(function, config, permutation);
    require(d.kinds.size() == 3, "wrong input kinds count");
    require(d.kinds[0] == sens::InputKind::Discrete, "wrong kind for TT axis 0");
    require(d.kinds[1] == sens::InputKind::Discrete, "wrong kind for TT axis 1");
    require(d.kinds[2] == sens::InputKind::Continuous, "wrong kind for TT axis 2");
    require(d.nodes[0] == std::vector<double>({-2.0, 0.0, 4.0}), "wrong integer grid");
    require(d.nodes[1] == std::vector<double>({-2.0, 0.0, 4.0}), "wrong second discrete grid");
    require(d.nodes[2].size() == 5, "wrong continuous nodes");
    require(d.integrationWeights[0] == std::vector<double>({1.0 / 3, 1.0 / 3, 1.0 / 3}), "wrong discrete weights");
    for(std::size_t k = 0; k < 3; ++k) {
        const auto &w = d.integrationWeights[k];
        require(near(std::accumulate(w.begin(), w.end(), 0.0), 1.0), "weight sum != 1");
    }

    // The TT's original argument order has been permuted; sampling and function
    // evaluation must agree on the TT-axis order.
    const auto         samples = sens::sampleFunctionValues(function, d, 10000, predictionRng);
    std::array<int, 3> integers{0, 0, 0};
    std::array<int, 3> secondDiscrete{0, 0, 0};
    for(const auto &sample : samples) {
        const auto &x = sample.point;
        require(x.size() == 3, "wrong sample dimension");
        const auto intIt = std::find(d.nodes[0].begin(), d.nodes[0].end(), x[0]);
        const auto secondIt = std::find(d.nodes[1].begin(), d.nodes[1].end(), x[1]);
        require(intIt != d.nodes[0].end(), "integer sampled outside configured values");
        require(secondIt != d.nodes[1].end(), "second discrete parameter sampled outside configured values");
        require(x[2] > -1 && x[2] < 1, "continuous sample outside domain");
        ++integers[static_cast<std::size_t>(intIt - d.nodes[0].begin())];
        ++secondDiscrete[static_cast<std::size_t>(secondIt - d.nodes[1].begin())];
        require(near(sample.value, function.evaluate(x)), "incorrect evaluated function value");
    }
    for(int i = 0; i < 3; ++i) {
        require(std::abs(integers[i] / double(samples.size()) - d.integrationWeights[0][i]) < 0.025,
                "incorrect sampled integer probability");
        require(std::abs(secondDiscrete[i] / double(samples.size()) - d.integrationWeights[1][i]) < 0.025,
                "incorrect sampled second discrete probability");
    }

    // Default uniform weights when probabilities are omitted.
    auto       uniform = config;
    const auto ud      = sens::discretizeInputs(function, uniform, permutation);
    for(int k = 0; k < 2; ++k)
        for(double weight : ud.integrationWeights[k]) require(near(weight, 1.0 / 3.0), "nonuniform default weights");
}
