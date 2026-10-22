#include "cfg/inputs.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>

namespace {
    void require(bool condition, const std::string &message) {
        if(!condition) throw std::runtime_error(message);
    }
}

int main(int argc, char **argv) {
    require(argc == 2, "usage: test-inputs <path-to-ishigami.toml>");
    const auto inputs = cfg::readInputConfiguration(argv[1]);
    require(inputs.parameters.size() == 3, "expected three input parameters");
    for(int k = 0; k < 3; ++k) {
        const auto &parameter = inputs.parameters[k];
        require(parameter.name == "x" + std::to_string(k + 1), "incorrect input name");
        const auto *continuous = std::get_if<cfg::ContinuousParameter>(&parameter.specification);
        require(continuous != nullptr, "expected continuous parameter");
        require(std::abs(continuous->domain[0] + 3.141592653589793) < 1e-14, "invalid lower bound");
        require(std::abs(continuous->domain[1] - 3.141592653589793) < 1e-14, "invalid upper bound");
        const auto *nodes = std::get_if<cfg::GeneratedNodes>(&continuous->nodes);
        require(nodes && nodes->rule == "gk15" && nodes->count == 15, "incorrect node rule");
    }
}
