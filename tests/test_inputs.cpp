#include "cfg/inputs.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>
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
        require(parameter.name == "x" + std::to_string(k), "incorrect input name");
        const auto *continuous = std::get_if<cfg::ContinuousParameter>(&parameter.specification);
        require(continuous != nullptr, "expected continuous parameter");
        require(std::abs(continuous->domain[0] + 3.141592653589793) < 1e-14, "invalid lower bound");
        require(std::abs(continuous->domain[1] - 3.141592653589793) < 1e-14, "invalid upper bound");
        const auto *nodes = std::get_if<cfg::GeneratedNodes>(&continuous->nodes);
        require(nodes && nodes->quadratureRule == "gk15" && nodes->count == 15, "incorrect node rule");
    }

    // Mixed comma-separated indices, half-open slices, omitted bounds and strides.
    const auto path = std::filesystem::current_path() / "tci_test_python_selectors.toml";
    {
        std::ofstream out(path);
        require(static_cast<bool>(out), "cannot create selector test file");
        out << R"([function]
name = "selector-test"
dimension = 29

[defaults]
type = "continuous"
domain = [-1.0, 1.0]
nodes = { rule = "gk15", count = 15 }

[[parameter_groups]]
select = ["0:3", "6,8,9", "10", "11", "14:19,20 , 22", "23:28:2"]
domain = [-2.0, 2.0]
)";
    }
    const auto mixed = cfg::readInputConfiguration(path);
    const std::vector<int> expected{0, 1, 2, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20, 22, 23, 25, 27};
    require(mixed.parameters.size() == 29, "expected 29 input parameters");
    for(std::size_t i = 0; i < mixed.parameters.size(); ++i) {
        const auto *continuous = std::get_if<cfg::ContinuousParameter>(&mixed.parameters[i].specification);
        require(continuous != nullptr, "expected continuous input");
        const bool selected = std::find(expected.begin(), expected.end(), static_cast<int>(i)) != expected.end();
        require(continuous->domain[0] == (selected ? -2.0 : -1.0), "incorrect mixed selector expansion");
    }

    // Duplicates must not be silently accepted.
    {
        std::ofstream out(path);
        out << R"([function]
name = "selector-test"
dimension = 3
[defaults]
type = "continuous"
domain = [-1.0, 1.0]
nodes = { rule = "gk15", count = 15 }
[[parameter_groups]]
select = ["0:3", "2"]
)";
    }
    bool rejected = false;
    try { (void) cfg::readInputConfiguration(path); }
    catch(const std::invalid_argument &) { rejected = true; }
    require(rejected, "duplicate selection should be rejected");
    std::filesystem::remove(path);
}