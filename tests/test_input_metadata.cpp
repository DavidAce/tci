#include "cfg/inputs.hpp"
#include "func/BaseFunction.hpp"
#include "quad/quadrature.hpp"
#include "sens/grid.hpp"
#include "sens/hdf5.hpp"
#include "sens/sobol.hpp"
#include <cstdint>
#include <filesystem>
#include <h5pp/h5pp.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
    void require(bool condition, const char *message) {
        if(!condition) throw std::runtime_error(message);
    }

    class TestFunction final : public BaseFunction {
        public:
        std::string_view name() const override { return "mixed"; }
        int              dimension() const override { return 3; }
        double evaluate(const std::vector<double> &x) const override { return x.at(0) + 2 * x.at(1) + 3 * x.at(2); }
        std::pair<double, double> domain(int) const override { return {-1.0, 1.0}; }
        sens::SobolReference      exactSobolIndices() const override { return {}; }
    };

    void testConfigured(const std::filesystem::path &path, const cfg::InputConfiguration &config,
                        const TestFunction &function) {
        const std::vector<int> permutation{2, 0, 1};
        const auto             d = sens::discretizeInputs(function, config, permutation);
        {
            h5pp::File file(path.string(), h5pp::FileAccess::REPLACE);
            sens::writeInputMetadata(file, d, config, permutation, "unused");
        }
        {
            h5pp::File        file(path.string(), h5pp::FileAccess::READWRITE);
            const std::string p = "param/inputs";
            require(file.readDataset<int>(p + "/schema_version") == 1, "schema version incorrect");
            require(file.readDataset<int>(p + "/dimension") == 3, "dimension incorrect");
            require(file.readDataset<std::string>(p + "/source") == "toml", "source incorrect");
            require(file.readDataset<std::vector<int>>(p + "/original_to_tt") == permutation,
                    "forward permutation incorrect");
            require(file.readDataset<std::vector<int>>(p + "/tt_to_original") == std::vector<int>({1, 2, 0}),
                    "inverse permutation incorrect");
            for(int k = 0; k < 3; ++k) {
                const auto axis     = p + "/axis_" + std::to_string(k);
                const int  original = file.readDataset<int>(axis + "/original_index");
                require(original == std::vector<int>({1, 2, 0})[k], "axis original index incorrect");
                require(file.readDataset<std::string>(axis + "/name") == "x" + std::to_string(original),
                        "input name incorrect");
                require(file.readDataset<std::vector<double>>(axis + "/nodes") == d.nodes[k], "nodes incorrect");
                require(file.readDataset<std::vector<double>>(axis + "/weights") == d.integrationWeights[k],
                        "integration weights incorrect");
            }
            for(int k = 0; k < 2; ++k) {
                const auto axis = p + "/axis_" + std::to_string(k);
                require(file.readDataset<std::string>(axis + "/type") == "discrete", "discrete type missing");
                const auto original = std::vector<int>({1, 2})[k];
                const auto expected =
                    std::get<cfg::DiscreteParameter>(config.parameters[original].specification).values;
                require(file.readDataset<std::vector<std::int64_t>>(axis + "/values") == expected,
                        "discrete values not preserved exactly");
                require(!file.linkExists(axis + "/domain"), "discrete axis must not have a continuous domain");
            }
            const auto axis = p + "/axis_2";
            require(file.readDataset<std::string>(axis + "/type") == "continuous", "continuous type missing");
            require(file.readDataset<std::vector<double>>(axis + "/domain") == std::vector<double>({-1.0, 1.0}),
                    "continuous domain incorrect");
            require(file.readDataset<std::string>(axis + "/rule") == "gl", "quadrature rule incorrect");
            require(file.readDataset<int>(axis + "/count") == 5, "node count incorrect");
        }
    }

    void testLegacy(const std::filesystem::path &path, const TestFunction &function) {
        const auto gl7 = [](double lo, double hi) { return sens::quadrature::gaussLegendre(7, lo, hi); };
        const auto d   = sens::discretizeInputs(function, gl7);
        {
            h5pp::File file(path.string(), h5pp::FileAccess::REPLACE);
            sens::writeInputMetadata(file, d, std::nullopt, {}, "gl");
        }
        {
            h5pp::File        file(path.string(), h5pp::FileAccess::READWRITE);
            const std::string p = "param/inputs";
            require(file.readDataset<std::string>(p + "/source") == "cli", "legacy source incorrect");
            require(file.readDataset<std::vector<int>>(p + "/original_to_tt") == std::vector<int>({0, 1, 2}),
                    "legacy permutation incorrect");
            for(int k = 0; k < 3; ++k) {
                const auto axis = p + "/axis_" + std::to_string(k);
                require(file.readDataset<std::string>(axis + "/type") == "continuous", "legacy type incorrect");
                require(file.readDataset<std::string>(axis + "/rule") == "gl", "legacy quadrature rule incorrect");
                require(file.readDataset<int>(axis + "/count") == 7, "legacy node count incorrect");
                require(file.readDataset<std::vector<double>>(axis + "/nodes") == d.nodes[k], "legacy nodes incorrect");
            }
        }
    }
}

int main(int argc, char **argv) {
    require(argc == 2, "usage: test-input-metadata <path-to-mixed-inputs.toml>");
    const auto   config = cfg::readInputConfiguration(argv[1]);
    TestFunction function;
    const auto   configuredPath = std::filesystem::current_path() / "test-input-metadata-configured.h5";
    const auto   legacyPath     = std::filesystem::current_path() / "test-input-metadata-legacy.h5";
    testConfigured(configuredPath, config, function);
    testLegacy(legacyPath, function);
    std::filesystem::remove(configuredPath);
    std::filesystem::remove(legacyPath);
}
