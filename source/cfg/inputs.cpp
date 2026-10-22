#include "inputs.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml++/toml.hpp>
#include <vector>

namespace cfg {
    namespace {
        using namespace std::literals;

        [[noreturn]] void fail(std::string_view where, std::string_view message) {
            throw std::invalid_argument(fmt::format("Input TOML {}: {}", where, message));
        }

        void checkAllowedKeys(const toml::table &table, std::string_view where,
                              std::initializer_list<std::string_view> allowed) {
            for(const auto &[key, value] : table) {
                (void) value;
                if(std::find(allowed.begin(), allowed.end(), key.str()) == allowed.end())
                    fail(where, fmt::format("unknown key '{}'", key.str()));
            }
        }

        const toml::table &getRequiredTable(const toml::table &table, std::string_view key) {
            const auto *node = table.get(key);
            if(!node || !node->is_table()) fail(key, "expected a table");
            return *node->as_table();
        }

        const toml::node &getRequiredNode(const toml::table &table, std::string_view key, std::string_view where) {
            const auto *node = table.get(key);
            if(!node) fail(where, fmt::format("missing '{}'", key));
            return *node;
        }

        double readNumber(const toml::node &node, std::string_view where) {
            if(const auto value = node.value<double>()) return *value;
            if(const auto value = node.value<std::int64_t>()) return static_cast<double>(*value);
            fail(where, "expected a number");
        }

        std::array<double, 2> readDomain(const toml::node &node, std::string_view where) {
            const auto *array = node.as_array();
            if(!array || array->size() != 2) fail(where, "domain must be [lower, upper]");
            const double lo = readNumber(*array->get(0), where);
            const double hi = readNumber(*array->get(1), where);
            if(!std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi))
                fail(where, "domain must have finite bounds with lower < upper");
            return {lo, hi};
        }

        GeneratedNodes readNodes(const toml::node &node, std::string_view where) {
            const auto *table = node.as_table();
            if(!table) fail(where, "stage 1 supports generated nodes only: {rule, count}");
            checkAllowedKeys(*table, where, {"rule", "count"});
            const auto rule  = getRequiredNode(*table, "rule", where).value<std::string>();
            const auto count = getRequiredNode(*table, "count", where).value<std::int64_t>();
            if(!rule || rule->empty()) fail(where, "'rule' must be a nonempty string");
            if(!count || *count <= 0 || *count > std::numeric_limits<int>::max())
                fail(where, "'count' must be a positive integer");
            if(*rule == "gk15" && *count != 15) fail(where, "gk15 requires 15 nodes");
            if(*rule != "gk15" && *rule != "gl" && *rule != "cc")
                fail(where, fmt::format("unsupported rule '{}' (use gk15, gl, or cc)", *rule));
            return {*rule, static_cast<int>(*count)};
        }

        struct Overrides {
            std::optional<std::string>           type;
            std::optional<std::array<double, 2>> domain;
            std::optional<GeneratedNodes>        nodes;
        };

        Overrides readParameterConfigurations(const toml::table &table, std::string_view where, TableType tableType) {
            switch(tableType) {
                case TableType::Group: checkAllowedKeys(table, where, {"select", "type", "domain", "nodes"}); break;
                case TableType::Single: checkAllowedKeys(table, where, {"type", "domain", "nodes"}); break;
            }

            Overrides result;
            if(const auto *node = table.get("type")) {
                auto type = node->value<std::string>();
                if(!type) fail(where, "'type' must be a string");
                result.type = *type;
            }
            if(const auto *node = table.get("domain")) result.domain = readDomain(*node, where);
            if(const auto *node = table.get("nodes")) result.nodes = readNodes(*node, where);
            return result;
        }

        void applyOverrides(Overrides &target, const Overrides &source) {
            if(source.type) target.type = source.type;
            if(source.domain) target.domain = source.domain;
            if(source.nodes) target.nodes = source.nodes;
        }

        std::size_t parseOneBasedIndex(std::string_view token, std::size_t count) {
            if(token.empty() || token.front() != 'x')
                fail("select", fmt::format("expected an input name like x3, got '{}'", token));
            std::size_t index     = 0;
            const char *first     = token.data() + 1;
            const char *last      = token.data() + token.size();
            const auto [ptr, err] = std::from_chars(first, last, index);
            if(err != std::errc{} || ptr != last || index == 0 || index > count)
                fail("select", fmt::format("input '{}' is invalid or outside x1-x{}", token, count));
            return index - 1;
        }

        std::vector<std::size_t> parseParameterSelection(const toml::table &table, std::size_t count) {
            const auto &node  = getRequiredNode(table, "select", "parameter_groups");
            const auto *array = node.as_array();
            if(!array || array->empty()) fail("select", "expected a nonempty array of input names or ranges");
            std::vector<std::size_t> selection;
            std::vector<bool>        seen(count, false);
            for(const auto &entry : *array) {
                const auto selector = entry.value<std::string>();
                if(!selector) fail("select", "each selector must be a string");
                const auto hyphen = selector->find('-');
                const auto start  = parseOneBasedIndex(std::string_view(*selector).substr(0, hyphen), count);
                const auto stop   = hyphen == std::string::npos
                                        ? start
                                        : parseOneBasedIndex(std::string_view(*selector).substr(hyphen + 1), count);
                if(start > stop) fail("select", fmt::format("reversed range '{}'", *selector));
                for(std::size_t index = start; index <= stop; ++index) {
                    if(seen[index]) fail("select", fmt::format("input x{} selected twice in one group", index + 1));
                    seen[index] = true;
                    selection.push_back(index);
                }
            }
            return selection;
        }

    } // namespace

    InputConfiguration readInputConfiguration(const std::filesystem::path &path) {
        // toml++ throws a parse_error for malformed files or unavailable paths.
        const auto tomlfile = toml::parse_file(path.string());
        checkAllowedKeys(tomlfile, "root", {"function", "defaults", "parameter_groups"});

        const auto &functionTable = getRequiredTable(tomlfile, "function");
        checkAllowedKeys(functionTable, "function", {"name", "dimension"});
        const auto dimPtr = getRequiredNode(functionTable, "dimension", "function").value<std::int64_t>();
        if(!dimPtr || *dimPtr <= 0 || static_cast<std::uint64_t>(*dimPtr) > 100000)
            fail("function", "'dimension' must be an integer between 1 and 100000");
        const std::size_t dimension = static_cast<std::size_t>(*dimPtr);

        const auto defaultsTable =
            readParameterConfigurations(getRequiredTable(tomlfile, "defaults"), "defaults", TableType::Single);
        std::vector<Overrides> resolved(dimension, defaultsTable);
        std::vector<bool>      groupAssigned(dimension, false);

        if(const auto *node = tomlfile.get("parameter_groups")) {
            const auto *groups = node->as_array();
            if(!groups) fail("parameter_groups", "expected an array of tables ([[parameter_groups]])");
            for(std::size_t j = 0; j < groups->size(); ++j) {
                const auto *group = groups->get(j)->as_table();
                if(!group) fail("parameter_groups", fmt::format("entry {} is not a table", j));
                const auto where          = fmt::format("parameter_groups[{}]", j);
                const auto overridesTable = readParameterConfigurations(*group, where, TableType::Group);
                for(const auto index : parseParameterSelection(*group, dimension)) {
                    if(groupAssigned[index])
                        fail(where, fmt::format("input x{} is assigned by multiple groups", index + 1));
                    groupAssigned[index] = true;
                    applyOverrides(resolved[index], overridesTable);
                }
            }
        }

        InputConfiguration configuration;
        configuration.parameters.reserve(dimension);
        for(std::size_t i = 0; i < dimension; ++i) {
            const auto &p    = resolved[i];
            const auto  name = fmt::format("x{}", i + 1);
            if(!p.type) fail(name, "missing type after applying defaults and group overrides");
            if(*p.type != "continuous") fail(name, fmt::format("type '{}' is not implemented in stage 1", *p.type));
            if(!p.domain) fail(name, "missing domain after applying defaults and group overrides");
            if(!p.nodes) fail(name, "missing nodes after applying defaults and group overrides");
            configuration.parameters.push_back({name, ContinuousParameter{*p.domain, *p.nodes}});
        }
        return configuration;
    }

} // namespace cfg
