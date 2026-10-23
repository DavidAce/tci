#include "inputs.hpp"
#include <algorithm>
#include <array>
#include <cctype>
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

        // Integers must therefore be exactly representable by double.
        constexpr std::int64_t maxExactInteger = 9007199254740992LL; // 2^53

        std::vector<std::int64_t> readDiscreteValues(const toml::node &node, std::string_view where) {
            const auto *array = node.as_array();
            if(!array || array->size() < 2) fail(where, "'values' must contain at least two integers");
            std::vector<std::int64_t> values;
            values.reserve(array->size());
            for(const auto &item : *array) {
                const auto value = item.value_exact<std::int64_t>();
                if(!value) fail(where, "expected an integer");
                if(*value < -maxExactInteger || *value > maxExactInteger)
                    fail(where, "integer values must be exactly representable by double (absolute value <= 2^53)");
                if(std::find(values.begin(), values.end(), *value) != values.end())
                    fail(where, "duplicate integer value");
                values.push_back(*value);
            }
            return values;
        }

        struct Overrides {
            std::optional<std::string>               type;
            std::optional<std::array<double, 2>>     domain;
            std::optional<GeneratedNodes>            nodes;
            std::optional<std::vector<std::int64_t>> values;
        };

        Overrides readParameterConfigurations(const toml::table &table, std::string_view where, TableType type) {
            switch(type) {
                case TableType::Group:
                    checkAllowedKeys(table, where, {"select", "type", "domain", "nodes", "values"});
                    break;
                case TableType::Single: checkAllowedKeys(table, where, {"type", "domain", "nodes", "values"}); break;
            }

            Overrides result;
            if(const auto *node = table.get("type")) {
                auto type = node->value<std::string>();
                if(!type) fail(where, "'type' must be a string");
                result.type = *type;
            }
            if(const auto *node = table.get("domain")) result.domain = readDomain(*node, where);
            if(const auto *node = table.get("nodes")) result.nodes = readNodes(*node, where);
            if(const auto *node = table.get("values")) result.values = readDiscreteValues(*node, where);
            return result;
        }

        void applyOverrides(Overrides &target, const Overrides &source) {
            // Switching input type clears settings inherited from the previous type.
            if(source.type && source.type != target.type) {
                target.domain.reset();
                target.nodes.reset();
                target.values.reset();
            }
            if(source.type) target.type = source.type;
            if(source.domain) target.domain = source.domain;
            if(source.nodes) target.nodes = source.nodes;
            if(source.values) target.values = source.values;
        }

        // Python-style selectors: index, start:stop, or start:stop:step.
        // Indices are zero-based, stops are exclusive, and strides are positive.
        std::uint64_t parseSelectionInteger(std::string_view token) {
            if(token.empty()) fail("select", "empty integer in selector");
            std::uint64_t value  = 0;
            const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
            if(ec != std::errc{} || ptr != token.data() + token.size())
                fail("select", fmt::format("invalid nonnegative integer '{}'", token));
            return value;
        }

        std::vector<std::uint64_t> parseParameterSelection(const toml::table &table, std::size_t count) {
            const auto &node  = getRequiredNode(table, "select", "parameter_groups");
            const auto *array = node.as_array();
            if(!array || array->empty()) fail("select", "expected a nonempty array of strings");

            // Join strings, remove whitespace, and accept only digits, commas and colons.
            std::string expression;
            for(const auto &entry : *array) {
                const auto part = entry.value<std::string>();
                if(!part || part->empty()) fail("select", "each selector must be a nonempty string");
                if(!expression.empty()) expression += ',';
                expression += *part;
            }
            expression.erase(std::remove_if(expression.begin(), expression.end(),
                                            [](unsigned char c) { return std::isspace(c) != 0; }),
                             expression.end());
            if(expression.empty() || std::any_of(expression.begin(), expression.end(), [](unsigned char c) {
                   return !std::isdigit(c) && c != ',' && c != ':';
               }))
                fail("select", "only nonnegative integers, commas and colons are allowed");

            std::vector<std::uint64_t> selection;
            std::vector<bool>          seen(count, false);
            const auto                 append = [&](std::uint64_t index) {
                if(index >= count) fail("select", fmt::format("index {} is outside [0, {})", index, count));
                if(seen[static_cast<std::size_t>(index)])
                    fail("select", fmt::format("index {} is selected more than once", index));
                seen[static_cast<std::size_t>(index)] = true;
                selection.push_back(index);
            };

            const std::string_view text   = expression;
            std::size_t            offset = 0;
            while(offset < text.size()) {
                const auto comma = text.find(',', offset);
                const auto item  = text.substr(offset, comma == std::string_view::npos ? comma : comma - offset);
                if(item.empty()) fail("select", "empty selector between commas");

                const auto colon = item.find(':');
                if(colon == std::string_view::npos) {
                    append(parseSelectionInteger(item));
                } else {
                    const auto second = item.find(':', colon + 1);
                    if(second != std::string_view::npos && item.find(':', second + 1) != std::string_view::npos)
                        fail("select", fmt::format("too many ':' in '{}'", item));

                    const auto startText = item.substr(0, colon);
                    const auto stopText =
                        item.substr(colon + 1, second == std::string_view::npos ? second : second - colon - 1);
                    const auto stepText =
                        second == std::string_view::npos ? std::string_view{} : item.substr(second + 1);
                    const std::uint64_t start = startText.empty() ? 0 : parseSelectionInteger(startText);
                    const std::uint64_t stop  = stopText.empty() ? count : parseSelectionInteger(stopText);
                    const std::uint64_t step  = stepText.empty() ? 1 : parseSelectionInteger(stepText);

                    if(step == 0) fail("select", "slice step cannot be zero");
                    if(start > count || stop > count)
                        fail("select", fmt::format("slice '{}' exceeds parameter count {}", item, count));
                    if(start > stop)
                        fail("select",
                             fmt::format("descending slice '{}' requires a negative step (not supported)", item));

                    for(std::uint64_t index = start; index < stop;) {
                        append(index);
                        if(step >= stop - index) break; // Prevent uint64_t overflow.
                        index += step;
                    }
                }

                if(comma == std::string_view::npos) break;
                offset = comma + 1;
                if(offset == text.size()) fail("select", "trailing comma");
            }
            if(selection.empty()) fail("select", "selector list selects no parameters");
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

        const auto namePtr = getRequiredNode(functionTable, "name", "function").value<std::string>();
        if(!namePtr) fail("function", "'name' must be a valid function name in [ishigami, morris, legendre, npoly]");
        const std::string functionName = static_cast<std::string>(*namePtr);

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
                        fail(where, fmt::format("input x{} is assigned by multiple groups", index));
                    groupAssigned[index] = true;
                    applyOverrides(resolved[index], overridesTable);
                }
            }
        }

        InputConfiguration configuration;
        configuration.functionName = functionName;
        configuration.dimension    = dimension;
        configuration.parameters.reserve(dimension);
        for(std::size_t i = 0; i < dimension; ++i) {
            const auto &p    = resolved[i];
            const auto  name = fmt::format("x{}", i);
            if(!p.type) fail(name, "missing type after applying defaults and group overrides");
            if(*p.type == "continuous") {
                if(!p.domain || !p.nodes) fail(name, "continuous input requires 'domain' and 'nodes'");
                if(p.values) fail(name, "continuous input cannot specify values");
                configuration.parameters.push_back({name, ContinuousParameter{*p.domain, *p.nodes}});
            } else if(*p.type == "discrete") {
                if(!p.values) fail(name, "integer input requires 'values'");
                if(p.domain || p.nodes) fail(name, "discrete input accepts values");
                configuration.parameters.push_back({name, DiscreteParameter{*p.values}});
            } else {
                fail(name, fmt::format("unknown type '{}' (use 'continuous' or 'discrete')", *p.type));
            }
        }
        return configuration;
    }

} // namespace cfg