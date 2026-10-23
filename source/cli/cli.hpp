#pragma once

#include "cfg/inputs.hpp"
#include <optional>

namespace cli {
    [[nodiscard]] std::optional<cfg::InputConfiguration> parse(int argc, char **argv);
}