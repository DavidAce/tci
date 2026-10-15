#pragma once

#include <filesystem>

namespace sens {
    struct RunResults;
}

namespace io {
    void writeResults(const sens::RunResults &results, const std::filesystem::path &path);
}
