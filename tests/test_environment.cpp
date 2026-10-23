#include "cfg/cfg.hpp"
#include "io/environment.hpp"
#include <env/environment.h>
#include <h5pp/h5pp.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
    void require(bool result, const char *message) {
        if(!result) throw std::runtime_error(message);
    }
}

int main() {
    cfg::seed  = 12345;
    cfg::input = "input/example.toml";
    if(setenv("SLURM_JOB_ID", "987654", 1) != 0)
        throw std::runtime_error("Cannot configure SLURM_JOB_ID for test");

    const auto path = "test-environment.h5";
    auto file = h5pp::File(path, h5pp::FileAccess::REPLACE);
    env::writeEnvironmentMetadata(file);

    require(file.readDataset<std::string>(".env/TCI/git/commit") == std::string(env::git::commit_hash),
            "Git commit differs from generated build header");
    require(file.readDataset<std::string>(".env/TCI/git/branch") == std::string(env::git::branch),
            "Git branch differs from generated build header");
    require(file.readDataset<std::string>(".env/TCI/build/compiler") == std::string(env::build::compiler),
            "Compiler differs from generated build header");
    require(file.readDataset<std::string>(".env/TCI/build/configuration") ==
                std::string(env::build::configuration),
            "Build configuration differs from generated build header");
    require(!file.readDataset<std::string>(".env/TCI/exec/hostname").empty(), "Missing execution hostname");
    require(!file.readDataset<std::string>(".env/TCI/exec/datetime").empty(), "Missing execution timestamp");
    require(file.readDataset<std::string>(".env/TCI/exec/slurm_job_id") == "987654", "Wrong Slurm job ID");
    require(file.readDataset<long>("common/seed") == 12345, "Wrong random seed");
    require(file.readDataset<std::string>("common/config_filename") == "input/example.toml", "Wrong input filename");
}
