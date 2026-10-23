#include "environment.hpp"
#include "cfg/cfg.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <env/environment.h>
#include <fstream>
#include <h5pp/h5pp.h>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>

namespace {
    std::string executionDatetime() {
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm    tm{};
        if(localtime_r(&now, &tm) == nullptr) return "unavailable";
        char buffer[32]{};
        if(std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm) == 0) return "unavailable";
        return buffer;
    }

    std::string executionHostname() {
        char buffer[256]{};
        if(gethostname(buffer, sizeof(buffer) - 1) != 0) return "unavailable";
        return buffer;
    }

    std::string executionCpuType() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string   line;
        while(std::getline(cpuinfo, line)) {
            if(!line.starts_with("model name")) continue;
            const auto colon = line.find(':');
            if(colon == std::string::npos) continue;
            const auto first = line.find_first_not_of(" \t", colon + 1);
            if(first != std::string::npos) return line.substr(first);
        }
        return "unavailable";
    }

    // Scheduler and thread settings are saved only when explicitly present.
    void writeExecutionEnvironmentVariables(h5pp::File &file) {
        constexpr std::array<std::pair<const char *, const char *>, 9> fields{{
            {"SLURM_JOB_ID", "slurm_job_id"},
            {"SLURM_JOB_NAME", "slurm_job_name"},
            {"SLURM_ARRAY_JOB_ID", "slurm_array_job_id"},
            {"SLURM_ARRAY_TASK_ID", "slurm_array_task_id"},
            {"SLURM_PROCID", "slurm_procid"},
            {"SLURM_CPUS_PER_TASK", "slurm_cpus_per_task"},
            {"SLURM_JOB_NODELIST", "slurm_job_nodelist"},
            {"OMP_NUM_THREADS", "omp_num_threads"},
            {"CUDA_VISIBLE_DEVICES", "cuda_visible_devices"},
        }};
        for(const auto &[variable, name] : fields) {
            if(const char *value = std::getenv(variable); value && *value)
                file.writeDataset(std::string(value), std::string(".env/TCI/exec/") + name);
        }
    }
}

void env::writeEnvironmentMetadata(h5pp::File &file) {
    file.writeDataset(executionDatetime(), ".env/TCI/exec/datetime");
    file.writeDataset(executionHostname(), ".env/TCI/exec/hostname");
    file.writeDataset(executionCpuType(), ".env/TCI/exec/cpu_type");
    file.writeDataset(build::datetime, ".env/TCI/build/datetime");
    file.writeDataset(build::hostname, ".env/TCI/build/hostname");
    file.writeDataset(build::cpu_type, ".env/TCI/build/cpu_type");
    file.writeDataset(build::os_name, ".env/TCI/build/os_name");
    file.writeDataset(build::os_release, ".env/TCI/build/os_release");
    file.writeDataset(build::os_version, ".env/TCI/build/os_version");
    file.writeDataset(build::os_platform, ".env/TCI/build/os_platform");
    file.writeDataset(build::compiler, ".env/TCI/build/compiler");
    file.writeDataset(build::compiler_flags, ".env/TCI/build/compiler_flags");
    file.writeDataset(build::linker_flags, ".env/TCI/build/linker_flags");
    file.writeDataset(build::configuration, ".env/TCI/build/configuration");
    file.writeDataset(build::generator, ".env/TCI/build/generator");
    file.writeDataset(build::cmake_version, ".env/TCI/build/cmake_version");
    file.writeDataset(git::branch, ".env/TCI/git/branch");
    file.writeDataset(git::commit_hash, ".env/TCI/git/commit");
    file.writeDataset(git::revision, ".env/TCI/git/revision");

    file.writeDataset(cfg::command, "common/TCI/exec/command");
    writeExecutionEnvironmentVariables(file);

    file.writeDataset(cfg::seed, "common/seed");
    file.writeDataset(cfg::input.empty() ? std::string("none") : cfg::input, "common/input_config");
}
