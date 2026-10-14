#include "cli.hpp"
#include "cfg/cfg.hpp"
#include "CLI/CLI.hpp"
#include <CLI/CLI.hpp>
#include <fmt/core.h>
namespace cli {

    void parse(int argc, char **argv) {
        using namespace cfg;
        CLI::App app{"TCI sensitivity analysis"};

        app.allow_extras(false);
        app.option_defaults()->always_capture_default();
        app.add_option("--function", functionName, "Test function")
            ->check(CLI::IsMember({"ishigami", "morris", "legendre", "npoly"}));
        app.add_option("--quadrature", quadratureName, "Quadrature rule")->check(CLI::IsMember({"gk15", "gl", "cc"}));
        app.add_option("--nnodes", nnodes, "Number of Nodes (discretization points) per parameter ")
            ->default_val(15)
            ->check(CLI::PositiveNumber);
        app.add_option("--bond-dim", bondDim, "Maximum MPS bond dimension")->check(CLI::PositiveNumber);
        app.add_option("--max-sweeps", maxSweeps, "Maximum half-sweeps")->check(CLI::PositiveNumber);
        app.add_option("--seed", seed, "PCG random seed");
        app.add_option("--dimension", dimension,
                       "The number of input parameters for the function (for morris, legendre and npoly)")
            ->check(CLI::PositiveNumber);
        app.add_option("--active", active, "The number of input parameters for the function (legendre and npoly)")
            ->check(CLI::PositiveNumber);

        app.add_option("--tols", targets, "Relative RMSE target tolerances")
            ->delimiter(',')
            ->allow_extra_args(false)
            ->check(CLI::PositiveNumber);
        try {
            app.parse(argc, argv);
        } catch(const CLI::ParseError &e) { std::exit(app.exit(e)); }

        if(functionName == "morris" && active >= dimension)
            throw std::invalid_argument(
                fmt::format("'--function=morris' requires --active ({}) < --dimension ({})", active, dimension));
        if(quadratureName == "gk15" && nnodes != 15) throw std::invalid_argument("GK15 requires 15 nodes");
    }

}