#include "io/io.hpp"
#include "sens/results.hpp"
#include <algorithm>
#include <cstdint>
#include <Eigen/Core>
#include <h5pp/h5pp.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace io {
    namespace {

        void writeSobol(h5pp::File &file, const sens::util::SensitivityResult &s, const std::string &prefix) {
            file.writeDataset(s.mean, prefix + "/mean");
            file.writeDataset(s.variance, prefix + "/variance");
            file.writeDataset(s.rmse, prefix + "/rmse");
            file.writeDataset(s.maxAbs, prefix + "/max_abs");

            std::vector<h5pp::vstr_t> names;
            std::vector<double>       estimated, exact, absError;
            names.reserve(s.indices.size());
            estimated.reserve(s.indices.size());
            exact.reserve(s.indices.size());
            absError.reserve(s.indices.size());
            for(const auto &idx : s.indices) {
                names.push_back(idx.name);
                estimated.push_back(idx.estimated);
                exact.push_back(idx.exact);
                absError.push_back(idx.error);
            }
            file.writeDataset(names, prefix + "/names");
            file.writeDataset(estimated, prefix + "/estimated");
            file.writeDataset(exact, prefix + "/exact");
            file.writeDataset(absError, prefix + "/abs_error");
        }

    } // namespace

    void writeResults(const sens::RunResults &r, const std::filesystem::path &path) {
        if(r.dimension <= 0) throw std::invalid_argument("Invalid run dimension");

        h5pp::File file(path.string(), h5pp::FileAccess::REPLACE);
        file.writeDataset(r.functionName, "run/function");
        file.writeDataset(r.dimension, "run/dimension");
        file.writeDataset(r.stopReason, "run/stop_reason");
        file.writeDataset(r.firstStableNEval, "run/first_stable_n_eval");
        file.writeDataset(r.finalNEval, "run/final_n_eval");
        file.writeDataset(r.firstStableHalfSweeps, "run/first_stable_half_sweeps");
        file.writeDataset(r.finalHalfSweeps, "run/final_half_sweeps");

        std::vector<int>                                                              halfSweeps;
        std::vector<std::uint64_t>                                                    nEval;
        std::vector<double>                                                           mean, variance, rmse, maxAbs;
        Eigen::Matrix<std::uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bonds(
            static_cast<Eigen::Index>(r.history.size()), static_cast<Eigen::Index>(r.dimension - 1));
        for(std::size_t i = 0; i < r.history.size(); ++i) {
            const auto &s = r.history[i];
            if(s.bondDimensions.size() != static_cast<std::size_t>(r.dimension - 1))
                throw std::invalid_argument("Wrong number of TT bond dimensions in sweep record");
            halfSweeps.push_back(s.halfSweeps);
            nEval.push_back(s.nEval);
            mean.push_back(s.mean);
            variance.push_back(s.variance);
            rmse.push_back(s.sobolRMSE);
            maxAbs.push_back(s.sobolMaxAbs);
            for(std::size_t k = 0; k < s.bondDimensions.size(); ++k)
                bonds(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = s.bondDimensions[k];
        }
        if(!r.history.empty()) {
            file.writeDataset(halfSweeps, "history/half_sweeps");
            file.writeDataset(nEval, "history/n_eval");
            file.writeDataset(mean, "history/mean");
            file.writeDataset(variance, "history/variance");
            file.writeDataset(rmse, "history/sobol_rmse");
            file.writeDataset(maxAbs, "history/sobol_max_abs");
            if(r.dimension > 1) file.writeDataset(bonds, "history/bond_dimensions");
        }

        writeSobol(file, r.firstStableSobol, "sobol/first_stable");
        writeSobol(file, r.finalSobol, "sobol/final");

        if(r.offGrid) {
            const auto &s = *r.offGrid;
            if(s.dimension != static_cast<std::size_t>(r.dimension) || s.yExact.size() != s.yTCI.size() ||
               s.x.size() != s.yExact.size() * s.dimension)
                throw std::invalid_argument("Off-grid x/y sizes are inconsistent");

            using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            RowMatrix x(static_cast<Eigen::Index>(s.yExact.size()), static_cast<Eigen::Index>(s.dimension));
            std::copy(s.x.begin(), s.x.end(), x.data());
            file.writeDataset(x, "test/x");
            file.writeDataset(s.yExact, "test/y_exact");
            file.writeDataset(s.yTCI, "test/y_tci");
        }
    }

} // namespace io
