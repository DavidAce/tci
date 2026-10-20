#include "file.hpp"
#include "func/BaseFunction.hpp"
#include "h5t.hpp"
#include "util.hpp"
#include <h5pp/h5pp.h>
#include <xfac/tensor/tensor_ci_2.h>

namespace sens {
    using namespace util;

    void writeParam(h5pp::File &file, const xfac::TensorCI2Param &p, const BaseFunction &f,
                    const std::vector<double> &targets) {
        const h5XfacParam     xfacParam{p.bondDim, p.reltol, p.pivot1, p.fullPiv, p.nRookIter, p.useCachedFunction};
        static h5pp::hid::h5t xfacType;
        constexpr const char *xfacPath = "param/xfac";
        if(!xfacType.valid()) {
            xfacType = H5Tcreate(H5T_COMPOUND, sizeof(h5XfacParam));
            H5Tinsert(xfacType, "bondDim", HOFFSET(h5XfacParam, bondDim), H5T_NATIVE_INT);
            H5Tinsert(xfacType, "reltol", HOFFSET(h5XfacParam, reltol), H5T_NATIVE_DOUBLE);
            H5Tinsert(xfacType, "pivot1", HOFFSET(h5XfacParam, pivot1), h5pp::varr_t<int>::get_h5type());
            H5Tinsert(xfacType, "fullPiv", HOFFSET(h5XfacParam, fullPiv), H5T_NATIVE_HBOOL);
            H5Tinsert(xfacType, "nRookIter", HOFFSET(h5XfacParam, nRookIter), H5T_NATIVE_INT);
            H5Tinsert(xfacType, "useCachedFunction", HOFFSET(h5XfacParam, useCachedFunction), H5T_NATIVE_HBOOL);
        }
        if(!file.linkExists(xfacPath)) file.createTable(xfacType, xfacPath, "xfac parameters");
        file.appendTableRecords(xfacParam, xfacPath);

        const h5TCIParam      tciParam{f.dimension(), h5pp::vstr_t(f.name()), targets};
        static h5pp::hid::h5t tciType;
        constexpr const char *tciPath = "param/tci";
        if(!tciType.valid()) {
            tciType = H5Tcreate(H5T_COMPOUND, sizeof(h5TCIParam));
            H5Tinsert(tciType, "dim", HOFFSET(h5TCIParam, dim), H5T_NATIVE_INT);
            H5Tinsert(tciType, "name", HOFFSET(h5TCIParam, name), h5pp::vstr_t::get_h5type());
            H5Tinsert(tciType, "reltols", HOFFSET(h5TCIParam, reltols), h5pp::varr_t<double>::get_h5type());
        }
        if(!file.linkExists(tciPath)) file.createTable(tciType, tciPath, "TCI parameters");
        file.appendTableRecords(tciParam, tciPath);
    }

    void writePredictions(h5pp::File &file, const std::vector<Predictions> &predictions) {
        if(predictions.empty()) return;
        static h5pp::hid::h5t type;
        constexpr const char *path = "tci/predictions";
        if(!type.valid()) {
            type = H5Tcreate(H5T_COMPOUND, sizeof(Predictions));
            H5Tinsert(type, "predicted", HOFFSET(Predictions, predicted), H5T_NATIVE_DOUBLE);
            H5Tinsert(type, "funcvalue", HOFFSET(Predictions, funcvalue), H5T_NATIVE_DOUBLE);
            H5Tinsert(type, "error", HOFFSET(Predictions, error), H5T_NATIVE_DOUBLE);
            H5Tinsert(type, "point", HOFFSET(Predictions, point), h5pp::varr_t<double>::get_h5type());
        }
        if(!file.linkExists(path)) file.createTable(type, path, "Function value predictions");
        file.writeTableRecords(predictions, path);
    }

    void appendStatus(h5pp::File &file, const Status &status) {
        struct H5Status {
            using vlen_type = h5pp::varr_t<int>;
            int               sweeps, sweepsUnchanged, minSweepsUnchanged;
            double            changeTol;
            int               nEval;
            double            stat_mean, stat_variance, sobol_rmse, sobol_maxDiff;
            h5pp::varr_t<int> bondDims;
            h5pp::vstr_t      stopMsg;
        };
        const H5Status        entry{status.sweeps,
                                    status.sweepsUnchanged,
                                    status.minSweepsUnchanged,
                                    status.changeTol,
                                    status.nEval,
                                    status.statistics.mean,
                                    status.statistics.variance,
                                    status.sobolResult.rmse,
                                    status.sobolResult.maxDiff,
                                    status.bondDims,
                                    h5pp::vstr_t(status.stopMsg)};
        static h5pp::hid::h5t statusType;
        constexpr const char *statusPath = "tci/status";
        if(!statusType.valid()) {
            statusType = H5Tcreate(H5T_COMPOUND, sizeof(H5Status));
            H5Tinsert(statusType, "sweeps", HOFFSET(H5Status, sweeps), H5T_NATIVE_INT);
            H5Tinsert(statusType, "sweepsUnchanged", HOFFSET(H5Status, sweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(statusType, "minSweepsUnchanged", HOFFSET(H5Status, minSweepsUnchanged), H5T_NATIVE_INT);
            H5Tinsert(statusType, "changeTol", HOFFSET(H5Status, changeTol), H5T_NATIVE_DOUBLE);
            H5Tinsert(statusType, "nEval", HOFFSET(H5Status, nEval), H5T_NATIVE_INT);
            H5Tinsert(statusType, "stat_mean", HOFFSET(H5Status, stat_mean), H5T_NATIVE_DOUBLE);
            H5Tinsert(statusType, "stat_variance", HOFFSET(H5Status, stat_variance), H5T_NATIVE_DOUBLE);
            H5Tinsert(statusType, "sobol_rmse", HOFFSET(H5Status, sobol_rmse), H5T_NATIVE_DOUBLE);
            H5Tinsert(statusType, "sobol_maxDiff", HOFFSET(H5Status, sobol_maxDiff), H5T_NATIVE_DOUBLE);
            H5Tinsert(statusType, "bondDims", HOFFSET(H5Status, bondDims), h5pp::varr_t<int>::get_h5type());
            H5Tinsert(statusType, "stopMsg", HOFFSET(H5Status, stopMsg), h5pp::vstr_t::get_h5type());
        }
        if(!file.linkExists(statusPath)) file.createTable(statusType, statusPath, "Status Log");
        file.appendTableRecords(entry, statusPath);

        static h5pp::hid::h5t indexType;
        const auto            indexPath = fmt::format("tci/sweep_{}/sobol", status.sweeps);
        if(!indexType.valid()) {
            indexType = H5Tcreate(H5T_COMPOUND, sizeof(SobolIndex));
            H5Tinsert(indexType, "name", HOFFSET(SobolIndex, name), h5pp::fstr_t<32>::get_h5type());
            H5Tinsert(indexType, "estimated", HOFFSET(SobolIndex, estimated), H5T_NATIVE_DOUBLE);
            H5Tinsert(indexType, "exact", HOFFSET(SobolIndex, exact), H5T_NATIVE_DOUBLE);
            H5Tinsert(indexType, "error", HOFFSET(SobolIndex, error), H5T_NATIVE_DOUBLE);
        }
        if(!file.linkExists(indexPath)) file.createTable(indexType, indexPath, "Sobol Indices");
        file.writeTableRecords(status.sobolResult.sobolIndices, indexPath);
    }

    void writeStatusLog(h5pp::File &file, const std::vector<Status> &statuses) {
        for(const auto &status : statuses) appendStatus(file, status);
    }
}