#pragma once

#include <h5pp/details/h5ppVarr.h>

namespace sens {

    struct h5XfacParam {
        using vlen_type = h5pp::varr_t<int>;
        int               bondDim;
        double            reltol;
        h5pp::varr_t<int> pivot1;
        bool              fullPiv;
        int               nRookIter;
        bool              useCachedFunction;
    };
    struct h5TCIParam {
        using vlen_type = h5pp::varr_t<double>;
        int                  dim;
        h5pp::vstr_t         name;
        h5pp::varr_t<double> reltols;
    };

}