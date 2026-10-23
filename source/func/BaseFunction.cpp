#include "BaseFunction.hpp"

BaseFunction::Domains BaseFunction::domains() const {
    Domains d(dimension());
    for(size_t k = 0; k < d.size(); ++k) { d[k] = domain(k); }
    return d;
}