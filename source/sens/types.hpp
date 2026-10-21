#pragma once

#include <vector>

namespace sens {

    using Node  = double;
    using Nodes = std::vector<Node>;
    using Grid  = std::vector<Nodes>;

    using Weight             = double;
    using Weights            = std::vector<Weight>;
    using IntegrationWeights = std::vector<Weights>;

    using Point       = std::vector<double>;
    using NodeIndices = std::vector<int>;

}
