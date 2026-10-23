#pragma once

namespace h5pp {
    class File;
}

namespace env {
    // Record build, Git and execution environment in the newly created HDF5 file.
    void writeEnvironmentMetadata(h5pp::File &file);
}
