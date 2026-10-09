
cmake_minimum_required(VERSION 3.24)

### Setup project name and language
project(TCI VERSION 0.1
        DESCRIPTION "Trying tensor cross interpolation techniques"
        HOMEPAGE_URL "https://github.com/DavidAce/tci"
        LANGUAGES C CXX)


### Set up options
option(TCI_ENABLE_OPENMP   "Enable OpenMP flags such as -fopenmp"                 OFF)
option(TCI_ENABLE_MPI      "Enables use of MPI (work in progress)"                OFF)
option(TCI_ENABLE_COVERAGE "Adds the --coverage flag (used with github actions)"  OFF)
option(TCI_BUILD_BENCH     "Builds benchmarks in ./bench"                         OFF)
option(TCI_BUILD_EXAMPLES  "Builds examples in ./examples"                        OFF)
option(TCI_BUILD_TESTS     "Builds unit tests in ./tests"                         OFF)
option(TCI_CMAKE_DEBUG     "Print info during CMake configuration"                OFF)



if(TCI_DEBUG_CMAKE)
    ### Print operating system details
    include(cmake/PrintHostInfo.cmake)
endif()

### Add all source files
add_executable(TCI
        source/main.cpp 
        source/func/Ishigami.cpp
        source/func/Morris2006.cpp
        source/sens/sens.cpp
        )

target_include_directories(TCI PRIVATE source)


### Find dependencies
find_package(h5pp       1.11.3  REQUIRED)
find_package(Eigen3     5.0.0   REQUIRED)
find_package(spdlog     1.16.0  REQUIRED)
find_package(pcg-cpp            REQUIRED)
find_package(doctest    2.5.3   REQUIRED)
find_package(nanobench  4.3.11  REQUIRED)
find_package(CLI11      2.7.2   REQUIRED)
find_package(xfac               REQUIRED)
# find_package(BLAS               REQUIRED)

### Create a helper target that links dependencies (this target can be used to build tests and benchmarks)
add_library(tci-libs INTERFACE)

# Link the libraries to the helper target
target_link_libraries(tci-libs INTERFACE h5pp::h5pp)
target_link_libraries(tci-libs INTERFACE Eigen3::Eigen )
target_link_libraries(tci-libs INTERFACE spdlog::spdlog )
target_link_libraries(tci-libs INTERFACE pcg-cpp::pcg-cpp )
target_link_libraries(tci-libs INTERFACE doctest::doctest)
target_link_libraries(tci-libs INTERFACE nanobench::nanobench)
target_link_libraries(tci-libs INTERFACE CLI11::CLI11)
target_link_libraries(tci-libs INTERFACE xfac::xfac)

if(TCI_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    target_link_libraries(tci-libs INTERFACE OpenMP::OpenMP_CXX)
endif()

if(TCI_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX REQUIRED)
    target_link_libraries(tci-libs INTERFACE MPI::MPI_CXX)
endif()


### Link targets to the main executable
target_link_libraries(TCI PRIVATE tci-libs)


################################################################
### Get git version number                                   ###
### Generates a header gitversion/gitversion.h               ###
### Include it using #include <gitversion.h>                 ###
### Gives a namespace GIT:: with several git identifiers     ###
################################################################
include(cmake/gitversion.cmake)

add_subdirectory(bench)    # Builds benchmarks if TCI_BUILD_BENCH==ON
add_subdirectory(examples) # Builds examples if TCI_BUILD_EXAMPLES==ON


### CTest
if(TCI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    if(TCI_ENABLE_COVERAGE)
        # Coverage is used to check what fraction of the source code is tested
        # We use this option in the github actions when compiling on Ubuntu with gcc
        target_compile_options(tci-libs INTERFACE --coverage)
        target_link_options(tci-libs INTERFACE --coverage)
    endif()
endif()


if(TCI_CMAKE_DEBUG)
    # Print summary of CMake configuration
    include(cmake/PrintTargetInfo.cmake)
    print_and_write_project_summary(TCI)
endif()
