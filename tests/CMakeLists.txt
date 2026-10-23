cmake_minimum_required(VERSION 3.24)
project(tests)

if(TARGET h5pp::h5pp)
    list(APPEND TESTS test-h5pp)
endif()

if(TARGET Eigen3::Eigen)
    list(APPEND TESTS test-eigen)
endif()

if(TARGET OpenMP::OpenMP_CXX)
    list(APPEND TESTS test-openmp)
endif()

foreach (test ${TESTS})
    add_executable(${test} ${test}.cpp)
    target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/source)
    target_link_libraries(${test} PRIVATE tci-libs)
    add_test(NAME ${test} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMAND ${test})
endforeach ()

add_executable(test-inputs test_inputs.cpp ${CMAKE_SOURCE_DIR}/source/cfg/inputs.cpp)
target_include_directories(test-inputs PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_link_libraries(test-inputs PRIVATE tci-libs)
add_test(NAME test-inputs COMMAND test-inputs ${CMAKE_SOURCE_DIR}/input/ishigami.toml)

add_executable(test-discretization
    test_discretization.cpp
    ${CMAKE_SOURCE_DIR}/source/cfg/inputs.cpp
    ${CMAKE_SOURCE_DIR}/source/func/BaseFunction.cpp
    ${CMAKE_SOURCE_DIR}/source/sens/grid.cpp
    ${CMAKE_SOURCE_DIR}/source/sens/accuracy.cpp
    ${CMAKE_SOURCE_DIR}/source/interp/interp.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/quadrature.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/GaussKronrod.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/GaussLegendre.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/ClenshawCurtis.cpp
)
target_include_directories(test-discretization PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_link_libraries(test-discretization PRIVATE tci-libs)
add_test(NAME test-discretization
    COMMAND test-discretization ${CMAKE_SOURCE_DIR}/tests/data/discretization_domains.toml)

# Continuous, integer and categorical input sampling and discretization.
add_executable(test-discrete-inputs
    test_discrete_inputs.cpp
    ${CMAKE_SOURCE_DIR}/source/cfg/inputs.cpp
    ${CMAKE_SOURCE_DIR}/source/func/BaseFunction.cpp
    ${CMAKE_SOURCE_DIR}/source/sens/grid.cpp
    ${CMAKE_SOURCE_DIR}/source/sens/accuracy.cpp
    ${CMAKE_SOURCE_DIR}/source/interp/interp.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/quadrature.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/GaussKronrod.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/GaussLegendre.cpp
    ${CMAKE_SOURCE_DIR}/source/quad/ClenshawCurtis.cpp
)
target_include_directories(test-discrete-inputs PRIVATE ${CMAKE_SOURCE_DIR}/source)
target_link_libraries(test-discrete-inputs PRIVATE tci-libs)
add_test(NAME test-discrete-inputs
    COMMAND test-discrete-inputs ${CMAKE_SOURCE_DIR}/tests/data/mixed_inputs.toml)
