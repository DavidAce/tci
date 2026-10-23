# Build-machine information captured when CMake configures TCI.
# Git variables are supplied by the existing cmake/gitversion.cmake.

cmake_host_system_information(RESULT TCI_BUILD_HOSTNAME QUERY HOSTNAME)
cmake_host_system_information(RESULT TCI_BUILD_CPU_TYPE QUERY PROCESSOR_DESCRIPTION)
cmake_host_system_information(RESULT TCI_BUILD_OS_NAME QUERY OS_NAME)
cmake_host_system_information(RESULT TCI_BUILD_OS_RELEASE QUERY OS_RELEASE)
cmake_host_system_information(RESULT TCI_BUILD_OS_VERSION QUERY OS_VERSION)
cmake_host_system_information(RESULT TCI_BUILD_OS_PLATFORM QUERY OS_PLATFORM)

set(TCI_BUILD_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(TCI_BUILD_CMAKE_VERSION "${CMAKE_VERSION}")
set(TCI_BUILD_GENERATOR "${CMAKE_GENERATOR}")


# Include the flags for the selected configuration in a single-config build.
# Multi-config generators select a configuration after CMake has configured.
get_property(_tci_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_tci_multi_config)
    set(TCI_BUILD_CONFIGURATION "multi-config")
    set(TCI_BUILD_COMPILER_FLAGS "${CMAKE_CXX_FLAGS}")
    set(TCI_BUILD_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
else()
    set(TCI_BUILD_CONFIGURATION "${CMAKE_BUILD_TYPE}")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _tci_configuration_upper)
    if(_tci_configuration_upper)
        set(TCI_BUILD_COMPILER_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_tci_configuration_upper}}")
        set(TCI_BUILD_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${_tci_configuration_upper}}")
    else()
        set(TCI_BUILD_COMPILER_FLAGS "${CMAKE_CXX_FLAGS}")
        set(TCI_BUILD_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    endif()
endif()
string(STRIP "${TCI_BUILD_COMPILER_FLAGS}" TCI_BUILD_COMPILER_FLAGS)
string(STRIP "${TCI_BUILD_LINKER_FLAGS}" TCI_BUILD_LINKER_FLAGS)

set(TCI_GIT_BRANCH "${GIT_BRANCH}")
set(TCI_GIT_COMMIT_HASH "${GIT_COMMIT_HASH}")
set(TCI_GIT_REVISION "${GIT_REVISION}")
foreach(_field IN ITEMS TCI_GIT_BRANCH TCI_GIT_COMMIT_HASH TCI_GIT_REVISION)
    if("${${_field}}" STREQUAL "")
        set(${_field} "unavailable")
    endif()
endforeach()


# Escape strings that will be inserted into C++ string literals.
function(tci_escape_cpp_string result input)
    set(_value "${input}")
    string(REPLACE "\\" "\\\\" _value "${_value}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    string(REPLACE "\n" "\\n" _value "${_value}")
    string(REPLACE "\r" "\\r" _value "${_value}")
    string(REPLACE "\t" "\\t" _value "${_value}")
    set(${result} "${_value}" PARENT_SCOPE)
endfunction()

foreach(_field IN ITEMS
    TCI_GIT_BRANCH TCI_GIT_COMMIT_HASH TCI_GIT_REVISION
    TCI_BUILD_HOSTNAME TCI_BUILD_CPU_TYPE TCI_BUILD_OS_NAME
    TCI_BUILD_OS_RELEASE TCI_BUILD_OS_VERSION TCI_BUILD_OS_PLATFORM
    TCI_BUILD_COMPILER TCI_BUILD_COMPILER_PATH TCI_BUILD_COMPILER_FLAGS
    TCI_BUILD_LINKER_FLAGS TCI_BUILD_CONFIGURATION TCI_BUILD_GENERATOR
    TCI_BUILD_CMAKE_VERSION
)
    tci_escape_cpp_string(ESC_${_field} "${${_field}}")
endforeach()

configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/environment.h.in"
    "${CMAKE_BINARY_DIR}/environment/include/env/environment.h"
    @ONLY
)
