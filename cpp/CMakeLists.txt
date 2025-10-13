cmake_minimum_required(VERSION 3.15)
project(RichermeIonAnalog VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Find Eigen3 (with fallback for Homebrew on macOS)
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    # Try to find Eigen3 in common Homebrew locations on macOS
    if(APPLE)
        set(EIGEN3_POSSIBLE_PATHS
            /usr/local/include/eigen3
            /opt/homebrew/include/eigen3
            /usr/local/Cellar/eigen/*/include/eigen3
        )

        foreach(EIGEN_PATH ${EIGEN3_POSSIBLE_PATHS})
            if(EXISTS ${EIGEN_PATH})
                set(EIGEN3_INCLUDE_DIR ${EIGEN_PATH})
                message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
                break()
            endif()
        endforeach()

        if(EIGEN3_INCLUDE_DIR)
            add_library(Eigen3::Eigen INTERFACE IMPORTED)
            set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
            )
            set(Eigen3_FOUND TRUE)
        endif()
    endif()

    if(NOT Eigen3_FOUND)
        message(FATAL_ERROR "Eigen3 not found. Please install it:\n"
                "  macOS: brew install eigen\n"
                "  Ubuntu/Debian: sudo apt-get install libeigen3-dev\n"
                "  Or set EIGEN3_INCLUDE_DIR manually")
    endif()
endif()

# Optional: Find CUDA-Q
find_package(CUDAQ QUIET)

# Core library
add_library(richerme_ion_analog
    richerme_ion_analog.cpp
    richerme_ion_analog.h
)

target_link_libraries(richerme_ion_analog
    PUBLIC
        Eigen3::Eigen
)

target_include_directories(richerme_ion_analog
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# TWA Framework library
add_library(twa_framework
    twa_framework.cpp
    twa_framework.hpp
)

# H2 TWA simulator library
add_library(h2_twa
    h2_twa_simulator.cpp
    h2_twa_simulator.hpp
)

target_link_libraries(h2_twa
    PUBLIC
        twa_framework
)

# H2O TWA simulator library
add_library(h2o_twa
    h2o_twa_simulator.cpp
    h2o_twa_simulator.hpp
)

target_link_libraries(h2o_twa
    PUBLIC
        twa_framework
)

# Optional: OpenMP for CPU parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(twa_framework PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - enabling parallel trajectory execution")
else()
    message(STATUS "OpenMP not found - sequential trajectory execution")
endif()

# TWA Simulation Executables
add_executable(h2_twa_simulation
    main_h2.cpp
)

target_link_libraries(h2_twa_simulation
    PRIVATE
        h2_twa
)

add_executable(h2o_twa_simulation
    main_h2o.cpp
)

target_link_libraries(h2o_twa_simulation
    PRIVATE
        h2o_twa
)

# H2 simulator library (TODO: implement C++ version)
# add_library(rich_sim_h2
#     rich_sim_h2.cpp
#     rich_sim_h2.h
# )
#
# target_link_libraries(rich_sim_h2
#     PUBLIC
#         richerme_ion_analog
#         Eigen3::Eigen
# )

# H2O simulator library (TODO: implement C++ version)
# add_library(rich_sim_h2o
#     rich_sim_h2o.cpp
#     rich_sim_h2o.h
# )
#
# target_link_libraries(rich_sim_h2o
#     PUBLIC
#         richerme_ion_analog
#         Eigen3::Eigen
# )

# Examples
add_executable(example_basic
    examples/example_basic.cpp
)

target_link_libraries(example_basic
    PRIVATE
        richerme_ion_analog
)

# H2 example (TODO: implement when rich_sim_h2 is ready)
# add_executable(example_h2
#     examples/example_h2.cpp
# )
#
# target_link_libraries(example_h2
#     PRIVATE
#         rich_sim_h2
# )

# Tests
enable_testing()

add_executable(test_richerme
    tests/test_richerme.cpp
)

target_link_libraries(test_richerme
    PRIVATE
        richerme_ion_analog
)

add_test(NAME RichermeTests COMMAND test_richerme)

# H2 test (TODO: implement when rich_sim_h2 is ready)
# add_executable(test_h2
#     tests/test_h2.cpp
# )
#
# target_link_libraries(test_h2
#     PRIVATE
#         rich_sim_h2
# )
#
# add_test(NAME H2Tests COMMAND test_h2)

# Installation
install(TARGETS richerme_ion_analog
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    richerme_ion_analog.h
    DESTINATION include/richerme
)

# TODO: Add rich_sim_h2 and rich_sim_h2o when implemented
# install(TARGETS rich_sim_h2 rich_sim_h2o
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )
#
# install(FILES
#     rich_sim_h2.h
#     rich_sim_h2o.h
#     DESTINATION include/richerme
# )

# Optional: CUDA-Q integration
if(CUDAQ_FOUND)
    message(STATUS "CUDA-Q found - enabling GPU acceleration")
    target_compile_definitions(richerme_ion_analog PUBLIC USE_CUDAQ)
    target_link_libraries(richerme_ion_analog PUBLIC cudaq::cudaq)
endif()

# Print configuration
message(STATUS "")
message(STATUS "Richerme Ion Analog C++ Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3 Version: ${Eigen3_VERSION}")
message(STATUS "  CUDA-Q Support: ${CUDAQ_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
