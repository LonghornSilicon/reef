cmake_minimum_required(VERSION 3.20)
project(reef_perf LANGUAGES C CXX)

# Sparta 3.x requires C++20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Third-party dependencies are built from the ext/ submodules by
# tools/build_deps.sh; the Docker image installs them in /opt/reef-perf.
if(DEFINED ENV{REEF_PERF_DEPS})
    set(_deps_default "$ENV{REEF_PERF_DEPS}")
else()
    set(_deps_default "/opt/reef-perf")
endif()
set(REEF_PERF_DEPS "${_deps_default}" CACHE PATH
    "Install prefix of yaml-cpp, Sparta and Spike built from ext/")
if(NOT EXISTS "${REEF_PERF_DEPS}/lib/libsparta.a")
    message(FATAL_ERROR "Dependencies not found in ${REEF_PERF_DEPS}. Build "
                        "inside the Docker image (tools/docker.sh shell).")
endif()

# Sparta. Prefer our yaml-cpp 0.8 over any system copy.
list(PREPEND CMAKE_PREFIX_PATH "${REEF_PERF_DEPS}")
list(APPEND CMAKE_MODULE_PATH "${REEF_PERF_DEPS}/lib/cmake/sparta")
set(SPARTA_SEARCH_DIR "${REEF_PERF_DEPS}")
find_package(Sparta REQUIRED)

# Spike, linked statically.
add_library(spike INTERFACE)
target_include_directories(spike SYSTEM INTERFACE "${REEF_PERF_DEPS}/include")
foreach(lib riscv disasm fesvr fdt softfloat)
    target_link_libraries(spike INTERFACE "${REEF_PERF_DEPS}/lib/lib${lib}.a")
endforeach()
target_link_libraries(spike INTERFACE ${CMAKE_DL_LIBS} pthread)

# The model as a library, so the simulator, tools and tests share it.
add_library(reef_perf_core STATIC
    src/csrc/src/dispatch.cpp
    src/csrc/src/execute.cpp
    src/csrc/src/fetch.cpp
    src/csrc/src/func_sim.cpp
    src/csrc/src/inst.cpp
    src/csrc/src/inst_decode.cpp
    src/csrc/src/reef_sim.cpp
    src/csrc/src/resource_pool.cpp
    src/csrc/src/rob.cpp
    src/csrc/src/spike_driver.cpp
)
target_include_directories(reef_perf_core PUBLIC src/csrc/include)
target_link_libraries(reef_perf_core PUBLIC SPARTA::sparta PRIVATE spike)

add_executable(reef_perf src/csrc/src/main.cpp)
target_link_libraries(reef_perf PRIVATE reef_perf_core)

add_executable(reef_trace tools/reef_trace.cpp)
target_link_libraries(reef_trace PRIVATE reef_perf_core)

set_target_properties(reef_perf reef_trace PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
