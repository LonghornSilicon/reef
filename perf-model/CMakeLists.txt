cmake_minimum_required(VERSION 3.19)
project(coralnpu_perf CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Normally configured by scripts/build.sh, which builds yaml-cpp and Sparta
# from the ext/ submodules and installs them into DEPS_PREFIX.
set(DEPS_PREFIX "" CACHE PATH "Install prefix of yaml-cpp + Sparta built from ext/")
# The MPACT driver shared library, also built by scripts/build.sh.
set(CORALNPU_DRIVER_SO "" CACHE FILEPATH "Path to libcoralnpu_driver.so")

if(NOT DEPS_PREFIX OR NOT EXISTS "${DEPS_PREFIX}/lib/libsparta.a")
  message(FATAL_ERROR "Build with scripts/build.sh (or set -DDEPS_PREFIX=<prefix with "
                      "lib/libsparta.a>).")
endif()
# Prefer our yaml-cpp 0.8 over the system's 0.7.
list(PREPEND CMAKE_PREFIX_PATH "${DEPS_PREFIX}")
list(APPEND CMAKE_MODULE_PATH "${DEPS_PREFIX}/lib/cmake/sparta")
set(SPARTA_SEARCH_DIR "${DEPS_PREFIX}")
find_package(Sparta REQUIRED)

if(NOT CORALNPU_DRIVER_SO OR NOT EXISTS "${CORALNPU_DRIVER_SO}")
  message(FATAL_ERROR "Set -DCORALNPU_DRIVER_SO=<path to libcoralnpu_driver.so> "
                      "(built by scripts/build.sh)")
endif()

add_executable(coralnpu_perf
  src/main.cpp
  src/CoralSim.cpp
  src/FuncSim.cpp
  src/InstDecode.cpp
  src/Fetch.cpp
  src/Dispatch.cpp
  src/Execute.cpp
  src/Rob.cpp
)
target_include_directories(coralnpu_perf PRIVATE src driver)
target_link_libraries(coralnpu_perf PRIVATE SPARTA::sparta "${CORALNPU_DRIVER_SO}")
# Find the driver .so next to the binary's recorded path at run time.
get_filename_component(_driver_dir "${CORALNPU_DRIVER_SO}" DIRECTORY)
set_target_properties(coralnpu_perf PROPERTIES BUILD_RPATH "${_driver_dir}")
