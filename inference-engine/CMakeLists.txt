cmake_minimum_required(VERSION 3.20)
project(inference_engine LANGUAGES CXX)

# C++17 is a conservative host baseline while the Coral target toolchain is
# unconfirmed. Revisit the standard when that toolchain is pinned.
# Every operator is a template, so the engine is header-only: consumers get
# the includes and the standard, and nothing is compiled into a library.
add_library(inference_engine INTERFACE)
target_compile_features(inference_engine INTERFACE cxx_std_17)
target_include_directories(inference_engine INTERFACE src/csrc/include)

add_executable(tensor_cli tools/tensor_cli.cpp)
target_link_libraries(tensor_cli PRIVATE inference_engine)
set_target_properties(tensor_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()

# TODO: Add a fixed-weight model executable once the first model is implemented.
# TODO: Add a Coral cross-build and simulator run after the toolchain is pinned.
