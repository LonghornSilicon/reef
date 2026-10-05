cmake_minimum_required(VERSION 3.20)
project(inference_engine LANGUAGES CXX)

# C++17 is a conservative host baseline while the Coral target toolchain is
# unconfirmed. Revisit the standard when that toolchain is pinned.
add_library(inference_engine STATIC
    src/csrc/src/operator/activation.cpp
    src/csrc/src/operator/attention.cpp
    src/csrc/src/operator/embedding.cpp
    src/csrc/src/tensor.cpp
    src/csrc/src/operator/mlp.cpp
    src/csrc/src/model.cpp
)
target_compile_features(inference_engine PUBLIC cxx_std_17)
target_include_directories(inference_engine PUBLIC src/csrc/include)

add_executable(matrix_cli tools/matrix_cli.cpp)
target_link_libraries(matrix_cli PRIVATE inference_engine)
set_target_properties(matrix_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()

# TODO: Add a fixed-weight model executable once the first model is implemented.
# TODO: Add a Coral cross-build and simulator run after the toolchain is pinned.
