cmake_minimum_required(VERSION 3.20)
project(inference_engine LANGUAGES CXX)

# TODO: Select the C++ standard and compiler options supported by the pinned
# Coral target toolchain. For now, this target only builds the host scaffold.
add_library(inference_engine STATIC
    csrc/src/activation.cpp
    csrc/src/attention.cpp
    csrc/src/embedding.cpp
    csrc/src/matrix.cpp
    csrc/src/mlp.cpp
    csrc/src/model.cpp
)
target_compile_features(inference_engine PUBLIC cxx_std_17)
target_include_directories(inference_engine PUBLIC csrc/include)

add_executable(matrix_cli csrc/tools/matrix_cli.cpp)
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
