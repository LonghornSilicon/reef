cmake_minimum_required(VERSION 3.20)
project(inference_engine LANGUAGES CXX)

# C++17 is a conservative host baseline while the Coral target toolchain is
# unconfirmed. Revisit the standard when that toolchain is pinned.
# Every operator is a template, so the engine may be header-only: consumers get
# the includes and the standard, and nothing is compiled into a library.
add_library(inference_engine INTERFACE)
target_compile_features(inference_engine INTERFACE cxx_std_17)
target_include_directories(inference_engine INTERFACE src/csrc/include)
# The engine is the bit-exact reference for the Coral kernels, so every float
# step must round on its own: no compiler-fused multiply-adds.
target_compile_options(inference_engine INTERFACE -ffp-contract=off)

add_executable(gpt_neo_cli tools/gpt_neo_cli.cpp)
target_link_libraries(gpt_neo_cli PRIVATE inference_engine)
set_target_properties(gpt_neo_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Host tools for coral/: golden outputs for the kernel tests and the packed
# model the firmware runs. coral/tools/test.sh builds and runs them.
foreach(tool gen_golden pack_model)
    add_executable(${tool} coral/host/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE inference_engine)
    target_include_directories(${tool} PRIVATE coral/include coral/host)
    set_target_properties(${tool} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests/cpp)
endif()
