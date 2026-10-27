cmake_minimum_required(VERSION 3.20)
project(inference_engine LANGUAGES CXX)

# C++17 is a conservative host baseline while the Coral target toolchain is
# unconfirmed. Revisit the standard when that toolchain is pinned.
# Every operator is a template, so the engine may be header-only: consumers get
# the includes and the standard, and nothing is compiled into a library.
add_library(inference_engine INTERFACE)
target_compile_features(inference_engine INTERFACE cxx_std_17)
target_include_directories(inference_engine INTERFACE include)
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
    include(FetchContent)
    set(_gtest_extra "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(_gtest_extra DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/063de7e9578f82b369302001269680b4b1553359.zip
        ${_gtest_extra}
    )
    set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Install Google Test" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Every Google Test file links into one binary; CTest still lists each
    # test case on its own.
    add_executable(engine_tests
        tests/test_attention.cpp
        tests/test_embedding.cpp
        tests/test_gpt_neo.cpp
        tests/test_int8.cpp
        tests/test_layernorm.cpp
        tests/test_linear.cpp
        tests/test_math.cpp
        tests/test_mlp.cpp
        tests/test_quant.cpp
        tests/test_tensor.cpp
    )
    target_link_libraries(engine_tests PRIVATE inference_engine GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(engine_tests)
endif()
