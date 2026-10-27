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

add_executable(test_tensor_cpp test_tensor.cpp)
target_link_libraries(test_tensor_cpp PRIVATE inference_engine GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_tensor_cpp)

# Embeddings
add_executable(test_embedding_cpp test_embedding.cpp)
target_link_libraries(test_embedding_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_embedding_cpp)

# Attention
add_executable(test_attention_cpp test_attention.cpp)
target_link_libraries(test_attention_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_attention_cpp)

# Linear
add_executable(test_linear_cpp test_linear.cpp)
target_link_libraries(test_linear_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_linear_cpp)

# MLP
add_executable(test_mlp_cpp test_mlp.cpp)
target_link_libraries(test_mlp_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_mlp_cpp)

# LayerNorm
add_executable(test_layernorm_cpp test_layernorm.cpp)
target_link_libraries(test_layernorm_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_layernorm_cpp)

# Reproducible exp and tanh
add_executable(test_math_cpp test_math.cpp)
target_link_libraries(test_math_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_math_cpp)

# W8A8 quantization
add_executable(test_quant_cpp test_quant.cpp)
target_link_libraries(test_quant_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_quant_cpp)

# GPT-Neo model, weights and checkpoint loading
add_executable(test_gpt_neo_cpp test_gpt_neo.cpp)
target_link_libraries(test_gpt_neo_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_gpt_neo_cpp)

# Integer-only (int8) operators
add_executable(test_int8_cpp test_int8.cpp)
target_link_libraries(test_int8_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_int8_cpp)
