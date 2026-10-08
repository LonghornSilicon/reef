include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/063de7e9578f82b369302001269680b4b1553359.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
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
# test_attention.cpp cannot link into a runnable test until the MLP team's
# linear() lands (the last undefined symbol in the attention pipeline). The
# object target keeps it compiling in the meantime and gives clang-tidy a
# compile command; swap it for the executable target when linear() lands.
add_library(test_attention_lint OBJECT test_attention.cpp)
target_link_libraries(test_attention_lint PRIVATE inference_engine GTest::gtest_main)

# MLP
add_executable(test_mlp_cpp test_mlp.cpp)
target_link_libraries(test_mlp_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_mlp_cpp)
