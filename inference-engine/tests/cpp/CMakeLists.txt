include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/063de7e9578f82b369302001269680b4b1553359.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Install Google Test" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_matrix_cpp test_matrix.cpp)
target_link_libraries(test_matrix_cpp PRIVATE inference_engine GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_matrix_cpp)

# TODO: Add focused tests for attention, embeddings, and MLP as implemented.
