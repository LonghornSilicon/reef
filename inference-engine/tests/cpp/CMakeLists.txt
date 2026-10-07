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

# TODO: Add focused tests for attention, embeddings, and MLP as implemented.



# Embeddings
add_executable(test_embedding_cpp test_embedding.cpp)
target_link_libraries(test_embedding_cpp PRIVATE inference_engine GTest::gtest_main)
gtest_discover_tests(test_embedding_cpp)