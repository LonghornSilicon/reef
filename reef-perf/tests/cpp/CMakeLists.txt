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

include(GoogleTest)
foreach(test test_inst_decode test_resource_pool)
    add_executable(${test} ${test}.cpp)
    target_link_libraries(${test} PRIVATE reef_perf_core GTest::gtest_main)
    gtest_discover_tests(${test})
endforeach()
