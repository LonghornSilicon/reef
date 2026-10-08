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
# One directory per module, matching src/csrc/src/<module>/.
set(REEF_PERF_TESTS
    common/test_interfaces
    common/test_resource_pool
    frontend/test_inst_decode
    mem/test_mem_timing
    vector/test_vxu
)
foreach(path ${REEF_PERF_TESTS})
    get_filename_component(test ${path} NAME)
    add_executable(${test} ${path}.cpp)
    target_link_libraries(${test} PRIVATE reef_perf_core GTest::gtest_main)
    gtest_discover_tests(${test})
endforeach()
