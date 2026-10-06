cmake_minimum_required(VERSION 3.14)
project(MiniTenCpp VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(minitencpp
    src/tensor.cpp
    #future files
)

target_include_directories(minitencpp PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(minitencpp_run src/main.cpp)
target_link_libraries(minitencpp_run PRIVATE minitencpp)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(minitencpp_tests tests/test_tensor.cpp)
target_link_libraries(minitencpp_tests PRIVATE minitencpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(minitencpp_tests)
