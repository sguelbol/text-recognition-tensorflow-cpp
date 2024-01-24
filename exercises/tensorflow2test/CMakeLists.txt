cmake_minimum_required(VERSION 3.27)
project(tensorflow2test)

set(CMAKE_CXX_STANDARD 14)

find_package(TensorflowCC REQUIRED)

add_executable(tensorflow2test main.cpp)

target_link_libraries(tensorflow2test TensorflowCC::TensorflowCC)