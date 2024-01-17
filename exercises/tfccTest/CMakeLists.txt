cmake_minimum_required(VERSION 3.22.1)
project(tfccTest)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorflowCC REQUIRED)

add_executable(tfccTest main.cpp)

target_link_libraries(tfccTest TensorflowCC::TensorflowCC)
