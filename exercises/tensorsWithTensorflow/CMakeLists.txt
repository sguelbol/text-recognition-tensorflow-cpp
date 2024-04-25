cmake_minimum_required(VERSION 3.26)
project(tensorsWithTensorflow)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorflowCC REQUIRED)

add_executable(tensorsWithTensorflow main.cpp)

target_link_libraries(tensorsWithTensorflow TensorflowCC::TensorflowCC)
