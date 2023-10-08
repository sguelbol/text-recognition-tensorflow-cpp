cmake_minimum_required(VERSION 3.26)
project(tensorsWithTensorflow)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorFlow REQUIRED)

add_executable(tensorsWithTensorflow main.cpp)

target_include_directories(tensorsWithTensorflow PRIVATE ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(tensorsWithTensorflow ${TensorFlow_LIBRARIES})
