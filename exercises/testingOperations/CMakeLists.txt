cmake_minimum_required(VERSION 3.26)
project(perceptronModelTensorflow)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorFlow REQUIRED)

add_executable(perceptronModelTensorflow main.cpp)

target_include_directories(perceptronModelTensorflow PRIVATE ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(perceptronModelTensorflow ${TensorFlow_LIBRARIES})
