cmake_minimum_required(VERSION 3.26)
project(PerceptronModelWithTensors)

set(CMAKE_CXX_STANDARD 17)


find_package(TensorflowCC REQUIRED)

add_executable(PerceptronModelWithTensors main.cpp)

target_link_libraries(PerceptronModelWithTensors PRIVATE TensorflowCC::TensorflowCC)


