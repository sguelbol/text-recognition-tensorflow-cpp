cmake_minimum_required(VERSION 3.26)
project(PerceptronModelWithTensors)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorFlow REQUIRED)

add_executable(PerceptronModelWithTensors main.cpp
)

target_include_directories(PerceptronModelWithTensors PRIVATE ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(PerceptronModelWithTensors ${TensorFlow_LIBRARIES})