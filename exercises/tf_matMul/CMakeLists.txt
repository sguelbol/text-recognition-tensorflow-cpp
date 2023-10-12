cmake_minimum_required(VERSION 3.26)
project(tf_matMul)

set(CMAKE_CXX_STANDARD 17)

find_package(TensorFlow REQUIRED)

add_executable(tf_matMul main.cpp)

target_include_directories(tf_matMul PRIVATE ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(tf_matMul ${TensorFlow_LIBRARIES})