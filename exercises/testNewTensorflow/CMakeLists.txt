cmake_minimum_required(VERSION 3.22)
project(testNewTensorflow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})


find_package(TensorFlow REQUIRED)

add_executable(testNewTensorflow main.cpp)

target_include_directories(testNewTensorflow PRIVATE ${TensorFlow_INCLUDE_DIRS})

target_link_libraries(testNewTensorflow ${TensorFlow_LIBRARIES})
target_include_directories(testNewTensorflow PRIVATE ${TENSORFLOW_INCLUDE_DIRS})
