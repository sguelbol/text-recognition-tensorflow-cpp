cmake_minimum_required(VERSION 3.25)
project(testtf)

set(CMAKE_CXX_STANDARD 17)

add_executable(testtf main.cpp)

set(TENSORFLOW_DIR "~/tensorflow")

target_include_directories(testtf PUBLIC ${TENSORFLOW_DIR})

# Create a custom target to execute the install_headers.genrule_scripts.sh script
target_link_libraries(testtf PRIVATE ~/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so)




