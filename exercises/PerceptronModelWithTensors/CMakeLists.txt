cmake_minimum_required(VERSION 3.26)
project(PerceptronModelWithTensors)

set(CMAKE_CXX_STANDARD 17)


set(TENSORFLOW_DIR "/home/sguelbol/tensorflow")
set(EIGEN_DIR "/home/sguelbol/eigen-3.4.0/")
set(ABSL_DIR "/home/sguelbol/absl/abseil-cpp/")
add_executable(PerceptronModelWithTensors main.cpp)


target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/bazel-bin/tensorflow/include)
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/bazel-bin/)
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/tensorflow/)
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/tensorflow/cc)
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/tensorflow/core)
target_include_directories(PerceptronModelWithTensors PUBLIC ${EIGEN_DIR})
target_include_directories(PerceptronModelWithTensors PUBLIC ${ABSL_DIR})
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR})
target_include_directories(PerceptronModelWithTensors PUBLIC ${TENSORFLOW_DIR}/third_party/xla/third_party/tsl)
target_include_directories(PerceptronModelWithTensors PUBLIC /home/sguelbol/.local/lib/python3.11/site-packages/tensorflow/include/external/)
target_include_directories(PerceptronModelWithTensors PUBLIC /home/sguelbol/.local/lib/python3.11/site-packages/tensorflow/include/)
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow/)

target_link_libraries(PerceptronModelWithTensors PRIVATE ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2)
target_link_libraries(PerceptronModelWithTensors PRIVATE ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so.2)


#find_package(TensorFlow REQUIRED)


