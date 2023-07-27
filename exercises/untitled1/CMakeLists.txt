cmake_minimum_required(VERSION 3.25)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(TENSORFLOW_DIR "~/tensorflow")

find_package(Qt5 COMPONENTS
        Core
        Gui
        Widgets
        REQUIRED)

add_executable(untitled1 main.cpp)
target_link_libraries(untitled1
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_cc.so.2.12.0
        ${TENSORFLOW_DIR}/bazel-bin/tensorflow/libtensorflow_framework.so
        )

target_include_directories(untitled1 PUBLIC ${TENSORFLOW_DIR}/bazel-bin/tensorflow/include)
target_include_directories(untitled1 PUBLIC ${TENSORFLOW_DIR}/tensorflow/cc)
target_include_directories(untitled1 PUBLIC ${TENSORFLOW_DIR})
