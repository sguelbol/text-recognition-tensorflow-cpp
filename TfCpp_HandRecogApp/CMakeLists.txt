cmake_minimum_required(VERSION 3.27)
project(TfCpp_HandRecogApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(TensorflowCC REQUIRED)
find_package(OpenCV REQUIRED)
set(Qt6_DIR "~/Qt/6.5.3/gcc_64/lib/cmake/Qt6")
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OPTIONAL_COMPONENTS PrintSupport)

include_directories(headers)

set(SOURCES sources/Model.cpp
        sources/DenseLayer.cpp
        sources/GraphLogger.cpp
        sources/MNISTReader.cpp
        sources/Helper.cpp
)

add_executable(TfCpp_HandRecogApp main.cpp ${SOURCES}
        enum/ActivationFunction.h headers/ScribbleArea.h headers/MainWindow.h sources/MainWindow.cpp
        sources/ScribbleArea.cpp
        sources/Optimizer.cpp
        headers/Optimizer.h)

target_link_libraries(TfCpp_HandRecogApp TensorflowCC::TensorflowCC ${OpenCV_LIBS} Qt6::Core Qt6::Gui Qt6::Widgets Qt6::PrintSupport)

include(FetchContent)
FetchContent_Declare(
        googletest
        # Specify the commit you depend on and update it regularly.
        URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()
add_executable(run_opt_test tests/model-test.cpp ${SOURCES})

target_link_libraries(run_opt_test GTest::gtest_main TensorflowCC::TensorflowCC ${OpenCV_LIBS})

include(GoogleTest)
gtest_discover_tests(run_opt_test)

