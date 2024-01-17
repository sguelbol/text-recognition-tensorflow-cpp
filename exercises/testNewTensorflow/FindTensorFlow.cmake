# Locates the tensorFlow library and include directories.

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow)

find_library(TensorFlow_LIBRARY NAMES tensorflow_cc
        HINTS
        /usr/lib
        /usr/local/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow2 DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)




## find dependencies
#find_package(Protobuf REQUIRED)
#
##include directories
#find_path(INCLUDE_DIR "~/.local/lib/python3.10/site-packages/tensorflow/include/tensorflow" PATH_SUFFIXES tensorflow2)
#list(APPEND INCLUDE_DIRS ${INCLUDE_DIR})
#if(INCLUDE_DIR)
#    list(APPEND INCLUDE_DIRS ${INCLUDE_DIR}/src)
#endif()
#
## find libraries
#find_library(LIBRARY "~/.local/lib/python3.10/site-packages/tensorflow/libtensorflow_cc.so.2" PATH_SUFFIXES tensorflow2)
#find_library(LIBRARY_FRAMEWORK "~/.local/lib/python3.10/site-packages/tensorflow/libtensorflow_framework.so.2" PATH_SUFFIXES tensorflow2)
#
## handle the QUIETLY and REQUIRED arguments and set *_FOUND
#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(TensorFlow2 DEFAULT_MSG INCLUDE_DIRS LIBRARY)
#mark_as_advanced(INCLUDE_DIRS LIBRARY LIBRARY_FRAMEWORK)
#
## set INCLUDE_DIRS and LIBRARIES
#if(TensorFlow2_FOUND)
#    set(TensorFlow2_INCLUDE_DIRS ${INCLUDE_DIRS})
#    if(LIBRARY_FRAMEWORK)
#        set(TensorFlow2_LIBRARIES ${LIBRARY} ${LIBRARY_FRAMEWORK} ${Protobuf_LIBRARY})
#    else()
#        set(TensorFlow2_LIBRARIES ${LIBRARY} ${Protobuf_LIBRARY})
#    endif()
#endif()