# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build

# Include any dependencies generated for this target.
include CMakeFiles/testNewTensorflow.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testNewTensorflow.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testNewTensorflow.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testNewTensorflow.dir/flags.make

CMakeFiles/testNewTensorflow.dir/main.cpp.o: CMakeFiles/testNewTensorflow.dir/flags.make
CMakeFiles/testNewTensorflow.dir/main.cpp.o: ../main.cpp
CMakeFiles/testNewTensorflow.dir/main.cpp.o: CMakeFiles/testNewTensorflow.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testNewTensorflow.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testNewTensorflow.dir/main.cpp.o -MF CMakeFiles/testNewTensorflow.dir/main.cpp.o.d -o CMakeFiles/testNewTensorflow.dir/main.cpp.o -c /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/main.cpp

CMakeFiles/testNewTensorflow.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testNewTensorflow.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/main.cpp > CMakeFiles/testNewTensorflow.dir/main.cpp.i

CMakeFiles/testNewTensorflow.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testNewTensorflow.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/main.cpp -o CMakeFiles/testNewTensorflow.dir/main.cpp.s

# Object files for target testNewTensorflow
testNewTensorflow_OBJECTS = \
"CMakeFiles/testNewTensorflow.dir/main.cpp.o"

# External object files for target testNewTensorflow
testNewTensorflow_EXTERNAL_OBJECTS =

testNewTensorflow: CMakeFiles/testNewTensorflow.dir/main.cpp.o
testNewTensorflow: CMakeFiles/testNewTensorflow.dir/build.make
testNewTensorflow: /usr/local/lib/libtensorflow_cc.so
testNewTensorflow: CMakeFiles/testNewTensorflow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testNewTensorflow"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testNewTensorflow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testNewTensorflow.dir/build: testNewTensorflow
.PHONY : CMakeFiles/testNewTensorflow.dir/build

CMakeFiles/testNewTensorflow.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testNewTensorflow.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testNewTensorflow.dir/clean

CMakeFiles/testNewTensorflow.dir/depend:
	cd /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build /home/sguelbol/CodeContext/text_recognition_eink/exercises/testNewTensorflow/build/CMakeFiles/testNewTensorflow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testNewTensorflow.dir/depend

