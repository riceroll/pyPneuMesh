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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Roll/desktop/eigen-3.4.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/desktop/eigen-3.4.0/build

# Utility rule file for lapack.

# Include any custom commands dependencies for this target.
include lapack/CMakeFiles/lapack.dir/compiler_depend.make

# Include the progress variables for this target.
include lapack/CMakeFiles/lapack.dir/progress.make

lapack: lapack/CMakeFiles/lapack.dir/build.make
.PHONY : lapack

# Rule to build all files generated by this target.
lapack/CMakeFiles/lapack.dir/build: lapack
.PHONY : lapack/CMakeFiles/lapack.dir/build

lapack/CMakeFiles/lapack.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/lapack && $(CMAKE_COMMAND) -P CMakeFiles/lapack.dir/cmake_clean.cmake
.PHONY : lapack/CMakeFiles/lapack.dir/clean

lapack/CMakeFiles/lapack.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/lapack /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/lapack /Users/Roll/desktop/eigen-3.4.0/build/lapack/CMakeFiles/lapack.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lapack/CMakeFiles/lapack.dir/depend
