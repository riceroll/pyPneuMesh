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

# Utility rule file for doc.

# Include any custom commands dependencies for this target.
include doc/CMakeFiles/doc.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/CMakeFiles/doc.dir/progress.make

doc/CMakeFiles/doc:
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && doxygen
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && doxygen Doxyfile-unsupported
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E copy /Users/Roll/desktop/eigen-3.4.0/build/doc/html/group__TopicUnalignedArrayAssert.html /Users/Roll/desktop/eigen-3.4.0/build/doc/html/TopicUnalignedArrayAssert.html
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E rename html eigen-doc
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E remove eigen-doc/eigen-doc.tgz eigen-doc/unsupported/_formulas.log eigen-doc/_formulas.log
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E tar cfz eigen-doc.tgz eigen-doc
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E rename eigen-doc.tgz eigen-doc/eigen-doc.tgz
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && /opt/local/bin/cmake -E rename eigen-doc html

doc: doc/CMakeFiles/doc
doc: doc/CMakeFiles/doc.dir/build.make
.PHONY : doc

# Rule to build all files generated by this target.
doc/CMakeFiles/doc.dir/build: doc
.PHONY : doc/CMakeFiles/doc.dir/build

doc/CMakeFiles/doc.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc && $(CMAKE_COMMAND) -P CMakeFiles/doc.dir/cmake_clean.cmake
.PHONY : doc/CMakeFiles/doc.dir/clean

doc/CMakeFiles/doc.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/doc /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/doc /Users/Roll/desktop/eigen-3.4.0/build/doc/CMakeFiles/doc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/CMakeFiles/doc.dir/depend
