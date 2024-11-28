# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(ParallelFinalProject LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the target macOS version
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")

# Specify the macOS SDK
set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}            # OpenCV headers
    ${CMAKE_SOURCE_DIR}/src/include   # Local include directory
)

# Add compiler definitions and flags
add_definitions(-D__FLOAT16__=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-deprecated-declarations")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the main executable
add_executable(main src/main.cpp src/filters.cpp)
target_link_libraries(main ${OpenCV_LIBS})

# Add the test executable
add_executable(test_filters tests/test_filters.cpp src/filters.cpp)
target_link_libraries(test_filters ${OpenCV_LIBS})
