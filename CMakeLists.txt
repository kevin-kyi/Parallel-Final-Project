cmake_minimum_required(VERSION 3.10)
project(ParallelFinalProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify the executable and source files
add_executable(main src/main.cpp src/filters.cpp src/createFilterBank.cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Deployment target and compatibility settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.15")
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")

# Ensure compatibility with the macOS SDK
set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

# Suppress `_Float16` warnings if unnecessary
add_definitions(-D__NO_FLOAT16__)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES")

# Link libraries in a unified command
target_link_libraries(main PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
