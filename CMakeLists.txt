cmake_minimum_required(VERSION 3.10)
project(ParallelFinalProject)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Specify the executable and source files
add_executable(main main.cpp filters.cpp)

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})
