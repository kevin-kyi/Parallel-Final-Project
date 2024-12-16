cmake_minimum_required(VERSION 3.18)
project(ParallelFinalProject LANGUAGES CXX CUDA)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust for your GPU architecture

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and try again.")
endif()

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Find YAML-CPP
find_package(yaml-cpp REQUIRED)
if (yaml-cpp_FOUND)
    message(STATUS "YAML-CPP found at ${YAML_CPP_LIBRARIES}")
else()
    message(FATAL_ERROR "YAML-CPP not found. Please install YAML-CPP and try again.")
endif()

# Source files
set(CPP_SOURCES
    src/filters.cpp
    src/createFilterBank.cpp
    src/batchToVisualWords.cpp
    src/create_dictionary.cpp
    src/getHarrisPoints.cpp
    src/create_word_maps.cpp
    src/dbscan.cpp
    src/classification.cpp
    src/save_dictionary_yml.cpp
    src/getVisualWords.cpp
)

set(CUDA_SOURCES
    src/filters.cu
)

# Compile filters.cpp into a static library
add_library(filters_cpu STATIC ${CPP_SOURCES})
target_include_directories(filters_cpu PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(filters_cpu PRIVATE ${OpenCV_LIBS} yaml-cpp OpenMP::OpenMP_CXX)

# Compile filters.cu into a static library
add_library(filters_cuda STATIC ${CUDA_SOURCES})
target_include_directories(filters_cuda PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(filters_cuda PRIVATE ${OpenCV_LIBS} yaml-cpp OpenMP::OpenMP_CXX)

# Create the final executable
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE filters_cpu filters_cuda yaml-cpp)

# Debugging and verbose build output
# set(CMAKE_VERBOSE_MAKEFILE ON)