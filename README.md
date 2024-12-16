# Parallel Final Project

This repository contains the implementation of a parallelized scene recognition system using C++ and CUDA. Below are instructions for compiling and running the code on different systems.

---

## Prerequisites

Ensure you have the following dependencies installed:

1. **CMake (>= 3.18)**
2. **CUDA Toolkit (>= 11.0)**  
   Make sure your GPU supports architecture 7.5 (or adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU).
3. **OpenCV (>= 4.0)**  
   Install OpenCV development libraries (`libopencv-dev`).
4. **YAML-CPP**  
   Install `yaml-cpp` library.
5. **OpenMP**  
   Typically bundled with your compiler (e.g., `gcc` or `clang`).

---

## Compilation

### Using CMake

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a build directory**:
   ```bash
   mkdir build && cd build
   ```

3. **Generate the Makefiles with CMake**:
   ```bash
   cmake ..
   ```

4. **Build the project**:
   ```bash
   make
   ```

5. **Run the program**:
   ```bash
   ./main
   ```

### Command-Line Compilation Option

If you prefer compiling without CMake, you can use the following commands directly, assuming you have the required libraries and include paths configured:

#### 1. Compile CPU Code
```bash
g++ -std=c++17 -fopenmp -I<opencv_include_dir> -I<yaml-cpp_include_dir> \
    src/filters.cpp src/createFilterBank.cpp src/batchToVisualWords.cpp \
    src/create_dictionary.cpp src/getHarrisPoints.cpp src/create_word_maps.cpp \
    src/dbscan.cpp src/classification.cpp src/save_dictionary_yml.cpp \
    src/getVisualWords.cpp -c -o filters_cpu.o
```

#### 2. Compile CUDA Code
```bash
nvcc -std=c++17 -arch=sm_75 -I<opencv_include_dir> -I<yaml-cpp_include_dir> \
    src/filters.cu -c -o filters_cuda.o
```

#### 3. Link Everything
```bash
g++ -std=c++17 -fopenmp filters_cpu.o filters_cuda.o src/main.cpp \
    -L<opencv_library_dir> -L<yaml-cpp_library_dir> \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
    -lyaml-cpp -lcudart -o main
```

Replace `<opencv_include_dir>`, `<yaml-cpp_include_dir>`, `<opencv_library_dir>`, and `<yaml-cpp_library_dir>` with the paths to your respective libraries and include directories.

#### 4. Run the Program
```bash
./main
```

---

## Notes

1. **Test Different Filter Implementations**: There is a commented chunk of code in main.cpp that automatically tests each filter implementation on the dataset and logs the data in the results directory.
2. **Results of filtering**: If you want to see the results of the filtering, go to filters.cpp and filters.cu and uncomment the calls to saveFilterResponseImage(). 
