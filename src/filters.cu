#include "include/filters.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

#define CHECK_CUDA_CALL(call)                                                     \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA Error: ") +               \
                                     cudaGetErrorString(err) +                   \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                        \
    }

// CUDA Kernel
__global__ void applyFilterKernel(const float* image, const float* filter, float* output,
                                  int width, int height, int filterWidth) {
    extern __shared__ float sharedMem[]; // Shared memory
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int halfFilter = filterWidth / 2;
    int sharedWidth = blockDim.x + 2 * halfFilter;

    float* sharedTile = sharedMem;
    float* sharedFilter = &sharedMem[sharedWidth * (blockDim.y + 2 * halfFilter)];

    // Load filter into shared memory
    if (ty < filterWidth && tx < filterWidth) {
        sharedFilter[ty * filterWidth + tx] = filter[ty * filterWidth + tx];
    }

    // Load main tile and halo into shared memory
    int sharedX = tx + halfFilter;
    int sharedY = ty + halfFilter;

    // Main pixel
    if (x < width && y < height) {
        sharedTile[sharedY * sharedWidth + sharedX] = image[y * width + x];
    } else {
        sharedTile[sharedY * sharedWidth + sharedX] = 0.0f;
    }

    // Halo regions
    if (tx < halfFilter) {
        // Left halo
        sharedTile[sharedY * sharedWidth + (sharedX - halfFilter)] =
            (x - halfFilter >= 0) ? image[y * width + (x - halfFilter)] : 0.0f;
        // Right halo
        sharedTile[sharedY * sharedWidth + (sharedX + blockDim.x)] =
            (x + blockDim.x < width) ? image[y * width + (x + blockDim.x)] : 0.0f;
    }
    if (ty < halfFilter) {
        // Top halo
        sharedTile[(sharedY - halfFilter) * sharedWidth + sharedX] =
            (y - halfFilter >= 0) ? image[(y - halfFilter) * width + x] : 0.0f;
        // Bottom halo
        sharedTile[(sharedY + blockDim.y) * sharedWidth + sharedX] =
            (y + blockDim.y < height) ? image[(y + blockDim.y) * width + x] : 0.0f;
    }

    __syncthreads();

    // Perform convolution
    float value = 0.0f;
    if (x < width && y < height) {
        for (int fy = 0; fy < filterWidth; ++fy) {
            for (int fx = 0; fx < filterWidth; ++fx) {
                value += sharedTile[(sharedY - halfFilter + fy) * sharedWidth + (sharedX - halfFilter + fx)] *
                         sharedFilter[fy * filterWidth + fx];
            }
        }
        output[y * width + x] = value;
    }
}

// CUDA implementation for applying filters
void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses) {
    if (channel.type() != CV_32F) {
        throw std::invalid_argument("Input channel must have type CV_32F");
    }

    int width = channel.cols;
    int height = channel.rows;
    size_t imageSize = width * height * sizeof(float);

    // Allocate device memory for image and output
    float* d_image = nullptr;
    float* d_output = nullptr;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_image, imageSize));
    CHECK_CUDA_CALL(cudaMemcpy(d_image, channel.ptr<float>(), imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_output, imageSize));

    // Allocate device memory for all filters in the filter bank
    int maxFilterWidth = 0;
    size_t totalFilterMemory = 0;
    for (const auto& filter : filterBank) {
        maxFilterWidth = std::max(maxFilterWidth, filter.cols);
        totalFilterMemory += filter.cols * filter.rows;
    }
    float* d_allFilters = nullptr;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_allFilters, totalFilterMemory * sizeof(float)));

    // Copy all filters to device memory
    size_t offset = 0;
    std::vector<size_t> filterOffsets;
    for (const auto& filter : filterBank) {
        cv::Mat filter32F;
        if (filter.type() != CV_32F) {
            filter.convertTo(filter32F, CV_32F);
        } else {
            filter32F = filter;
        }

        size_t filterSize = filter32F.cols * filter32F.rows;
        CHECK_CUDA_CALL(cudaMemcpy(d_allFilters + offset, filter32F.ptr<float>(), filterSize * sizeof(float), cudaMemcpyHostToDevice));
        filterOffsets.push_back(offset);
        offset += filterSize;
    }

    // Kernel configuration
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Process each filter
    for (size_t i = 0; i < filterBank.size(); ++i) {
        int filterWidth = filterBank[i].cols;
        int filterHeight = filterBank[i].rows;

        size_t sharedMemSize = ((blockSize.x + 2 * (filterWidth / 2)) * (blockSize.y + 2 * (filterHeight / 2)) + filterWidth * filterHeight) * sizeof(float);

        applyFilterKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_image,
            d_allFilters + filterOffsets[i],
            d_output,
            width,
            height,
            filterWidth
        );

        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        // Copy the result back to the host
        cv::Mat response(height, width, CV_32F);
        CHECK_CUDA_CALL(cudaMemcpy(response.ptr<float>(), d_output, imageSize, cudaMemcpyDeviceToHost));
        responses.push_back(response);
    }

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_allFilters);
}

// The missing `extractFilterResponsesCUDA` function
std::vector<cv::Mat> extractFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }
    std::filesystem::create_directories(outputPath);

    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    for (int i = 0; i < 3; ++i) {
        labChannels[i].convertTo(labChannels[i], CV_32F);
    }

    int totalResponses = filterBank.size() * 3;
    std::vector<cv::Mat> filterResponses(totalResponses);

    for (int i = 0; i < 3; ++i) {
        const cv::Mat& channel = labChannels[i];
        std::vector<cv::Mat> channelResponses;
        applyFiltersCUDA(channel, filterBank, channelResponses);
        for (int filterIdx = 0; filterIdx < filterBank.size(); ++filterIdx) {
            filterResponses[filterIdx * 3 + i] = channelResponses[filterIdx];
        }
    }

    return filterResponses;
}
