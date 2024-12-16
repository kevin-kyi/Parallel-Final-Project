#include "include/filters.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <omp.h>

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int halfFilter = filterWidth / 2;

    if (x < width && y < height) {
        float value = 0.0f;

        // Apply convolution 
        for (int fy = -halfFilter; fy <= halfFilter; ++fy) {
            for (int fx = -halfFilter; fx <= halfFilter; ++fx) {
                int imageX = min(max(x + fx, 0), width - 1); 
                int imageY = min(max(y + fy, 0), height - 1);

                int filterIdx = (halfFilter - fy) * filterWidth + (halfFilter - fx);
                value += image[imageY * width + imageX] * filter[filterIdx];
            }
        }

        output[y * width + x] = value;
    }
}

// CUDA Implementation for Filters
void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses) {
    if (channel.type() != CV_32F) {
        throw std::invalid_argument("Input channel must have type CV_32F");
    }

    int width = channel.cols;
    int height = channel.rows;
    size_t imageSize = width * height * sizeof(float);

    // Allocate device memory for the input image and output
    float* d_image;
    float* d_output;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_image, imageSize));
    CHECK_CUDA_CALL(cudaMemcpy(d_image, channel.ptr<float>(), imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_output, imageSize));

    // Determine the total filter size for batching
    size_t totalFilterMemory = 0;
    std::vector<int> filterSizes;
    std::vector<int> filterWidths;
    for (const auto& filter : filterBank) {
        totalFilterMemory += filter.total() * sizeof(float);
        filterSizes.push_back(filter.total());
        filterWidths.push_back(filter.cols);
    }

    // Allocate and copy all filters in a single transfer
    float* d_filters;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_filters, totalFilterMemory));

    size_t offset = 0;
    for (size_t i = 0; i < filterBank.size(); ++i) {
        cv::Mat filter32F;
        if (filterBank[i].type() != CV_32F) {
            filterBank[i].convertTo(filter32F, CV_32F);
        } else {
            filter32F = filterBank[i];
        }
        CHECK_CUDA_CALL(cudaMemcpy(d_filters + offset, filter32F.ptr<float>(), filterSizes[i] * sizeof(float), cudaMemcpyHostToDevice));
        offset += filterSizes[i];
    }

    // Kernel config
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Iterate over the filters
    offset = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < filterBank.size(); ++i) {
        int filterWidth = filterWidths[i];
        int filterSize = filterSizes[i];

        // Launch kernel
        size_t sharedMemSize = ((blockSize.x + 2 * (filterWidth / 2)) * (blockSize.y + 2 * (filterWidth / 2))) * sizeof(float);
        applyFilterKernel<<<gridSize, blockSize, sharedMemSize>>>(
            d_image,
            d_filters + offset,
            d_output,
            width,
            height,
            filterWidth
        );
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        // Copy the result back to host memory
        cv::Mat response(height, width, CV_32F);
        CHECK_CUDA_CALL(cudaMemcpy(response.ptr<float>(), d_output, imageSize, cudaMemcpyDeviceToHost));
        responses.push_back(response);

        offset += filterSize;
    }

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_filters);
}

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
            std::string channelName = (i == 0) ? "L" : (i == 1) ? "a" : "b";
            // saveFilterResponseImage(channelResponses[filterIdx], outputPath, filterIdx + 1, channelName);
            filterResponses[filterIdx * 3 + i] = channelResponses[filterIdx];
        }
    }

    return filterResponses;
}
