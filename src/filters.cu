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
    extern __shared__ float sharedMemory[]; // Shared memory buffer for the image tile
    float* sharedImage = sharedMemory;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int halfFilter = filterWidth / 2;

    // Compute shared memory dimensions (including padding for boundaries)
    int sharedWidth = blockDim.x + 2 * halfFilter;
    int sharedHeight = blockDim.y + 2 * halfFilter;

    // Position in shared memory
    int sharedX = tx + halfFilter;
    int sharedY = ty + halfFilter;

    // Load image data into shared memory with boundary handling
    if (x < width && y < height) {
        sharedImage[sharedY * sharedWidth + sharedX] = image[y * width + x];
    } else {
        // Handle out-of-bounds replication
        int clampedX = max(0, min(x, width - 1));
        int clampedY = max(0, min(y, height - 1));
        sharedImage[sharedY * sharedWidth + sharedX] = image[clampedY * width + clampedX];
    }

    // Handle boundary padding (replication)
    if (tx < halfFilter) {
        sharedImage[sharedY * sharedWidth + (sharedX - halfFilter)] = (x >= halfFilter) ? image[y * width + (x - halfFilter)] : image[y * width];
        sharedImage[sharedY * sharedWidth + (sharedX + blockDim.x)] = (x + blockDim.x < width) ? image[y * width + (x + blockDim.x)] : image[y * width + (width - 1)];
    }
    if (ty < halfFilter) {
        sharedImage[(sharedY - halfFilter) * sharedWidth + sharedX] = (y >= halfFilter) ? image[(y - halfFilter) * width + x] : image[x];
        sharedImage[(sharedY + blockDim.y) * sharedWidth + sharedX] = (y + blockDim.y < height) ? image[(y + blockDim.y) * width + x] : image[(height - 1) * width + x];
    }

    __syncthreads();

    // Perform convolution
    if (x < width && y < height) {
        float value = 0.0f;

        // Apply filter (flipped kernel)
        for (int fy = -halfFilter; fy <= halfFilter; ++fy) {
            for (int fx = -halfFilter; fx <= halfFilter; ++fx) {
                int imageIdx = (sharedY + fy) * sharedWidth + (sharedX + fx);
                int filterIdx = (halfFilter - fy) * filterWidth + (halfFilter - fx); // Flipping the filter
                value += sharedImage[imageIdx] * filter[filterIdx];
            }
        }

        output[y * width + x] = value;
    }
}

void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses) {
    if (channel.type() != CV_32F) {
        throw std::invalid_argument("Input channel must have type CV_32F");
    }

    int width = channel.cols;
    int height = channel.rows;

    // Allocate device memory
    float* d_image;
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_image, width * height * sizeof(float)));
    CHECK_CUDA_CALL(cudaMemcpy(d_image, channel.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice));

    float* d_filter;
    float* d_output;
    int maxFilterSize = 0;
    for (const auto& filter : filterBank) {
        maxFilterSize = std::max(maxFilterSize, filter.rows);
    }
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_filter, maxFilterSize * maxFilterSize * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_output, width * height * sizeof(float)));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    for (const auto& filter : filterBank) {
        int filterWidth = filter.cols;
        cv::Mat filter32F;
        if (filter.type() != CV_32F) {
            filter.convertTo(filter32F, CV_32F);
        } else {
            filter32F = filter;
        }

        CHECK_CUDA_CALL(cudaMemcpy(d_filter, filter32F.ptr<float>(), filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        size_t sharedMemSize = ((blockSize.x + 2 * (filterWidth / 2)) * (blockSize.y + 2 * (filterWidth / 2))) * sizeof(float);
        applyFilterKernel<<<gridSize, blockSize, sharedMemSize>>>(d_image, d_filter, d_output, width, height, filterWidth);
        CHECK_CUDA_CALL(cudaDeviceSynchronize());

        cv::Mat response(height, width, CV_32F);
        CHECK_CUDA_CALL(cudaMemcpy(response.ptr<float>(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));
        responses.push_back(response);
    }

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
}


void extractFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
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

    std::vector<std::vector<cv::Mat>> channelResponses(3);

    for (int i = 0; i < 3; ++i) {
        applyFiltersCUDA(labChannels[i], filterBank, channelResponses[i]);
    }

    for (int i = 0; i < filterBank.size(); ++i) {
        saveFilterResponseImage(channelResponses[0][i], outputPath, i + 1, "L");
        saveFilterResponseImage(channelResponses[1][i], outputPath, i + 1, "a");
        saveFilterResponseImage(channelResponses[2][i], outputPath, i + 1, "b");
    }
}