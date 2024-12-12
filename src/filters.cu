#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <filesystem> // C++17 for creating directories
#include <omp.h>
#include "include/filters.h"


//Sequential filter implementation
cv::Mat extractFilterResponsesSequential(const cv::Mat& image, const std::vector<cv::Mat>& filterBank) {
    // Convert image to Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    // Create a vector to store filter responses for each pixel
    std::vector<cv::Mat> filterResponses;

    cv::Mat L_channel, a_channel, b_channel;

    // Iterate over filters and channels
    for (const auto& filter : filterBank) {
        cv::Mat response_L, response_a, response_b;

        L_channel = labChannels[0];
        a_channel = labChannels[1];
        b_channel = labChannels[2];

        // Apply filter to each channel
        cv::filter2D(L_channel, response_L, CV_32F, filter);
        cv::filter2D(a_channel, response_a, CV_32F, filter);
        cv::filter2D(b_channel, response_b, CV_32F, filter);

        // Concatenate responses for the current pixel into a single row
        std::vector<float> pixelResponse = {response_L.at<float>(0, 0), response_a.at<float>(0, 0), response_b.at<float>(0, 0)};
        cv::Mat pixelResponseMat = cv::Mat(1, pixelResponse.size(), CV_64FC1, &pixelResponse[0]); // Ensure 64FC1 data type
        filterResponses.push_back(pixelResponseMat);
    }

    // Concatenate all filter responses along the vertical dimension
    cv::Mat allResponses;
    cv::vconcat(filterResponses, allResponses);

    return allResponses;
}

// Parallelized filter response extraction using OpenMP
void extractFilterResponsesOpenMP(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Create output directory if it doesn't exist
    std::__fs::filesystem::create_directories(outputPath);

    // Convert image to CIE Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    // Parallelize filter application
    #pragma omp parallel for
    for (int filterIdx = 0; filterIdx < filterBank.size(); ++filterIdx) {
        const auto& filter = filterBank[filterIdx];

        try {
            // Process each channel
            cv::Mat response_L, response_a, response_b;

            cv::filter2D(L_channel, response_L, CV_32F, filter);
            saveFilterResponseImage(response_L, outputPath, filterIdx + 1, "L");

            cv::filter2D(a_channel, response_a, CV_32F, filter);
            saveFilterResponseImage(response_a, outputPath, filterIdx + 1, "a");

            cv::filter2D(b_channel, response_b, CV_32F, filter);
            saveFilterResponseImage(response_b, outputPath, filterIdx + 1, "b");
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error processing filter " << filterIdx + 1 << ": " << e.what() << std::endl;
            }
        }
    }
}

__global__ void applyFilterKernel(const float* image, const float* filter, float* output, int width, int height, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfFilter = filterWidth / 2;
    float value = 0.0f;

    for (int fy = -halfFilter; fy <= halfFilter; ++fy) {
        for (int fx = -halfFilter; fx <= halfFilter; ++fx) {
            int ix = min(max(x + fx, 0), width - 1);
            int iy = min(max(y + fy, 0), height - 1);
            float imageVal = image[iy * width + ix];
            float filterVal = filter[(fy + halfFilter) * filterWidth + (fx + halfFilter)];
            value += imageVal * filterVal;
        }
    }

    output[y * width + x] = value;
}

void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses) {
    int width = channel.cols;
    int height = channel.rows;

    // Allocate device memory for the input channel
    float* d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(float));
    cudaMemcpy(d_image, channel.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Preallocate memory for filter and output
    float* d_filter, *d_output;
    int maxFilterSize = 0;
    for (const auto& filter : filterBank) maxFilterSize = std::max(maxFilterSize, filter.rows);
    cudaMalloc((void**)&d_filter, maxFilterSize * maxFilterSize * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    for (const auto& filter : filterBank) {
        int filterWidth = filter.cols;

        cudaMemcpy(d_filter, filter.ptr<float>(), filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        applyFilterKernel<<<gridSize, blockSize>>>(d_image, d_filter, d_output, width, height, filterWidth);

        // Copy the output back to the host
        cv::Mat response(height, width, CV_32F);
        cudaMemcpy(response.ptr<float>(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        responses.push_back(response);        
    }

    // Free device memory for the input channel
    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
}

void extractAndSaveFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Convert image to Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    std::vector<std::vector<cv::Mat>> channelResponses(3);

    for (int i = 0; i < 3; ++i) {
        applyFiltersCUDA(labChannels[i], filterBank, channelResponses[i]);
    }

    // Save the responses
    for (int i = 0; i < filterBank.size(); ++i) {
        saveFilterResponseImage(channelResponses[0][i], outputPath, i + 1, "L");
        saveFilterResponseImage(channelResponses[1][i], outputPath, i + 1, "a");
        saveFilterResponseImage(channelResponses[2][i], outputPath, i + 1, "b");
    }
}

void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath, const std::string& method) {
    if (method == "sequential") {
        extractFilterResponsesSequential(image, filterBank, outputPath);
    } else if (method == "openmp") {
        extractFilterResponsesOpenMP(image, filterBank, outputPath);
    } else if (method == "cuda") {
        extractFilterResponsesCUDA(image, filterBank, outputPath);
    } else {
        throw std::invalid_argument("Unknown method: " + method);
    }
}



void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel) {
    // Normalize response for better visualization
    cv::Mat normalizedResponse;
    cv::normalize(response, normalizedResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Construct file name
    std::string filename = outputPath + "/filter_response_" + std::to_string(filterIdx) + "_" + channel + ".jpg";

    // Save the image
    if (!cv::imwrite(filename, normalizedResponse)) {
        throw std::runtime_error("Failed to save filter response image: " + filename);
    }

    // std::cout << "Saved filter response: " << filename << std::endl;
}

