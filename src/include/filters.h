#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Sequential implementation of filter response extraction
cv::Mat extractFilterResponsesSequential(const cv::Mat& image, const std::vector<cv::Mat>& filterBank);

// OpenMP-based parallel implementation of filter response extraction
void extractFilterResponsesOpenMP(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// CUDA-based parallel implementation of filter response extraction
void extractFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// Wrapper function to handle different methods (sequential, OpenMP, CUDA)
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath, const std::string& method);

// Function to save filter response images
void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel);

// CUDA kernel for applying filters
__global__ void applyFilterKernel(const float* image, const float* filter, float* output, int width, int height, int filterWidth);

// Function to apply filters on a single channel using CUDA
void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses);

#endif // FILTERS_H
