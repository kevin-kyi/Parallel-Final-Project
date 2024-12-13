#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Sequential implementation
void extractFilterResponsesSequential(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// OpenMP-based implementation
void extractFilterResponsesOpenMP(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// CUDA-specific functions (declarations always visible, definitions guarded)
void extractFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

#ifdef __CUDACC__
__global__ void applyFilterKernel(const float* image, const float* filter, float* output, int width, int height, int filterWidth);
void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses);
#endif

// Wrapper function to select the method
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath, const std::string& method);

// Helper function to save filter response images
void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel);

#endif // FILTERS_H
