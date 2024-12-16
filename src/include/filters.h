#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>



cv::Mat extractFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank);


// Sequential implementation
std::vector<cv::Mat> extractFilterResponsesSequential(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// OpenMP implementation
cv::Mat extractFilterResponsesOpenMP(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// CUDA implementation
std::vector<cv::Mat> extractFilterResponsesCUDA(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

#ifdef __CUDACC__
// CUDA Kernel for applying a single filter
__global__ void applyFilterKernel(const float* image, const float* filter, float* output, 
                                  int width, int height, int filterWidth);

// CUDA function for applying a set of filters
void applyFiltersCUDA(const cv::Mat& channel, const std::vector<cv::Mat>& filterBank, std::vector<cv::Mat>& responses);
#endif

// Wrapper function to select implementation
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath, const std::string& method);

// Helper function to save filter response images
void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel);

#endif // FILTERS_H