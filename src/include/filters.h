// #ifndef FILTERS_H
// #define FILTERS_H

// #include <vector>
// #include <opencv2/opencv.hpp>

// // Function to extract filter responses for an image
// // Input: image in cv::Mat format
// // Output: A vector of feature vectors for each pixel
// std::vector<std::vector<float>> extractFilterResponses(const cv::Mat& image);

// #endif // FILTERS_H

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Function to apply filters and save responses as images
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);

// Helper function to save individual filter response images
void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel);

#endif
