#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <opencv2/opencv.hpp>   // Core OpenCV functions and data structures
#include <vector>              // For std::vector
#include <string>              // For std::string

// Function to get keypoints using Harris or Random methods
std::vector<cv::Point> getKeypoints(const cv::Mat& image, int alpha, const std::string& method);

// Function to build a feature matrix from keypoints and filter responses
cv::Mat buildFeatureMatrix(const std::vector<cv::Point>& points, const std::string& responseDir, int numFilters);

#endif // FEATURE_EXTRACTION_H
