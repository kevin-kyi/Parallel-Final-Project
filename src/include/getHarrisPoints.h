#ifndef GET_HARRIS_POINTS_H
#define GET_HARRIS_POINTS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Function declarations
std::vector<cv::Point> getHarrisPoints(const cv::Mat& image, int alpha, double k);
void saveHarrisPointsImage(const cv::Mat& image, const std::vector<cv::Point>& points, const std::string& outputPath);

#endif