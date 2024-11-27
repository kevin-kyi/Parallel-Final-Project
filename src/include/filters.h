#ifndef FILTERS_H
#define FILTERS_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<std::vector<float>> extractFeatures(const cv::Mat& image);

#endif
