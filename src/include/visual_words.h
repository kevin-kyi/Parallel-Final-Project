#ifndef VISUAL_WORDS_H
#define VISUAL_WORDS_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "filters.h"

cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank);

#endif