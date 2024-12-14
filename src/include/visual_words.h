#ifndef VISUAL_WORDS_H
#define VISUAL_WORDS_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "filters.h"

cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank);

cv::Mat getVisualWordsGMM(const cv::Mat& image, const cv::Mat& means, 
                          const std::vector<cv::Mat>& covariances, const cv::Mat& weights);

#endif