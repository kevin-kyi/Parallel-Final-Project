#ifndef RUN_GMM_H
#define RUN_GMM_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "create_dictionary.h"


double gaussianDensity(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& covariance);

void trainGMM(const cv::Mat& data, int K, int maxIter, double tol, cv::Mat& means, std::vector<cv::Mat>& covariances, cv::Mat& weights);

#endif 