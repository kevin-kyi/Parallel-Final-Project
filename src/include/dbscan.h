#ifndef DBSCAN_H
#define DBSCAN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <omp.h>

/*
 * Sequential Version of DBSCAN
 * Parameters:
 *   - data: Input data matrix (cv::Mat), rows are points, cols are dimensions
 *   - eps: Radius parameter for neighborhood search
 *   - minSamples: Minimum points to form a cluster
 * Returns:
 *   - std::vector<int>: Cluster labels for each data point
 */
std::vector<int> dbscan_Sequential(const cv::Mat& data, double eps, int minSamples);

/*
 * OpenMP Version of DBSCAN
 * Parameters:
 *   - data: Input data matrix (cv::Mat), rows are points, cols are dimensions
 *   - eps: Radius parameter for neighborhood search
 *   - minSamples: Minimum points to form a cluster
 * Returns:
 *   - std::vector<int>: Cluster labels for each data point
 */
std::vector<int> dbscan_OpenMP(const cv::Mat& data, double eps, int minSamples);

#endif 
