#ifndef FILTER_BANK_H
#define FILTER_BANK_H

#include <opencv2/opencv.hpp>   // Core OpenCV functions and data structures
#include <iostream>            // For console I/O (e.g., std::cout)
#include <vector>              // For std::vector

// Function declaration for creating a filter bank
std::vector<cv::Mat> createFilterBank();

#endif // FILTER_BANK_H
