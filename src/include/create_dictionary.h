#ifndef CREATE_DICTIONARY_H
#define CREATE_DICTIONARY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


#include "getHarrisPoints.h"
#include "filters.h"
#include "visual_words.h"



// Function to create a visual word dictionary
cv::Mat get_kmeans_dictionary(const std::vector<std::string>& imgPaths, int alpha, int K, const std::string& method);

cv::Mat get_gmm_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method);

// Function to save the dictionary to a file
void save_dictionary(const cv::Mat& dictionary, const std::string& filename);

#endif
