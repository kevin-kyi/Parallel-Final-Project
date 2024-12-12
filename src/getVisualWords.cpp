#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "include/visual_words.h"
#include "include/filters.h"

cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank) {
    // Extract filter responses
    cv::Mat filterResponses = extractFilterResponses(image, filterBank);

    // Reshape filter responses to a 2D matrix where each row is a feature vector
    int numPixels = filterResponses.rows * filterResponses.cols;
    int numChannels = filterResponses.channels();
    filterResponses = filterResponses.reshape(numPixels, numChannels);

    // Ensure data type consistency
    filterResponses.convertTo(filterResponses, dictionary.type());

    // Compute distances to dictionary entries
    std::vector<float> distances;
    for (int i = 0; i < numPixels; ++i) {
        float minDist = std::numeric_limits<float>::max();
        int minIndex = -1;
        for (int j = 0; j < dictionary.rows; ++j) {
            float dist = cv::norm(filterResponses.row(i), dictionary.row(j), cv::NORM_L2);
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }
        distances.push_back(minIndex);
    }

    // Reshape the distance map
    cv::Mat wordMap = cv::Mat(image.rows, image.cols, CV_32SC1, distances.data());

    return wordMap;
}