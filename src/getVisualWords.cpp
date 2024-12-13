#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "include/visual_words.h"
#include "include/filters.h"




cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank) {
    // Extract filter responses
    cv::Mat filterResponses = extractFilterResponses(image, filterBank);

    // // Reshape filter responses to [numPixels x numChannels]
    int numPixels = filterResponses.rows * filterResponses.cols; // Total number of pixels
    int numChannels = filterResponses.channels();               // Number of channels (filters * color channels)

    // filterResponses = filterResponses.reshape(numChannels, numPixels); // Reshape to [numPixels x numChannels]

    // Debug output for dimensions
    std::cout << "Extracted Filter Responses: Rows=" << filterResponses.rows
              << ", Cols=" << filterResponses.cols
              << ", Channels=" << numChannels << std::endl;

    std::cout << "DICTIONARY: Rows=" << dictionary.rows
              << ", Cols=" << dictionary.cols
              << ", Channels=" << dictionary.channels() << std::endl;
    

    if (filterResponses.cols != dictionary.cols) {
        throw std::runtime_error("Feature dimensions mismatch between filter responses and dictionary.");
    }

    // Compute distances between filter responses and dictionary entries
    cv::Mat distances;
    cv::batchDistance(filterResponses, dictionary, distances, CV_32F, cv::noArray(), cv::NORM_L2);

    std::cout << "Distances: Rows=" << distances.rows << ", Cols=" << distances.cols << std::endl;


    // Map each pixel to its closest visual word
    cv::Mat wordMap(image.rows, image.cols, CV_32SC1);
    // for (int i = 0; i < numPixels; ++i) {
    //     double minVal;
    //     cv::Point minIdx;
    //     cv::minMaxLoc(distances.row(i), &minVal, nullptr, &minIdx, nullptr);
    //     wordMap.at<int>(i / image.cols, i % image.cols) = minIdx.x;
    // }
    for (int i = 0; i < distances.rows; ++i) {
        double minVal;
        cv::Point minIdx;
        cv::minMaxLoc(distances.row(i), &minVal, nullptr, &minIdx, nullptr);

        int row = i / image.cols;
        int col = i % image.cols;

        if (row < image.rows && col < image.cols) {
            wordMap.at<int>(row, col) = minIdx.x;
        } else {
            std::cerr << "Out of bounds pixel mapping: (" << row << ", " << col << ")\n";
        }
    }

    return wordMap;
}
