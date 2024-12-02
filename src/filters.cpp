#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <filesystem> // C++17 for creating directories

void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel) {
    // Normalize response for better visualization
    cv::Mat normalizedResponse;
    cv::normalize(response, normalizedResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Construct file name
    std::string filename = outputPath + "/filter_response_" + std::to_string(filterIdx) + "_" + channel + ".jpg";

    // Save the image
    if (!cv::imwrite(filename, normalizedResponse)) {
        throw std::runtime_error("Failed to save filter response image: " + filename);
    }

    std::cout << "Saved filter response: " << filename << std::endl;
}


void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputPath);

    // Convert image to CIE Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    // Filter index
    int filterIdx = 1;

    // Iterate over filters
    for (const auto& filter : filterBank) {
        // Process each channel
        cv::Mat response_L, response_a, response_b;

        // Apply filter to each channel
        cv::filter2D(L_channel, response_L, CV_32F, filter);
        saveFilterResponseImage(response_L, outputPath, filterIdx, "L");

        cv::filter2D(a_channel, response_a, CV_32F, filter);
        saveFilterResponseImage(response_a, outputPath, filterIdx, "a");

        cv::filter2D(b_channel, response_b, CV_32F, filter);
        saveFilterResponseImage(response_b, outputPath, filterIdx, "b");

        // Increment filter index
        ++filterIdx;
    }
}

