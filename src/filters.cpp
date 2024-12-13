#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <filesystem> // C++17 for creating directories

#include "include/filters.h"



// cv::Mat extractFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank) {
//     // Convert image to Lab color space
//     cv::Mat labImage;
//     cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

//     // Split the Lab channels
//     std::vector<cv::Mat> labChannels(3);
//     cv::split(labImage, labChannels);

//     // Create a vector to store filter responses for each pixel
//     std::vector<cv::Mat> responseChannels;

//     // Apply each filter to each channel and store the results
//     for (const auto& filter : filterBank) {
//         for (const auto& channel : labChannels) {
//             cv::Mat response;
//             cv::filter2D(channel, response, CV_32F, filter);
//             responseChannels.push_back(response);
//         }
//     }

//     // Ensure the correct number of channels
//     if (responseChannels.size() != 3 * filterBank.size()) {
//         throw std::runtime_error("Filter responses do not match expected dimensions.");
//     }

//     // Combine all responses into a single multi-channel matrix
//     cv::Mat combinedResponses;
//     cv::merge(responseChannels, combinedResponses);

//     std::cout << "Extracted Filter Responses: Rows=" << combinedResponses.rows 
//               << ", Cols=" << combinedResponses.cols 
//               << ", Channels=" << combinedResponses.channels() << std::endl;

//     return combinedResponses;
// }


cv::Mat extractFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank) {
    // Convert image to Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);

    // Create a vector to store filter responses
    std::vector<cv::Mat> responseChannels;

    // Apply each filter to each channel and store the results
    for (const auto& filter : filterBank) {
        for (const auto& channel : labChannels) {
            cv::Mat response;
            cv::filter2D(channel, response, CV_32F, filter); // Ensure CV_32F type
            responseChannels.push_back(response);
        }
    }

    // Combine all responses into a single matrix
    cv::Mat combinedResponses;
    cv::merge(responseChannels, combinedResponses); // Combine into a multi-channel matrix

    // Reshape to [rows * cols x numChannels] for use with dictionary
    int numPixels = labImage.rows * labImage.cols;
    combinedResponses = combinedResponses.reshape(1, numPixels);

    // Debug output for combined responses
    std::cout << "Extracted Filter Responses: Rows=" << combinedResponses.rows
              << ", Cols=" << combinedResponses.cols
              << ", Type=" << combinedResponses.type() << std::endl;

    return combinedResponses;
}




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

