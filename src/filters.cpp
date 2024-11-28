// #include "include/filters.h"
// #include <opencv2/opencv.hpp>
// #include <vector>

// std::vector<std::vector<float>> extractFilterResponses(const cv::Mat& image) {
//     // Check image is valid
//     if (image.empty()) {
//         throw std::invalid_argument("Input image empty");
//     }

//     // Convert the image to CIE Lab color space
//     cv::Mat labImage;
//     cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

//     int rows = labImage.rows;
//     int cols = labImage.cols;
//     std::vector<std::vector<float>> featureResponses(rows * cols);

//     cv::Mat gaussian, laplacian;

//     // Apply Gaussian blur on each channel of the Lab image
//     for (int channel = 0; channel < 3; ++channel) {
//         cv::GaussianBlur(labImage, gaussian, cv::Size(3, 3), 1.0);

//         // Apply Laplacian filter
//         cv::Mat singleChannel = labImage.clone();
//         cv::extractChannel(labImage, singleChannel, channel); 
//         cv::Laplacian(singleChannel, laplacian, CV_32F);

//         // Iterate over pixels to store filter responses
//         for (int row = 0; row < rows; ++row) {
//             for (int col = 0; col < cols; ++col) {
//                 int idx = row * cols + col; // Flattened index
//                 if (featureResponses[idx].empty()) {
//                     featureResponses[idx].resize(6); // Three filters * 2
//                 }
//                 // Add Gaussian and Laplacian filter responses for this channel
//                 featureResponses[idx][channel * 2] = gaussian.at<cv::Vec3b>(row, col)[channel];
//                 featureResponses[idx][channel * 2 + 1] = laplacian.at<float>(row, col);
//             }
//         }
//     }

//     return featureResponses;
// }

#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>

// Function to apply filters and save responses as images
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    // Ensure the input image is valid
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Convert the image to CIE Lab color space for better feature extraction
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    // Create directory for output images if needed
    // (This step may require external libraries like filesystem in modern C++)

    // Apply filters to each channel and save results
    int filterIdx = 1;
    for (const auto& filter : filterBank) {
        // Apply filter to L channel
        cv::Mat response_L;
        cv::filter2D(L_channel, response_L, CV_32F, filter);
        saveFilterResponseImage(response_L, outputPath, filterIdx++, "L");

        // Apply filter to a channel
        cv::Mat response_a;
        cv::filter2D(a_channel, response_a, CV_32F, filter);
        saveFilterResponseImage(response_a, outputPath, filterIdx++, "a");

        // Apply filter to b channel
        cv::Mat response_b;
        cv::filter2D(b_channel, response_b, CV_32F, filter);
        saveFilterResponseImage(response_b, outputPath, filterIdx++, "b");
    }
}

// Helper function to normalize and save filter response images
void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel) {
    // Normalize the response for visualization
    cv::Mat normalizedResponse;
    cv::normalize(response, normalizedResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Construct output file path
    std::string filename = outputPath + "/filter_response_" + std::to_string(filterIdx) + "_" + channel + ".jpg";

    // Save the image
    if (!cv::imwrite(filename, normalizedResponse)) {
        throw std::runtime_error("Failed to save filter response image: " + filename);
    }

    std::cout << "Saved filter response: " << filename << std::endl;
}
