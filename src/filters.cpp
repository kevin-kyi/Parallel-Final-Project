#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <filesystem> // C++17 for creating directories
#include <omp.h>


#include "include/filters.h"

//Sequential filter implementation

// cv::Mat extractFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank) {
//     // Convert image to Lab color space
//     cv::Mat labImage;
//     cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

//     // Split the Lab channels
//     std::vector<cv::Mat> labChannels(3);
//     cv::split(labImage, labChannels);

//     // Create a vector to store filter responses for each pixel
//     std::vector<cv::Mat> filterResponses;

//     cv::Mat L_channel, a_channel, b_channel;

//     // Iterate over filters and channels
//     for (const auto& filter : filterBank) {
//         cv::Mat response_L, response_a, response_b;

//         L_channel = labChannels[0];
//         a_channel = labChannels[1];
//         b_channel = labChannels[2];

//         // Apply filter to each channel
//         cv::filter2D(L_channel, response_L, CV_32F, filter);
//         cv::filter2D(a_channel, response_a, CV_32F, filter);
//         cv::filter2D(b_channel, response_b, CV_32F, filter);

//         // Concatenate responses for the current pixel into a single row
//         std::vector<float> pixelResponse = {response_L.at<float>(0, 0), response_a.at<float>(0, 0), response_b.at<float>(0, 0)};
//         cv::Mat pixelResponseMat = cv::Mat(1, pixelResponse.size(), CV_64FC1, &pixelResponse[0]); // Ensure 64FC1 data type
//         filterResponses.push_back(pixelResponseMat);
//     }

//     // Concatenate all filter responses along the vertical dimension
//     cv::Mat allResponses;
//     cv::vconcat(filterResponses, allResponses);

//     return allResponses;
// }

// Optimized function to extract and save filter responses
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Create output directory if it doesn't exist
    std::__fs::filesystem::create_directories(outputPath);

    // Convert image to CIE Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    // Parallelize filter application
    #pragma omp parallel for
    for (int filterIdx = 0; filterIdx < filterBank.size(); ++filterIdx) {
        const auto& filter = filterBank[filterIdx];

        try {
            // Process each channel
            cv::Mat response_L, response_a, response_b;

            cv::filter2D(L_channel, response_L, CV_32F, filter);
            saveFilterResponseImage(response_L, outputPath, filterIdx + 1, "L");

            cv::filter2D(a_channel, response_a, CV_32F, filter);
            saveFilterResponseImage(response_a, outputPath, filterIdx + 1, "a");

            cv::filter2D(b_channel, response_b, CV_32F, filter);
            saveFilterResponseImage(response_b, outputPath, filterIdx + 1, "b");
        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error processing filter " << filterIdx + 1 << ": " << e.what() << std::endl;
            }
        }
    }
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

    // std::cout << "Saved filter response: " << filename << std::endl;
}

