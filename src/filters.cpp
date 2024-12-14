#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <filesystem> // C++17 for creating directories

#include "include/filters.h"



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
    // std::cout << "Extracted Filter Responses: Rows=" << combinedResponses.rows
    //           << ", Cols=" << combinedResponses.cols
    //           << ", Type=" << combinedResponses.type() << std::endl;

    return combinedResponses;
}




// void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel) {
//     // Normalize response for better visualization
//     cv::Mat normalizedResponse;
//     cv::normalize(response, normalizedResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

//     // Construct file name
//     std::string filename = outputPath + "/filter_response_" + std::to_string(filterIdx) + "_" + channel + ".jpg";

//     // Save the image
//     if (!cv::imwrite(filename, normalizedResponse)) {
//         throw std::runtime_error("Failed to save filter response image: " + filename);
//     }

//     std::cout << "Saved filter response: " << filename << std::endl;
// }


// void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
//     if (image.empty()) {
//         throw std::invalid_argument("Input image is empty!");
//     }

//     // Create output directory if it doesn't exist
//     std::filesystem::create_directories(outputPath);

//     // Convert image to CIE Lab color space
//     cv::Mat labImage;
//     cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

//     // Split the Lab channels
//     std::vector<cv::Mat> labChannels(3);
//     cv::split(labImage, labChannels);
//     const cv::Mat& L_channel = labChannels[0];
//     const cv::Mat& a_channel = labChannels[1];
//     const cv::Mat& b_channel = labChannels[2];

//     // Filter index
//     int filterIdx = 1;

//     // Iterate over filters
//     for (const auto& filter : filterBank) {
//         // Process each channel
//         cv::Mat response_L, response_a, response_b;

//         // Apply filter to each channel
//         cv::filter2D(L_channel, response_L, CV_32F, filter);
//         saveFilterResponseImage(response_L, outputPath, filterIdx, "L");

//         cv::filter2D(a_channel, response_a, CV_32F, filter);
//         saveFilterResponseImage(response_a, outputPath, filterIdx, "a");

//         cv::filter2D(b_channel, response_b, CV_32F, filter);
//         saveFilterResponseImage(response_b, outputPath, filterIdx, "b");

//         // Increment filter index
//         ++filterIdx;
//     }
// }



std::vector<cv::Mat> extractFilterResponsesSequential(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }
    
    // Create output directory if it doesn't exist
    // std::filesystem::create_directories(outputPath);

    // Convert image to CIE Lab color space
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    std::vector<cv::Mat> filterResponses;

    // Filter index
    int filterIdx = 1;

    // Iterate over filters
    for (const auto& filter : filterBank) {
        cv::Mat response_L, response_a, response_b;

        cv::filter2D(L_channel, response_L, CV_32F, filter);
        // saveFilterResponseImage(response_L, outputPath, filterIdx, "L");
        filterResponses.push_back(response_L);

        cv::filter2D(a_channel, response_a, CV_32F, filter);
        // saveFilterResponseImage(response_a, outputPath, filterIdx, "a");
        filterResponses.push_back(response_a);

        cv::filter2D(b_channel, response_b, CV_32F, filter);
        // saveFilterResponseImage(response_b, outputPath, filterIdx, "b");
        filterResponses.push_back(response_b);

        ++filterIdx;
    }
    return filterResponses;
}

// OpenMP filter implementation
std::vector<cv::Mat> extractFilterResponsesOpenMP(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    std::filesystem::create_directories(outputPath);

    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> labChannels(3);
    cv::split(labImage, labChannels);
    const cv::Mat& L_channel = labChannels[0];
    const cv::Mat& a_channel = labChannels[1];
    const cv::Mat& b_channel = labChannels[2];

    int totalResponses = filterBank.size()*3;
    std::vector<cv::Mat> filterResponses(totalResponses);

    #pragma omp parallel for
    for (int filterIdx = 0; filterIdx < filterBank.size(); ++filterIdx) {
        const auto& filter = filterBank[filterIdx];

        try {
            cv::Mat response_L, response_a, response_b;

            cv::filter2D(L_channel, response_L, CV_32F, filter);
            saveFilterResponseImage(response_L, outputPath, filterIdx + 1, "L");
            filterResponses[filterIdx * 3 + 0] = response_L;

            cv::filter2D(a_channel, response_a, CV_32F, filter);
            saveFilterResponseImage(response_a, outputPath, filterIdx + 1, "a");
            filterResponses[filterIdx * 3 + 1] = response_a;

            cv::filter2D(b_channel, response_b, CV_32F, filter);
            saveFilterResponseImage(response_b, outputPath, filterIdx + 1, "b");
            filterResponses[filterIdx * 3 + 2] = response_b;
            

        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "Error processing filter " << filterIdx + 1 << ": " << e.what() << std::endl;
            }
        }
    }
    return filterResponses;
}

// Wrapper to select method
void extractAndSaveFilterResponses(const cv::Mat& image, const std::vector<cv::Mat>& filterBank, const std::string& outputPath, const std::string& method) {
    if (method == "sequential") {
        extractFilterResponsesSequential(image, filterBank, outputPath);
    } else if (method == "openmp") {
        extractFilterResponsesOpenMP(image, filterBank, outputPath);
    } 
    // else if (method == "cuda") {
    //     // Call the CUDA function, which is defined in filters.cu
    //     extractFilterResponsesCUDA(image, filterBank, outputPath);
    // } 
    else {
        throw std::invalid_argument("Unknown method: " + method);
    }
}

void saveFilterResponseImage(const cv::Mat& response, const std::string& outputPath, int filterIdx, const std::string& channel) {

    cv::Mat normalizedResponse;
    cv::normalize(response, normalizedResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

    std::string filename = outputPath + "/filter_response_" + std::to_string(filterIdx) + "_" + channel + ".jpg";

    if (!cv::imwrite(filename, normalizedResponse)) {
        throw std::runtime_error("Failed to save filter response image: " + filename);
    }
}
