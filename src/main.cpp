// TESTING GIT OOOO

#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// Declare the function to create the filter bank
std::vector<cv::Mat> createFilterBank();

int main() {
    // Define the image path and results output folder
    std::string imagePath = "../data/airport_image1.jpg";
    std::string resultsPath = "../results";

    // Load the input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not open image at " << imagePath << std::endl;
        return -1;
    }

    // Create the filter bank
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Apply filters and save the responses
    try {
        extractAndSaveFilterResponses(image, filterBank, resultsPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Processing completed successfully. Filtered images saved in " << resultsPath << std::endl;

    return 0;
}