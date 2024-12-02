#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


// MAIN FUNCTION TO TEST FILTER EXTRACTION

// std::vector<cv::Mat> createFilterBank();

// int main() {
//     // Define the image path and results output folder
//     std::string imagePath = "data/airport_image1.jpg";
//     std::string resultsPath = "results";

//     // Load the input image
//     cv::Mat image = cv::imread(imagePath);
//     if (image.empty()) {
//         std::cerr << "Error: Could not open image at " << imagePath << std::endl;
//         return -1;
//     }

//     // Create the filter bank
//     std::vector<cv::Mat> filterBank = createFilterBank();

//     // Apply filters and save the responses
//     try {
//         extractAndSaveFilterResponses(image, filterBank, resultsPath);
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }

//     std::cout << "Processing completed successfully. Filtered images saved in " << resultsPath << std::endl;

//     return 0;
// }



// MAIN FUNCTION TO TEST GETHARRISPOINTS

// #include "include/getHarrisPoints.h"
// #include <iostream>
// #include <filesystem>

// int main() {
//     try {
//         // Load the image
//         std::string inputPath = "data/campus_image1.jpg";  // Replace with your image path
//         cv::Mat image = cv::imread(inputPath);

//         // Parameters for Harris detector
//         int alpha = 500;     // Number of top responses
//         double k = 0.04;     // Harris constant

//         // Perform Harris corner detection
//         std::vector<cv::Point> harrisPoints = getHarrisPoints(image, alpha, k);

//         // Save result
//         std::string outputPath = "results/harris_points_image2.jpg";
//         saveHarrisPointsImage(image, harrisPoints, outputPath);

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }



