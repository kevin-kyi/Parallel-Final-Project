// #include "include/filters.h"
// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <vector>
// #include <filesystem>


// // MAIN FUNCTION TO TEST FILTER EXTRACTION

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
#include "include/filters.h"
#include "include/createFilterBank.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // C++17 for iterating over directory contents
#include <chrono>     // For timing
#include <fstream>
#include <omp.h>

// MAIN FUNCTION TO TEST FILTER EXTRACTION

std::vector<cv::Mat> createFilterBank();

int main() {

    // Define the directory containing images and results output folder
    std::string inputDir = "../data/Testing/test_swimming_pool"; //change dir name
    std::string resultsDir = "../results/parallel/test_swimming_pool"; //change dir name
    std::string csvPath = "../results/parallel/performance_data/performance_swimming_pool.csv"; //change csv name

    // Create the filter bank
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Open the CSV file for writing
    std::ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing: " << csvPath << std::endl;
        return -1;
    }

    // Write the header for the CSV
    csvFile << "test_swimming_pool\n"; //change name
    csvFile << "Image,Time (seconds)\n";

    // Measure the total directory processing time
    auto totalStart = std::chrono::high_resolution_clock::now();

    // Iterate over all files in the input directory
    for (const auto& entry : std::__fs::filesystem::directory_iterator(inputDir)) {
        const std::string imagePath = entry.path().string();

        // Ensure the file has a .jpg extension (case-insensitive check)
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
            // std::cout << "Processing image: " << imagePath << std::endl;

            // Measure the time for processing a single image
            auto imageStart = std::chrono::high_resolution_clock::now();

            // Load the image
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Error: Could not open image at " << imagePath << std::endl;
                continue; // Skip to the next file
            }

            // Define the output path for this image
            std::string imageResultsDir = resultsDir + "/" + entry.path().stem().string();

            // Apply filters and save the responses
            try {
                extractAndSaveFilterResponses(image, filterBank, imageResultsDir);
            } catch (const std::exception& e) {
                std::cerr << "Error processing image " << imagePath << ": " << e.what() << std::endl;
                continue; // Skip to the next file
            }

            // Measure and log the time for the current image
            auto imageEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> imageDuration = imageEnd - imageStart;
            std::cout << "Time taken for " << imagePath << ": " << imageDuration.count() << " seconds" << std::endl;
            // Write the result to the CSV file
            csvFile << entry.path().filename().string() << "," << imageDuration.count() << "\n";
        } else {
            std::cout << "Skipping non-image file: " << imagePath << std::endl;
        }
    }

    // Measure and log the total processing time
    auto totalEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalDuration = totalEnd - totalStart;
    std::cout << "Total time taken for directory: " << totalDuration.count() << " seconds" << std::endl;

    // Close the CSV file
    
    csvFile << "Total," << totalDuration.count() << "\n";
    csvFile.close();
    std::cout << "Performance times written to " << csvPath << std::endl;

    return 0;
}




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
