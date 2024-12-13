#include "include/filters.h"
#include "include/createFilterBank.h"
#include "include/batchToVisualWords.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // C++17 for iterating over directory contents
#include <chrono>     // For timing
#include <fstream>
#include <omp.h>


// MAIN FUNCTION TO TEST FILTER EXTRACTION

void test_filtering_implementations() {

    std::string inputDir = "../data/Testing/test_swimming_pool"; //change dir name
    std::string resultsDir = "../results/cuda/test_swimming_pool"; //change dir name
    std::string csvPath = "../results/cuda/performance_data/performance_swimming_pool.csv"; //change csv name

    std::vector<cv::Mat> filterBank = createFilterBank();

    // Open the CSV file for writing
    std::ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open CSV file for writing: " << csvPath << std::endl;
        return;
    }

    csvFile << "test_swimming_pool\n"; //change name
    csvFile << "Image,Time (seconds)\n";

    auto totalStart = std::chrono::high_resolution_clock::now();
    // Define implementation we want to use ("sequential", "openmp", "cuda")
    std::string method = "cuda";
    std::cout << "testing with " << method << " implementation\n";

    // Iterate over all files in the input directory
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        const std::string imagePath = entry.path().string();

        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
            // std::cout << "Processing image: " << imagePath << std::endl;

            auto imageStart = std::chrono::high_resolution_clock::now();

            cv::Mat image = cv::imread(imagePath);
            if (image.empty()) {
                std::cerr << "Error: Could not open image at " << imagePath << std::endl;
                continue; // Skip to the next file
            }

            std::string imageResultsDir = resultsDir + "/" + entry.path().stem().string();

            // Apply filters and save the responses
            try {
                extractAndSaveFilterResponses(image, filterBank, imageResultsDir, method);
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

}

int test_visual_words() {
    try {
        // Paths to required files and directories
        std::string dictionaryPath = "../kmeans_dictionary.yml"; // Path to dictionary file
        std::string csvPath = "../traintest.csv";                // Path to CSV file
        int numCores = 4;                                       // Number of threads to use

        // Start batch processing
        std::cout << "Starting batch processing of visual words..." << std::endl;
        batchToVisualWords(numCores, dictionaryPath, csvPath);
        std::cout << "Visual word processing completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main() {

    test_visual_words();
    
    return 0;
}
