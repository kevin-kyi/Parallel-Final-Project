#include "include/filters.h"
#include "include/createFilterBank.h"
#include "include/batchToVisualWords.h"
#include "include/save_dictionary_yml.h"
#include "include/create_dictionary.h"
#include "include/create_word_maps.h"
#include "include/classification.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // C++17 for iterating over directory contents
#include <chrono>     // For timing
#include <fstream>
#include <omp.h>


// Function to test each filter implementation and visually see results
void apply_filters(const std::string& method) {
    std::cout << "testing with " << method << " implementation\n";
    std::vector<std::string> datasets = {"airport_terminal", "campus", "desert", "elevator", "forest", "kitchen", "lake", "swimming_pool"};
    for (const auto& dataset: datasets) {
        std::string inputDir = "../data/Testing/test_" + dataset;
        std::string resultsDir = "../results/" + method + "/test_" + dataset;
        
        std::string csvPath = "../results/" + method + "/performance_data/performance_" + dataset + ".csv";
        std::vector<cv::Mat> filterBank = createFilterBank();

        // Open the CSV file for writing
        std::ofstream csvFile(csvPath);
        if (!csvFile.is_open()) {
            std::cerr << "Error: Could not open CSV file for writing: " << csvPath << std::endl;
            return;
        }

        csvFile << dataset + "\n"; //change name
        csvFile << "Image,Time (seconds)\n";

        auto totalStart = std::chrono::high_resolution_clock::now();
        

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
}


int main() {

    

    // uncomment below to test filtering

    // //apply filters to every image in each directory with all implementations
    // std::vector<std::string> methods = {"sequential", "openmp", "cuda"};

    // for (const auto& method : methods) {
    //     apply_filters(method);
    // }

    

    save_dbscan_dictionary();
    DBSCANcreateWordMaps();
    DBSCANtestDictionaryEvaluation();



    return 0;
}