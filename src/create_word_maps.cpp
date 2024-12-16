#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"

#include <fstream>

#include "include/getVisualWords.h"
#include "include/createFilterBank.h"
#include <yaml-cpp/yaml.h>
#include "include/create_csv.h"

#include "include/save_dictionary_yml.h"

#include "include/create_word_maps.h"






// *************** DBSCAN CREATING WORD MAPS FOR EVERY IMAGE ***************
// This function applied the dbscan dictionary to all training images which create a word map
// for every image. The resulting images are stored in "results/dbscan" in /Training or /Testing
// folders which have the same category directories as kmeans
void DBSCANloadImageNames(const std::string& filePath, std::vector<std::string>& imagePaths) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filePath);
    }

    std::string line;
    std::getline(file, line); // Skip the header line

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string filename;
        std::getline(stream, filename, ','); // Get the first column (filename)
        imagePaths.push_back(filename);     // Add the filename to the vector
    }
}

// Helper function to determine the input directory based on the image name
std::string DBSCANdetermineInputPath(const std::string& imageName) {
    std::string basePath;

    if (imageName.find("test_") == 0) {
        basePath = "data/Testing/";
    } else {
        basePath = "data/Training/";
    }

    return basePath + imageName;
}

// Helper function to determine the base output directory based on the image name
std::string DBSCANdetermineOutputPath(const std::string& imageName, const std::string& resultsDir) {
    std::string basePath;

    if (imageName.find("test_") == 0) {
        basePath = resultsDir + "/dbscan/Testing/";
    } else {
        basePath = resultsDir + "/dbscan/Training/";
    }

    // Append the category subdirectory (e.g., airport_terminal or test_airport_terminal)
    size_t slashPos = imageName.find('/');
    if (slashPos != std::string::npos) {
        std::string category = imageName.substr(0, slashPos);
        basePath += category + "/";
    }

    return basePath;
}

void DBSCANcreateWordMaps() {
    // Paths to input and output data
    std::string csvPath = "../traintest.csv"; // Adjust the path to your CSV file
    std::string dictionaryPath = "../dbscan_dictionary.yml"; // Path to the YAML dictionary
    std::string resultsDir = "../results"; // Base directory for saving results

    // Load image names from CSV
    std::vector<std::string> imagePaths;
    try {
        DBSCANloadImageNames(csvPath, imagePaths);
    } catch (const std::exception& ex) {
        std::cerr << "Error loading image names from CSV: " << ex.what() << std::endl;
    }

    // Create filter bank
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Load the dictionary from YAML
    cv::Mat dictionary;
    cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
    }
    fs["means"] >> dictionary;
    dictionary.convertTo(dictionary, CV_32F);
    fs.release();

    // Process each image
    for (const std::string& imageName : imagePaths) {
        std::cout << "Processing: " << imageName << std::endl;

        // Determine the input path
        std::string inputImagePath = DBSCANdetermineInputPath(imageName);
        inputImagePath = "../" + inputImagePath;
        // Load the input image
        cv::Mat image = cv::imread(inputImagePath);
        if (image.empty()) {
            std::cerr << "Error: Unable to load the image at " << inputImagePath << std::endl;
            continue;
        }

        try {
            // Generate the word map
            if (image.rows * image.cols > 1000000) { // Threshold: 1 million pixels
                std::cout << "Resizing large image: " << imageName << " (" << image.rows << "x" << image.cols << ")" << std::endl;
                cv::resize(image, image, cv::Size(), 0.7, 0.7, cv::INTER_AREA); // Adjust scaling factor as needed
                std::cout << "Image resized to: " << image.size() << std::endl;
            }

            // cv::resize(image, image, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);

            std::cout << "Generated word map successfully for " << imageName << std::endl;

            // Determine the output directory
            std::string outputBasePath = DBSCANdetermineOutputPath(imageName, resultsDir);
            // Ensure the output directory exists
            std::string command = "mkdir -p " + outputBasePath;
            system(command.c_str());
            
            // Construct the full output path
            std::string outputPath = outputBasePath + imageName.substr(imageName.find('/') + 1);
            outputPath = outputPath.substr(0, outputPath.find_last_of('.')) + ".yml";

            // Save the word map as a YML file
            cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
                        
            std::cout << "saving wordmap to: " << outputPath << std::endl;
            if (fs.isOpened()) {
                fs << "wordMap" << wordMap;
                fs.release();
                std::cout << "Word map YML saved successfully at " << outputPath << std::endl;
            } else {
                std::cerr << "Failed to save word map YML for " << imageName << std::endl;
            }
                        

            wordMap.release();
            image.release();
        } catch (const std::exception& ex) {
            std::cerr << "Error during visual words computation for " << imageName << ": " << ex.what() << std::endl;
        }
    }

    std::cout << "Processing completed for all images." << std::endl;
}











// *************** KMEANS WORKING IMPLENTATION FOR EXTRACTUBG VISUAL WORDS ***************
void KMEANSloadImageNames(const std::string& filePath, std::vector<std::string>& imagePaths) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filePath);
    }

    std::string line;
    std::getline(file, line); // Skip the header line

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string filename;
        std::getline(stream, filename, ','); // Get the first column (filename)
        imagePaths.push_back(filename);     // Add the filename to the vector
    }
}

// Helper function to determine the input directory based on the image name
std::string KMEANSdetermineInputPath(const std::string& imageName) {
    std::string basePath;

    if (imageName.find("test_") == 0) {
        basePath = "../data/Testing/";
    } else {
        basePath = "../data/Training/";
    }

    return basePath + imageName;
}

// Helper function to determine the base output directory based on the image name
std::string KMEANSdetermineOutputPath(const std::string& imageName, const std::string& resultsDir) {
    std::string basePath;

    if (imageName.find("test_") == 0) {
        basePath = resultsDir + "/Testing/";
    } else {
        basePath = resultsDir + "/Training/";
    }

    // Append the category subdirectory (e.g., airport_terminal or test_airport_terminal)
    size_t slashPos = imageName.find('/');
    if (slashPos != std::string::npos) {
        std::string category = imageName.substr(0, slashPos);
        basePath += category + "/";
    }

    return basePath;
}

void KMEANScreateWordMaps() {
    // Paths to input and output data
    std::string csvPath = "../traintest.csv"; // Adjust the path to your CSV file
    std::string dictionaryPath = "../kmeans_dictionary.yml"; // Path to the YAML dictionary
    std::string resultsDir = "../results"; // Base directory for saving results

    // Load image names from CSV
    std::vector<std::string> imagePaths;
    try {
        KMEANSloadImageNames(csvPath, imagePaths);
    } catch (const std::exception& ex) {
        std::cerr << "Error loading image names from CSV: " << ex.what() << std::endl;
    }

    // Create filter bank
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Load the dictionary from YAML
    cv::Mat dictionary;
    cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
    }
    fs["dictionary"] >> dictionary;
    fs.release();

    // std::cout << "Loaded Dictionary - Rows: " << dictionary.rows
    //           << ", Cols: " << dictionary.cols
    //           << ", Type: " << dictionary.type() << std::endl;

    // Process each image
    for (const std::string& imageName : imagePaths) {
        std::cout << "Processing: " << imageName << std::endl;

        // Determine the input path
        std::string inputImagePath = KMEANSdetermineInputPath(imageName);

        // Load the input image
        cv::Mat image = cv::imread(inputImagePath);
        if (image.empty()) {
            std::cerr << "Error: Unable to load the image at " << inputImagePath << std::endl;
            continue;
        }

        try {
            // Generate the word map
            if (image.rows * image.cols > 1000000) { // Threshold: 1 million pixels
                // std::cout << "Resizing large image: " << imageName << " (" << image.rows << "x" << image.cols << ")" << std::endl;
                cv::resize(image, image, cv::Size(), 0.7, 0.7, cv::INTER_AREA); // Adjust scaling factor as needed
                // std::cout << "Image resized to: " << image.size() << std::endl;
            }

            // cv::resize(image, image, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);

            std::cout << "Generated word map successfully for " << imageName << std::endl;

            // Determine the output directory
            std::string outputBasePath = KMEANSdetermineOutputPath(imageName, resultsDir);

            // Ensure the output directory exists
            std::string command = "mkdir -p " + outputBasePath;
            system(command.c_str());

            // Construct the full output path
            std::string outputPath = outputBasePath + imageName.substr(imageName.find('/') + 1);
            outputPath = outputPath.substr(0, outputPath.find_last_of('.')) + ".yml";

            // Save the word map as a YML file
            cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "wordMap" << wordMap;
                fs.release();
                std::cout << "Word map YML saved successfully at " << outputPath << std::endl;
            } else {
                std::cerr << "Failed to save word map YML for " << imageName << std::endl;
            }
            wordMap.release();
            image.release();
        } catch (const std::exception& ex) {
            std::cerr << "Error during visual words computation for " << imageName << ": " << ex.what() << std::endl;
        }
    }

    std::cout << "Processing completed for all images." << std::endl;
}

