#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include "include/createFilterBank.h"
#include "include/filters.h"

std::string changeExtensionToYml(const std::string& path) {
    // Find the last dot in the string to identify the file extension
    size_t dotPosition = path.find_last_of('.');
    if (dotPosition != std::string::npos) {
        // Replace the extension with ".yml"
        return path.substr(0, dotPosition) + ".yml";
    }
    // If no dot is found, just append ".yml"
    return path + ".yml";
}


void printImageNames(const std::vector<std::string>& allImageNames) {
    if (allImageNames.empty()) {
        std::cout << "The vector is empty." << std::endl;
        return;
    }

    std::cout << "Image Names:" << std::endl;
    for (size_t i = 0; i < allImageNames.size(); ++i) {
        std::cout << i + 1 << ": " << allImageNames[i] << std::endl;
    }
}

cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank, const std::string& outputPath) {
    // Extract filter responses
    std::vector<cv::Mat> filterResponses = extractFilterResponsesSequential(image, filterBank, outputPath);



    // Combine all filter responses into a single matrix
    int rows = filterResponses[0].rows * filterResponses[0].cols; // Total pixels in the image
    int cols = static_cast<int>(filterResponses.size());         // Number of filter responses
    cv::Mat combinedResponses(rows, cols, CV_32F);

    for (int i = 0; i < filterResponses.size(); ++i) {
        cv::Mat reshaped = filterResponses[i].reshape(1, rows); // Flatten each filter response
        reshaped.copyTo(combinedResponses.col(i));             // Copy into the combined matrix
    }
    
    // Ensure data type consistency
    // combinedResponses.convertTo(combinedResponses, dictionary.type());

    // Check for dimensional mismatch
    if (dictionary.cols != combinedResponses.cols) {
        throw std::runtime_error("Mismatch between dictionary dimensions and combined responses!");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Compute distances to dictionary entries
    cv::Mat wordMap(image.rows, image.cols, CV_32SC1);
    for (int i = 0; i < combinedResponses.rows; ++i) {
        double minDist = std::numeric_limits<double>::max();
        int minIndex = -1;

        for (int j = 0; j < dictionary.rows; ++j) {
            double dist = cv::norm(combinedResponses.row(i), dictionary.row(j), cv::NORM_L2);
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }

        wordMap.at<int>(i / image.cols, i % image.cols) = minIndex;
    }

    return wordMap;
}



// Load dictionary from YAML into a cv::Mat
void loadYamlDictionary(const std::string& filePath, cv::Mat& dictionary) {
    YAML::Node config = YAML::LoadFile(filePath);

    if (!config["dictionary"] || !config["dictionary"]["data"]) {
        throw std::runtime_error("Invalid dictionary format in YAML file.");
    }

    const YAML::Node& data = config["dictionary"]["data"];
    const int rows = config["dictionary"]["rows"].as<int>();
    const int cols = config["dictionary"]["cols"].as<int>();

    // Flatten the YAML data into a single vector
    std::vector<float> flatData;
    for (const auto& it : data) {
        flatData.push_back(it.as<float>());
    }

    // Convert the flat vector into a cv::Mat
    dictionary = cv::Mat(rows, cols, CV_32F, flatData.data()).clone(); // Clone to ensure ownership
}

// Load image names from a CSV file
void loadImageNamesFromCsv(const std::string& filePath, std::vector<std::string>& allImageNames) {
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
        allImageNames.push_back(filename);  // Add the filename to the vector
    }

    // printImageNames(allImageNames); // Optional: print the loaded filenames for debugging
}


// Process a batch of images to compute visual words
void processImageToVisualWords(
    int threadId, 
    const std::vector<std::string>& allImageNames, 
    const std::vector<cv::Mat>& loadedImages, 
    const cv::Mat& dictionary, 
    const std::vector<cv::Mat>& filterBank, 
    std::mutex& ioMutex, 
    std::atomic<int>& imageCounter) 
{
    while (true) {
        int imgIndex = imageCounter.fetch_add(1); // Atomically get the next image index
        if (imgIndex >= allImageNames.size()) break; // Exit when no more images to process

        const std::string& imgName = allImageNames[imgIndex];
        const cv::Mat& image = loadedImages[imgIndex];

        if (image.empty()) {
            std::lock_guard<std::mutex> lock(ioMutex);
            std::cerr << "Failed to load image: " << imgName << std::endl;
            continue;
        }

        cv::Mat wordMap = getVisualWords(image, dictionary, filterBank, imgName);

        // Determine the base path
        std::string basePath;
        if (imgName.find("test_") == 0) { // If the name starts with "test_"
            basePath = "../results/Testing/test_";
        } else {
            basePath = "../results/Training/";
        }

        // Construct the full path
        std::string yml_path = basePath + imgName;
        

        // Save results
        {
            std::lock_guard<std::mutex> lock(ioMutex);
            // Save word map as a PNG image
            std::string outputPath = "wordmaps/visual_words_map_" + std::to_string(imgIndex) + ".png";
            cv::Mat normalizedWordMap;
            cv::normalize(wordMap, normalizedWordMap, 0, 255, cv::NORM_MINMAX, CV_8U);
            if (cv::imwrite(outputPath, normalizedWordMap)) {
                std::cout << "Word map image saved successfully at " << outputPath << std::endl;
            } else {
                std::cerr << "Failed to save word map image." << std::endl;
            }

            // Save word map as a YAML file
            yml_path = changeExtensionToYml(yml_path);
            cv::FileStorage fs(yml_path, cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "wordMap" << wordMap;
                fs.release();
                std::cout << "Word map YAML saved successfully at " << yml_path << std::endl;
            } else {
                std::cout << yml_path << "FAIL" << std::endl;
                std::cerr << "Failed to save word map YAML." << std::endl;
            }
        }
    }
}



// Batch process images using multithreading
void batchToVisualWords(int numCores, const std::string& dictionaryPath, const std::string& csvPath) {
    // Load image names
    std::vector<std::string> allImageNames;
    loadImageNamesFromCsv(csvPath, allImageNames);

    // Preload images
    std::vector<cv::Mat> loadedImages(allImageNames.size());
    for (size_t i = 0; i < allImageNames.size(); ++i) {
        std::string basePath = allImageNames[i].find("test_") == 0 ? "../data/Testing/" : "../data/Training/";
        loadedImages[i] = cv::imread(basePath + allImageNames[i]);
    }

    // Load dictionary and filter bank
    cv::Mat dictionary;
    loadYamlDictionary(dictionaryPath, dictionary);
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Multithreading
    std::mutex ioMutex;
    std::atomic<int> imageCounter(0); // Shared atomic counter for dynamic load balancing
    std::vector<std::thread> workers;

    for (int i = 0; i < numCores; ++i) {
        workers.emplace_back(processImageToVisualWords, i, std::ref(allImageNames), std::ref(loadedImages), 
                             std::ref(dictionary), std::ref(filterBank), std::ref(ioMutex), std::ref(imageCounter));
    }
    for (auto& worker : workers) {
        worker.join();
    }

    std::cout << "Batch to visual words done!" << std::endl;
}
