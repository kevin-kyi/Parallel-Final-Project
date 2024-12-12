#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "include/createFilterBank.h"   
#include "include/featureExtraction.h" // Assumes you implemented buildFeatureMatrix
#include "include/getHarrisPoints.h"                    // Replace this with a C++ KMeans implementation
#include <filesystem>                  // For directory handling

void getDictionary(const std::vector<std::string>& imgPaths, int alpha, int K, const std::string& method, const std::string& resultsDir, const std::string& outputFile) {
    // Create filter bank (used for size and consistency checks)
    std::vector<cv::Mat> filterBank = createFilterBank();
    int numFilters = filterBank.size();

    // Initialize feature matrix
    cv::Mat pixelResponses;

    for (const auto& path : imgPaths) {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Error loading image: " << path << std::endl;
            continue;
        }

        // Get keypoints
        std::vector<cv::Point> harrispoints = getHarrisPoints(image, alpha, 0.05);

        // Load precomputed filter responses from results directory
        cv::Mat features = buildFeatureMatrix(harrispoints, resultsDir, numFilters);

        // Append to global pixel responses
        if (pixelResponses.empty()) {
            pixelResponses = features;
        } else {
            cv::vconcat(pixelResponses, features, pixelResponses);
        }
    }

    // Apply KMeans clustering
    // Apply KMeans clustering
    cv::Mat labels;
    cv::Mat dictionary;

    cv::kmeans(pixelResponses, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0),
               3, cv::KMEANS_PP_CENTERS, dictionary);


    // Save dictionary to .pkl file
    // cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
    // fs << "dictionary" << dictionary;
    // fs.release();
}

