#include "include/pipeline.h"
#include "include/FeatureExtraction.h"
#include "Autoencoder.h"
#include "KMeans.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat runPipeline(
    const std::vector<std::string>& images,
    const std::vector<cv::Mat>& filterBank,
    const std::string& outputDir,
    const std::string& keypointMethod,
    int dictionarySize,
    int featureDim,
    int latentDim,
    const std::string& modelPath
) {
    fs::create_directories(outputDir);
    cv::Mat featureMatrix;

    for (const auto& imgPath : images) {
        // Step 1: Extract and Save Filter Responses
        std::string responseDir = outputDir + "/" + fs::path(imgPath).stem().string();
        fs::create_directories(responseDir);
        extractAndSaveFilterResponses(imgPath, filterBank, responseDir);

        // Step 2: Detect Keypoints
        auto keypoints = detectKeypoints(imgPath, keypointMethod, 100);

        // Step 3: Build Feature Matrix
        auto features = buildFeatureMatrix(keypoints, responseDir, filterBank.size());
        if (featureMatrix.empty()) {
            featureMatrix = features;
        } else {
            cv::vconcat(featureMatrix, features, featureMatrix);
        }
    }

    // Step 4: Train Autoencoder
    trainAutoencoder(featureMatrix, modelPath, featureDim, latentDim);

    // Step 5: Create Dictionary Using K-Means
    cv::Mat labels, dictionary;
    kmeans(featureMatrix, dictionarySize, labels, dictionary);

    // Save dictionary
    cv::FileStorage fs(outputDir + "/dictionary.yml", cv::FileStorage::WRITE);
    fs << "dictionary" << dictionary;
    fs.release();

    return dictionary;
}
