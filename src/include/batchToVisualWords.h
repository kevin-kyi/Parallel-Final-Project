#ifndef BATCH_TO_VISUAL_WORDS_H
#define BATCH_TO_VISUAL_WORDS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

// Function to compute visual words for a single image
// cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank, std::string& outputPath);
cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank, const std::string& outputPath);


// Function to load a dictionary from a YAML file
void loadYamlDictionary(const std::string& filePath, cv::Mat& dictionary);

// Function to load image names from a CSV file
void loadImageNamesFromCsv(const std::string& filePath, std::vector<std::string>& allImageNames);

// Function to process a single image to compute visual words
void processImageToVisualWords(
    int index, 
    const std::vector<std::string>& allImageNames,
    const std::vector<cv::Mat>& loadedImages,
    const cv::Mat& dictionary, 
    const std::vector<cv::Mat>& filterBank,
    std::mutex& ioMutex,
    std::atomic<int>& imageCounter);

// Batch processing function to compute visual words for a dataset
void batchToVisualWords(int numCores, const std::string& dictionaryPath, const std::string& csvPath);


#endif // BATCH_TO_VISUAL_WORDS_H