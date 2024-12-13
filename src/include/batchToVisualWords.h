#ifndef VISUAL_WORDS_H
#define VISUAL_WORDS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Function to compute visual words for a single image
cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank, std::string& outputPath);

// Function to load a dictionary from a YAML file
void loadYamlDictionary(const std::string& filePath, cv::Mat& dictionary);

// Function to load image names from a CSV file
void loadImageNamesFromCsv(const std::string& filePath, std::vector<std::string>& allImageNames);

// Function to process a single image to compute visual words
void processImageToVisualWords(
    int index, const std::vector<std::string>& allImageNames,
    const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank,
    int numCores, std::mutex& ioMutex);

// Batch processing function to compute visual words for a dataset
void batchToVisualWords(int numCores, const std::string& dictionaryPath, const std::string& csvPath);

#endif // VISUAL_WORDS_H
