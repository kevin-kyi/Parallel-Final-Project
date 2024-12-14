#ifndef KMEANS_KNN_CLASSIFICATION
#define KMEANS_KNN_CLASSIFICATION

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <filesystem>

// Function to load word maps and labels
std::vector<std::pair<cv::Mat, int>> loadWordMaps(const std::string& baseDirectory, const std::vector<std::string>& categoryDirs, const std::vector<int>& labels);
// Function to compute image features (histograms)
cv::Mat getImageFeatures(const cv::Mat& wordMap, int dictionarySize);

// Function to compute distances between histograms
float getImageDistance(const cv::Mat& hist1, const cv::Mat& hist2, const std::string& method);

// Function to classify a single image using k-NN
int classifyImage(const cv::Mat& testFeature, const std::vector<cv::Mat>& trainFeatures, 
                  const std::vector<int>& trainLabels, int k, const std::string& distanceMethod);

// Function to evaluate the system and compute accuracy and confusion matrix
void evaluateSystem(const std::vector<std::pair<cv::Mat, int>>& testWordMaps,
                    const std::vector<cv::Mat>& trainFeatures, const std::vector<int>& trainLabels,
                    int dictionarySize, int k, const std::string& distanceMethod);

#endif // EVALUATION_H