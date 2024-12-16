#ifndef DICTIONARY_EVALUATION_H
#define DICTIONARY_EVALUATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

/*
 * loadWordMaps
 * 
 * Loads visual word maps from the specified directories under the given base directory.
 * Filters and loads only `.yml` files.
 * 
 * Parameters:
 *   - baseDirectory: Path to the root directory containing the category subdirectories.
 *   - categoryDirs: Vector of category directory names (e.g., "Training", "Testing").
 *   - labels: Vector of labels corresponding to the categories.
 * 
 * Returns:
 *   - A vector of pairs containing the word maps and their corresponding category labels.
 */
std::vector<std::pair<cv::Mat, int>> loadWordMaps(const std::string& baseDirectory, 
                                                  const std::vector<std::string>& categoryDirs, 
                                                  const std::vector<int>& labels);

/*
 * getImageFeatures
 * 
 * Computes a histogram of visual word frequencies for a given word map.
 * 
 * Parameters:
 *   - wordMap: Input word map (integer matrix).
 *   - dictionarySize: The size of the visual word dictionary.
 * 
 * Returns:
 *   - A normalized histogram as a 1xN cv::Mat (where N = dictionarySize).
 */
cv::Mat getImageFeatures(const cv::Mat& wordMap, int dictionarySize);

/*
 * getImageDistance
 * 
 * Computes the distance between two histograms.
 * 
 * Parameters:
 *   - hist1: First histogram.
 *   - hist2: Second histogram.
 *   - method: Distance metric ("euclidean" or "chi2").
 * 
 * Returns:
 *   - The computed distance as a float.
 */
float getImageDistance(const cv::Mat& hist1, const cv::Mat& hist2, const std::string& method);

/*
 * classifyImage
 * 
 * Classifies a test image feature histogram using k-nearest neighbors (k-NN).
 * 
 * Parameters:
 *   - testFeature: Feature histogram of the test image.
 *   - trainFeatures: Vector of feature histograms for training images.
 *   - trainLabels: Vector of labels corresponding to the training features.
 *   - k: Number of neighbors to consider for k-NN.
 *   - distanceMethod: Distance metric ("euclidean" or "chi2").
 * 
 * Returns:
 *   - The predicted label for the test image.
 */
int classifyImage(const cv::Mat& testFeature, const std::vector<cv::Mat>& trainFeatures, 
                  const std::vector<int>& trainLabels, int k, const std::string& distanceMethod);

/*
 * evaluateSystem
 * 
 * Evaluates the classification system by computing accuracy and confusion matrix.
 * 
 * Parameters:
 *   - testWordMaps: Vector of word maps and their actual labels (test data).
 *   - trainFeatures: Vector of feature histograms for training images.
 *   - trainLabels: Vector of labels corresponding to the training features.
 *   - dictionarySize: Size of the visual word dictionary.
 *   - k: Number of neighbors for k-NN.
 *   - distanceMethod: Distance metric ("euclidean" or "chi2").
 */
void evaluateSystem(const std::vector<std::pair<cv::Mat, int>>& testWordMaps,
                    const std::vector<cv::Mat>& trainFeatures, const std::vector<int>& trainLabels,
                    int dictionarySize, int k, const std::string& distanceMethod);

/*
 * KMEANtestDictionaryEvaluation
 * 
 * Tests and evaluates the K-Means dictionary by generating visual word maps and computing accuracy.
 * Runs multiple evaluations with different k values for k-NN classification.
 */
void KMEANtestDictionaryEvaluation();

/*
 * DBSCANtestDictionaryEvaluation
 * 
 * Tests and evaluates the DBSCAN dictionary by generating visual word maps and computing accuracy.
 * Runs multiple evaluations with different k values for k-NN classification.
 */
void DBSCANtestDictionaryEvaluation();

#endif // DICTIONARY_EVALUATION_H