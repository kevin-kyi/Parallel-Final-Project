
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "include/getVisualWords.h"
#include "include/filters.h"
#include "include/create_dictionary.h"

#include "include/createFilterBank.h"



cv::Mat getVisualWords(const cv::Mat& image, const cv::Mat& dictionary, const std::vector<cv::Mat>& filterBank) {
    // Extract filter responses
    cv::Mat filterResponses = extractFilterResponses(image, filterBank);

    // // Reshape filter responses to [numPixels x numChannels]
    int numPixels = filterResponses.rows * filterResponses.cols; // Total number of pixels
    int numChannels = filterResponses.channels();               // Number of channels (filters * color channels)

    // filterResponses = filterResponses.reshape(numChannels, numPixels); // Reshape to [numPixels x numChannels]

    // Debug output for dimensions
    std::cout << "Extracted Filter Responses: Rows=" << filterResponses.rows
              << ", Cols=" << filterResponses.cols
              << ", Channels=" << numChannels << " Type: " << filterResponses.type() << std::endl;

    std::cout << "DICTIONARY: Rows=" << dictionary.rows
              << ", Cols=" << dictionary.cols
              << ", Channels=" << dictionary.channels() << " Type: " << dictionary.type() << std::endl;
    

    if (filterResponses.cols != dictionary.cols) {
        throw std::runtime_error("Feature dimensions mismatch between filter responses and dictionary.");
    }

    // Compute distances between filter responses and dictionary entries
    cv::Mat distances;
    cv::batchDistance(filterResponses, dictionary, distances, CV_32F, cv::noArray(), cv::NORM_L2);

    std::cout << "Distances: Rows=" << distances.rows << ", Cols=" << distances.cols << std::endl;


    // Map each pixel to its closest visual word
    cv::Mat wordMap(image.rows, image.cols, CV_32SC1);
    for (int i = 0; i < distances.rows; ++i) {
        double minVal;
        cv::Point minIdx;
        cv::minMaxLoc(distances.row(i), &minVal, nullptr, &minIdx, nullptr);

        int row = i / image.cols;
        int col = i % image.cols;

        if (row < image.rows && col < image.cols) {
            wordMap.at<int>(row, col) = minIdx.x;
        } else {
            std::cerr << "Out of bounds pixel mapping: (" << row << ", " << col << ")\n";
        }
    }

    return wordMap;
}

// cv::Mat getVisualWordsGMM(const cv::Mat& image,
//                           const cv::Mat& means,
//                           const std::vector<cv::Mat>& covariances,
//                           const cv::Mat& weights,
//                           const std::vector<cv::Mat>& filterBank) {
//     // Extract filter responses
//     cv::Mat filterResponses = extractFilterResponses(image, filterBank);

//     int numPixels = filterResponses.rows * filterResponses.cols;
//     int numChannels = means.cols;



//     std::cout << "Extracted Filter Responses: Rows=" << filterResponses.rows
//               << ", Cols=" << filterResponses.cols
//               << ", Channels=" << numChannels << " Type: " << filterResponses.type() << std::endl;

//     std::cout << "MEAN: Rows=" << means.rows
//               << ", Cols=" << means.cols
//               << ", Channels=" << means.channels() << " Type: " << means.type() << std::endl;

//     // Ensure the GMM dictionary matches the filter response dimensions
//     // if (filterResponses.channels() != numChannels) {
//     //     throw std::runtime_error("Feature dimensions mismatch between filter responses and GMM dictionary.");
//     // }

//     // Reshape filter responses to [numPixels x numChannels]
//     cv::Mat reshapedResponses = filterResponses.reshape(1, numPixels); // [numPixels x numChannels]
//     reshapedResponses.convertTo(reshapedResponses, CV_64F); // Ensure consistency with GMM (CV_64F)

//     // Prepare word map
//     cv::Mat wordMap(image.rows, image.cols, CV_32SC1);

//     // Precompute inverse covariances and log-determinants
//     std::vector<cv::Mat> invCovariances(covariances.size());
//     std::vector<double> logDetCov(covariances.size());
//     for (int k = 0; k < covariances.size(); ++k) {
//         cv::Mat invCov;
//         double detCov = cv::determinant(covariances[k]);
//         if (detCov <= 1e-12) {
//             // Regularize covariance if determinant is too small
//             invCov = covariances[k] + cv::Mat::eye(covariances[k].rows, covariances[k].cols, CV_64F) * 1e-6;
//             detCov = cv::determinant(invCov);
//         } else {
//             invCov = covariances[k];
//         }
//         cv::invert(invCov, invCovariances[k], cv::DECOMP_CHOLESKY);
//         logDetCov[k] = std::log(detCov);
//     }

//     // Process each pixel
//     for (int i = 0; i < reshapedResponses.rows; ++i) {
//         cv::Mat pixel = reshapedResponses.row(i); // 1 x numChannels

//         // Compute likelihoods for all components
//         std::vector<double> likelihoods(covariances.size());
//         for (int k = 0; k < covariances.size(); ++k) {
//             double likelihood = gaussianDensity(pixel, means.row(k), covariances[k], logDetCov[k], invCovariances[k]);
//             likelihoods[k] = weights.at<double>(0, k) * likelihood;
//         }

//         // Find the component with the maximum likelihood
//         auto maxIt = std::max_element(likelihoods.begin(), likelihoods.end());
//         int bestComponent = std::distance(likelihoods.begin(), maxIt);

//         // Map the pixel to the visual word corresponding to the best component
//         int row = i / image.cols;
//         int col = i % image.cols;
//         wordMap.at<int>(row, col) = bestComponent;
//     }

//     return wordMap;
// }

