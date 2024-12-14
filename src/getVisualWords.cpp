#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "include/visual_words.h"
#include "include/filters.h"
#include "include/create_dictionary.h"
#include "include/run_gmm.h"


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


// double gausDensity(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& cov, double weight) {
//     std::cout << "x size: " << x.size() << ", type: " << x.type() << std::endl;
//     std::cout << "mean size: " << mean.size() << ", type: " << mean.type() << std::endl;
//     std::cout << "cov size: " << cov.size() << ", type: " << cov.type() << std::endl;


//     int dim = x.rows;
//     cv::Mat diff = x - mean;

//     double det_cov = cv::determinant(cov);
//     if (det_cov <= 0) return 0;

//     cv::Mat inv_cov = cov.inv();
//     double norm_factor = weight / (std::pow(2 * M_PI, dim / 2.0) * std::sqrt(det_cov));
//     double exponent = -0.5 * cv::Mat(diff.t() * inv_cov * diff).at<double>(0, 0);

//     return norm_factor * std::exp(exponent);
// }

double gausDensity(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& cov) {
    // if (x.rows != mean.rows || x.cols != mean.cols) {
    //     throw std::runtime_error("Feature vector and mean dimensions do not match.");
    // }

    // if (cov.rows != cov.cols) {
    //     throw std::runtime_error("Covariance matrix must be square.");
    // }

    std::cout << "x size: " << x.size() << ", type: " << x.type() << std::endl;
    std::cout << "mean size: " << mean.size() << ", type: " << mean.type() << std::endl;

    cv::Mat diff = x - mean;

    // Compute determinant and inverse of the covariance matrix
    double detCov = cv::determinant(cov);
    if (detCov <= 0.0) {
        throw std::runtime_error("Covariance matrix determinant is non-positive.");
    }

    cv::Mat covInv = cov.inv();

    std::cout << "diff size: " << diff.size() << ", type: " << diff.type() << std::endl;
    std::cout << "covInv size: " << covInv.size() << ", type: " << covInv.type() << std::endl;

    double exponent = -0.5 * cv::Mat(diff.t() * covInv * diff).at<double>(0, 0);

    double normalizationFactor = std::pow(2 * CV_PI, cov.rows / 2.0) * std::sqrt(detCov);

    return std::exp(exponent) / normalizationFactor;
}




// GMM-based visual words computation
cv::Mat getVisualWordsGMM(const cv::Mat& image, const cv::Mat& means, 
                          const std::vector<cv::Mat>& covariances, const cv::Mat& weights) {

    // std::cout << "Image size: " << image.size() << ", Type: " << image.type() << std::endl;
    // std::cout << "Means size: " << means.size() << ", Type: " << means.type() << std::endl;
    // std::cout << "Weights size: " << weights.size() << ", Type: " << weights.type() << std::endl;

    // for (size_t i = 0; i < covariances.size(); i++) {
    //     std::cout << "Covariance " << i << " size: " << covariances[i].size() 
    //               << ", Type: " << covariances[i].type() << std::endl;
    // }

    // // Extract feature responses
    cv::Mat featureResponses = extractFilterResponses(image, createFilterBank());
    // featureResponses.convertTo(featureResponses, CV_64F);


    // std::cout << "Feature responses size: " << featureResponses.size() 
    //           << ", Type: " << featureResponses.type() << std::endl;

    // if (featureResponses.cols != means.cols) {
    //     throw std::runtime_error("Feature dimensions mismatch between GMM means and feature responses.");
    // }





    cv::Mat image32F;
    image.convertTo(image32F, CV_32F); // Ensure the image is in float format

    // int numPixels = image32F.rows * image32F.cols;
    // int K = means.rows; // Number of Gaussians
    // cv::Mat wordMap(image.rows, image.cols, CV_32SC1);

    // // For each pixel, compute responsibilities and assign to the most likely Gaussian
    // for (int i = 0; i < image.rows; i++) {
    //     for (int j = 0; j < image.cols; j++) {
    //         cv::Vec3f pixelVec = image32F.at<cv::Vec3f>(i, j);
    //         cv::Mat pixelMat(3, 1, CV_32F, &pixelVec); // Convert Vec3f to Mat

    //         double maxResp = -1.0;
    //         int bestComponent = -1;

    //         for (int k = 0; k < K; k++) {
    //             // double resp = gaussianDensity(pixelMat, means.row(k).t(), covariances[k], weights.at<double>(0, k));
    //             double resp = gausDensity(pixelMat, means.row(k).t(), covariances[k]);

    //             if (resp > maxResp) {
    //                 maxResp = resp;
    //                 bestComponent = k;
    //             }
    //         }

    //         wordMap.at<int>(i, j) = bestComponent;
    //     }
    // }



    int numPixels = featureResponses.rows;
    int K = means.rows; // Number of Gaussians
    cv::Mat wordMap(image.rows, image.cols, CV_32SC1);

    for (int p = 0; p < numPixels; p++) {
        cv::Mat pixelMat = featureResponses.row(p).t(); // Extract feature vector

        double maxResp = -1.0;
        int bestComponent = -1;

        for (int k = 0; k < K; k++) {
            double resp = gausDensity(pixelMat, means.row(k).t(), covariances[k]);

            if (resp > maxResp) {
                maxResp = resp;
                bestComponent = k;
            }
        }

        int row = p / image.cols;
        int col = p % image.cols;
        wordMap.at<int>(row, col) = bestComponent;
    }

    return wordMap;
}
