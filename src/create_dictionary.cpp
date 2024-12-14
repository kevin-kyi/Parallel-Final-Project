#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"
#include "include/visual_words.h"

#include "include/run_gmm.h"
#include "include/createFilterBank.h"


#include <yaml-cpp/yaml.h>

// #include <matplotlibcpp.h>
// #include "include/matplotlib-cpp/matplotlibcpp.h"
// namespace plt = matplotlibcpp;



cv::Mat get_kmeans_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method) {
    std::vector<cv::Mat> filterBank = createFilterBank();
    int filterCount = (int)filterBank.size();
    int dim = 3 * filterCount;
    int totalPoints = alpha * imgPaths.size();

    if (totalPoints < K) {
        std::cerr << "Error: Insufficient total points for clustering. Increase alpha or reduce K." << std::endl;
        return cv::Mat();
    }

    cv::Mat pixelResponses(totalPoints, dim, CV_32F, cv::Scalar(0));
    std::vector<std::string> problematicImages;

    for (size_t i = 0; i < imgPaths.size(); i++) {
        std::cout << "-- processing " << i + 1 << "/" << imgPaths.size() << std::endl;

        cv::Mat image = cv::imread("data/" + imgPaths[i]);
        if (image.empty()) {
            std::cerr << "Error loading image: " << imgPaths[i] << std::endl;
            problematicImages.push_back(imgPaths[i]);
            continue;
        }

        cv::Mat imageRGB;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

        cv::Mat response = extractFilterResponses(imageRGB, filterBank);

        std::vector<cv::Point> points = getHarrisPoints(imageRGB, alpha, 0.05);
        if (points.size() < alpha) {
            std::cerr << "Insufficient Harris points in image: " << imgPaths[i] << std::endl;
            problematicImages.push_back(imgPaths[i]);
            continue;
        }

        int rowOffset = i * alpha;
        for (int j = 0; j < alpha; j++) {
            cv::Point p = points[j];
            const float* pix_ptr = response.ptr<float>(p.y, p.x);
            for (int d = 0; d < dim; d++) {
                float value = pix_ptr[d];
                if (std::isnan(value) || std::isinf(value)) {
                    std::cerr << "Invalid value detected in image: " << imgPaths[i] << " at (" << p.y << ", " << p.x << ")\n";
                    problematicImages.push_back(imgPaths[i]);
                    break;
                }
                pixelResponses.at<float>(rowOffset + j, d) = value;
            }
        }
    }

    // Remove duplicates from problematicImages
    std::sort(problematicImages.begin(), problematicImages.end());
    problematicImages.erase(std::unique(problematicImages.begin(), problematicImages.end()), problematicImages.end());

    // Print all problematic images
    if (!problematicImages.empty()) {
        std::cerr << "Problematic images with invalid values:\n";
        for (const auto &img : problematicImages) {
            std::cerr << img << std::endl;
        }
    }

    std::cout << "PixelResponses size: " << pixelResponses.rows << " x " << pixelResponses.cols << std::endl;
    std::cout << "PixelResponses type: " << pixelResponses.type() << " (Expected: " << CV_32F << ")" << std::endl;



    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 500, 1e-4);
    cv::kmeans(pixelResponses, K, labels, criteria, 20, cv::KMEANS_PP_CENTERS, centers);

    std::cout << "Centers dimensions: " << centers.rows << " x " << centers.cols << std::endl;
    std::cout << "Centers type: " << centers.type() << " (Expected: " << CV_32F << ")" << std::endl;


    return centers;  // KxD
}



// void plotInitialGMMParams(const cv::Mat& means, const cv::Mat& weights, const std::vector<cv::Mat>& covariances) {
//     // Convert means to vector of points for plotting
//     std::vector<double> meanX, meanY;
//     for (int i = 0; i < means.rows; i++) {
//         meanX.push_back(means.at<double>(i, 0)); // First dimension
//         meanY.push_back(means.at<double>(i, 1)); // Second dimension
//     }

//     // Plot the means as a scatter plot
//     plt::figure();
//     plt::scatter(meanX, meanY);
//     plt::title("Initial Means");
//     plt::xlabel("Dimension 1");
//     plt::ylabel("Dimension 2");
//     plt::save("initial_means.png");

//     // Plot the weights as a bar chart
//     std::vector<double> weightValues;
//     for (int i = 0; i < weights.cols; i++) {
//         weightValues.push_back(weights.at<double>(0, i));
//     }

//     plt::figure();
//     plt::bar(weightValues);
//     plt::title("Initial Weights");
//     plt::xlabel("Cluster Index");
//     plt::ylabel("Weight");
//     plt::save("initial_weights.png");

//     // Plot covariance matrices as heatmaps
//     for (size_t k = 0; k < covariances.size(); k++) {
//         cv::Mat cov = covariances[k];
//         cv::Mat covNorm;
//         cv::normalize(cov, covNorm, 0, 1, cv::NORM_MINMAX); // Normalize for better visualization

//         // Convert covariance matrix to a std::vector for plotting
//         std::vector<double> covData(covNorm.begin<double>(), covNorm.end<double>());

//         plt::figure();
//         plt::imshow(covData, cov.rows, cov.cols, {{"cmap", "hot"}});
//         plt::title("Covariance Matrix " + std::to_string(k));
//         plt::save("covariance_matrix_" + std::to_string(k) + ".png");
//     }
// }







cv::Mat get_gmm_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method) {
    std::vector<cv::Mat> filterBank = createFilterBank();
    int filterCount = (int)filterBank.size();
    int dim = 3 * filterCount;
    int totalPoints = alpha * imgPaths.size();

    if (totalPoints < K) {
        std::cerr << "Error: Insufficient total points for clustering. Increase alpha or reduce K." << std::endl;
        return cv::Mat();
    }

    cv::Mat pixelResponses(totalPoints, dim, CV_32F, cv::Scalar(0));
    std::vector<std::string> problematicImages;

    for (size_t i = 0; i < imgPaths.size(); i++) {
        std::cout << "-- processing " << i + 1 << "/" << imgPaths.size() << std::endl;

        cv::Mat image = cv::imread("data/" + imgPaths[i]);
        if (image.empty()) {
            std::cerr << "Error loading image: " << imgPaths[i] << std::endl;
            problematicImages.push_back(imgPaths[i]);
            continue;
        }

        cv::Mat imageRGB;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

        cv::Mat response = extractFilterResponses(imageRGB, filterBank);

        std::vector<cv::Point> points = getHarrisPoints(imageRGB, alpha, 0.05);
        if (points.size() < alpha) {
            std::cerr << "Insufficient Harris points in image: " << imgPaths[i] << std::endl;
            problematicImages.push_back(imgPaths[i]);
            continue;
        }

        int rowOffset = i * alpha;
        for (int j = 0; j < alpha; j++) {
            cv::Point p = points[j];
            const float* pix_ptr = response.ptr<float>(p.y, p.x);
            for (int d = 0; d < dim; d++) {
                float value = pix_ptr[d];
                if (std::isnan(value) || std::isinf(value)) {
                    std::cerr << "Invalid value detected in image: " << imgPaths[i] << " at (" << p.y << ", " << p.x << ")\n";
                    problematicImages.push_back(imgPaths[i]);
                    break;
                }
                pixelResponses.at<float>(rowOffset + j, d) = value;
            }
        }
    }

    // Remove duplicates from problematicImages
    std::sort(problematicImages.begin(), problematicImages.end());
    problematicImages.erase(std::unique(problematicImages.begin(), problematicImages.end()), problematicImages.end());

    // Print all problematic images
    if (!problematicImages.empty()) {
        std::cerr << "Problematic images with invalid values:\n";
        for (const auto &img : problematicImages) {
            std::cerr << img << std::endl;
        }
    }

    std::cout << "PixelResponses size: " << pixelResponses.rows << " x " << pixelResponses.cols << std::endl;
    std::cout << "PixelResponses type: " << pixelResponses.type() << " (Expected: " << CV_32F << ")" << std::endl;




    cv::Mat pixelResponses64F;
    pixelResponses.convertTo(pixelResponses64F, CV_64F);

    cv::Mat mean, stddev;
    cv::meanStdDev(pixelResponses64F, mean, stddev);
    pixelResponses64F = (pixelResponses64F - mean) / stddev; // Normalize to zero mean, unit variance

    // Initialize GMM parameters using K-Means++
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(pixelResponses, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-6), 
            5, cv::KMEANS_PP_CENTERS, centers);

    cv::Mat means;
    centers.convertTo(means, CV_64F);
    std::vector<cv::Mat> covariances(K, cv::Mat::eye(dim, dim, CV_64F));
    cv::Mat weights = cv::Mat::ones(1, K, CV_64F) / K; // Uniform initial weights

    cv::RNG rng; // Random generator for handling empty clusters

    // Compute initial weights and covariances
    for (int k = 0; k < K; k++) {
        int clusterSize = 0;
        cv::Mat clusterVariance = cv::Mat::zeros(dim, dim, CV_64F);

        for (int i = 0; i < pixelResponses64F.rows; i++) {
            if (labels.at<int>(i) == k) {
                clusterSize++;
                cv::Mat diff = pixelResponses64F.row(i) - means.row(k);
                clusterVariance += diff.t() * diff;
            }
        }

        if (clusterSize > 0) {
            // Set initial weight
            weights.at<double>(0, k) = static_cast<double>(clusterSize) / pixelResponses64F.rows;

            // Compute covariance with regularization
            covariances[k] = clusterVariance / clusterSize + cv::Mat::eye(dim, dim, CV_64F) * 1e-4; // Stronger regularization
        } else {
            std::cerr << "Cluster " << k << " is empty. Reinitializing...\n";

            // Reinitialize mean, weight, and covariance for empty cluster
            means.row(k) = pixelResponses64F.row(rng.uniform(0, pixelResponses64F.rows));
            weights.at<double>(0, k) = 1.0 / K; // Redistribute weight evenly
            covariances[k] = cv::Mat::eye(dim, dim, CV_64F) * 1e-3; // Small but valid covariance
        }
    }


    // plotInitialGMMParams(means, weights, covariances);


    // cv::Mat labels, centers;
    // cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 500, 1e-4);
    // cv::kmeans(pixelResponses, K, labels, criteria, 20, cv::KMEANS_PP_CENTERS, centers);



    // // *************** GMM PORTION ***************

    // cv::Mat pixelResponses64F;
    // pixelResponses.convertTo(pixelResponses64F, CV_64F);

    // // Initialize GMM parameters using K-Means results
    // cv::Mat means;
    // centers.convertTo(means, CV_64F);
    // std::vector<cv::Mat> covariances(K, cv::Mat::eye(dim, dim, CV_64F));
    // cv::Mat weights = cv::Mat::ones(1, K, CV_64F) / K;  // Uniform initial weights

    // cv::RNG rng;  // Random generator for handling empty clusters

    // // Compute initial weights and covariances
    // for (int k = 0; k < K; k++) {
    //     int clusterSize = 0;
    //     cv::Mat clusterVariance = cv::Mat::zeros(dim, dim, CV_64F);

    //     for (int i = 0; i < pixelResponses.rows; i++) {
    //         if (labels.at<int>(i) == k) {
    //             clusterSize++;
    //             cv::Mat diff = pixelResponses64F.row(i) - means.row(k);
    //             clusterVariance += diff.t() * diff;
    //         }
    //     }

    //     if (clusterSize > 0) {
    //         // weights.at<double>(0, k) = static_cast<double>(clusterSize) / pixelResponses.rows;
    //         covariances[k] = clusterVariance / clusterSize + cv::Mat::eye(dim, dim, CV_64F) * 1e-3;  // Regularization
    //     } else {
    //         std::cerr << "Cluster " << k << " is empty. Reinitializing...\n";
    //         // means.row(k) = pixelResponses64F.row(rng.uniform(0, pixelResponses64F.rows));
    //         // weights.at<double>(0, k) = 1.0 / K;  // Redistribute weight
    //         covariances[k] = cv::Mat::eye(dim, dim, CV_64F) * 1e-3;  // Small variance
    //     }

    // }


    std::cout << "Initial Means:\n" << means << std::endl;
    std::cout << "Initial Weights:\n" << weights << std::endl;
    std::cout << "Covariance:\n" << covariances[0] << std::endl;

    // Run GMM
    std::cout << "Starting GMM training..." << std::endl;
    trainGMM(pixelResponses64F, K, 100, 1e-6, means, covariances, weights);
    std::cout << "GMM training completed." << std::endl;


    // Save GMM parameters to .yml
    cv::FileStorage fs("gmm_dictionary.yml", cv::FileStorage::WRITE);
    fs << "means" << means;
    fs << "weights" << weights;
    fs << "covariances" << "[";
    for (const auto& cov : covariances) {
        fs << cov;
    }
    fs << "]";
    fs.release();

    std::cout << "GMM dictionary saved to gmm_dictionary.yml" << std::endl;
    return means;

}

















void save_dictionary(const cv::Mat& dictionary, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "dictionary" << dictionary;
    fs.release();
    std::cout << "Dictionary saved to " << filename << std::endl;
}

