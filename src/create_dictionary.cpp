#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"
#include "include/getVisualWords.h"

#include "include/createFilterBank.h"


#include <yaml-cpp/yaml.h>

#include "include/dbscan.h"

#include <chrono>

// Parallelization Libraries

#include <omp.h>





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


cv::Mat get_dictionary_dbscan(const std::vector<std::string> &imgPaths, int alpha, double eps, int minSamples) {
    // Create filter bank and set up dimensions
    std::vector<cv::Mat> filterBank = createFilterBank();
    int filterCount = (int)filterBank.size();
    int dim = 3 * filterCount;
    int totalPoints = alpha * (int)imgPaths.size();

    if (totalPoints == 0) {
        std::cerr << "No points collected. Check alpha or image paths." << std::endl;
        return cv::Mat();
    }

    cv::Mat pixelResponses(totalPoints, dim, CV_32F, cv::Scalar(0));
    int currentRow = 0;

    // Collect pixel responses from images
    auto start1 = std::chrono::high_resolution_clock::now();

    // #pragma omp parallel
    for (const std::string& imgPath : imgPaths) {
        std::cout << "PIXEL RESPONSE For image: " << imgPath << std::endl; 

        cv::Mat image = cv::imread(imgPath);
        if (image.empty()) {
            std::cerr << "Unable to load image: " << imgPath << std::endl;
            continue;
        }

        cv::Mat imageRGB;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
        cv::Mat response = extractFilterResponsesOpenMP(imageRGB, filterBank, imgPath);

        // Get Harris points
        std::vector<cv::Point> points = getHarrisPoints(imageRGB, alpha, 0.05);
        if ((int)points.size() < alpha) {
            std::cerr << "Insufficient Harris points in image: " << imgPath << std::endl;
            continue;
        }

        // Store responses
        for (int j = 0; j < alpha; j++) {
            cv::Point p = points[j];
            const float* pix_ptr = response.ptr<float>(p.y, p.x);
            for (int d = 0; d < dim; d++) {
                pixelResponses.at<float>(currentRow + j, d) = pix_ptr[d];
            }
        }
        currentRow += alpha;
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
    // Output the time taken
    std::cout << "Time taken: " << duration1 << " microseconds" << std::endl;

    std::cout << "Finish Gathering PixelResponses" << std::endl;



    // Adjust pixelResponses if needed
    pixelResponses = pixelResponses.rowRange(0, currentRow);

    // Convert to double
    cv::Mat data;
    pixelResponses.convertTo(data, CV_64F);

    // Run DBSCAN
    std::cout << "Running DBSCAN with eps=" << eps << " and minSamples=" << minSamples << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // std::vector<int> labels = dbscan_OpenMP(data, eps, minSamples);
    // std::vector<int> labels = dbscan(data, eps, minSamples);
    std::vector<int> labels = dbscan_Sequential(data, eps, minSamples);



    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // Output the time taken
    std::cout << "Time taken: " << duration << " microseconds" << std::endl;

    std::cout << "Finish DBSCAN" << std::endl;

    int maxLabel = -1;
    for (int lbl : labels) {
        if (lbl > maxLabel) maxLabel = lbl;
    }
    int numClusters = maxLabel + 1;

    if (numClusters == 0) {
        std::cerr << "DBSCAN found no clusters. Consider adjusting eps or minSamples." << std::endl;
        return cv::Mat();
    }

    // Compute cluster centers (means)
    cv::Mat means = cv::Mat::zeros(numClusters, dim, CV_64F);
    std::vector<int> clusterCounts(numClusters, 0);

    for (int i = 0; i < data.rows; i++) {
        int c = labels[i];
        if (c >= 0) {
            clusterCounts[c]++;
            for (int d = 0; d < dim; d++) {
                means.at<double>(c, d) += data.at<double>(i, d);
            }
        }
    }

    for (int c = 0; c < numClusters; c++) {
        if (clusterCounts[c] > 0) {
            for (int d = 0; d < dim; d++) {
                means.at<double>(c, d) /= clusterCounts[c];
            }
        } else {
            // Shouldn't happen if c is a valid cluster, but just in case
            means.row(c).setTo(0);
        }
    }

    // Save dictionary to a yml file
    std::string outputPath = "../dbscan_dictionary.yml";
    {
        cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open " << outputPath << " for writing." << std::endl;
        } else {
            fs << "means" << means; 
            fs.release();
            std::cout << "DBSCAN iteration 6 dictionary saved to: " << outputPath << std::endl;
        }
    }

    return means;
}




void save_dictionary(const cv::Mat& dictionary, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "dictionary" << dictionary;
    fs.release();
    std::cout << "Dictionary saved to " << filename << std::endl;
}

void saveGMM(const cv::Mat& means, const cv::Mat& weights, const std::vector<cv::Mat>& covariances, const std::string& outputPath) {
    cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open file for writing GMM parameters: " << outputPath << std::endl;
        return;
    }

    // Save means
    fs << "means" << means;

    // Save weights
    fs << "weights" << weights;

    // Save covariances
    fs << "covariances" << "[";
    for (const auto& cov : covariances) {
        fs << cov;
    }
    fs << "]";

    fs.release();
    std::cout << "GMM parameters saved to: " << outputPath << std::endl;
}




// cv::Mat get_gmm_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K) {
//     std::vector<cv::Mat> filterBank = createFilterBank();
//     int filterCount = (int)filterBank.size();
//     int dim = 3 * filterCount; // Number of filter responses
//     int totalPoints = alpha * imgPaths.size();

//     if (totalPoints < K) {
//         std::cerr << "Error: Insufficient total points for clustering. Increase alpha or reduce K." << std::endl;
//         return cv::Mat();
//     }

//     cv::Mat pixelResponses(totalPoints, dim, CV_32F, cv::Scalar(0));
//     int currentRow = 0;

//     for (const std::string& imgPath : imgPaths) {
//         std::cout << "Collecting Pixel Responses for: " << imgPath << std::endl; 

//         cv::Mat image = cv::imread("data/" + imgPath); // Assuming category is in Training directory
//         if (image.empty()) {
//             std::cerr << "Error: Unable to load image " << imgPath << std::endl;
//             continue;
//         }

//         // Convert to RGB and extract filter responses
//         cv::Mat imageRGB;
//         cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
//         cv::Mat response = extractFilterResponses(imageRGB, filterBank);

//         // Get Harris points
//         std::vector<cv::Point> points = getHarrisPoints(imageRGB, alpha, 0.05);
//         if (points.size() < alpha) {
//             std::cerr << "Insufficient Harris points in image: " << imgPath << std::endl;
//             continue;
//         }

//         // Collect pixel responses at Harris points
//         for (int j = 0; j < alpha; j++) {
//             cv::Point p = points[j];
//             const float* pix_ptr = response.ptr<float>(p.y, p.x);
//             for (int d = 0; d < dim; d++) {
//                 pixelResponses.at<float>(currentRow + j, d) = pix_ptr[d];
//             }
//         }
//         currentRow += alpha; // Increment row counter
//     }

//     // Resize to match the actual number of collected points
//     pixelResponses = pixelResponses.rowRange(0, currentRow);

//     // Convert pixel responses to CV_64F for consistency
//     cv::Mat pixelResponses64F;
//     pixelResponses.convertTo(pixelResponses64F, CV_64F);


//     // Initialize GMM parameters using K-Means
//     std::cout << "Collected Pixel Responses, Running Kmeans" << std::endl;

//     cv::Mat labels, centers;
//     cv::kmeans(pixelResponses, K, labels, 
//                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-6), 
//                5, cv::KMEANS_PP_CENTERS, centers);

//     cv::Mat means;
//     centers.convertTo(means, CV_64F);
//     std::vector<cv::Mat> covariances(K, cv::Mat::eye(dim, dim, CV_64F));
//     cv::Mat weights = cv::Mat::ones(1, K, CV_64F) / K; // Uniform initial weights

//     // Compute initial covariances and weights
//     for (int k = 0; k < K; k++) {
//         int clusterSize = 0;
//         cv::Mat clusterVariance = cv::Mat::zeros(dim, dim, CV_64F);

//         for (int i = 0; i < labels.rows; i++) {
//             if (labels.at<int>(i) == k) {
//                 clusterSize++;
//                 cv::Mat diff = pixelResponses64F.row(i).t() - means.row(k).t();
//                 clusterVariance += diff * diff.t();
//             }
//         }

//         if (clusterSize > 0) {
//             weights.at<double>(0, k) = static_cast<double>(clusterSize) / labels.rows;
//             covariances[k] = clusterVariance / clusterSize + cv::Mat::eye(dim, dim, CV_64F) * 1e-6; // Stronger regularization
//         } else {
//             std::cout << "ENTERED EMPTY CLUSTER" << std::endl;
//             weights.at<double>(0, k) = 1.0 / K;
//             covariances[k] = cv::Mat::eye(dim, dim, CV_64F) * 1e-5; // Handle empty clusters
//         }
//     }




//     std::cout << "Initial Mean: " << means << std::endl;
//     std::cout << "Initial Covariance: " << covariances[0] << std::endl;
//     std::cout << "Initial Weights: " << weights << std::endl;
        



//     // Train GMM
//     // trainGMM(pixelResponses64F, K, 10, 1e-4, means, covariances, weights);
//     // trainGMM(pixelResponses64F, K, 100, 1e-3, means, covariances, weights);
//     trainGMM_Simple(pixelResponses, K, means, covariances, weights);


//     // Save the GMM for this category
//     saveGMM(means, weights, covariances, "gmm_airport_terminal.yml");

//     return means;
// }