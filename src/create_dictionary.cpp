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


#include <yaml-cpp/yaml.h>




// Assume you have these functions implemented:
// std::vector<cv::Mat> createFilterBank();




// cv::Mat get_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method) {
//     std::vector<cv::Mat> filterBank = createFilterBank();
//     int filterCount = (int)filterBank.size();
//     int totalPoints = alpha * (int)imgPaths.size();
//     int dim = 3 * filterCount;
//     cv::Mat pixelResponses(totalPoints, dim, CV_32F);

//     for (int r = 0; r < pixelResponses.rows; ++r) {
//         for (int c = 0; c < pixelResponses.cols; ++c) {
//             pixelResponses.at<float>(r, c) = 0.0f;
            
//         }
//     }

//     for (size_t i = 0; i < imgPaths.size(); i++) {
//         std::cout << "-- processing " << i+1 << "/" << imgPaths.size() << std::endl;

//         cv::Mat image = cv::imread("data/" + imgPaths[i]);
//         if (image.empty()) {
//             std::cerr << "Error loading image: " << imgPaths[i] << std::endl;
//             continue;
//         }

//         cv::Mat imageRGB;
//         cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

//         cv::Mat response = extractFilterResponses(imageRGB, filterBank);
//         // cv::Mat response = extractAndSaveFilterResponses(imageRGB, filterBank, "");


//         std::vector<cv::Point> points;
//         if (method == "Harris") {
//             // points = getHarrisPoints(imageRGB, alpha, 0.05);
//             points = getHarrisPoints(imageRGB, 500, 0.05);

//         } else {
//             std::cerr << "Invalid method\n";
//             return cv::Mat();
//         }

//         int rowOffset = (int)i * alpha;
//         int H = response.rows;
//         int W = response.cols;
//         int channels = response.channels();

//         // response expected to be CV_32F with dim channels.
//         // Access pixel data:
//         for (int j = 0; j < alpha; j++) {
//             cv::Point p = points[j];
//             // Extract feature vector at p
//             const float* pix_ptr = response.ptr<float>(p.y, p.x);
//             for (int d = 0; d < dim; d++) {
//                 // pixelResponses.at<float>(rowOffset + j, d) = pix_ptr[d];
//                 if (std::isnan(pix_ptr[d]) || std::isinf(pix_ptr[d])) {
//                     pixelResponses.at<float>(rowOffset + j, d) = 0.0f;
//                 } else {
//                     pixelResponses.at<float>(rowOffset + j, d) = pix_ptr[d];
//                 }
//             }
//         }
//     }
//     for (int r = 0; r < pixelResponses.rows; ++r) {
//         for (int c = 0; c < pixelResponses.cols; ++c) {
//             float val = pixelResponses.at<float>(r, c);
//             if (std::isnan(val) || std::isinf(val)) {
//                 std::cerr << "Invalid value detected at (" << r << ", " << c << "): " << val << std::endl;
//                 // Replace with a default value or remove the row
//                 pixelResponses.at<float>(r, c) = 0.0f; // Example: set to zero
//             }
//         }
//     }

//     // Run KMeans
//     cv::Mat labels, centers;
//     cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 1e-5);
//     cv::kmeans(pixelResponses, K, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);

//     return centers; // KxD
// }


cv::Mat get_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method) {
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







void save_dictionary(const cv::Mat& dictionary, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "dictionary" << dictionary;
    fs.release();
    std::cout << "Dictionary saved to " << filename << std::endl;
}

// void save_dictionary(const cv::Mat& dictionary, const std::string& filename) {
//   std::stringstream yaml_stream;
//   yaml_stream << YAML::BeginMap;
//   yaml_stream << YAML::Key << "dictionary";
//   yaml_stream << YAML::BeginMap;
//   yaml_stream << YAML::Key << "rows";
//   yaml_stream << dictionary.rows;
//   yaml_stream << YAML::Key << "cols";
//   yaml_stream << dictionary.cols;
//   yaml_stream << YAML::Key << "data";
//   yaml_stream << YAML::BeginSeq;

//   // Iterate through each element in the dictionary matrix
//   for (int i = 0; i < dictionary.rows; ++i) {
//     for (int j = 0; j < dictionary.cols; ++j) {
//       yaml_stream << dictionary.at<float>(i, j);
//     }
//   }

//   yaml_stream << YAML::EndSeq;
//   yaml_stream << YAML::EndMap;
//   yaml_stream << YAML::EndMap;

//   std::ofstream yaml_file(filename);
//   yaml_file << yaml_stream.str();
//   yaml_file.close();
//   std::cout << "Dictionary saved to " << filename << " (YAML)" << std::endl;
// }








