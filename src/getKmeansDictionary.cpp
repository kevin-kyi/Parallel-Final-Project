#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>



cv::Mat getDictionary(const std::vector<std::string>& imgPaths, int alpha, int K, const std::string& method) {
    std::vector<cv::Mat> filterBank; // Replace with your filter bank creation
    int featureSize = filterBank.size() * 3; // Assuming 3 channels (Lab color space)

    cv::Mat pixelResponses(alpha * imgPaths.size(), featureSize, CV_32F);

    for (size_t i = 0; i < imgPaths.size(); ++i) {
        std::cout << "-- processing " << (i + 1) << "/" << imgPaths.size() << std::endl;
        cv::Mat image = cv::imread(imgPaths[i]);
        cv::cvtColor(image, image, cv::COLOR_BGR2Lab);

        auto response = extractFilterResponses(image, filterBank);
        std::vector<cv::Point> points;
        
        // if (method == "Random") {
        //     points = getRandomPoints(image, alpha);
        // } else if (method == "Harris") {
        //     points = getHarrisPoints(image, alpha, 0.05);
        // } else {
        //     throw std::invalid_argument("Method must be 'Random' or 'Harris'");
        // }
        if (method == "Harris") {
            points = getHarrisPoints(image, alpha);

        } else {
            throw std::invalid_argument("Method must be 'Harris'");
        }


        for (int j = 0; j < alpha; ++j) {
            int x = points[j].x, y = points[j].y;
            // Flatten the response at (x, y) and store in pixelResponses
            cv::Mat featureVector; // Extract feature vector at (x, y)
            pixelResponses.row(i * alpha + j) = featureVector.reshape(1, 1);
        }
    }

    cv::Mat labels, dictionary;
    cv::kmeans(pixelResponses, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-4),
               3, cv::KMEANS_PP_CENTERS, dictionary);

    return dictionary;
}