#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <filesystem>

#include <numeric>
#include "include/getHarrisPoints.h"


// Function to perform Harris corner detection
std::vector<cv::Point> getHarrisPoints(const cv::Mat& image, int alpha, double k) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Convert to grayscale if needed
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Normalize the image to [0, 1]
    grayImage.convertTo(grayImage, CV_32F, 1.0 / 255.0);

    // Compute gradients
    cv::Mat Ix, Iy;
    cv::Sobel(grayImage, Ix, CV_32F, 1, 0, 3);
    cv::Sobel(grayImage, Iy, CV_32F, 0, 1, 3);

    // Compute products of derivatives
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Apply Gaussian filter to derivatives
    cv::Mat Sxx, Syy, Sxy;
    cv::GaussianBlur(Ixx, Sxx, cv::Size(3, 3), 1);
    cv::GaussianBlur(Iyy, Syy, cv::Size(3, 3), 1);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(3, 3), 1);

    // Compute Harris response
    cv::Mat detM = Sxx.mul(Syy) - Sxy.mul(Sxy);
    cv::Mat traceM = Sxx + Syy;
    cv::Mat R = detM - k * traceM.mul(traceM);

    // Flatten the response matrix and sort to find top alpha responses
    std::vector<cv::Point> points;
    std::vector<float> flatR;
    R.reshape(1, 1).copyTo(flatR);
    std::vector<int> indices(flatR.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + alpha, indices.end(),
                      [&flatR](int i1, int i2) { return flatR[i1] > flatR[i2]; });

    for (int i = 0; i < alpha; i++) {
        int idx = indices[i];
        points.emplace_back(idx % R.cols, idx / R.cols);
    }

    return points;
}

// Function to save Harris points on an image
void saveHarrisPointsImage(const cv::Mat& image, const std::vector<cv::Point>& points, const std::string& outputPath) {
    cv::Mat outputImage = image.clone();
    for (const auto& point : points) {
        cv::circle(outputImage, point, 2, cv::Scalar(0, 255, 0), -1);
    }
    if (!cv::imwrite(outputPath, outputImage)) {
        throw std::runtime_error("Failed to save Harris points image: " + outputPath);
    }
    std::cout << "Saved Harris points image: " << outputPath << std::endl;
}