#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::vector<cv::Mat> createFilterBank() {
    std::vector<cv::Mat> filterBank;
    std::vector<double> scales = {1, 2, 4};
    // std::vector<double> scales = {1};


    for (double scale : scales) {
        int size = 2 * std::ceil(scale * 2.5) + 1;  // Kernel size

        // Gaussian filter
        cv::Mat gaussianKernel = cv::getGaussianKernel(size, scale, CV_64F);
        cv::Mat gaussianFilter = gaussianKernel * gaussianKernel.t();
        filterBank.push_back(gaussianFilter);

        // Laplacian of Gaussian (LoG)
        cv::Mat logFilter = gaussianFilter.clone();
        cv::Laplacian(gaussianFilter, logFilter, CV_64F);
        filterBank.push_back(logFilter);

        // X Gradient
        cv::Mat sobelX;
        cv::Sobel(gaussianFilter, sobelX, CV_64F, 1, 0);
        filterBank.push_back(sobelX);

        // Y Gradient
        cv::Mat sobelY;
        cv::Sobel(gaussianFilter, sobelY, CV_64F, 0, 1);
        filterBank.push_back(sobelY);
    }

    return filterBank;
}