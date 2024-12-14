#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::vector<cv::Mat> createFilterBank() {
    std::vector<cv::Mat> filterBank;
    std::vector<double> scales = {1};  // Adjust scales as needed
    // std::vector<double> scales = {1, 2, 4, 8};  // Adjust scales as needed

    for (double scale : scales) {
        int size = 2 * std::ceil(scale * 2.5) + 1;  // Kernel size

        // Gaussian filter
        cv::Mat gaussianKernel = cv::getGaussianKernel(size, scale, CV_32F);  // Use CV_32F
        cv::Mat gaussianFilter = gaussianKernel * gaussianKernel.t();
        gaussianFilter.convertTo(gaussianFilter, CV_32F);  // Ensure CV_32F
        filterBank.push_back(gaussianFilter);

        // Laplacian of Gaussian (LoG)
        cv::Mat logFilter;
        cv::Laplacian(gaussianFilter, logFilter, CV_32F);  // Use CV_32F
        filterBank.push_back(logFilter);

        // X Gradient
        cv::Mat sobelX;
        cv::Sobel(gaussianFilter, sobelX, CV_32F, 1, 0);  // Use CV_32F
        filterBank.push_back(sobelX);

        // Y Gradient
        cv::Mat sobelY;
        cv::Sobel(gaussianFilter, sobelY, CV_32F, 0, 1);  // Use CV_32F
        filterBank.push_back(sobelY);
    }

    return filterBank;
}