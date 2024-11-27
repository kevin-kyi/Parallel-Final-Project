#include "filters.h"
#include <opencv2/opencv.hpp>
#include <vector>

// Function to extract filter responses for an image
std::vector<std::vector<float>> extractFilterResponses(const cv::Mat& image) {
    // Ensure the input image is valid
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty!");
    }

    // Convert the image to CIE Lab color space for better feature extraction
    cv::Mat labImage;
    cv::cvtColor(image, labImage, cv::COLOR_BGR2Lab);

    // Prepare to store the filter responses
    int rows = labImage.rows;
    int cols = labImage.cols;
    std::vector<std::vector<float>> featureResponses(rows * cols);

    // Define the filter kernels (adjust parameters as needed)
    cv::Mat gaussian, laplacian;

    // Apply Gaussian blur on each channel of the Lab image
    for (int channel = 0; channel < 3; ++channel) {
        cv::GaussianBlur(labImage, gaussian, cv::Size(3, 3), 1.0);

        // Apply Laplacian filter
        cv::Mat singleChannel = labImage.clone();
        cv::extractChannel(labImage, singleChannel, channel); // Extract the specific channel
        cv::Laplacian(singleChannel, laplacian, CV_32F);

        // Iterate over pixels to store filter responses
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                int idx = row * cols + col; // Flattened index
                if (featureResponses[idx].empty()) {
                    featureResponses[idx].resize(6); // Three filters * 2
                }
                // Add Gaussian and Laplacian filter responses for this channel
                featureResponses[idx][channel * 2] = gaussian.at<cv::Vec3b>(row, col)[channel];
                featureResponses[idx][channel * 2 + 1] = laplacian.at<float>(row, col);
            }
        }
    }

    return featureResponses;
}
