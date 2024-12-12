#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat buildFeatureMatrix(const cv::Mat& image, const std::vector<cv::Mat>& responses, const std::vector<cv::Point>& points) {
    int numFilters = responses.size();
    int numPoints = points.size();

    cv::Mat featureMatrix(numPoints, numFilters, CV_32F);

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < responses.size(); ++j) {
            featureMatrix.at<float>(i, j) = responses[j].at<float>(points[i].y, points[i].x);
        }
    }

    return featureMatrix;
}
