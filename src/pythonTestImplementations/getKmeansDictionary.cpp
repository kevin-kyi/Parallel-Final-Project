#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>




cv::Mat get_dictionary(const std::vector<std::string> &imgPaths, int alpha, int K, const std::string &method) {
    std::vector<cv::Mat> filterBank = createFilterBank();

    int filterCount = (int)filterBank.size();
    int totalPoints = alpha * (int)imgPaths.size();
    int dim = 3 * filterCount; // Each pixel response has 3*filterCount dimensions.

    // Prepare a matrix for all pixel responses: totalPoints x dim (CV_32F)
    cv::Mat pixelResponses(totalPoints, dim, CV_32F);

    for (size_t i = 0; i < imgPaths.size(); i++) {
        std::cout << "-- processing " << i+1 << "/" << imgPaths.size() << std::endl;

        cv::Mat image = cv::imread("../data/" + imgPaths[i]);
        if (image.empty()) {
            std::cerr << "Error loading image: " << imgPaths[i] << std::endl;
            continue;
        }

        // Convert BGR to RGB if needed
        cv::Mat imageRGB;
        cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);

        cv::Mat response = extractFilterResponses(imageRGB, filterBank);

        // Get points
        std::vector<cv::Point> points;
        if (method == "Random") {
            points = getRandomPoints(imageRGB, alpha);
        } else if (method == "Harris") {
            points = getHarrisPoints(imageRGB, alpha, 0.05);
        } else {
            std::cerr << "Method must be 'Random' or 'Harris'" << std::endl;
            return cv::Mat();
        }

        // For each selected point, extract its response vector
        // response is expected to be HxWx(3*filterCount). Let's assume it's stored in a known format.
        int rowOffset = (int)i * alpha;
        int H = response.size().height;
        int W = response.size().width;

        // response should have dimension: H x W x dim. 
        // If stored as 3*filterCount channels, ensure indexing is correct.
        // Let's assume response is CV_32F and has shape [H, W, dim], stored as HxW with dim-channels.
        // Access as response.at<cv::Vec<float, dim>>(y, x)
        for (int j = 0; j < alpha; j++) {
            cv::Point p = points[j];
            int y = p.y;
            int x = p.x;
            cv::Vec<float, 60> vals; // adjust 60 if dim differs

            // Extract the pixel's filter response. If dim = 3*filterCount,
            // and we have response with dim channels, we can do:
            std::vector<float> feature(dim);
            cv::Vec<float, 3* /*filterCount*/> pixelRes = response.at<cv::Vec<float, 3* /*filterCount*/>>(y, x);
            for (int d = 0; d < dim; d++) {
                pixelResponses.at<float>(rowOffset + j, d) = pixelRes[d];
            }
        }
    }

    // Run KMeans on pixelResponses
    cv::Mat labels;
    cv::Mat centers;
    // K-means parameters
    int attempts = 3;
    cv::TermCriteria criteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 100, 1e-4);
    cv::kmeans(pixelResponses, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
    // centers will be Kxdim

    return centers;
}

