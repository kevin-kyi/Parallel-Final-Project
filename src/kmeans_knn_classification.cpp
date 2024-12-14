#include <filesystem>
#include <opencv2/opencv.hpp>




// RETURNS WORD MAPS UNDER EACH THE TRAINING AND TESTING DIRS
std::vector<std::pair<cv::Mat, int>> loadWordMaps(const std::string& baseDirectory, const std::vector<std::string>& categoryDirs, const std::vector<int>& labels) {
    std::vector<std::pair<cv::Mat, int>> wordMaps;

    for (size_t i = 0; i < categoryDirs.size(); ++i) {
        const std::string categoryPath = baseDirectory + "/" + categoryDirs[i];
        const int categoryLabel = labels[i];

        for (const auto& file : std::filesystem::directory_iterator(categoryPath)) {
            if (file.path().extension() == ".yml") { // Only load YML files
                cv::Mat wordMap;
                cv::FileStorage fs(file.path().string(), cv::FileStorage::READ);
                fs["wordMap"] >> wordMap;
                fs.release();

                if (!wordMap.empty()) {
                    wordMaps.emplace_back(wordMap, categoryLabel);
                } else {
                    std::cerr << "Warning: Empty word map at " << file.path() << std::endl;
                }
            }
        }
    }
    return wordMaps;
}



// CREATES HISTOGRAM OF WORD MAP VALUES
cv::Mat getImageFeatures(const cv::Mat& wordMap, int dictionarySize) {
    cv::Mat histogram = cv::Mat::zeros(1, dictionarySize, CV_32F);

    for (int i = 0; i < wordMap.rows; ++i) {
        for (int j = 0; j < wordMap.cols; ++j) {
            int word = wordMap.at<int>(i, j);
            histogram.at<float>(0, word)++;
        }
    }

    // L1 Normalization
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}



// COMPUTES DISTANCE BETWEEN TWO HISTOGRAMS TO FIND BEST FIT
float getImageDistance(const cv::Mat& hist1, const cv::Mat& hist2, const std::string& method) {
    if (method == "euclidean") {
        return cv::norm(hist1, hist2, cv::NORM_L2);
    } else if (method == "chi2") {
        cv::Mat diff = hist1 - hist2;
        cv::Mat sum = hist1 + hist2 + 1e-10; // Avoid division by zero
        cv::Mat chi2 = diff.mul(diff) / sum;
        return cv::sum(chi2)[0];
    }
    throw std::invalid_argument("Unknown distance method: " + method);
}



// RETURNS CATEGORY OF BEST FIT
int classifyImage(const cv::Mat& testFeature, const std::vector<cv::Mat>& trainFeatures, 
                  const std::vector<int>& trainLabels, int k, const std::string& distanceMethod) {
    std::vector<std::pair<float, int>> distances;

    for (size_t i = 0; i < trainFeatures.size(); ++i) {
        float distance = getImageDistance(testFeature, trainFeatures[i], distanceMethod);
        distances.emplace_back(distance, trainLabels[i]);
    }

    // Sort by distance
    std::sort(distances.begin(), distances.end());

    // Collect k nearest neighbors
    std::map<int, int> labelCounts;
    for (int i = 0; i < k; ++i) {
        labelCounts[distances[i].second]++;
    }

    // Return the most frequent label
    return std::max_element(labelCounts.begin(), labelCounts.end(), 
                            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
}


// RETURN EVALUATION METRICS: CONFUSION MATRIX AND ACCURACY
void evaluateSystem(const std::vector<std::pair<cv::Mat, int>>& testWordMaps,
                    const std::vector<cv::Mat>& trainFeatures, const std::vector<int>& trainLabels,
                    int dictionarySize, int k, const std::string& distanceMethod) {
    int correctCount = 0;
    cv::Mat confusionMatrix = cv::Mat::zeros(8, 8, CV_32S); // Assuming 8 categories

    for (const auto& [wordMap, actualLabel] : testWordMaps) {
        cv::Mat testFeature = getImageFeatures(wordMap, dictionarySize);
        int predictedLabel = classifyImage(testFeature, trainFeatures, trainLabels, k, distanceMethod);

        confusionMatrix.at<int>(actualLabel - 1, predictedLabel - 1)++;
        if (predictedLabel == actualLabel) {
            correctCount++;
        }
    }

    float accuracy = static_cast<float>(correctCount) / testWordMaps.size();
    std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    std::cout << "Confusion Matrix:\n" << confusionMatrix << std::endl;
}

