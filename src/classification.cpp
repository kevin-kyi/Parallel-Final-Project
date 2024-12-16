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



void KMEANtestDictionaryEvaluation() {
    // Base directories
    std::string trainingDir = "results/Training";
    std::string testingDir = "results/Testing";

    // Categories for training and testing
    std::vector<std::string> trainingCategories = {
        "airport_terminal", "campus", "desert", "elevator", 
        "forest", "kitchen", "lake", "swimming_pool"
    };
    std::vector<std::string> testingCategories = {
        "test_airport_terminal", "test_campus", "test_desert", "test_elevator", 
        "test_forest", "test_kitchen", "test_lake", "test_swimming_pool"
    };

    // Labels corresponding to categories (assuming 8 categories with labels 1 to 8)
    std::vector<int> labels = {1, 2, 3, 4, 5, 6, 7, 8};

    // Dictionary parameters
    std::string dictionaryPath = "kmeans_dictionary.yml";
    int dictionarySize = 500;
    int k = 8;
    std::string distanceMethod = "chi2";

    // Load the dictionary
    cv::Mat dictionary;
    cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
    }
    fs["dictionary"] >> dictionary;
    fs.release();

    // Load training word maps
    std::cout << "Loading training word maps..." << std::endl;
    std::vector<std::pair<cv::Mat, int>> trainWordMaps = loadWordMaps(trainingDir, trainingCategories, labels);

    std::vector<cv::Mat> trainFeatures;
    std::vector<int> trainLabels;
    for (const auto& [wordMap, label] : trainWordMaps) {
        trainFeatures.push_back(getImageFeatures(wordMap, dictionarySize));
        trainLabels.push_back(label);
    }

    // Load testing word maps
    std::cout << "Loading testing word maps..." << std::endl;
    std::vector<std::pair<cv::Mat, int>> testWordMaps = loadWordMaps(testingDir, testingCategories, labels);

    // Evaluate the system
    std::cout << "Evaluating system..." << std::endl;
    // evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, k, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 1, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 5, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 8, distanceMethod);

    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 10, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 20, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 30, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 40, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 50, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 100, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 200, distanceMethod);
}



void DBSCANtestDictionaryEvaluation() {
    // Base directories
    std::string trainingDir = "../results/dbscan/Training";
    std::string testingDir = "../results/dbscan/Testing";

    // Categories for training and testing
    std::vector<std::string> trainingCategories = {
        "airport_terminal", "campus", "desert", "elevator", 
        "forest", "kitchen", "lake", "swimming_pool"
    };
    std::vector<std::string> testingCategories = {
        "test_airport_terminal", "test_campus", "test_desert", "test_elevator", 
        "test_forest", "test_kitchen", "test_lake", "test_swimming_pool"
    };

    // Labels corresponding to categories (assuming 8 categories with labels 1 to 8)
    std::vector<int> labels = {1, 2, 3, 4, 5, 6, 7, 8};

    // Dictionary parameters
    std::string dictionaryPath = "../dbscan_dictionary.yml";
    int dictionarySize = 500;
    int k = 8;
    std::string distanceMethod = "chi2";

    // Load the dictionary
    cv::Mat dictionary;
    cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
    }
    fs["means"] >> dictionary;
    fs.release();

    // Load training word maps
    std::cout << "Loading training word maps..." << std::endl;
    std::vector<std::pair<cv::Mat, int>> trainWordMaps = loadWordMaps(trainingDir, trainingCategories, labels);

    std::vector<cv::Mat> trainFeatures;
    std::vector<int> trainLabels;
    for (const auto& [wordMap, label] : trainWordMaps) {
        trainFeatures.push_back(getImageFeatures(wordMap, dictionarySize));
        trainLabels.push_back(label);
    }

    // Load testing word maps
    std::cout << "Loading testing word maps..." << std::endl;
    std::vector<std::pair<cv::Mat, int>> testWordMaps = loadWordMaps(testingDir, testingCategories, labels);

    // Evaluate the system
    std::cout << "Evaluating system..." << std::endl;
    // evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, k, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 1, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 5, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 8, distanceMethod);

    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 10, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 10, "euclidean");

    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 20, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 30, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 40, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 50, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 100, distanceMethod);
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, 200, distanceMethod);
}
