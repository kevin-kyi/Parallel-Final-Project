#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


// MAIN FUNCTION TO TEST FILTER EXTRACTION

// std::vector<cv::Mat> createFilterBank();

// int main() {
//     // Define the image path and results output folder
//     std::string imagePath = "data/airport_image1.jpg";
//     std::string resultsPath = "results";

//     // Load the input image
//     cv::Mat image = cv::imread(imagePath);
//     if (image.empty()) {
//         std::cerr << "Error: Could not open image at " << imagePath << std::endl;
//         return -1;
//     }

//     // Create the filter bank
//     std::vector<cv::Mat> filterBank = createFilterBank();

//     // Apply filters and save the responses
//     try {
//         extractAndSaveFilterResponses(image, filterBank, resultsPath);
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }

//     std::cout << "Processing completed successfully. Filtered images saved in " << resultsPath << std::endl;

//     return 0;
// }



// MAIN FUNCTION TO TEST GETHARRISPOINTS

// #include "include/getHarrisPoints.h"
// #include <iostream>
// #include <filesystem>

// int main() {
//     try {
//         // Load the image
//         std::string inputPath = "data/campus_image1.jpg";  // Replace with your image path
//         cv::Mat image = cv::imread(inputPath);

//         // Parameters for Harris detector
//         int alpha = 500;     // Number of top responses
//         double k = 0.04;     // Harris constant

//         // Perform Harris corner detection
//         std::vector<cv::Point> harrisPoints = getHarrisPoints(image, alpha, k);

//         // Save result
//         std::string outputPath = "results/harris_points_image2.jpg";
//         saveHarrisPointsImage(image, harrisPoints, outputPath);

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }




// TESTING MAIN FOR CREATING TRAIN/TEST CSV
#include "include/create_csv.h"
#include "include/create_dictionary.h"


// int main() {
//     std::string train_dir = "data/Training";
//     std::string test_dir = "data/Testing";
//     std::string output_csv = "traintest.csv";

//     create_csv(train_dir, test_dir, output_csv);



//     return 0;
// }


#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"

#include <fstream>

#include "include/visual_words.h"
#include "include/createFilterBank.h"
#include <yaml-cpp/yaml.h>



// KMEANS VISUAL WORD MAP REPRESENTATION

// int main() {
//     // std::string imagePath = "data/airport_image1.jpg";
//     std::string imagePath = "data/Testing/test_airport_terminal/sun_acklfjjyqbayxbll.jpg";


//     // Load the input image
//     cv::Mat image = cv::imread(imagePath);
//     if (image.empty()) {
//         std::cerr << "Error: Unable to load the image at " << imagePath << std::endl;
//         return -1;
//     }

//     // Create filter bank
//     std::vector<cv::Mat> filterBank = createFilterBank();

//     // Load the dictionary from YAML
//     cv::FileStorage fs("kmeans_dictionary.yml", cv::FileStorage::READ);
//     if (!fs.isOpened()) {
//         std::cerr << "Error: Unable to open the dictionary file." << std::endl;
//         return -1;
//     }

//     cv::Mat dictionary;
//     fs["dictionary"] >> dictionary;
//     fs.release();

//     std::cout << "Loaded Dictionary - Rows: " << dictionary.rows 
//           << ", Cols: " << dictionary.cols 
//           << ", Type: " << dictionary.type() << std::endl;



//     try {
//         cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);
//         std::cout << "Generated word map successfully!" << std::endl;

//         // Normalize the word map for visualization
//         cv::Mat normalizedWordMap;
//         cv::normalize(wordMap, normalizedWordMap, 0, 255, cv::NORM_MINMAX, CV_8U);

//         // Save the word map as an image
//         std::string outputPath = "kmeans_visual_words_map.png";
//         if (cv::imwrite(outputPath, normalizedWordMap)) {
//             std::cout << "Word map saved successfully at " << outputPath << std::endl;
//         } else {
//             std::cerr << "Failed to save word map image." << std::endl;
//         }

//     } catch (const std::exception& ex) {
//         std::cerr << "Error during visual words computation: " << ex.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }







// // ****************** GMM VISUAL WORD MAP REPRESENTATION ******************
// int main() {
//     std::string imagePath = "data/Testing/test_airport_terminal/sun_acklfjjyqbayxbll.jpg";

//     // Load the input image
//     cv::Mat image = cv::imread(imagePath);
//     if (image.empty()) {
//         std::cerr << "Error: Unable to load the image at " << imagePath << std::endl;
//         return -1;
//     }

//     // Load the GMM dictionary from YAML
//     cv::FileStorage fs("gmm_dictionary.yml", cv::FileStorage::READ);
//     if (!fs.isOpened()) {
//         std::cerr << "Error: Unable to open the GMM dictionary file." << std::endl;
//         return -1;
//     }

//     // Read means and weights
//     cv::Mat means, weights;
//     std::vector<cv::Mat> covariances;
//     fs["means"] >> means;
//     fs["weights"] >> weights;



//     // Ensure means and weights are converted to CV_64F
//     // if (means.type() != CV_64F) {
//     //     means.convertTo(means, CV_64F);
//     //     std::cout << "Converted means to CV_64F." << std::endl;
//     // }
//     // if (weights.type() != CV_64F) {
//     //     weights.convertTo(weights, CV_64F);
//     //     std::cout << "Converted weights to CV_64F." << std::endl;
//     // }
//     if (means.type() != CV_32F) {
//         means.convertTo(means, CV_32F);
//         std::cout << "Converted means to CV_32F." << std::endl;
//     }
//     if (weights.type() != CV_32F) {
//         weights.convertTo(weights, CV_32F);
//         std::cout << "Converted weights to CV_32F." << std::endl;
//     }

//     std::cout << "Means size: " << means.size() << ", Type: " << means.type() << std::endl;
//     std::cout << "Weights size: " << weights.size() << ", Type: " << weights.type() << std::endl;

//     // Read covariance matrices
//     int K = means.rows;
//     for (int i = 0; i < K; i++) {
//         cv::Mat cov;
//         fs["covariances"][i] >> cov;

//         // Debug output for covariance matrix
//         if (cov.empty()) {
//             std::cerr << "Error: Covariance matrix " << i << " is empty." << std::endl;
//             return -1;
//         }
//         if (cov.rows != cov.cols) {
//             std::cerr << "Error: Covariance matrix " << i << " is not square (size: " << cov.size() << ")." << std::endl;
//             return -1;
//         }
//         // std::cout << "Covariance matrix " << i << " size: " << cov.size() << ", Type: " << cov.type() << std::endl;

//         // Convert covariance to CV_64F for consistency
//         // if (cov.type() != CV_64F) {
//         //     cov.convertTo(cov, CV_64F);
//         //     std::cout << "Converted covariance matrix " << i << " to CV_64F." << std::endl;
//         // }
//         if (cov.type() != CV_32F) {
//             cov.convertTo(cov, CV_32F);
//             std::cout << "Converted covariance matrix " << i << " to CV_32F." << std::endl;
//         }
//         std::cout << "Covariance matrix " << i << " size: " << cov.size() << ", Type: " << cov.type() << std::endl;

//         covariances.push_back(cov);
//     }
//     fs.release();

//     // Compute the word map using the GMM dictionary
//     try {
//         cv::Mat wordMap = getVisualWordsGMM(image, means, covariances, weights);
//         std::cout << "Generated word map successfully!" << std::endl;

//         // Normalize the word map for visualization
//         cv::Mat normalizedWordMap;
//         cv::normalize(wordMap, normalizedWordMap, 0, 255, cv::NORM_MINMAX, CV_8U);

//         // Save the word map as an image
//         std::string outputPath = "gmm_visual_words_map.png";
//         if (cv::imwrite(outputPath, normalizedWordMap)) {
//             std::cout << "Word map saved successfully at " << outputPath << std::endl;
//         } else {
//             std::cerr << "Failed to save word map image." << std::endl;
//         }
//     } catch (const std::exception& ex) {
//         std::cerr << "Error during visual words computation: " << ex.what() << std::endl;
//         return -1;
//     }

//     return 0;
// }
























// CREATING YML DICTIONARY FOR KMEANS

// int main() {
//     // Read the CSV file
//     std::ifstream infile("traintest.csv");
//     if (!infile.is_open()) {
//         std::cerr << "Cannot open traintest.csv\n";
//         return -1;
//     }

//     std::vector<std::string> image_paths;
//     std::string line;
//     std::getline(infile, line); // Skip the header line

//     while (std::getline(infile, line)) {
//         std::istringstream ss(line);
//         std::string filename, label, split;
//         std::getline(ss, filename, ',');
//         std::getline(ss, label, ',');
//         std::getline(ss, split, ',');

//         // Construct the full image path based on the split
//         std::cout << "FileName: " << filename << std::endl; 

//         if (filename == ".DS_Store"){ continue; }
//         std::string category = (split == "train" ? "Training/" : "Testing/");

//         // std::string full_path = "data/" + category + filename;
//         std::string full_path = category + filename;

//         image_paths.push_back(full_path);
//     }

//     // Parameters for dictionary creation
//     int alpha = 10;  // Number of points per image
//     int K = 500;     // Number of visual words
//     std::string method = "Harris";

//     // Create the dictionary
//     cv::Mat dictionary = get_kmeans_dictionary(image_paths, alpha, K, method);

//     // Save the dictionary
//     save_dictionary(dictionary, "kmeans_dictionary.yml");

//     return 0;
// }



// CREATING YML DICTIONARY FOR GMM

// int main() {
//     // Read the CSV file
//     std::ifstream infile("traintest.csv");
//     if (!infile.is_open()) {
//         std::cerr << "Cannot open traintest.csv\n";
//         return -1;
//     }

//     std::vector<std::string> image_paths;
//     std::string line;
//     std::getline(infile, line); // Skip the header line

//     while (std::getline(infile, line)) {
//         std::istringstream ss(line);
//         std::string filename, label, split;
//         std::getline(ss, filename, ',');
//         std::getline(ss, label, ',');
//         std::getline(ss, split, ',');

//         // Construct the full image path based on the split
//         if (filename == ".DS_Store") continue;
//         std::string category = (split == "train" ? "Training/" : "Testing/");
//         std::string full_path = category + filename;

//         image_paths.push_back(full_path);
//     }

//     // Parameters for dictionary creation
//     int alpha = 10;  // Number of points per image
//     int K = 50;     // Number of visual words
//     std::string method = "Harris";

//     // Create the GMM-based dictionary
//     cv::Mat dictionary = get_gmm_dictionary(image_paths, alpha, K, method);

//     // GMM parameters are saved in get_gmm_dictionary itself
//     std::cout << "GMM dictionary saved to gmm_dictionary.yml" << std::endl;

//     return 0;
// }











// // RUNNING BATCH TO VISUAL WORDS FOR EVALUATION
// #include "include/batchToVisualWords.h"


// int test_visual_words() {
//     try {
//         // Paths to required files and directories
//         // std::string dictionaryPath = "../kmeans_dictionary.yml"; // Path to dictionary file
//         // std::string csvPath = "../traintest.csv";                // Path to CSV file
//         std::string dictionaryPath = "kmeans_dictionary.yml"; // Path to dictionary file
//         std::string csvPath = "traintest.csv";                // Path to CSV file
//         int numCores = 4;                                       // Number of threads to use

//         // Start batch processing
//         std::cout << "Starting batch processing of visual words..." << std::endl;
//         batchToVisualWords(numCores, dictionaryPath, csvPath);
//         // batchToVisualWords(dictionaryPath, csvPath);

//         std::cout << "Visual word processing completed successfully!" << std::endl;

//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return EXIT_FAILURE;
//     }

//     return EXIT_SUCCESS;
// }


// int main() {

//     test_visual_words();
    
//     return 0;
// }











// *************** WORKING IMPLENTATION FOR EXTRACTUBG VISUAL WORDS ***************
// void loadImageNames(const std::string& filePath, std::vector<std::string>& imagePaths) {
//     std::ifstream file(filePath);
//     if (!file.is_open()) {
//         throw std::runtime_error("Failed to open CSV file: " + filePath);
//     }

//     std::string line;
//     std::getline(file, line); // Skip the header line

//     while (std::getline(file, line)) {
//         std::istringstream stream(line);
//         std::string filename;
//         std::getline(stream, filename, ','); // Get the first column (filename)
//         imagePaths.push_back(filename);     // Add the filename to the vector
//     }
// }

// // Helper function to determine the input directory based on the image name
// std::string determineInputPath(const std::string& imageName) {
//     std::string basePath;

//     if (imageName.find("test_") == 0) {
//         basePath = "data/Testing/";
//     } else {
//         basePath = "data/Training/";
//     }

//     return basePath + imageName;
// }

// // Helper function to determine the base output directory based on the image name
// std::string determineOutputPath(const std::string& imageName, const std::string& resultsDir) {
//     std::string basePath;

//     if (imageName.find("test_") == 0) {
//         basePath = resultsDir + "/Testing/";
//     } else {
//         basePath = resultsDir + "/Training/";
//     }

//     // Append the category subdirectory (e.g., airport_terminal or test_airport_terminal)
//     size_t slashPos = imageName.find('/');
//     if (slashPos != std::string::npos) {
//         std::string category = imageName.substr(0, slashPos);
//         basePath += category + "/";
//     }

//     return basePath;
// }

// int main() {
//     // Paths to input and output data
//     std::string csvPath = "traintest.csv"; // Adjust the path to your CSV file
//     std::string dictionaryPath = "kmeans_dictionary.yml"; // Path to the YAML dictionary
//     std::string resultsDir = "results"; // Base directory for saving results

//     // Load image names from CSV
//     std::vector<std::string> imagePaths;
//     try {
//         loadImageNames(csvPath, imagePaths);
//     } catch (const std::exception& ex) {
//         std::cerr << "Error loading image names from CSV: " << ex.what() << std::endl;
//         return -1;
//     }

//     // Create filter bank
//     std::vector<cv::Mat> filterBank = createFilterBank();

//     // Load the dictionary from YAML
//     cv::Mat dictionary;
//     cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
//     if (!fs.isOpened()) {
//         std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
//         return -1;
//     }
//     fs["dictionary"] >> dictionary;
//     fs.release();

//     // std::cout << "Loaded Dictionary - Rows: " << dictionary.rows
//     //           << ", Cols: " << dictionary.cols
//     //           << ", Type: " << dictionary.type() << std::endl;

//     // Process each image
//     for (const std::string& imageName : imagePaths) {
//         std::cout << "Processing: " << imageName << std::endl;

//         // Determine the input path
//         std::string inputImagePath = determineInputPath(imageName);

//         // Load the input image
//         cv::Mat image = cv::imread(inputImagePath);
//         if (image.empty()) {
//             std::cerr << "Error: Unable to load the image at " << inputImagePath << std::endl;
//             continue;
//         }

//         try {
//             // Generate the word map
//             // cv::Mat wordMap = getVisualWords(image, dictionary, filterBank, imageName);
//             cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);

//             std::cout << "Generated word map successfully for " << imageName << std::endl;

//             // Determine the output directory
//             std::string outputBasePath = determineOutputPath(imageName, resultsDir);

//             // Ensure the output directory exists
//             std::string command = "mkdir -p " + outputBasePath;
//             system(command.c_str());

//             // Construct the full output path
//             std::string outputPath = outputBasePath + imageName.substr(imageName.find('/') + 1);
//             outputPath = outputPath.substr(0, outputPath.find_last_of('.')) + ".yml";

//             // Save the word map as a YML file
//             cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
//             if (fs.isOpened()) {
//                 fs << "wordMap" << wordMap;
//                 fs.release();
//                 std::cout << "Word map YML saved successfully at " << outputPath << std::endl;
//             } else {
//                 std::cerr << "Failed to save word map YML for " << imageName << std::endl;
//             }
//             wordMap.release();
//             image.release();
//         } catch (const std::exception& ex) {
//             std::cerr << "Error during visual words computation for " << imageName << ": " << ex.what() << std::endl;
//         }
//     }

//     std::cout << "Processing completed for all images." << std::endl;
//     return 0;
// }














#include "include/kmeans_knn_classification.h"


int main() {
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
    int k = 5;
    std::string distanceMethod = "chi2";

    // Load the dictionary
    cv::Mat dictionary;
    cv::FileStorage fs(dictionaryPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file at " << dictionaryPath << std::endl;
        return -1;
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
    evaluateSystem(testWordMaps, trainFeatures, trainLabels, dictionarySize, k, distanceMethod);

    return 0;
}


