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
#include <yaml-cpp/yaml.h>





int main() {
    // std::string imagePath = "data/airport_image1.jpg";
    std::string imagePath = "data/Testing/test_airport_terminal/sun_acklfjjyqbayxbll.jpg";


    // Load the input image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Unable to load the image at " << imagePath << std::endl;
        return -1;
    }

    // Create filter bank
    std::vector<cv::Mat> filterBank = createFilterBank();

    // Load the dictionary from YAML
    cv::FileStorage fs("dictionary.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open the dictionary file." << std::endl;
        return -1;
    }

    cv::Mat dictionary;
    fs["dictionary"] >> dictionary;
    fs.release();

    std::cout << "Loaded Dictionary - Rows: " << dictionary.rows 
          << ", Cols: " << dictionary.cols 
          << ", Type: " << dictionary.type() << std::endl;



    // try {
    //     cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);
    //     std::cout << "Generated word map successfully!" << std::endl;

    //     // Visualize the word map (optional)
    //     cv::normalize(wordMap, wordMap, 0, 255, cv::NORM_MINMAX, CV_8U);
    //     cv::imshow("Visual Words Map", wordMap);
    //     cv::waitKey(0);


    // } catch (const std::exception& ex) {
    //     std::cerr << "Error during visual words computation: " << ex.what() << std::endl;
    //     return -1;
    // }

    try {
        cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);
        std::cout << "Generated word map successfully!" << std::endl;

        // Normalize the word map for visualization
        cv::Mat normalizedWordMap;
        cv::normalize(wordMap, normalizedWordMap, 0, 255, cv::NORM_MINMAX, CV_8U);

        // Save the word map as an image
        std::string outputPath = "visual_words_map.png";
        if (cv::imwrite(outputPath, normalizedWordMap)) {
            std::cout << "Word map saved successfully at " << outputPath << std::endl;
        } else {
            std::cerr << "Failed to save word map image." << std::endl;
        }

        // // Optionally display the image
        // cv::imshow("Visual Words Map", normalizedWordMap);
        // cv::waitKey(0);

    } catch (const std::exception& ex) {
        std::cerr << "Error during visual words computation: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}









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
//     cv::Mat dictionary = get_dictionary(image_paths, alpha, K, method);

//     // Save the dictionary
//     save_dictionary(dictionary, "dictionary.yml");

//     return 0;
// }



