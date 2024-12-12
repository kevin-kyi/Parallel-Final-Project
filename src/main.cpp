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
// #include "include/create_csv.h"
// #include "include/create_dictionary.h"


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


// int main() {
//     std::string imagePath = "data/airport_image1.jpg";

//     // Load the input image
//     cv::Mat image = cv::imread(imagePath);

    
//     std::vector<cv::Mat> filterBank = createFilterBank();

   





//     YAML::Node config = YAML::LoadFile("dictionary.yml");

//     // Extract matrix dimensions and data from YAML
//     int rows = config["dictionary"]["rows"].as<int>();
//     int cols = config["dictionary"]["cols"].as<int>();
//     std::vector<double> data = config["dictionary"]["data"].as<std::vector<double>>();
//     // Create a cv::Mat from the extracted data
//     cv::Mat dictionary = cv::Mat(rows, cols, CV_64FC1, data.data()).t();



//     std::cout << "Here is dict Rows: " << rows << std::endl;
//     std::cout << "Here is dict Cols: " << cols << std::endl;


//     cv::Mat filterResponses = extractFilterResponses(image, filterBank);

//     // Reshape filter responses to a 2D matrix where each row is a feature vector
//     int numPixels = filterResponses.rows * filterResponses.cols;
//     int numChannels = filterResponses.channels();
//     filterResponses = filterResponses.reshape(numPixels, numChannels);

//     // Ensure data type consistency
//     filterResponses.convertTo(filterResponses, dictionary.type());

//     // std::cout << "Filter Response Row: " << filterResponses.row(0) << std::endl;
//     std::cout << "Filter Response Row: " << filterResponses.row(0).size() << std::endl;





//     cv::Mat wordMap = getVisualWords(image, dictionary, filterBank);

//     // // Reconstruct the image using the wordMap (optional)
//     // // ... (implementation for reconstructing the image)

//     // // Display or save the reconstructed image
//     // cv::imshow("Reconstructed Image", wordMap);
//     // cv::waitKey(0);
//     // cv::imwrite("reconstructed_image.jpg", wordMap);

//     // return 0;
// }




int main() {
    // Read the CSV file
    std::ifstream infile("traintest.csv");
    if (!infile.is_open()) {
        std::cerr << "Cannot open traintest.csv\n";
        return -1;
    }

    std::vector<std::string> image_paths;
    std::string line;
    std::getline(infile, line); // Skip the header line

    while (std::getline(infile, line)) {
        std::istringstream ss(line);
        std::string filename, label, split;
        std::getline(ss, filename, ',');
        std::getline(ss, label, ',');
        std::getline(ss, split, ',');

        // Construct the full image path based on the split

        std::string category = (split == "train" ? "Training/" : "Testing/");

        // std::string full_path = "data/" + category + filename;
        std::string full_path = category + filename;

        image_paths.push_back(full_path);
    }

    // Parameters for dictionary creation
    int alpha = 10;  // Number of points per image
    int K = 500;     // Number of visual words
    std::string method = "Harris";

    // Create the dictionary
    cv::Mat dictionary = get_dictionary(image_paths, alpha, K, method);

    // Save the dictionary
    save_dictionary(dictionary, "dictionary.yml");

    return 0;
}



