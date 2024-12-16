#include "include/filters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"

#include <fstream>

#include "include/getVisualWords.h"
#include "include/createFilterBank.h"
#include <yaml-cpp/yaml.h>
#include "include/create_csv.h"

#include "include/save_dictionary_yml.h"





// *************** DBSCAN GET DICTIONARY ***************
// This function uses the csv to collect training image paths then for each image
// our Harris point detector collects "alpha" points which creates pixel resposnes
// that we can use in DBSCAN to find noise reduced cluster centers

void save_dbscan_dictionary() {
    // Read the CSV file
    std::ifstream infile("../traintest.csv");
    if (!infile.is_open()) {
        std::cerr << "Cannot open traintest.csv\n";
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
        // std::cout << "FileName: " << filename << std::endl; 

        if (filename == ".DS_Store"){ continue; }
        std::string category = (split == "train" ? "Training/" : "Testing/");

        std::string full_path = "../data/" + category + filename;
        
        if (split == "train") {
            image_paths.push_back(full_path); 
        }

    }

    // Parameters for dictionary creation
    int alpha = 40;
    double eps = 0.24;      
    int minSamples = 6;   

    cv::Mat dbscan_means = get_dictionary_dbscan(image_paths, alpha, eps, minSamples);
}



// *************** KMEANS GET DICTINOARY ***************

void save_kmeans_dictinonary() {
    // Read the CSV file
    std::ifstream infile("../traintest.csv");
    if (!infile.is_open()) {
        std::cerr << "Cannot open traintest.csv\n";
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
        std::cout << "FileName: " << filename << std::endl; 

        if (filename == ".DS_Store"){ continue; }
        std::string category = (split == "train" ? "Training/" : "Testing/");

        // std::string full_path = "data/" + category + filename;
        std::string full_path = category + filename;

        if (split == "train") {
            image_paths.push_back(full_path);
        }
    }

    // Parameters for dictionary creation
    int alpha = 10;  // Number of points per image
    int K = 500;     // Number of visual words
    std::string method = "Harris";

    // Create the dictionary
    cv::Mat dictionary = get_kmeans_dictionary(image_paths, alpha, K, method);

    // Save the dictionary
    save_dictionary(dictionary, "kmeans_dictionary.yml");
}
