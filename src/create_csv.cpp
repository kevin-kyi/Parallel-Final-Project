#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

#include "include/create_csv.h"


using namespace std;
namespace fs = std::filesystem;

void create_csv(const std::string &train_dir, const std::string &test_dir, const std::string &output_csv) {
    vector<string> mapping = {
        "airport_terminal",
        "campus",
        "desert",
        "elevator",
        "forest",
        "kitchen",
        "lake",
        "swimming_pool"
    };

    ofstream csv(output_csv);
    if (!csv.is_open()) {
        cerr << "Error: Unable to open " << output_csv << " for writing." << endl;
        return;
    }

    csv << "filename,label,split\n";  // header line

    for (size_t i = 0; i < mapping.size(); i++) {
        int label = (int)i + 1;
        string category = mapping[i];

        // Training images
        fs::path cat_train_path = fs::path(train_dir) / category;
        if (fs::exists(cat_train_path) && fs::is_directory(cat_train_path)) {
            for (auto &entry : fs::directory_iterator(cat_train_path)) {
                if (entry.is_regular_file()) {
                    string rel = category + "/" + entry.path().filename().string();
                    csv << rel << "," << label << ",train\n";
                }
            }
        }

        // Testing images
        fs::path cat_test_path = fs::path(test_dir) / ("test_" + category);
        if (fs::exists(cat_test_path) && fs::is_directory(cat_test_path)) {
            for (auto &entry : fs::directory_iterator(cat_test_path)) {
                if (entry.is_regular_file()) {
                    string rel = "test_" + category + "/" + entry.path().filename().string();
                    csv << rel << "," << label << ",test\n";
                }
            }
        }
    }

    csv.close();
    cout << "CSV file created: " << output_csv << endl;
}