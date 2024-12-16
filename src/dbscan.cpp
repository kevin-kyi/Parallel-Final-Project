#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/create_dictionary.h"
#include "include/getHarrisPoints.h"
#include "include/filters.h"
#include "include/getVisualWords.h"

#include "include/createFilterBank.h"

#include <omp.h>
#include <yaml-cpp/yaml.h>

#include "include/dbscan.h"





/*
 * OpenMP Version of DBSCAN
*/
std::vector<int> dbscan_OpenMP(const cv::Mat& data, double eps, int minSamples) {
    int N = data.rows;
    std::vector<int> labels(N, -1); // -1 means noise
    std::vector<bool> visited(N, false);
    int clusterId = 0;

    // Precompute distances (parallelized)
    cv::Mat distMat(N, N, CV_64F);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            cv::Mat diff = data.row(i) - data.row(j);
            double dist = std::sqrt(diff.dot(diff));
            distMat.at<double>(i, j) = dist;
            distMat.at<double>(j, i) = dist; // Symmetric
        }
    }

    auto regionQuery = [&](int idx) {
        std::vector<int> neighbors;
        for (int i = 0; i < N; i++) {
            if (distMat.at<double>(idx, i) <= eps) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    };

    std::function<void(int, std::vector<int>&)> expandCluster =
    [&](int idx, std::vector<int>& neighbors) {
        labels[idx] = clusterId;

        #pragma omp parallel for
        for (size_t i = 0; i < neighbors.size(); i++) {
            int nIdx = neighbors[i];
            if (!visited[nIdx]) {
                #pragma omp critical
                {
                    if (!visited[nIdx]) {
                        visited[nIdx] = true;
                        auto nNeighbors = regionQuery(nIdx);
                        if ((int)nNeighbors.size() >= minSamples) {
                            neighbors.insert(neighbors.end(), nNeighbors.begin(), nNeighbors.end());
                        }
                    }
                }
            }
            if (labels[nIdx] == -1) {
                labels[nIdx] = clusterId;
            }
        }
    };

    // Main DBSCAN Loop (sequential)
    for (int i = 0; i < N; i++) {
        if (visited[i]) continue;

        visited[i] = true;
        auto neighbors = regionQuery(i);

        if ((int)neighbors.size() < minSamples) {
            labels[i] = -1; // Noise
        } else {
            labels[i] = clusterId;
            expandCluster(i, neighbors);
            #pragma omp atomic
            clusterId++;
        }
    }

    return labels;
}


/*
 * Sequential Version of DBSCAN
*/
std::vector<int> dbscan_Sequential(const cv::Mat& data, double eps, int minSamples) {
    int N = data.rows;
    std::vector<int> labels(N, -1); // -1 means noise
    std::vector<bool> visited(N, false);
    int clusterId = 0;

    // Precompute distances
    cv::Mat distMat(N, N, CV_64F);

    // std::cout << "# Iterations: " << N << std::endl;
    for (int i = 0; i < N; i++) {
        if (i % 1000 == 0) {
            std::cout << "DBSCAN Distance Iteration: " << i << "/" << N << std::endl;
        }

        for (int j = i; j < N; j++) {
            cv::Mat diff = data.row(i) - data.row(j);
            double dist = std::sqrt(diff.dot(diff));
            distMat.at<double>(i, j) = dist;
            distMat.at<double>(j, i) = dist;
        }
    }

    auto regionQuery = [&](int idx) {
        std::vector<int> neighbors;
        for (int i = 0; i < N; i++) {
            if (distMat.at<double>(idx, i) <= eps) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    };

    std::function<void(int, std::vector<int>&)> expandCluster =
    [&](int idx, std::vector<int>& neighbors) {
        labels[idx] = clusterId;
        for (size_t i = 0; i < neighbors.size(); i++) {
            int nIdx = neighbors[i];
            if (!visited[nIdx]) {
                visited[nIdx] = true;
                auto nNeighbors = regionQuery(nIdx);
                if ((int)nNeighbors.size() >= minSamples) {
                    neighbors.insert(neighbors.end(), nNeighbors.begin(), nNeighbors.end());
                }
            }
            if (labels[nIdx] == -1) {
                labels[nIdx] = clusterId;
            }
        }
    };

    // DBSCAN main loop
    std::cout << "# Iterations: " << N << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "DBSCAN Iteration: " << i << std::endl;

        if (visited[i]) continue;
        visited[i] = true;
        auto neighbors = regionQuery(i);
        if ((int)neighbors.size() < minSamples) {
            // noise
            labels[i] = -1;
        } else {
            labels[i] = clusterId;
            expandCluster(i, neighbors);
            clusterId++;
        }
    }

    return labels;
}