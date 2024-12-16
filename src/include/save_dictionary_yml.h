#ifndef SAVE_DICTIONARY_H
#define SAVE_DICTIONARY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "dbscan.h"

/*
 * save_dbscan_dictionary
 * 
 * Reads training image paths from a CSV file ("traintest.csv") and uses the DBSCAN
 * algorithm to create a noise-reduced dictionary. The dictionary is built by:
 * - Detecting Harris points from images
 * - Collecting pixel responses at these points
 * - Using DBSCAN to compute cluster centers
 * 
 * Parameters:
 *   - CSV file "traintest.csv" must contain columns: filename, label, split
 *   - "data/Training/" and "data/Testing/" directories must contain images
 * 
 * Output:
 *   - Saves the resulting dictionary to a file (if implemented in get_dictionary_dbscan)
 */
void save_dbscan_dictionary();

/*
 * save_kmeans_dictionary
 * 
 * Reads training image paths from a CSV file ("traintest.csv") and uses the K-Means
 * clustering algorithm to create a visual dictionary. The dictionary is built by:
 * - Detecting Harris points or other feature points
 * - Collecting pixel responses at these points
 * - Using K-Means to compute visual words
 * 
 * Parameters:
 *   - CSV file "traintest.csv" must contain columns: filename, label, split
 *   - "data/Training/" and "data/Testing/" directories must contain images
 * 
 * Output:
 *   - Saves the resulting dictionary to "kmeans_dictionary.yml"
 */
void save_kmeans_dictionary();

#endif // SAVE_DICTIONARY_H