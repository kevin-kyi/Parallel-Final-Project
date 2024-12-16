#ifndef CREATE_WORD_MAPS_H
#define CREATE_WORD_MAPS_H

#include <string>

/*
 * DBSCANcreateWordMaps
 * 
 * Generates visual word maps for all training/testing images using a DBSCAN-based
 * dictionary. The function reads image paths from "traintest.csv", processes the images
 * using the provided dictionary and filter bank, and saves the resulting word maps in YML format.
 * 
 * Input:
 *   - CSV file: "traintest.csv"
 *   - Dictionary file: "dbscan_iteration12_dictionary.yml"
 * 
 * Output:
 *   - Word maps are saved in "results/Training/dbscan" and "results/Testing/dbscan".
 */
void DBSCANcreateWordMaps();

/*
 * KMEANScreateWordMaps
 * 
 * Generates visual word maps for all training/testing images using a K-Means-based
 * dictionary. The function reads image paths from "traintest.csv", processes the images
 * using the provided dictionary and filter bank, and saves the resulting word maps in YML format.
 * 
 * Input:
 *   - CSV file: "traintest.csv"
 *   - Dictionary file: "kmeans_dictionary.yml"
 * 
 * Output:
 *   - Word maps are saved in "results/Training/" and "results/Testing/".
 */
void KMEANScreateWordMaps();

#endif // CREATE_WORD_MAPS_H