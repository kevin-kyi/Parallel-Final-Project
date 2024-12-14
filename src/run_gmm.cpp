#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <iostream>

#include "include/run_gmm.h"
#include "include/create_dictionary.h"

#include <random>


// Gaussian density function


// double gaussianDensity(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& cov) {
//     int dim = x.rows;

//     // Compute the difference vector
//     cv::Mat diff = x - mean;

//     // Regularize covariance for numerical stability
//     cv::Mat reg_cov = cov + cv::Mat::eye(dim, dim, CV_64F) * 1e-3;

//     // Compute determinant and check validity
//     double det_cov = cv::determinant(reg_cov);
//     if (det_cov <= 0) {
//         throw std::runtime_error("Covariance matrix determinant is non-positive!");
//     }

//     // Compute the inverse of the covariance matrix
//     cv::Mat inv_cov;
//     cv::invert(reg_cov, inv_cov, cv::DECOMP_SVD);

//     // Compute the normalization constant
//     double norm_const = 1.0 / (std::pow(2.0 * M_PI, dim / 2.0) * std::sqrt(det_cov));

//     // Compute the Mahalanobis distance
//     double mahalanobis = diff.dot(inv_cov * diff);

//     // Compute the probability density
//     double pdf = norm_const * std::exp(-0.5 * mahalanobis);

//     return pdf;
// }


double gaussianDensity(const cv::Mat& x, const cv::Mat& mean, const cv::Mat& cov) {
    cv::Mat diff = x - mean;

    double det_cov = cv::determinant(cov);
    if (det_cov < 1e-10) {
        // Regularize covariance matrix if determinant is too small
        cv::Mat reg_cov = cov + cv::Mat::eye(cov.rows, cov.cols, cov.type()) * 1e-6;
        det_cov = cv::determinant(reg_cov);
    }

    double normalizer = 1.0 / (std::pow(2 * M_PI, x.rows / 2.0) * std::sqrt(det_cov));
    cv::Mat inv_cov;
    cv::invert(cov, inv_cov, cv::DECOMP_SVD); // Use stable inversion
    double quad_term = diff.t().dot(inv_cov * diff);

    return normalizer * std::exp(-0.5 * quad_term);
}






void trainGMM(const cv::Mat& data, int K, int maxIter, double tol, cv::Mat& means, std::vector<cv::Mat>& covariances, cv::Mat& weights) {
    cv::Mat data64F;
    if (data.type() != CV_64F) {
        data.convertTo(data64F, CV_64F);
    } else {
        data64F = data;
    }

    cv::normalize(data64F, data64F, 0, 1, cv::NORM_MINMAX);


    int N = data64F.rows;
    int dim = data64F.cols;

    // Initialize GMM parameters
    cv::RNG rng;
    cv::Mat responsibilities(N, K, CV_64F);
    double prevLogLikelihood = -std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < maxIter; iter++) {
        // E-Step: Compute responsibilities
        for (int i = 0; i < N; i++) {
            double sumResp = 0.0;
            for (int k = 0; k < K; k++) {
                cv::Mat x = data64F.row(i).t();
                double gausDensity = gaussianDensity(x, means.row(k).t(), covariances[k]);
                double resp = weights.at<double>(0, k) * gausDensity;

                std::cout << "Weight: " << weights.at<double>(0, k) << " Gaussian Density: " << gausDensity << " RESP: " << resp << std::endl;


                responsibilities.at<double>(i, k) = resp;
                sumResp += resp;

                std::cout << "SUMRESP: " << sumResp << std::endl;
            }

            if (sumResp < 1e-10) {
                responsibilities.row(i).setTo(1.0 / K);  // Assign uniform responsibility
            } else {
                for (int k = 0; k < K; k++) {
                    responsibilities.at<double>(i, k) /= sumResp;
                }
            }
        }

        // M-Step: Update parameters
        for (int k = 0; k < K; k++) {
            // cv::Mat Nk = cv::sum(responsibilities.col(k));
            cv::Mat Nk(1, 1, CV_64F);
            Nk.at<double>(0, 0) = cv::sum(responsibilities.col(k))[0];
            if (Nk.at<double>(0) > 0) {
                weights.at<double>(0, k) = Nk.at<double>(0) / N;

                cv::Mat mean = cv::Mat::zeros(dim, 1, CV_64F);
                for (int i = 0; i < N; i++) {
                    mean += responsibilities.at<double>(i, k) * data64F.row(i).t();
                }
                means.row(k) = (mean / Nk.at<double>(0)).t();

                cv::Mat covariance = cv::Mat::zeros(dim, dim, CV_64F);
                for (int i = 0; i < N; i++) {
                    cv::Mat diff = data64F.row(i).t() - means.row(k).t();
                    covariance += responsibilities.at<double>(i, k) * diff * diff.t();
                }
                covariances[k] = covariance / Nk.at<double>(0) + cv::Mat::eye(dim, dim, CV_64F) * 1e-6; // Regularization
            } else {
                // Reinitialize empty cluster
                means.row(k) = data64F.row(rng.uniform(0, N)).clone();
                weights.at<double>(0, k) = 1.0 / K;
                covariances[k] = cv::Mat::eye(dim, dim, CV_64F) * 1e-3;
            }
        }

        // Compute log-likelihood
        double logLikelihood = 0.0;
        for (int i = 0; i < N; i++) {
            double likelihood = 0.0;
            for (int k = 0; k < K; k++) {
                likelihood += weights.at<double>(0, k) * gaussianDensity(data64F.row(i).t(), means.row(k).t(), covariances[k]);

                std::cout << "likelihood: " << likelihood << std::endl;
            }
            logLikelihood += std::log(likelihood);
        }

        std::cout << "LOG likelihood: " << logLikelihood << std::endl;

        if (std::abs(logLikelihood - prevLogLikelihood) < tol) {
            break;
        }
        prevLogLikelihood = logLikelihood;
    }
}

