#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#pragma once


cv::Mat generateFingerMask(const cv::Mat& img);

void save_sift_vis(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
                   const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
                   const std::vector<cv::DMatch>& matches, int score, const std::string& out);

cv::Mat extract_gmfs_mask(const cv::Mat& image,
    float sigma = 13.0f / 3.0f,
    float percentile = 95.0f,
    float threshold_ratio = 0.2f,
    int closing_count = 6,
    int opening_count = 2,
    int image_dpi = 500);

#endif
