#ifndef BINARY_OVERLAP_H
#define BINARY_OVERLAP_H

#include <opencv2/opencv.hpp>
#include <string>

enum class BinaryBinarizationMode {
    ClaheGaussianOtsu = 0,
    ClaheMedianOtsu,
    ClaheAdaptiveMean,
    ClaheAdaptiveGaussian
};

struct BinaryOverlapParams {
    BinaryBinarizationMode mode = BinaryBinarizationMode::ClaheGaussianOtsu;
    double clahe_clip_limit = 3.0;
    cv::Size clahe_tile_grid_size = cv::Size(8, 8);
    int gaussian_kernel_size = 5;
    double gaussian_sigma = 0.8;
    int median_kernel_size = 5;
    int adaptive_block_size = 31;
    double adaptive_c = 6.0;
};

struct BinaryOverlapStats {
    int overlap_pixels = 0;
    int matched_pixels = 0;
    int mask1_pixels = 0;
    int mask2_pixels = 0;
    double overlap_ratio = 0.0;
    double binary_match_ratio = 0.0;

    cv::Mat binary1;
    cv::Mat binary2;
    cv::Mat mask1;
    cv::Mat mask2;
    cv::Mat warped_mask1;
    cv::Mat warped_binary1;
    cv::Mat overlap_mask;
    cv::Mat match_mask;
    cv::Mat mismatch_mask;
};

bool compute_binary_overlap_with_affine(
    const cv::Mat& gray1,
    const cv::Mat& gray2,
    const cv::Mat& affine_2x3,
    BinaryOverlapStats& stats,
    const BinaryOverlapParams& params = BinaryOverlapParams(),
    const cv::Mat& finger_mask1 = cv::Mat(),
    const cv::Mat& finger_mask2 = cv::Mat());

cv::Mat render_binary_overlap_visualization(
    const cv::Mat& gray2,
    const BinaryOverlapStats& stats);

const char* binary_binarization_mode_to_string(BinaryBinarizationMode mode);
bool binary_binarization_mode_from_string(const std::string& text, BinaryBinarizationMode& mode);

#endif
