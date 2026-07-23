#include "BinaryOverlap.h"

#include <algorithm>
#include <cctype>

// 本文件负责二值前景重叠评估：
// 1. 对灰度指纹图做可配置二值化；
// 2. 在给定仿射矩阵下把图 1 对齐到图 2；
// 3. 统计 overlap ratio 与 binary match ratio，供主链路分析使用。
namespace {
int ensure_odd_at_least_3(int value) {
    value = std::max(3, value);
    if ((value % 2) == 0) {
        value += 1;
    }
    return value;
}

cv::Mat apply_clahe(const cv::Mat& gray, const BinaryOverlapParams& params) {
    cv::Mat enhanced = gray.clone();
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(params.clahe_clip_limit, params.clahe_tile_grid_size);
    clahe->apply(enhanced, enhanced);
    return enhanced;
}

cv::Mat to_binary_with_params(const cv::Mat& gray, const BinaryOverlapParams& params) {
    cv::Mat enhanced = apply_clahe(gray, params);
    cv::Mat binary;

    switch (params.mode) {
        case BinaryBinarizationMode::ClaheGaussianOtsu: {
            const int kernel = ensure_odd_at_least_3(params.gaussian_kernel_size);
            cv::GaussianBlur(enhanced, enhanced, cv::Size(kernel, kernel), params.gaussian_sigma, params.gaussian_sigma);
            cv::threshold(enhanced, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        }
        case BinaryBinarizationMode::ClaheMedianOtsu: {
            const int kernel = ensure_odd_at_least_3(params.median_kernel_size);
            cv::medianBlur(enhanced, enhanced, kernel);
            cv::threshold(enhanced, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        }
        case BinaryBinarizationMode::ClaheAdaptiveMean: {
            const int block_size = ensure_odd_at_least_3(params.adaptive_block_size);
            cv::adaptiveThreshold(
                enhanced, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                block_size, params.adaptive_c);
            break;
        }
        case BinaryBinarizationMode::ClaheAdaptiveGaussian: {
            const int block_size = ensure_odd_at_least_3(params.adaptive_block_size);
            cv::adaptiveThreshold(
                enhanced, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                block_size, params.adaptive_c);
            break;
        }
        default: {
            cv::threshold(enhanced, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
            break;
        }
    }
    return binary;
}

cv::Mat normalize_mask(const cv::Mat& in_mask, const cv::Size& size) {
    cv::Mat mask;
    if (in_mask.empty()) {
        mask = cv::Mat(size, CV_8U, cv::Scalar(255));
        return mask;
    }
    if (in_mask.size() != size) {
        cv::resize(in_mask, mask, size, 0, 0, cv::INTER_NEAREST);
    } else {
        mask = in_mask.clone();
    }
    if (mask.type() != CV_8U) {
        cv::Mat tmp;
        mask.convertTo(tmp, CV_8U);
        mask = tmp;
    }
    cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);
    return mask;
}

std::string normalize_mode_name(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch == '-' || ch == ' ' || ch == '.') {
            out.push_back('_');
        } else {
            out.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return out;
}
}  // namespace

const char* binary_binarization_mode_to_string(BinaryBinarizationMode mode) {
    switch (mode) {
        case BinaryBinarizationMode::ClaheGaussianOtsu:
            return "clahe_gaussian_otsu";
        case BinaryBinarizationMode::ClaheMedianOtsu:
            return "clahe_median_otsu";
        case BinaryBinarizationMode::ClaheAdaptiveMean:
            return "clahe_adaptive_mean";
        case BinaryBinarizationMode::ClaheAdaptiveGaussian:
            return "clahe_adaptive_gaussian";
        default:
            return "unknown";
    }
}

bool binary_binarization_mode_from_string(const std::string& text, BinaryBinarizationMode& mode) {
    const std::string normalized = normalize_mode_name(text);
    if (normalized == "clahe_gaussian_otsu" || normalized == "gaussian_otsu" || normalized == "otsu") {
        mode = BinaryBinarizationMode::ClaheGaussianOtsu;
        return true;
    }
    if (normalized == "clahe_median_otsu" || normalized == "median_otsu") {
        mode = BinaryBinarizationMode::ClaheMedianOtsu;
        return true;
    }
    if (normalized == "clahe_adaptive_mean" || normalized == "adaptive_mean") {
        mode = BinaryBinarizationMode::ClaheAdaptiveMean;
        return true;
    }
    if (normalized == "clahe_adaptive_gaussian" || normalized == "adaptive_gaussian") {
        mode = BinaryBinarizationMode::ClaheAdaptiveGaussian;
        return true;
    }
    return false;
}

bool compute_binary_overlap_with_affine(
    const cv::Mat& gray1,
    const cv::Mat& gray2,
    const cv::Mat& affine_2x3,
    BinaryOverlapStats& stats,
    const BinaryOverlapParams& params,
    const cv::Mat& finger_mask1,
    const cv::Mat& finger_mask2) {
    // affine_2x3 约定为把 gray1 映射到 gray2 坐标系下的 2x3 仿射矩阵。
    // 后续所有 overlap/match 统计都在 gray2 坐标系里完成，方便和匹配结果直接对照。
    stats = BinaryOverlapStats{};

    if (gray1.empty() || gray2.empty()) return false;
    if (gray1.type() != CV_8U || gray2.type() != CV_8U) return false;
    if (affine_2x3.empty() || affine_2x3.rows != 2 || affine_2x3.cols != 3) return false;

    const cv::Mat bin1 = to_binary_with_params(gray1, params);
    const cv::Mat bin2 = to_binary_with_params(gray2, params);
    stats.binary1 = bin1.clone();
    stats.binary2 = bin2.clone();

    const cv::Mat mask1 = normalize_mask(finger_mask1, gray1.size());
    const cv::Mat mask2 = normalize_mask(finger_mask2, gray2.size());
    stats.mask1 = mask1.clone();
    stats.mask2 = mask2.clone();

    stats.mask1_pixels = cv::countNonZero(mask1);
    stats.mask2_pixels = cv::countNonZero(mask2);

    cv::Mat warped_bin1;
    cv::warpAffine(bin1, warped_bin1, affine_2x3, gray2.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    stats.warped_binary1 = warped_bin1;

    cv::Mat warped_mask1;
    cv::warpAffine(mask1, warped_mask1, affine_2x3, gray2.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::threshold(warped_mask1, warped_mask1, 0, 255, cv::THRESH_BINARY);
    stats.warped_mask1 = warped_mask1.clone();

    cv::Mat overlap_mask;
    cv::bitwise_and(warped_mask1, mask2, overlap_mask);
    stats.overlap_mask = overlap_mask;
    stats.overlap_pixels = cv::countNonZero(overlap_mask);

    const int denom_overlap = std::max(1, std::min(stats.mask1_pixels, stats.mask2_pixels));
    stats.overlap_ratio = static_cast<double>(stats.overlap_pixels) / static_cast<double>(denom_overlap);

    if (stats.overlap_pixels <= 0) {
        stats.binary_match_ratio = 0.0;
        return true;
    }

    cv::Mat equal_mask;
    cv::compare(warped_bin1, bin2, equal_mask, cv::CMP_EQ);

    cv::Mat match_mask;
    cv::bitwise_and(equal_mask, overlap_mask, match_mask);
    stats.match_mask = match_mask;
    stats.matched_pixels = cv::countNonZero(match_mask);
    stats.binary_match_ratio = static_cast<double>(stats.matched_pixels) / static_cast<double>(stats.overlap_pixels);

    cv::Mat mismatch_mask;
    cv::Mat not_equal_mask;
    cv::bitwise_not(equal_mask, not_equal_mask);
    cv::bitwise_and(not_equal_mask, overlap_mask, mismatch_mask);
    stats.mismatch_mask = mismatch_mask;

    return true;
}

cv::Mat render_binary_overlap_visualization(
    const cv::Mat& gray2,
    const BinaryOverlapStats& stats) {
    // 绿色表示两张二值图在重叠前景区域内取值一致，
    // 红色表示同样落在重叠区域内，但二值结果不一致。
    cv::Mat canvas;
    cv::cvtColor(gray2, canvas, cv::COLOR_GRAY2BGR);

    if (stats.overlap_mask.empty()) {
        return canvas;
    }

    cv::Mat green(canvas.size(), canvas.type(), cv::Scalar(40, 200, 40));
    cv::Mat red(canvas.size(), canvas.type(), cv::Scalar(40, 40, 220));

    green.copyTo(canvas, stats.match_mask);
    red.copyTo(canvas, stats.mismatch_mask);

    return canvas;
}
