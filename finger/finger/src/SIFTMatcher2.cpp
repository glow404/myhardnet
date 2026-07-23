#include "SIFTMatcher2.h"

SIFTMatcher2::SIFTMatcher2(const SIFTParams& p, const SIFTConstants& c)
    : P(p), C(c), sift(p, c) {}

void SIFTMatcher2::extractFeatures(const cv::Mat& img, const cv::Mat& mask,
    std::vector<cv::KeyPoint>& kp, cv::Mat& des) const {
    sift.detectAndCompute(img, mask, kp, des);
}

int SIFTMatcher2::match(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
    const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
    std::vector<cv::DMatch>& final_matches) const {

    final_matches.clear();

    // 1. 安全检查
    if (des1.empty() || des2.empty() || des1.rows < 2 || des2.rows < 2) return 0;

    // =========================================================================
    // 第一阶段：【纯双向交叉验证】(无 Ratio，无距离门槛)
    // true 表示开启交叉验证：A找B最近，B找A最近，互为第一才放行
    // =========================================================================
    cv::BFMatcher bf(cv::NORM_L2, true);
    std::vector<cv::DMatch> mutual_matches;
    bf.match(des1, des2, mutual_matches);

    // 至少需要 4 对点来拟合仿射模型
    if (mutual_matches.size() < 4) return 0;

    // =========================================================================
    // 第二阶段：【MAGSAC++ 空间一致性校验】
    // =========================================================================
    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : mutual_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    cv::Mat mask;
    std::vector<uchar> inliers_mask;

    if (P.use_affine_model) {
        // 使用针对大形变指纹的最佳模型：全仿射变换 
        // P.ransac_thresh 建议在 SiftConfig 里设置为 3.0 ~ 5.0
        cv::estimateAffine2D(pts1, pts2, inliers_mask, cv::USAC_MAGSAC, P.ransac_thresh);
        mask = cv::Mat(inliers_mask).clone();
    }
    else {
        // 单应性变换模型
        mask = cv::findHomography(pts1, pts2, cv::USAC_MAGSAC, P.ransac_thresh);
    }

    // =========================================================================
    // 第三阶段：【收网提取最终真点】
    // =========================================================================
    if (mask.empty()) return 0;

    int num_elements = std::max(mask.rows, mask.cols);
    for (int i = 0; i < num_elements; ++i) {
        // 只有被 MAGSAC++ 判定为符合物理空间规律的内点 (Inlier) 才会被保留
        if (mask.at<uchar>(i)) {
            final_matches.push_back(mutual_matches[i]);
        }
    }

    return (int)final_matches.size();
}