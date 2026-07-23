#ifndef SIFT_MATCHER2_H
#define SIFT_MATCHER2_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "SiftConfig.h"
#include "ManualSIFT.h"

// 全新的 SIFTMatcher2 类，专为指纹高重复纹理设计
class SIFTMatcher2 {
public:
    SIFTMatcher2(const SIFTParams& p, const SIFTConstants& c);

    void extractFeatures(const cv::Mat& img, const cv::Mat& mask,
        std::vector<cv::KeyPoint>& kp, cv::Mat& des) const;

    // 终极匹配逻辑：纯双向验证 + MAGSAC++ 空间校验
    int match(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
        const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
        std::vector<cv::DMatch>& final_matches) const;

private:
    SIFTParams P;
    SIFTConstants C;
    ManualSIFT sift;
};

#endif