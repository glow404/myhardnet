#ifndef SIFT_MATCHER_H
#define SIFT_MATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "SiftConfig.h"
#include "ManualSIFT.h"

struct SIFTMatchDebugStats {
    int mutual_candidate_count = 0;
    int final_inlier_count = 0;
};

class SIFTMatcher {
public:
    SIFTMatcher(const SIFTParams& p, const SIFTConstants& c);

    void extractFeatures(const cv::Mat& img, const cv::Mat& mask,
        std::vector<cv::KeyPoint>& kp, cv::Mat& des) const;

    int match(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
        const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
        std::vector<cv::DMatch>& final_matches) const;

    int matchWithStats(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
        const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
        std::vector<cv::DMatch>& final_matches,
        SIFTMatchDebugStats* stats) const;

private:
    SIFTParams P;
    SIFTConstants C;
    ManualSIFT sift;
    cv::BFMatcher bf;
};

#endif
