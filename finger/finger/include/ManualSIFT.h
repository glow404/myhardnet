#ifndef MANUAL_SIFT_H
#define MANUAL_SIFT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "SiftConfig.h"

class ManualSIFT {
public:
    struct DetectionContext {
        cv::Mat preprocessed_gray;
        cv::Mat base;
        cv::Mat mask_base;
        int firstOctave = -1;
        int nOctaves = 0;
        std::vector<cv::Mat> gpyr;
        std::vector<cv::Mat> dogpyr;
        std::vector<cv::Mat> maskpyr;
    };

    struct CanonicalPatch {
        cv::Mat patch;
        cv::KeyPoint keypoint;
        int octave = 0;
        int layer = 0;
        float pyramid_scale = 1.0f;
    };

    ManualSIFT(const SIFTParams& p, const SIFTConstants& c);

    void detectAndCompute(const cv::Mat& img_input,
        const cv::Mat& mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) const;

    void detectKeypoints(const cv::Mat& img_input,
        const cv::Mat& mask,
        std::vector<cv::KeyPoint>& keypoints,
        DetectionContext* context = nullptr,
        bool apply_dedup = true) const;

    std::vector<CanonicalPatch> extractCanonicalPatches(
        const DetectionContext& context,
        const std::vector<cv::KeyPoint>& keypoints,
        int patch_size = 32) const;

    void computeDescriptorFromPatches(
        const DetectionContext& context,
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<CanonicalPatch>& patches,
        cv::Mat& descriptors) const;

private:
    SIFTParams P;
    SIFTConstants C;

    void preprocessImageAndMask(const cv::Mat& img_input,
        const cv::Mat& mask,
        DetectionContext& context) const;
    cv::Mat createInitialImage(const cv::Mat& img, bool doubleImageSize) const;
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
    void buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const;
    bool adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv, int& layer, int& r, int& c) const;
    float calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius, float sigma, std::vector<float>& hist, int n) const;
    std::vector<cv::KeyPoint> findScaleSpaceExtrema(const std::vector<cv::Mat>& gpyr, const std::vector<cv::Mat>& dog_pyr, int nOctaves, const std::vector<cv::Mat>& maskpyr) const;
    void calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl, int d, int n, cv::Mat& dstRow) const;
    cv::Mat calcDescriptors(const std::vector<cv::Mat>& gpyr, const std::vector<cv::KeyPoint>& keypoints, int nOctaveLayers, int firstOctave) const;
};

#endif
