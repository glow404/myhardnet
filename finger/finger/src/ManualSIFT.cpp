#include "ManualSIFT.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "Hadamard.h"

// 本文件是手写 SIFT 特征提取主实现，负责：
// 1. 图像预处理与尺度空间构建
// 2. 关键点检测、主方向估计与去重
// 3. 描述子生成，以及后续二值描述子入口准备
namespace {
bool useNoUpsampleCompensation(const SIFTParams& params) {
    return !params.enable_initial_upsample && params.enable_no_upsample_compensation;
}

int getInitialFirstOctave(const SIFTParams& params) {
    return params.enable_initial_upsample ? -1 : 0;
}

int getEffectiveOctaveLayers(const SIFTParams& params) {
    // 关闭上采样后，尺度层数自动切换到补偿参数。
    return params.enable_initial_upsample ? params.nOctaveLayers : params.no_upsample_nOctaveLayers;
}

double getEffectiveBaseSigma(const SIFTParams& params) {
    return params.enable_initial_upsample ? params.sigma : params.no_upsample_sigma;
}

double getEffectiveContrastThreshold(const SIFTParams& params) {
    return params.enable_initial_upsample ? params.contrastThreshold : params.no_upsample_contrast_threshold;
}

double getEffectiveEdgeThreshold(const SIFTParams& params) {
    return params.enable_initial_upsample ? params.edgeThreshold : params.no_upsample_edgeThreshold;
}

double getEffectiveClaheClip(const SIFTParams& params) {
    return params.enable_initial_upsample ? params.clahe_clip : params.no_upsample_clahe_clip;
}

double getEffectiveBlurSigma(const SIFTParams& params) {
    return params.enable_initial_upsample ? params.blur_sigma : params.no_upsample_blur_sigma;
}

float wrapAngleDiffDeg(float a, float b) {
    float diff = std::fabs(a - b);
    while (diff >= 360.0f) {
        diff -= 360.0f;
    }
    return std::min(diff, 360.0f - diff);
}

void deduplicateKeypoints(std::vector<cv::KeyPoint>& keypoints) {
    // 去重要在最终输出坐标系下完成，
    // 否则上采样阶段看似不同、缩回原图后几乎重合的点会漏掉。
    if (keypoints.size() < 2) {
        return;
    }

    std::stable_sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response > b.response;
    });

    std::vector<cv::KeyPoint> unique_keypoints;
    unique_keypoints.reserve(keypoints.size());

    for (const auto& candidate : keypoints) {
        const int candidate_octave = candidate.octave & 255;
        const int candidate_layer = (candidate.octave >> 8) & 255;
        bool duplicated = false;

        for (const auto& kept : unique_keypoints) {
            const int kept_octave = kept.octave & 255;
            const int kept_layer = (kept.octave >> 8) & 255;
            if (candidate_octave != kept_octave || candidate_layer != kept_layer || candidate.class_id != kept.class_id) {
                continue;
            }

            const float size_diff = std::fabs(candidate.size - kept.size);
            const float size_tol = std::max(0.25f, 0.05f * std::max(candidate.size, kept.size));
            if (size_diff > size_tol) {
                continue;
            }

            const float angle_diff = wrapAngleDiffDeg(candidate.angle, kept.angle);
            if (angle_diff > 5.0f) {
                continue;
            }

            const float dx = candidate.pt.x - kept.pt.x;
            const float dy = candidate.pt.y - kept.pt.y;
            const float dist2 = dx * dx + dy * dy;
            if (dist2 > 0.25f) {
                continue;
            }

            duplicated = true;
            break;
        }

        if (!duplicated) {
            unique_keypoints.push_back(candidate);
        }
    }

    keypoints.swap(unique_keypoints);
}
} // namespace

/**
 * ManualSIFT constructor
 */
ManualSIFT::ManualSIFT(const SIFTParams& p, const SIFTConstants& c) : P(p), C(c) {}

void ManualSIFT::preprocessImageAndMask(const cv::Mat& img_input,
    const cv::Mat& mask,
    DetectionContext& context) const {
    // 统一完成灰度化、CLAHE、模糊、初始上采样、mask 对齐，
    // 后续检测与描述子阶段都复用这个上下文。
    if (img_input.channels() == 3) {
        cv::cvtColor(img_input, context.preprocessed_gray, cv::COLOR_BGR2GRAY);
    }
    else {
        context.preprocessed_gray = img_input.clone();
    }
    if (context.preprocessed_gray.depth() != CV_8U) {
        context.preprocessed_gray.convertTo(context.preprocessed_gray, CV_8U);
    }

    if (P.enable_clahe) {
        auto clahe = cv::createCLAHE(getEffectiveClaheClip(P), P.clahe_tile);
        clahe->apply(context.preprocessed_gray, context.preprocessed_gray);
    }
    const double blur_sigma = getEffectiveBlurSigma(P);
    if (P.enable_blur && blur_sigma > 0) {
        cv::GaussianBlur(context.preprocessed_gray, context.preprocessed_gray, cv::Size(), blur_sigma, blur_sigma);
    }

    context.firstOctave = getInitialFirstOctave(P);
    context.base = createInitialImage(context.preprocessed_gray, context.firstOctave < 0);

    context.mask_base.release();
    if (!mask.empty()) {
        cv::Mat mask_float;
        mask.convertTo(mask_float, CV_32F);
        if (context.firstOctave < 0) {
            cv::resize(
                mask_float,
                context.mask_base,
                cv::Size(mask_float.cols * 2, mask_float.rows * 2),
                0,
                0,
                cv::INTER_NEAREST);
        }
        else {
            context.mask_base = mask_float;
        }
        context.mask_base.convertTo(context.mask_base, CV_8U);
    }

    context.nOctaves = static_cast<int>(
        std::round(std::log((double)std::min(context.base.cols, context.base.rows)) / std::log(2.0) - 2.0))
        - context.firstOctave;

    buildGaussianPyramid(context.base, context.gpyr, context.nOctaves);
    buildDoGPyramid(context.gpyr, context.dogpyr);

    context.maskpyr.clear();
    if (!context.mask_base.empty()) {
        const int nL = getEffectiveOctaveLayers(P);
        context.maskpyr.resize(context.nOctaves * (nL + 3));
        for (int o = 0; o < context.nOctaves; ++o) {
            for (int i = 0; i < nL + 3; ++i) {
                cv::Mat& dst = context.maskpyr[o * (nL + 3) + i];
                if (o == 0 && i == 0) {
                    dst = context.mask_base;
                }
                else if (i == 0) {
                    const cv::Mat& src = context.maskpyr[(o - 1) * (nL + 3) + nL];
                    cv::resize(src, dst, cv::Size(src.cols / 2, src.rows / 2), 0, 0, cv::INTER_NEAREST);
                }
                else {
                    dst = context.maskpyr[o * (nL + 3) + i - 1];
                }
            }
        }
    }
}

void ManualSIFT::detectKeypoints(const cv::Mat& img_input,
    const cv::Mat& mask,
    std::vector<cv::KeyPoint>& keypoints,
    DetectionContext* context,
    bool apply_dedup) const {
    // detectKeypoints 只负责“找到点”，不负责最终描述子。
    // 这样未来替换 learned descriptor 时可以直接复用检测结果。
    DetectionContext local_context;
    DetectionContext& active_context = context ? *context : local_context;
    preprocessImageAndMask(img_input, mask, active_context);

    keypoints = findScaleSpaceExtrema(
        active_context.gpyr,
        active_context.dogpyr,
        active_context.nOctaves,
        active_context.maskpyr);

    if (active_context.firstOctave < 0) {
        const float scale = 0.5f;
        for (auto& kpt : keypoints) {
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + active_context.firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }
    }

    // 去重要在最终输出坐标系下做，否则上采样阶段相差半个像素以上、
    // 缩回原图后又几乎重合的点会漏掉。
    if (apply_dedup) {
        deduplicateKeypoints(keypoints);
    }

    if (P.nfeatures > 0 && static_cast<int>(keypoints.size()) > P.nfeatures) {
        std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
            return a.response > b.response;
            });
        keypoints.resize(P.nfeatures);
    }
}

std::vector<ManualSIFT::CanonicalPatch> ManualSIFT::extractCanonicalPatches(
    const DetectionContext& context,
    const std::vector<cv::KeyPoint>& keypoints,
    int patch_size) const {
    std::vector<CanonicalPatch> patches;
    if (patch_size <= 0) {
        return patches;
    }

    patches.reserve(keypoints.size());
    for (const auto& keypoint : keypoints) {
        int octave = keypoint.octave & 255;
        octave = (octave < 128) ? octave : (-128 | octave);
        const int layer = (keypoint.octave >> 8) & 255;
        const float pyramid_scale = (octave >= 0) ? 1.f / (1 << octave) : static_cast<float>(1 << -octave);

        const int nL = getEffectiveOctaveLayers(P);
        const int pyramid_index = (octave - context.firstOctave) * (nL + 3) + layer;
        if (pyramid_index < 0 || pyramid_index >= static_cast<int>(context.gpyr.size())) {
            continue;
        }

        const cv::Mat& pyramid_img = context.gpyr[pyramid_index];
        const cv::Point2f center(keypoint.pt.x * pyramid_scale, keypoint.pt.y * pyramid_scale);
        const float angle_rad = (360.0f - keypoint.angle) * static_cast<float>(CV_PI / 180.0);
        const float half_window = std::max(1.0f, keypoint.size * pyramid_scale * 0.5f * C.SIFT_DESCR_SCL_FCTR);
        const float scale_factor = (2.0f * half_window) / static_cast<float>(patch_size);
        const float cos_a = std::cos(angle_rad) * scale_factor;
        const float sin_a = std::sin(angle_rad) * scale_factor;
        const float center_offset = static_cast<float>(patch_size - 1) * 0.5f;

        cv::Matx23f warp(
            cos_a, -sin_a, center.x - cos_a * center_offset + sin_a * center_offset,
            sin_a, cos_a, center.y - sin_a * center_offset - cos_a * center_offset);

        CanonicalPatch patch_info;
        patch_info.keypoint = keypoint;
        patch_info.octave = octave;
        patch_info.layer = layer;
        patch_info.pyramid_scale = pyramid_scale;
        cv::warpAffine(
            pyramid_img,
            patch_info.patch,
            warp,
            cv::Size(patch_size, patch_size),
            cv::INTER_LINEAR,
            cv::BORDER_REFLECT101);
        patches.push_back(std::move(patch_info));
    }

    return patches;
}

void ManualSIFT::computeDescriptorFromPatches(
    const DetectionContext& context,
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<CanonicalPatch>& patches,
    cv::Mat& descriptors) const {
    // 当前仍走传统 SIFT 描述子主路径，
    // patches 已经准备好，后面接 AI descriptor 时可以直接切入。
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    // Canonical patches are prepared for the future learned descriptor path.
    // The current implementation intentionally keeps the original descriptor
    // computation so all existing executables preserve their previous behavior.
    (void)patches;
    cv::Mat float_descriptors = calcDescriptors(context.gpyr, keypoints, getEffectiveOctaveLayers(P), context.firstOctave);

    if (P.enable_hamming_match) {
        descriptors.create(float_descriptors.rows, 4, CV_32F);
        for (int i = 0; i < float_descriptors.rows; ++i) {
            uint64_t bin[2];
            Hadamard::applyHadamardProjection(float_descriptors.ptr<float>(i), bin);
            std::memcpy(descriptors.ptr<float>(i), bin, 16);
        }
    }
    else {
        if (P.enable_rootsift) {
            for (int i = 0; i < float_descriptors.rows; ++i) {
                cv::Mat row = float_descriptors.row(i);
                double s = cv::sum(row)[0] + 1e-7;
                row /= s;
                cv::sqrt(row, row);
            }
        }
        descriptors = float_descriptors;
    }
}

/**
 * Main entry for SIFT detection and descriptor computation.
 * Internally split into:
 * 1. detectKeypoints
 * 2. extractCanonicalPatches
 * 3. computeDescriptorFromPatches
 */
void ManualSIFT::detectAndCompute(const cv::Mat& img_input, const cv::Mat& mask,
    std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const {
    // 兼容旧接口的总入口：
    // 外部仍按 detectAndCompute 调用，内部已拆成三段复用。
    DetectionContext context;
    detectKeypoints(img_input, mask, keypoints, &context);
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    const std::vector<CanonicalPatch> patches = extractCanonicalPatches(context, keypoints);
    computeDescriptorFromPatches(context, keypoints, patches, descriptors);
}

cv::Mat ManualSIFT::createInitialImage(const cv::Mat& img, bool doubleImageSize) const {
    cv::Mat gray_fpt;
    img.convertTo(gray_fpt, CV_32F, C.SIFT_FIXPT_SCALE, 0);
    float sig_diff;
    const float base_sigma = static_cast<float>(getEffectiveBaseSigma(P));
    if (doubleImageSize) {
        sig_diff = std::sqrt(std::max(base_sigma * base_sigma - C.SIFT_INIT_SIGMA * C.SIFT_INIT_SIGMA * 4.f, 0.01f));
        cv::Mat dbl;
        cv::resize(gray_fpt, dbl, cv::Size(gray_fpt.cols * 2, gray_fpt.rows * 2), 0, 0, cv::INTER_LINEAR);
        cv::Mat result;
        cv::GaussianBlur(dbl, result, cv::Size(), sig_diff, sig_diff);
        return result;
    }
    else {
        sig_diff = std::sqrt(std::max(base_sigma * base_sigma - C.SIFT_INIT_SIGMA * C.SIFT_INIT_SIGMA, 0.01f));
        cv::Mat result;
        cv::GaussianBlur(gray_fpt, result, cv::Size(), sig_diff, sig_diff);
        return result;
    }
}

void ManualSIFT::buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const {
    const int nL = getEffectiveOctaveLayers(P);
    std::vector<double> sig(nL + 3);
    const double base_sigma = getEffectiveBaseSigma(P);
    sig[0] = base_sigma;
    const double k = std::pow(2.0, 1.0 / nL);
    for (int i = 1; i < nL + 3; ++i) {
        const double sig_prev = std::pow(k, i - 1) * base_sigma;
        const double sig_total = sig_prev * k;
        sig[i] = std::sqrt(std::max(sig_total * sig_total - sig_prev * sig_prev, 0.0));
    }

    pyr.resize(nOctaves * (nL + 3));
    for (int o = 0; o < nOctaves; ++o) {
        for (int i = 0; i < nL + 3; ++i) {
            cv::Mat& dst = pyr[o * (nL + 3) + i];
            if (o == 0 && i == 0) {
                dst = base;
            }
            else if (i == 0) {
                const cv::Mat& src = pyr[(o - 1) * (nL + 3) + nL];
                cv::resize(src, dst, cv::Size(src.cols / 2, src.rows / 2), 0, 0, cv::INTER_NEAREST);
            }
            else {
                const cv::Mat& src = pyr[o * (nL + 3) + i - 1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

void ManualSIFT::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const {
    const int nL = getEffectiveOctaveLayers(P);
    const int nOctaves = (int)gpyr.size() / (nL + 3);
    dogpyr.resize(nOctaves * (nL + 2));
    for (int o = 0; o < nOctaves; ++o) {
        for (int i = 0; i < nL + 2; ++i) {
            const cv::Mat& s1 = gpyr[o * (nL + 3) + i];
            const cv::Mat& s2 = gpyr[o * (nL + 3) + i + 1];
            cv::subtract(s2, s1, dogpyr[o * (nL + 2) + i], cv::noArray(), CV_32F);
        }
    }
}

bool ManualSIFT::adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv, int& layer, int& r, int& c) const {
    const int nL = getEffectiveOctaveLayers(P);
    const float img_scale = 1.f / (255.f * C.SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale * 0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale * 0.25f;
    float xi = 0;
    float xr = 0;
    float xc = 0;
    int i = 0;

    for (; i < C.SIFT_MAX_INTERP_STEPS; ++i) {
        const int idx = octv * (nL + 2) + layer;
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx - 1];
        const cv::Mat& next = dog_pyr[idx + 1];

        cv::Vec3f dD(
            (img.at<float>(r, c + 1) - img.at<float>(r, c - 1)) * deriv_scale,
            (img.at<float>(r + 1, c) - img.at<float>(r - 1, c)) * deriv_scale,
            (next.at<float>(r, c) - prev.at<float>(r, c)) * deriv_scale);

        const float v2 = img.at<float>(r, c) * 2.f;
        const float dxx = (img.at<float>(r, c + 1) + img.at<float>(r, c - 1) - v2) * second_deriv_scale;
        const float dyy = (img.at<float>(r + 1, c) + img.at<float>(r - 1, c) - v2) * second_deriv_scale;
        const float dss = (next.at<float>(r, c) + prev.at<float>(r, c) - v2) * second_deriv_scale;
        const float dxy = (img.at<float>(r + 1, c + 1) - img.at<float>(r + 1, c - 1) - img.at<float>(r - 1, c + 1) + img.at<float>(r - 1, c - 1)) * cross_deriv_scale;
        const float dxs = (next.at<float>(r, c + 1) - next.at<float>(r, c - 1) - prev.at<float>(r, c + 1) + prev.at<float>(r, c - 1)) * cross_deriv_scale;
        const float dys = (next.at<float>(r + 1, c) - next.at<float>(r - 1, c) - prev.at<float>(r + 1, c) + prev.at<float>(r - 1, c)) * cross_deriv_scale;

        cv::Matx33f H(dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss);
        cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);
        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f) {
            break;
        }

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);
        if (layer < 1 || layer > nL || c < C.SIFT_IMG_BORDER || c >= img.cols - C.SIFT_IMG_BORDER || r < C.SIFT_IMG_BORDER || r >= img.rows - C.SIFT_IMG_BORDER) {
            return false;
        }
    }

    if (i >= C.SIFT_MAX_INTERP_STEPS) {
        return false;
    }

    const cv::Mat& img = dog_pyr[octv * (nL + 2) + layer];
    cv::Matx31f dD(
        (img.at<float>(r, c + 1) - img.at<float>(r, c - 1)) * deriv_scale,
        (img.at<float>(r + 1, c) - img.at<float>(r - 1, c)) * deriv_scale,
        (dog_pyr[octv * (nL + 2) + layer + 1].at<float>(r, c) - dog_pyr[octv * (nL + 2) + layer - 1].at<float>(r, c)) * deriv_scale);
    const float contr = img.at<float>(r, c) * img_scale + dD.dot(cv::Matx31f(xc, xr, xi)) * 0.5f;

    if (std::abs(contr) * nL < getEffectiveContrastThreshold(P)) {
        return false;
    }

    const float v2 = img.at<float>(r, c) * 2.f;
    const float dxx = (img.at<float>(r, c + 1) + img.at<float>(r, c - 1) - v2) * second_deriv_scale;
    const float dyy = (img.at<float>(r + 1, c) + img.at<float>(r - 1, c) - v2) * second_deriv_scale;
    const float dxy = (img.at<float>(r + 1, c + 1) - img.at<float>(r + 1, c - 1) - img.at<float>(r - 1, c + 1) + img.at<float>(r - 1, c - 1)) * cross_deriv_scale;
    const float tr = dxx + dyy;
    const float det = dxx * dyy - dxy * dxy;
    const double edge_threshold = getEffectiveEdgeThreshold(P);
    if (det <= 0 || tr * tr * edge_threshold >= (edge_threshold + 1) * (edge_threshold + 1) * det) {
        return false;
    }

    kpt.pt = cv::Point2f((c + xc) * (1 << octv), (r + xr) * (1 << octv));
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5f) * 255) << 16);
    kpt.size = (float)getEffectiveBaseSigma(P) * std::pow(2.f, (layer + xi) / (float)nL) * (1 << octv) * 2.f;
    kpt.response = std::abs(contr);
    return true;
}

float ManualSIFT::calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius, float sigma, std::vector<float>& hist, int n) const {
    hist.assign(n, 0.f);
    const float expf_scale = -1.f / (2.f * sigma * sigma);
    std::vector<float> temphist(n + 4, 0.f);
    for (int i = -radius; i <= radius; ++i) {
        const int y = pt.y + i;
        if (y <= 0 || y >= img.rows - 1) {
            continue;
        }
        for (int j = -radius; j <= radius; ++j) {
            const int x = pt.x + j;
            if (x <= 0 || x >= img.cols - 1) {
                continue;
            }
            const float dx = img.at<float>(y, x + 1) - img.at<float>(y, x - 1);
            const float dy = img.at<float>(y - 1, x) - img.at<float>(y + 1, x);
            const float W = std::exp((float)(i * i + j * j) * expf_scale);
            const float Mag = std::sqrt(dx * dx + dy * dy);
            float Ori = std::atan2(dy, dx) * 180.f / (float)CV_PI;
            if (Ori < 0) {
                Ori += 360.f;
            }
            int bin = cvRound((n / 360.f) * Ori);
            if (bin >= n) {
                bin -= n;
            }
            if (bin < 0) {
                bin += n;
            }
            temphist[bin + 2] += W * Mag;
        }
    }
    temphist[0] = temphist[n];
    temphist[1] = temphist[n + 1];
    temphist[n + 2] = temphist[2];
    temphist[n + 3] = temphist[3];
    for (int i = 0; i < n; ++i) {
        hist[i] = (temphist[i] + temphist[i + 4]) * (1.f / 16.f) + (temphist[i + 1] + temphist[i + 3]) * (4.f / 16.f) + temphist[i + 2] * (6.f / 16.f);
    }
    float maxval = hist[0];
    for (int i = 1; i < n; ++i) {
        maxval = std::max(maxval, hist[i]);
    }
    return maxval;
}

std::vector<cv::KeyPoint> ManualSIFT::findScaleSpaceExtrema(const std::vector<cv::Mat>& gpyr,
    const std::vector<cv::Mat>& dog_pyr,
    int nOctaves,
    const std::vector<cv::Mat>& maskpyr) const {
    std::vector<cv::KeyPoint> keypoints;
    const int nL = getEffectiveOctaveLayers(P);
    const float threshold = cvFloor(0.5 * getEffectiveContrastThreshold(P) / nL * 255 * C.SIFT_FIXPT_SCALE);

    for (int o = 0; o < nOctaves; ++o) {
        for (int i = 1; i <= nL; ++i) {
            const cv::Mat& img = dog_pyr[o * (nL + 2) + i];
            const cv::Mat& prev = dog_pyr[o * (nL + 2) + i - 1];
            const cv::Mat& next = dog_pyr[o * (nL + 2) + i + 1];
            const cv::Mat* maskPtr = maskpyr.empty() ? nullptr : &maskpyr[o * (nL + 3) + i];

            for (int r = C.SIFT_IMG_BORDER; r < img.rows - C.SIFT_IMG_BORDER; ++r) {
                for (int c = C.SIFT_IMG_BORDER; c < img.cols - C.SIFT_IMG_BORDER; ++c) {
                    if (maskPtr && maskPtr->at<uchar>(r, c) == 0) {
                        continue;
                    }

                    const float val = img.at<float>(r, c);
                    if (std::abs(val) <= threshold) {
                        continue;
                    }

                    bool isExtrema = false;
                    int kpt_type = -1;

                    if (val > 0) {
                        const float vMax = std::max({ img.at<float>(r - 1, c - 1), img.at<float>(r - 1, c), img.at<float>(r - 1, c + 1),
                            img.at<float>(r, c - 1), img.at<float>(r, c + 1),
                            img.at<float>(r + 1, c - 1), img.at<float>(r + 1, c), img.at<float>(r + 1, c + 1) });
                        if (val < vMax) {
                            continue;
                        }
                        const float pMax = std::max({ prev.at<float>(r - 1, c - 1), prev.at<float>(r - 1, c), prev.at<float>(r - 1, c + 1),
                            prev.at<float>(r, c - 1), prev.at<float>(r, c), prev.at<float>(r, c + 1),
                            prev.at<float>(r + 1, c - 1), prev.at<float>(r + 1, c), prev.at<float>(r + 1, c + 1) });
                        if (val < pMax) {
                            continue;
                        }
                        const float nMax = std::max({ next.at<float>(r - 1, c - 1), next.at<float>(r - 1, c), next.at<float>(r - 1, c + 1),
                            next.at<float>(r, c - 1), next.at<float>(r, c), next.at<float>(r, c + 1),
                            next.at<float>(r + 1, c - 1), next.at<float>(r + 1, c), next.at<float>(r + 1, c + 1) });
                        if (val < nMax) {
                            continue;
                        }
                        isExtrema = true;
                        kpt_type = 1;
                    }
                    else {
                        const float vMin = std::min({ img.at<float>(r - 1, c - 1), img.at<float>(r - 1, c), img.at<float>(r - 1, c + 1),
                            img.at<float>(r, c - 1), img.at<float>(r, c + 1),
                            img.at<float>(r + 1, c - 1), img.at<float>(r + 1, c), img.at<float>(r + 1, c + 1) });
                        if (val > vMin) {
                            continue;
                        }
                        const float pMin = std::min({ prev.at<float>(r - 1, c - 1), prev.at<float>(r - 1, c), prev.at<float>(r - 1, c + 1),
                            prev.at<float>(r, c - 1), prev.at<float>(r, c), prev.at<float>(r, c + 1),
                            prev.at<float>(r + 1, c - 1), prev.at<float>(r + 1, c), prev.at<float>(r + 1, c + 1) });
                        if (val > pMin) {
                            continue;
                        }
                        const float nMin = std::min({ next.at<float>(r - 1, c - 1), next.at<float>(r - 1, c), next.at<float>(r - 1, c + 1),
                            next.at<float>(r, c - 1), next.at<float>(r, c), next.at<float>(r, c + 1),
                            next.at<float>(r + 1, c - 1), next.at<float>(r + 1, c), next.at<float>(r + 1, c + 1) });
                        if (val > nMin) {
                            continue;
                        }
                        isExtrema = true;
                        kpt_type = 0;
                    }

                    if (!isExtrema) {
                        continue;
                    }

                    cv::KeyPoint kpt;
                    int r1 = r;
                    int c1 = c;
                    int layer = i;
                    if (adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1)) {
                        kpt.class_id = kpt_type;

                        const float scl = kpt.size * 0.5f / (1 << o);
                        std::vector<float> hist;
                        calcOrientationHist(
                            gpyr[o * (nL + 3) + layer],
                            cv::Point(c1, r1),
                            cvRound(C.SIFT_ORI_RADIUS * scl),
                            C.SIFT_ORI_SIG_FCTR * scl,
                            hist,
                            C.SIFT_ORI_HIST_BINS);

                        int max_bin = -1;
                        float max_val = -1.0f;
                        for (int j = 0; j < C.SIFT_ORI_HIST_BINS; ++j) {
                            if (hist[j] > max_val) {
                                max_val = hist[j];
                                max_bin = j;
                            }
                        }

                        if (max_bin != -1) {
                            const int l = (max_bin > 0) ? max_bin - 1 : C.SIFT_ORI_HIST_BINS - 1;
                            const int r2 = (max_bin < C.SIFT_ORI_HIST_BINS - 1) ? max_bin + 1 : 0;

                            float bin = max_bin + 0.5f * (hist[l] - hist[r2]) / (hist[l] - 2 * hist[max_bin] + hist[r2] + 1e-7f);
                            bin = (bin < 0) ? bin + C.SIFT_ORI_HIST_BINS : (bin >= C.SIFT_ORI_HIST_BINS ? bin - C.SIFT_ORI_HIST_BINS : bin);

                            kpt.angle = 360.f - (360.f / C.SIFT_ORI_HIST_BINS) * bin;
                            if (std::abs(kpt.angle - 360.f) < FLT_EPSILON) {
                                kpt.angle = 0.f;
                            }

                            keypoints.push_back(kpt);
                        }
                    }
                }
            }
        }
    }
    return keypoints;
}

void ManualSIFT::calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl, int d, int n, cv::Mat& dstRow) const {
    const cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = std::cos(ori * (float)(CV_PI / 180));
    float sin_t = std::sin(ori * (float)(CV_PI / 180));
    const float bins_per_rad = n / 360.f;
    const float exp_scale = -1.f / (d * d * 0.5f);
    const float hist_width = C.SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.41421356f * (d + 1) * 0.5f);
    radius = std::min(radius, (int)std::sqrt((double)img.cols * img.cols + (double)img.rows * img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    std::vector<float> hist((d + 2) * (d + 2) * (n + 2), 0.f);
    for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
            const float c_rot = j * cos_t - i * sin_t;
            const float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d / 2.f - 0.5f;
            float cbin = c_rot + d / 2.f - 0.5f;
            const int r = pt.y + i;
            const int c = pt.x + j;

            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d && r > 0 && r < img.rows - 1 && c > 0 && c < img.cols - 1) {
                const float dx = img.at<float>(r, c + 1) - img.at<float>(r, c - 1);
                const float dy = img.at<float>(r - 1, c) - img.at<float>(r + 1, c);
                const float mag = std::sqrt(dx * dx + dy * dy);
                const float weight = std::exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
                float obin = (std::atan2(dy, dx) * 180.f / (float)CV_PI - ori) * bins_per_rad;

                const int r0 = cvFloor(rbin);
                const int c0 = cvFloor(cbin);
                int o0 = cvFloor(obin);
                rbin -= r0;
                cbin -= c0;
                obin -= o0;
                o0 = (o0 % n + n) % n;

                const float v_r1 = mag * weight * rbin;
                const float v_r0 = mag * weight - v_r1;
                const float v_rc11 = v_r1 * cbin;
                const float v_rc10 = v_r1 - v_rc11;
                const float v_rc01 = v_r0 * cbin;
                const float v_rc00 = v_r0 - v_rc01;

                hist[((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0] += v_rc00 * (1 - obin);
                hist[((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + (o0 + 1) % n] += v_rc00 * obin;
                hist[((r0 + 1) * (d + 2) + c0 + 2) * (n + 2) + o0] += v_rc01 * (1 - obin);
                hist[((r0 + 1) * (d + 2) + c0 + 2) * (n + 2) + (o0 + 1) % n] += v_rc01 * obin;
                hist[((r0 + 2) * (d + 2) + c0 + 1) * (n + 2) + o0] += v_rc10 * (1 - obin);
                hist[((r0 + 2) * (d + 2) + c0 + 1) * (n + 2) + (o0 + 1) % n] += v_rc10 * obin;
                hist[((r0 + 2) * (d + 2) + c0 + 2) * (n + 2) + o0] += v_rc11 * (1 - obin);
                hist[((r0 + 2) * (d + 2) + c0 + 2) * (n + 2) + (o0 + 1) % n] += v_rc11 * obin;
            }
        }
    }

    std::vector<float> rawDst(d * d * n, 0.f);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int k = 0; k < n; ++k) {
                rawDst[(i * d + j) * n + k] = hist[((i + 1) * (d + 2) + j + 1) * (n + 2) + k];
            }
        }
    }

    double nrm2 = 0;
    for (float v : rawDst) {
        nrm2 += v * v;
    }
    nrm2 = std::sqrt(nrm2);
    const float thr = (float)(nrm2 * C.SIFT_DESCR_MAG_THR);
    for (float& v : rawDst) {
        v = std::min(v, thr);
    }
    nrm2 = 0;
    for (float v : rawDst) {
        nrm2 += v * v;
    }
    nrm2 = std::sqrt(nrm2);
    const float scale = (float)(C.SIFT_INT_DESCR_FCTR / std::max((float)nrm2, FLT_EPSILON));
    dstRow.create(1, d * d * n, CV_32F);
    for (int k = 0; k < d * d * n; ++k) {
        dstRow.at<float>(0, k) = std::min(std::max(std::round(rawDst[k] * scale), 0.f), 255.f);
    }
}

cv::Mat ManualSIFT::calcDescriptors(const std::vector<cv::Mat>& gpyr, const std::vector<cv::KeyPoint>& keypoints, int nOctaveLayers, int firstOctave) const {
    const int d = C.SIFT_DESCR_WIDTH;
    const int n = C.SIFT_DESCR_HIST_BINS;
    cv::Mat descriptors((int)keypoints.size(), d * d * n, CV_32F);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        cv::KeyPoint kpt = keypoints[i];
        int octave = kpt.octave & 255;
        octave = (octave < 128) ? octave : (-128 | octave);
        const int layer = (kpt.octave >> 8) & 255;
        const float scale = (octave >= 0) ? 1.f / (1 << octave) : (float)(1 << -octave);
        cv::Mat row;
        calcSIFTDescriptor(
            gpyr[(octave - firstOctave) * (nOctaveLayers + 3) + layer],
            cv::Point2f(kpt.pt.x * scale, kpt.pt.y * scale),
            360.f - kpt.angle,
            kpt.size * scale * 0.5f,
            d,
            n,
            row);
        row.copyTo(descriptors.row((int)i));
    }
    return descriptors;
}
