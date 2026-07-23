#include "SIFTMatcher.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>
#include <limits>
#include <unordered_map>
#include <vector>

#include "Hadamard.h"
#include "SiftConfig.h"

// 本文件是当前主链路里“匹配与几何验证”的核心：
// 1. 先做粗匹配（支持 Hamming 与 180 度翻转补偿）
// 2. 再做方向一致性筛选
// 3. 最后做几何/方向联合 RANSAC，输出最终内点
namespace {
struct MatchCandidate {
    cv::DMatch match;
    bool used_flip = false;
    float compensated_delta_theta = 0.0f;
};

inline float wrap_angle_deg(float angle) {
    while (angle > 180.0f) angle -= 360.0f;
    while (angle <= -180.0f) angle += 360.0f;
    return angle;
}

inline float compute_compensated_delta_theta(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, bool used_flip) {
    const float kp2_angle = used_flip ? (kp2.angle + 180.0f) : kp2.angle;
    return wrap_angle_deg(kp2_angle - kp1.angle);
}

inline cv::Point2d apply_affine(const cv::Matx23d& affine, const cv::Point2f& pt) {
    return cv::Point2d(
        affine(0, 0) * pt.x + affine(0, 1) * pt.y + affine(0, 2),
        affine(1, 0) * pt.x + affine(1, 1) * pt.y + affine(1, 2));
}

float mean_angle_deg(float a, float b) {
    const double s = std::sin(a * CV_PI / 180.0) + std::sin(b * CV_PI / 180.0);
    const double c = std::cos(a * CV_PI / 180.0) + std::cos(b * CV_PI / 180.0);
    if (std::abs(s) < 1e-9 && std::abs(c) < 1e-9) {
        return wrap_angle_deg((a + b) * 0.5f);
    }
    return wrap_angle_deg(static_cast<float>(std::atan2(s, c) * 180.0 / CV_PI));
}

float mean_angle_deg(const std::vector<float>& angles) {
    double s = 0.0;
    double c = 0.0;
    for (float angle : angles) {
        s += std::sin(angle * CV_PI / 180.0);
        c += std::cos(angle * CV_PI / 180.0);
    }
    if (std::abs(s) < 1e-9 && std::abs(c) < 1e-9) {
        return 0.0f;
    }
    return wrap_angle_deg(static_cast<float>(std::atan2(s, c) * 180.0 / CV_PI));
}

inline float extract_rotation_deg_from_affine(const cv::Matx23d& affine) {
    return wrap_angle_deg(static_cast<float>(std::atan2(affine(1, 0), affine(0, 0)) * 180.0 / CV_PI));
}

bool estimate_similarity_from_matches(
    const std::vector<MatchCandidate>& samples,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    cv::Matx23d& affine) {
    if (samples.size() < 2) {
        return false;
    }

    cv::Point2d centroid1(0.0, 0.0);
    cv::Point2d centroid2(0.0, 0.0);
    for (const auto& sample : samples) {
        centroid1 += cv::Point2d(kp1[sample.match.queryIdx].pt);
        centroid2 += cv::Point2d(kp2[sample.match.trainIdx].pt);
    }
    centroid1 *= 1.0 / static_cast<double>(samples.size());
    centroid2 *= 1.0 / static_cast<double>(samples.size());

    double cross = 0.0;
    double dot = 0.0;
    double norm1 = 0.0;
    for (const auto& sample : samples) {
        const cv::Point2d p1 = cv::Point2d(kp1[sample.match.queryIdx].pt) - centroid1;
        const cv::Point2d p2 = cv::Point2d(kp2[sample.match.trainIdx].pt) - centroid2;
        dot += p1.x * p2.x + p1.y * p2.y;
        cross += p1.x * p2.y - p1.y * p2.x;
        norm1 += p1.x * p1.x + p1.y * p1.y;
    }
    if (norm1 < 1e-6) {
        return false;
    }

    const double theta = std::atan2(cross, dot);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);

    double scale_num = 0.0;
    for (const auto& sample : samples) {
        const cv::Point2d p1 = cv::Point2d(kp1[sample.match.queryIdx].pt) - centroid1;
        const cv::Point2d p2 = cv::Point2d(kp2[sample.match.trainIdx].pt) - centroid2;
        const double rx = cos_theta * p1.x - sin_theta * p1.y;
        const double ry = sin_theta * p1.x + cos_theta * p1.y;
        scale_num += p2.x * rx + p2.y * ry;
    }
    const double scale = scale_num / norm1;
    if (std::abs(scale) < 1e-6) {
        return false;
    }

    const double cos_t = cos_theta * scale;
    const double sin_t = sin_theta * scale;
    const double tx = centroid2.x - (cos_t * centroid1.x - sin_t * centroid1.y);
    const double ty = centroid2.y - (sin_t * centroid1.x + cos_t * centroid1.y);
    affine = cv::Matx23d(cos_t, -sin_t, tx,
                         sin_t,  cos_t, ty);
    return true;
}

bool orientation_peak_filter_candidates(
    const SIFTParams& params,
    std::vector<MatchCandidate>& candidates) {
    if (!params.enable_orientation_peak_filter) {
        return candidates.size() >= 4;
    }
    if (candidates.size() < 4) {
        return false;
    }

    const int num_bins = std::max(1, static_cast<int>(std::ceil(360.0f / params.orientation_peak_bin_deg)));
    std::vector<int> hist(num_bins, 0);
    for (const auto& candidate : candidates) {
        const float wrapped = wrap_angle_deg(candidate.compensated_delta_theta);
        const float shifted = wrapped + 180.0f;
        int bin = static_cast<int>(std::floor(shifted / params.orientation_peak_bin_deg));
        bin = std::max(0, std::min(num_bins - 1, bin));
        hist[bin] += 1;
    }

    const int best_bin = static_cast<int>(std::distance(hist.begin(), std::max_element(hist.begin(), hist.end())));
    const float peak_center = -180.0f + (static_cast<float>(best_bin) + 0.5f) * params.orientation_peak_bin_deg;

    std::vector<MatchCandidate> filtered;
    filtered.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const float diff = std::abs(wrap_angle_deg(candidate.compensated_delta_theta - peak_center));
        if (diff <= params.orientation_peak_band_deg) {
            filtered.push_back(candidate);
        }
    }

    const float peak_ratio = static_cast<float>(filtered.size()) / static_cast<float>(candidates.size());
    if (peak_ratio < params.orientation_peak_min_ratio || filtered.size() < 4) {
        candidates.clear();
        return false;
    }

    candidates.swap(filtered);
    return true;
}

bool triplet_has_consistent_orientation(
    const SIFTParams& params,
    const std::array<const MatchCandidate*, 3>& samples) {
    const float theta01 = std::abs(wrap_angle_deg(samples[0]->compensated_delta_theta - samples[1]->compensated_delta_theta));
    const float theta02 = std::abs(wrap_angle_deg(samples[0]->compensated_delta_theta - samples[2]->compensated_delta_theta));
    const float theta12 = std::abs(wrap_angle_deg(samples[1]->compensated_delta_theta - samples[2]->compensated_delta_theta));
    return theta01 <= params.orientation_sample_thresh_deg &&
           theta02 <= params.orientation_sample_thresh_deg &&
           theta12 <= params.orientation_sample_thresh_deg;
}

double triangle_area2(const cv::Point2d& a, const cv::Point2d& b, const cv::Point2d& c) {
    return std::abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}

bool triplet_has_consistent_structure(
    const SIFTParams& params,
    const std::array<const MatchCandidate*, 3>& samples,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2) {
    std::array<cv::Point2d, 3> src_pts;
    std::array<cv::Point2d, 3> dst_pts;
    for (int i = 0; i < 3; ++i) {
        src_pts[i] = kp1[samples[i]->match.queryIdx].pt;
        dst_pts[i] = kp2[samples[i]->match.trainIdx].pt;
    }

    const double src_area2 = triangle_area2(src_pts[0], src_pts[1], src_pts[2]);
    const double dst_area2 = triangle_area2(dst_pts[0], dst_pts[1], dst_pts[2]);
    if (src_area2 < params.orientation_triplet_area_min || dst_area2 < params.orientation_triplet_area_min) {
        return false;
    }

    std::array<double, 3> src_len = {
        cv::norm(src_pts[0] - src_pts[1]),
        cv::norm(src_pts[0] - src_pts[2]),
        cv::norm(src_pts[1] - src_pts[2])
    };
    std::array<double, 3> dst_len = {
        cv::norm(dst_pts[0] - dst_pts[1]),
        cv::norm(dst_pts[0] - dst_pts[2]),
        cv::norm(dst_pts[1] - dst_pts[2])
    };
    std::sort(src_len.begin(), src_len.end());
    std::sort(dst_len.begin(), dst_len.end());
    if (src_len[0] < 1e-6 || dst_len[0] < 1e-6) {
        return false;
    }

    const double src_ratio1 = src_len[1] / src_len[0];
    const double src_ratio2 = src_len[2] / src_len[0];
    const double dst_ratio1 = dst_len[1] / dst_len[0];
    const double dst_ratio2 = dst_len[2] / dst_len[0];
    return std::abs(src_ratio1 - dst_ratio1) <= params.orientation_triplet_side_ratio_thresh &&
           std::abs(src_ratio2 - dst_ratio2) <= params.orientation_triplet_side_ratio_thresh;
}

bool sample_has_consistent_orientation(
    const SIFTParams& params,
    const std::vector<const MatchCandidate*>& samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t j = i + 1; j < samples.size(); ++j) {
            const float delta = std::abs(wrap_angle_deg(
                samples[i]->compensated_delta_theta - samples[j]->compensated_delta_theta));
            if (delta > params.orientation_sample_thresh_deg) {
                return false;
            }
        }
    }
    return true;
}

int run_legacy_geometry_verification(
    const SIFTParams& params,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& mutual_candidates,
    std::vector<cv::DMatch>& final_matches) {
    if (mutual_candidates.size() < 4) return 0;

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    pts1.reserve(mutual_candidates.size());
    pts2.reserve(mutual_candidates.size());
    for (const auto& m : mutual_candidates) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    cv::Mat mask;
    if (params.use_affine_model) {
        cv::estimateAffinePartial2D(pts1, pts2, mask, cv::RANSAC, params.ransac_thresh);
    }
    else {
        mask = cv::findHomography(pts1, pts2, cv::RANSAC, params.ransac_thresh);
    }

    if (mask.empty()) return 0;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            final_matches.push_back(mutual_candidates[i]);
        }
    }
    return static_cast<int>(final_matches.size());
}

int run_orientation_guided_affine_ransac(
    const SIFTParams& params,
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<MatchCandidate>& input_candidates,
    std::vector<cv::DMatch>& final_matches) {
    // 这是当前最重要的一层约束：
    // 候选匹配不仅要几何位置对得上，还要和整体方向模型相互解释。
    final_matches.clear();
    std::vector<MatchCandidate> candidates = input_candidates;
    if (!orientation_peak_filter_candidates(params, candidates)) {
        return 0;
    }

    cv::RNG rng(static_cast<uint64>(cv::getTickCount()));
    int best_count = 0;
    double best_pos_error_sum = std::numeric_limits<double>::max();
    std::vector<int> best_inlier_indices;
    const int sample_size = std::max(2, params.orientation_sample_size);

    if (static_cast<int>(candidates.size()) < sample_size) {
        return 0;
    }

    for (int iter = 0; iter < params.orientation_ransac_iterations; ++iter) {
        std::vector<int> sample_indices;
        sample_indices.reserve(sample_size);
        while (static_cast<int>(sample_indices.size()) < sample_size) {
            const int idx = rng.uniform(0, static_cast<int>(candidates.size()));
            if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                sample_indices.push_back(idx);
            }
        }

        std::vector<const MatchCandidate*> sample_ptrs;
        sample_ptrs.reserve(sample_size);
        std::vector<MatchCandidate> sample_vec;
        sample_vec.reserve(sample_size);
        for (int idx : sample_indices) {
            sample_ptrs.push_back(&candidates[idx]);
            sample_vec.push_back(candidates[idx]);
        }

        if (!sample_has_consistent_orientation(params, sample_ptrs)) {
            continue;
        }

        if (sample_size >= 3 && params.enable_orientation_triplet_presample) {
            const std::array<const MatchCandidate*, 3> triplet = {
                sample_ptrs[0], sample_ptrs[1], sample_ptrs[2]
            };
            if (!triplet_has_consistent_structure(params, triplet, kp1, kp2)) {
                continue;
            }
        }

        cv::Matx23d affine;
        if (!estimate_similarity_from_matches(sample_vec, kp1, kp2, affine)) {
            continue;
        }

        float theta_model_deg = 0.0f;
        if (params.enable_orientation_geometry_binding) {
            theta_model_deg = extract_rotation_deg_from_affine(affine);
        }
        else {
            std::vector<float> sample_angles;
            sample_angles.reserve(sample_vec.size());
            for (const auto& sample : sample_vec) {
                sample_angles.push_back(sample.compensated_delta_theta);
            }
            theta_model_deg = mean_angle_deg(sample_angles);
        }

        std::vector<int> inlier_indices;
        inlier_indices.reserve(candidates.size());
        double pos_error_sum = 0.0;

        for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
            const MatchCandidate& candidate = candidates[i];
            const cv::Point2d pred = apply_affine(affine, kp1[candidate.match.queryIdx].pt);
            const cv::Point2f& target = kp2[candidate.match.trainIdx].pt;
            const double dx = pred.x - target.x;
            const double dy = pred.y - target.y;
            const double pos_error = std::sqrt(dx * dx + dy * dy);
            if (pos_error > params.ransac_thresh) {
                continue;
            }

            const float dir_error = std::abs(wrap_angle_deg(candidate.compensated_delta_theta - theta_model_deg));
            if (dir_error > params.orientation_inlier_thresh_deg) {
                continue;
            }

            inlier_indices.push_back(i);
            pos_error_sum += pos_error;
        }

        const int inlier_count = static_cast<int>(inlier_indices.size());
        if (inlier_count > best_count || (inlier_count == best_count && pos_error_sum < best_pos_error_sum)) {
            best_count = inlier_count;
            best_pos_error_sum = pos_error_sum;
            best_inlier_indices = std::move(inlier_indices);
        }
    }

    if (best_count < 4) {
        return 0;
    }

    final_matches.reserve(best_inlier_indices.size());
    for (int idx : best_inlier_indices) {
        final_matches.push_back(candidates[idx].match);
    }
    return static_cast<int>(final_matches.size());
}
} // namespace

/**
 * SIFTMatcher constructor
 */
SIFTMatcher::SIFTMatcher(const SIFTParams& p, const SIFTConstants& c)
    : P(p), C(c), sift(p, c), bf(cv::NORM_L2, false) {}

/**
 * Feature extraction wrapper
 */
void SIFTMatcher::extractFeatures(const cv::Mat& img, const cv::Mat& mask, std::vector<cv::KeyPoint>& kp, cv::Mat& des) const {
    sift.detectAndCompute(img, mask, kp, des);
}

/**
 * Core matching function.
 * When enable_orientation_verification is false, the legacy matching path is preserved.
 */
int SIFTMatcher::match(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
    const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
    std::vector<cv::DMatch>& final_matches) const {
    return matchWithStats(kp1, des1, kp2, des2, final_matches, nullptr);
}

int SIFTMatcher::matchWithStats(const std::vector<cv::KeyPoint>& kp1, const cv::Mat& des1,
    const std::vector<cv::KeyPoint>& kp2, const cv::Mat& des2,
    std::vector<cv::DMatch>& final_matches,
    SIFTMatchDebugStats* stats) const {
    // 这是匹配总入口，外面看到的分数本质上就是最终内点数。
    final_matches.clear();
    if (stats) {
        stats->mutual_candidate_count = 0;
        stats->final_inlier_count = 0;
    }
    if (des1.empty() || des2.empty() || des1.rows < 2 || des2.rows < 2) return 0;

    std::vector<MatchCandidate> mutual_candidates;

    if (!P.enable_hamming_match) {
        std::vector<std::vector<cv::DMatch>> knn12;
        bf.knnMatch(des1, des2, knn12, 2);

        auto pass = [&](const std::vector<cv::DMatch>& v) {
            if (v.size() < 2) return false;
            if (v[0].distance >= P.ratio_thresh * v[1].distance) return false;
            return true;
        };

        std::unordered_map<int, MatchCandidate> bestTrain;
        for (auto& pair : knn12) {
            if (!pass(pair)) {
                continue;
            }
            const cv::DMatch& match = pair[0];
            MatchCandidate candidate;
            candidate.match = match;
            candidate.used_flip = false;
            candidate.compensated_delta_theta = compute_compensated_delta_theta(
                kp1[match.queryIdx], kp2[match.trainIdx], false);

            auto it = bestTrain.find(match.trainIdx);
            if (it == bestTrain.end() || match.distance < it->second.match.distance) {
                bestTrain[match.trainIdx] = candidate;
            }
        }
        for (auto& kv : bestTrain) mutual_candidates.push_back(kv.second);
    }
    else {
        std::unordered_map<int, MatchCandidate> bestTrain;

        for (int i = 0; i < des1.rows; ++i) {
            int best_dist = 129;
            int second_best_dist = 129;
            int best_idx = -1;
            bool best_used_flip = false;

            const uint64_t* d1 = reinterpret_cast<const uint64_t*>(des1.ptr<float>(i));
            const int type1 = kp1[i].class_id;

            for (int j = 0; j < des2.rows; ++j) {
                if (P.enable_max_min_filter && type1 != kp2[j].class_id) continue;

                const uint64_t* d2 = reinterpret_cast<const uint64_t*>(des2.ptr<float>(j));
                int d_norm = 0;
                int d_flip = 0;
                Hadamard::calculateForwardAndFlipHamming128(d1, d2, d_norm, d_flip);

                const bool use_flip = d_flip < d_norm;
                const int d = std::min(d_norm, d_flip);

                if (d < best_dist) {
                    second_best_dist = best_dist;
                    best_dist = d;
                    best_idx = j;
                    best_used_flip = use_flip;
                }
                else if (d < second_best_dist) {
                    second_best_dist = d;
                }
            }

            if (best_idx != -1 && static_cast<float>(best_dist) < P.ratio_thresh * static_cast<float>(second_best_dist)) {
                MatchCandidate candidate;
                candidate.match = cv::DMatch(i, best_idx, static_cast<float>(best_dist));
                candidate.used_flip = best_used_flip;
                candidate.compensated_delta_theta = compute_compensated_delta_theta(
                    kp1[i], kp2[best_idx], best_used_flip);

                auto it = bestTrain.find(best_idx);
                if (it == bestTrain.end() || candidate.match.distance < it->second.match.distance) {
                    bestTrain[best_idx] = candidate;
                }
            }
        }
        for (auto& kv : bestTrain) mutual_candidates.push_back(kv.second);
    }

    if (stats) {
        stats->mutual_candidate_count = static_cast<int>(mutual_candidates.size());
    }

    if (mutual_candidates.size() < 4) {
        return 0;
    }

    if (!P.enable_orientation_verification || !P.use_affine_model) {
        std::vector<cv::DMatch> legacy_candidates;
        legacy_candidates.reserve(mutual_candidates.size());
        for (const auto& candidate : mutual_candidates) {
            legacy_candidates.push_back(candidate.match);
        }
        const int inliers = run_legacy_geometry_verification(P, kp1, kp2, legacy_candidates, final_matches);
        if (stats) {
            stats->final_inlier_count = inliers;
        }
        return inliers;
    }

    const int inliers = run_orientation_guided_affine_ransac(P, kp1, kp2, mutual_candidates, final_matches);
    if (stats) {
        stats->final_inlier_count = inliers;
    }
    return inliers;
}
