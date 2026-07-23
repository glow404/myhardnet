#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "BinaryOverlap.h"
#include "SiftConfig.h"
#include "SIFTMatcher.h"
#include "UnicodeIO.h"
#include "Utils.h"

namespace fs = std::filesystem;

struct ImgFeature {
    std::vector<cv::KeyPoint> kp;
    cv::Mat des;
    cv::Mat gray;
    cv::Mat mask;
    std::string filename;
};

struct FingerDB {
    std::string finger_name;
    std::vector<ImgFeature> models;
    std::vector<ImgFeature> tests;
};

struct PairScore {
    int inlier_count = 0;
    double overlap_ratio = 0.0;
    double binary_match_ratio = 0.0;
};

enum class DatasetLayoutMode {
    FullBatch = 1,// 原来的 30 指纹大批量
    SmallBatch = 2// 现在这种单指纹 / 小批量
};

// Keep this file saved as UTF-8 when editing Chinese paths.
static const std::wstring kDefaultMainDir = utf8_to_wstring(u8R"(E:\超声数据\部分不贴屏)");
static const std::wstring kDefaultOutCsv = utf8_to_wstring(u8R"(E:\超声数据\部分不贴屏\scores_hamming_overlap_fvc_ClaheAdaptiveMean_fangxiang_0.5_r0.5.csv)");

// Supported modes:
//   clahe_gaussian_otsu
//   clahe_median_otsu
//   clahe_adaptive_mean
//   clahe_adaptive_gaussian
static const BinaryBinarizationMode kBatchBinarizationMode = BinaryBinarizationMode::ClaheAdaptiveMean;
static const int kBatchAdaptiveBlockSize = 31;
static const double kBatchAdaptiveC = 6.0;
static const int kBatchMedianKernelSize = 5;
static const int kBatchGaussianKernelSize = 5;
static const double kBatchGaussianSigma = 0.8;

// ========================================================================
// !!! 测试前请先确认这里 !!!
// FullBatch : 根目录/手指名/手指名_model + 手指名_test
// SmallBatch: 根目录/单个手指目录/model + test
// ========================================================================
static const DatasetLayoutMode kDatasetLayoutMode = DatasetLayoutMode::FullBatch;

static void print_supported_modes() {
    std::cout << "Supported modes:\n";
    std::cout << "  " << binary_binarization_mode_to_string(BinaryBinarizationMode::ClaheGaussianOtsu) << "\n";
    std::cout << "  " << binary_binarization_mode_to_string(BinaryBinarizationMode::ClaheMedianOtsu) << "\n";
    std::cout << "  " << binary_binarization_mode_to_string(BinaryBinarizationMode::ClaheAdaptiveMean) << "\n";
    std::cout << "  " << binary_binarization_mode_to_string(BinaryBinarizationMode::ClaheAdaptiveGaussian) << "\n";
}

static const char* dataset_layout_mode_to_string(DatasetLayoutMode mode) {
    switch (mode) {
        case DatasetLayoutMode::FullBatch:
            return "FullBatch";
        case DatasetLayoutMode::SmallBatch:
            return "SmallBatch";
        default:
            return "Unknown";
    }
}

static fs::path append_mode_suffix_to_csv_path(
    const std::wstring& base_csv,
    const std::string& bin_mode_name,
    bool enable_initial_upsample) {
    fs::path out_path(base_csv);
    const fs::path parent = out_path.parent_path();
    const std::string stem = wstring_to_utf8(out_path.stem().wstring());
    const std::string ext = wstring_to_utf8(out_path.extension().wstring());
    const std::string filename = stem + "_" + bin_mode_name +
                                 (enable_initial_upsample ? "_upsample_on" : "_upsample_off") + ext;
    return parent / fs::u8path(filename);
}

static SIFTParams make_sift_params() {
    SIFTParams P;
    P.nfeatures = 300;
    P.enable_hamming_match = true;
    P.enable_max_min_filter = true;
    P.enable_rootsift = true;
    P.use_affine_model = true;
    P.ratio_thresh = 0.85f;
    P.ransac_thresh = 0.5;
    P.enable_orientation_verification = true;
    P.enable_orientation_peak_filter = true;
    P.orientation_peak_bin_deg = 10.0f;
    P.orientation_peak_band_deg = 15.0f;
    P.orientation_peak_min_ratio = 0.4f;
    P.orientation_sample_thresh_deg = 20.0f;
    P.orientation_inlier_thresh_deg = 20.0f;
    P.enable_orientation_triplet_presample = true;
    P.orientation_triplet_side_ratio_thresh = 0.25f;
    P.orientation_triplet_area_min = 25.0;

    P.enable_initial_upsample = true;

    return P;
}

static PairScore evaluate_pair(
    const ImgFeature& probe_img,
    const ImgFeature& model_img,
    const SIFTMatcher& matcher,
    const SIFTParams& params,
    const BinaryOverlapParams& overlap_params) {
    PairScore result;

    std::vector<cv::DMatch> inlier_matches;
    result.inlier_count = matcher.match(
        probe_img.kp, probe_img.des,
        model_img.kp, model_img.des,
        inlier_matches);

    if (result.inlier_count < 4) {
        return result;
    }

    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    pts1.reserve(inlier_matches.size());
    pts2.reserve(inlier_matches.size());
    for (const auto& m : inlier_matches) {
        pts1.push_back(probe_img.kp[m.queryIdx].pt);
        pts2.push_back(model_img.kp[m.trainIdx].pt);
    }

    cv::Mat affine_inlier_mask;
    cv::Mat affine = cv::estimateAffinePartial2D(
        pts1, pts2, affine_inlier_mask, cv::RANSAC, params.ransac_thresh);
    if (affine.empty()) {
        return result;
    }

    BinaryOverlapStats overlap_stats;
    if (!compute_binary_overlap_with_affine(
            probe_img.gray, model_img.gray, affine, overlap_stats, overlap_params,
            probe_img.mask, model_img.mask)) {
        return result;
    }

    result.overlap_ratio = overlap_stats.overlap_ratio;
    result.binary_match_ratio = overlap_stats.binary_match_ratio;
    return result;
}

static void extract_folder_features(
    const fs::path& folder_path,
    const SIFTMatcher& matcher,
    std::vector<ImgFeature>& target_vec,
    int& total_images) {
    if (!fs::exists(folder_path)) {
        return;
    }

    for (const auto& img_entry : fs::directory_iterator(folder_path)) {
        if (img_entry.path().extension() != L".bmp") {
            continue;
        }

        cv::Mat img = imread_unicode(img_entry.path().wstring(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            continue;
        }

        ImgFeature feat;
        feat.gray = img;
        feat.mask = extract_gmfs_mask(img);
        feat.filename = wstring_to_utf8(img_entry.path().filename().wstring());

        matcher.extractFeatures(feat.gray, feat.mask, feat.kp, feat.des);
        if (feat.des.empty()) {
            continue;
        }

        target_vec.push_back(std::move(feat));
        total_images++;
    }
}

static void load_database_full_batch(
    const std::wstring& main_dir,
    const SIFTMatcher& matcher,
    std::vector<FingerDB>& database,
    int& total_images) {
    for (const auto& entry : fs::directory_iterator(main_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        FingerDB fdb;
        fdb.finger_name = wstring_to_utf8(entry.path().filename().wstring());

        const fs::path model_path = entry.path() / fs::path(entry.path().filename().wstring() + L"_model");
        const fs::path test_path = entry.path() / fs::path(entry.path().filename().wstring() + L"_test");

        extract_folder_features(model_path, matcher, fdb.models, total_images);
        extract_folder_features(test_path, matcher, fdb.tests, total_images);

        if (!fdb.models.empty() && !fdb.tests.empty()) {
            std::cout << "  [Loaded] " << fdb.finger_name
                      << " (M: " << fdb.models.size()
                      << ", T: " << fdb.tests.size() << ")" << std::endl;
            database.push_back(std::move(fdb));
        }
    }
}

static void load_database_small_batch(
    const std::wstring& main_dir,
    const SIFTMatcher& matcher,
    std::vector<FingerDB>& database,
    int& total_images) {
    for (const auto& entry : fs::directory_iterator(main_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const fs::path finger_dir = entry.path();
        const fs::path model_path = finger_dir / L"model";
        const fs::path test_path = finger_dir / L"test";
        if (!fs::exists(model_path) || !fs::exists(test_path)) {
            continue;
        }

        FingerDB fdb;
        fdb.finger_name = wstring_to_utf8(finger_dir.filename().wstring());

        extract_folder_features(model_path, matcher, fdb.models, total_images);
        extract_folder_features(test_path, matcher, fdb.tests, total_images);

        if (!fdb.models.empty() && !fdb.tests.empty()) {
            std::cout << "  [Loaded] " << fdb.finger_name
                      << " (M: " << fdb.models.size()
                      << ", T: " << fdb.tests.size() << ")" << std::endl;
            database.push_back(std::move(fdb));
        }
    }
}

int main(int argc, char** argv) {
    const std::wstring main_dir = kDefaultMainDir;
    BinaryBinarizationMode selected_mode = kBatchBinarizationMode;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (binary_binarization_mode_from_string(arg, selected_mode)) {
            continue;
        }

        std::cerr << "Unsupported argument: " << arg << std::endl;
        print_supported_modes();
        return 2;
    }
    SIFTParams P = make_sift_params();
    const fs::path out_csv = append_mode_suffix_to_csv_path(
        kDefaultOutCsv,
        binary_binarization_mode_to_string(selected_mode),
        P.enable_initial_upsample);

    SIFTConstants C;
    SIFTMatcher matcher(P, C);

    BinaryOverlapParams overlap_params;
    overlap_params.mode = selected_mode;
    overlap_params.adaptive_block_size = kBatchAdaptiveBlockSize;
    overlap_params.adaptive_c = kBatchAdaptiveC;
    overlap_params.median_kernel_size = kBatchMedianKernelSize;
    overlap_params.gaussian_kernel_size = kBatchGaussianKernelSize;
    overlap_params.gaussian_sigma = kBatchGaussianSigma;

    std::vector<FingerDB> database;
    int total_images = 0;

    std::cout << ">>> Stage 1: scanning folders and extracting binary features..." << std::endl;
    std::cout << ">>> Binarization mode: " << binary_binarization_mode_to_string(overlap_params.mode) << std::endl;
    std::cout << ">>> enable_initial_upsample: " << (P.enable_initial_upsample ? "true" : "false") << std::endl;
    std::cout << ">>> dataset_layout_mode: " << dataset_layout_mode_to_string(kDatasetLayoutMode) << std::endl;
    if (argc < 2) {
        print_supported_modes();
    }

    if (kDatasetLayoutMode == DatasetLayoutMode::FullBatch) {
        load_database_full_batch(main_dir, matcher, database, total_images);
    } else {
        load_database_small_batch(main_dir, matcher, database, total_images);
    }

    std::ofstream ofs(out_csv, std::ios::binary);
    ofs << "\xEF\xBB\xBF";
    ofs << "label,probe_finger,model_finger,probe_img,best_model_img,max_score,overlap_ratio,binary_match_ratio,binarization_mode\n";

    std::cout << "\n>>> Stage 2: batch matching with overlap metrics..." << std::endl;

    for (size_t i = 0; i < database.size(); ++i) {
        const auto& probe_finger = database[i];
        int test_idx = 1;
        const int total_tests = static_cast<int>(probe_finger.tests.size());

        std::cout << "Processing finger: " << probe_finger.finger_name << " ..." << std::endl;

        for (const auto& probe_img : probe_finger.tests) {
            std::cout << "\r  -> Progress: " << test_idx++ << " / " << total_tests
                      << " (" << probe_img.filename << ")   " << std::flush;

            PairScore best_genuine;
            std::string best_genuine_model;
            for (const auto& model_img : probe_finger.models) {
                const PairScore score = evaluate_pair(probe_img, model_img, matcher, P, overlap_params);
                if (score.inlier_count > best_genuine.inlier_count) {
                    best_genuine = score;
                    best_genuine_model = model_img.filename;
                }
            }

            ofs << "1," << probe_finger.finger_name << "," << probe_finger.finger_name << ","
                << probe_img.filename << "," << best_genuine_model << ","
                << best_genuine.inlier_count << ","
                << best_genuine.overlap_ratio << ","
                << best_genuine.binary_match_ratio << ","
                << binary_binarization_mode_to_string(overlap_params.mode) << "\n";

            for (size_t j = 0; j < database.size(); ++j) {
                if (i == j) {
                    continue;
                }

                const auto& impostor_finger = database[j];
                PairScore best_impostor;
                std::string best_impostor_model;
                for (const auto& model_img : impostor_finger.models) {
                    const PairScore score = evaluate_pair(probe_img, model_img, matcher, P, overlap_params);
                    if (score.inlier_count > best_impostor.inlier_count) {
                        best_impostor = score;
                        best_impostor_model = model_img.filename;
                    }
                }

                ofs << "0," << probe_finger.finger_name << "," << impostor_finger.finger_name << ","
                    << probe_img.filename << "," << best_impostor_model << ","
                    << best_impostor.inlier_count << ","
                    << best_impostor.overlap_ratio << ","
                    << best_impostor.binary_match_ratio << ","
                    << binary_binarization_mode_to_string(overlap_params.mode) << "\n";
            }
        }
        std::cout << std::endl;
    }

    ofs.close();
    std::cout << "\n>>> Done. CSV saved to: " << wstring_to_utf8(out_csv.wstring()) << std::endl;
    std::cout << ">>> Total images loaded: " << total_images << std::endl;
    return 0;
}
