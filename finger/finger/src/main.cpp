#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 包含核心头文件
#include "SiftConfig.h"
#include "SIFTMatcher.h"
#include "Utils.h"

/**
 * 实验组 A：标准 RANSAC 匹配程序 (Baseline)
 * 逻辑：128维浮点描述子 -> 欧氏距离 L2 -> KNN 匹配 -> Ratio Test -> 基础 RANSAC
 */
int main() {
    // =========================================================================
    // 1. 设置路径 (请在此处修改输入图片和输出可视化路径)
    // =========================================================================
    std::string p1 = R"(E:/超声数据/FullData/FVCtest/dhy_R0/dhy_R0_test/wi-Cycle=2Freq=41Rgd=1245-0-20260318165421.161250_pair1.bmp)";
    std::string p2 = R"(E:/超声数据/FullData/FVCtest/dhy_R0/dhy_R0_test/wi-Cycle=2Freq=41Rgd=1245-0-20260318165520.434729_pair2.bmp)";

    // 【自定义输出路径】
    std::string output_visual_path = "E:/C++Finger/output/baseline_ransac_match.png";

    // =========================================================================
    // 2. 读取图像
    // =========================================================================
    cv::Mat img1 = cv::imread(p1, 0);
    cv::Mat img2 = cv::imread(p2, 0);

    if (img1.empty() || img2.empty()) {
        std::cerr << "错误：图像加载失败，请检查路径。" << std::endl;
        return -1;
    }

    // =========================================================================
    // 3. 配置标准 SIFT 参数 (关闭所有二进制优化)
    // =========================================================================
    SIFTParams P;

    // 确保关闭新开发的优化模块，回归标准 SIFT 流程
    P.enable_hamming_match = false;   // 禁用汉明匹配，强制使用欧氏距离
    P.enable_max_min_filter = false;  // 禁用极值分类过滤，全量特征点比对

    // 基准测试常用参数
    P.nfeatures = 300;                  // 0 表示不限制点数，保留所有检测到的点
    P.ratio_thresh = 0.85f;            // 标准 Lowe's Ratio
    P.use_affine_model = true;       // 使用标准单应性矩阵 (Homography) 校验
    P.enable_rootsift = true;         // 保持 RootSIFT 开启，这是目前 128D 匹配的标配

    SIFTConstants C;
    SIFTMatcher matcher(P, C);

    // =========================================================================
    // 4. 特征提取与匹配
    // =========================================================================
    std::cout << ">>> [Baseline] 正在提取 128D 浮点特征..." << std::endl;
    cv::Mat m1 = generateFingerMask(img1);
    cv::Mat m2 = generateFingerMask(img2);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;

    matcher.extractFeatures(img1, m1, kp1, des1);
    matcher.extractFeatures(img2, m2, kp2, des2);

    std::cout << ">>> [Baseline] 执行 BFMatcher + KNN + RANSAC 匹配..." << std::endl;
    std::vector<cv::DMatch> matches;

    // 内部会根据 P.enable_hamming_match = false 自动调用 cv::BFMatcher (NORM_L2)
    int score = matcher.match(kp1, des1, kp2, des2, matches);

    // =========================================================================
    // 5. 结果输出与保存
    // =========================================================================
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "基准测试得分 (RANSAC Inliers): " << score << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 保存可视化图片到指定路径
    save_sift_vis(img1, kp1, img2, kp2, matches, score, output_visual_path);
    std::cout << ">>> 匹配图已保存至: " << output_visual_path << std::endl;

    return 0;
}