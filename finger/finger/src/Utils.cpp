#include "Utils.h"
#include <vector>
#include <algorithm>
#include <cmath>
/**
 * 自动生成指纹有效区域掩膜 (Mask)
 * 对应优化方案 Step 6：预处理约束
 * 目的：通过图像处理手段提取出指纹的实心区域，防止 SIFT 在背景噪点或边缘处提取伪特征。
 * * @param img 输入的原始指纹灰度图
 * @return 返回二值化的掩膜图像（有效区域为白色 255，背景为黑色 0）
 */
cv::Mat generateFingerMask(const cv::Mat& img) {
    if (img.empty()) return cv::Mat();

    cv::Mat mask;
    // 1. Otsu 大津法自动阈值分割
    // 使用 cv::THRESH_BINARY_INV 是因为指纹脊线通常为深色，背景为浅色
    // 反转后，指纹纹路变为高亮区域，背景变为黑色
    cv::threshold(img, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 2. 定义形态学操作的结构元素（15x15 的矩形核）
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

    // 3. 闭运算 (MORPH_CLOSE)：先膨胀后腐蚀
    // 作用：填充指纹脊线之间的谷线空隙，使指纹区域连接成一个连通的整体实心块
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);

    // 4. 开运算 (MORPH_OPEN)：先腐蚀后膨胀
    // 作用：消除指纹区域外部细小的孤立噪点（如灰尘或传感器伪影）
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);

    // 5. 腐蚀操作 (Erode)
    // 作用：将生成的掩膜边缘向内收缩 5 像素
    // 理由：指纹采集时边缘往往存在严重的压力形变或不完整纹理，收缩掩膜可以强迫 SIFT 避开这些不稳定区域
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    return mask;
}

/**
 * 保存 SIFT 匹配结果的可视化图像
 * * @param img1 图像 1
 * @param kp1 图像 1 的特征点
 * @param img2 图像 2
 * @param kp2 图像 2 的特征点
 * @param matches 最终通过 RANSAC 校验的匹配对 (Inliers)
 * @param score 匹配分数（内点数量）
 * @param out 输出文件路径
 */
void save_sift_vis(const cv::Mat& img1, const std::vector<cv::KeyPoint>& kp1,
    const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches, int score, const std::string& out) {

    cv::Mat res;
    // 调用 OpenCV 标准绘制函数，使用绿色 (0, 255, 0) 连线展示匹配关系
    // DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS 表示不画出没有匹配上的孤立点，使画面更清晰
    cv::drawMatches(img1, kp1, img2, kp2, matches, res,
        cv::Scalar(0, 255, 0), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // 在图像左上角标注最终的比对分数（内点数）
    // 对应优化方案中的最终评估逻辑
    cv::putText(res, "Inliers: " + std::to_string(score), cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    // 将可视化结果保存到本地，方便后续人工核对和 Step 2/Step 3 的验证分析 [cite: 1, 2]
    cv::imwrite(out, res);
}

/**
* gmfs
* =================================================================================
// 函数功能：使用 GMFS (Gradient-Magnitude Fingerprint Segmentation) 算法提取指纹前景掩膜
// 参数说明：
//   - image: 输入的灰度指纹图像
//   - sigma: 高斯平滑的尺度（默认 13/3）
//   - percentile: 用于计算阈值的百分位数（默认 95%，代表取图像中极强的梯度值参考）
//   - threshold_ratio: 阈值系数（默认 0.2，即取 95% 梯度的 20% 作为门槛）
//   - closing_count: 形态学闭运算迭代次数（用于填补小坑，默认 6）
//   - opening_count: 形态学开运算迭代次数（用于去除边缘毛刺，默认 12）
//   - image_dpi: 输入图像的 DPI（默认 500，若不是 500 会在内部先缩放处理）
// 返回值：
//   - cv::Mat: 提取出的二值掩膜 (前景 255，背景 0，类型为 CV_8UC1)
// =================================================================================
*/
cv::Mat extract_gmfs_mask(const cv::Mat& image,
    float sigma,
    float percentile,
    float threshold_ratio,
    int closing_count,
    int opening_count,
    int image_dpi) {
    int image_h = image.rows;
    int image_w = image.cols;
    cv::Mat img = image.clone();

    // 1. DPI 缩放处理 (如果是 500 dpi 则不缩放)
    if (image_dpi != 500) {
        float f = 500.0f / image_dpi;
        cv::resize(img, img, cv::Size(), f, f, cv::INTER_CUBIC);
    }

    // 2. 计算梯度幅值
    cv::Mat dx, dy;
    // 使用 Sobel 算子求梯度 (核大小 3x3)，等价于 python 的 spatialGradient
    cv::Sobel(img, dx, CV_32F, 1, 0, 3);
    cv::Sobel(img, dy, CV_32F, 0, 1, 3);

    cv::Mat m;
    cv::magnitude(dx, dy, m);

    // 3. 高斯平滑梯度幅值图
    int gs = std::ceil(3 * sigma) * 2 + 1;
    cv::Mat m_a;
    cv::GaussianBlur(m, m_a, cv::Size(gs, gs), sigma);

    // 4. 计算自适应阈值 (基于百分位数)
    // 将矩阵展平以便排序求 percentile
    cv::Mat m_flat = m.reshape(1, 1).clone();
    std::vector<float> m_vec;
    m_flat.copyTo(m_vec);

    // 使用 nth_element 快速求取第 percentile 分位的值
    int k = std::min((int)std::round(m_vec.size() * (percentile / 100.0f)), (int)m_vec.size() - 1);
    std::nth_element(m_vec.begin(), m_vec.begin() + k, m_vec.end());
    float p95 = m_vec[k];
    float norm_t = p95 * threshold_ratio;

    // 5. 阈值分割得到初始二值掩膜
    cv::Mat mask;
    cv::threshold(m_a, mask, norm_t, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U); // 确保转为 8 位单通道图

    // 预设形态学核
    cv::Mat se3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // 6. 闭运算 (填充小空洞和凹陷)
    if (closing_count > 0) {
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, se3x3, cv::Point(-1, -1), closing_count);
    }

    // 帮助提取最大连通域的 Lambda 函数
    auto keep_largest_cc = [](cv::Mat& binary_mask) {
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8, CV_32S);
        if (num_labels > 1) {
            int max_area = 0;
            int max_label = 0;
            for (int i = 1; i < num_labels; ++i) { // 从 1 开始，跳过背景 (label 0)
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > max_area) {
                    max_area = area;
                    max_label = i;
                }
            }
            // 只保留面积最大的连通域
            binary_mask = (labels == max_label) * 255;
        }
    };

    // 7. 移除除最大连通域之外的所有小色块
    keep_largest_cc(mask);

    // 8. 填充连通域内部的空洞 (不包括连接到边缘的)
    cv::Mat bg_mask;
    cv::bitwise_not(mask, bg_mask);
    cv::Mat labels_bg, stats_bg, centroids_bg;
    int num_bg = cv::connectedComponentsWithStats(bg_mask, labels_bg, stats_bg, centroids_bg, 8, CV_32S);

    int h = img.rows;
    int w = img.cols;

    for (int i = 1; i < num_bg; ++i) {
        int left = stats_bg.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats_bg.at<int>(i, cv::CC_STAT_TOP);
        int width = stats_bg.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats_bg.at<int>(i, cv::CC_STAT_HEIGHT);

        // 判断这个背景块是否不接触图像边界
        if (left > 0 && (left + width < w - 1) && top > 0 && (top + height < h - 1)) {
            // 是内部空洞，填满为 255
            mask.setTo(255, labels_bg == i);
        }
    }

    // 9. 开运算 (移除小的凸起和毛刺)
    if (opening_count > 0) {
        // borderValue 设置为 0，防止图像边缘的连通域被误操作
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, se3x3, cv::Point(-1, -1), opening_count, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    // 10. 再次保留最大连通域 (清除开运算可能切断的小碎片)
    keep_largest_cc(mask);

    // 11. 如果之前缩放过，现在还原回原尺寸
    if (image_dpi != 500) {
        cv::resize(mask, mask, cv::Size(image_w, image_h), 0, 0, cv::INTER_NEAREST);
    }

    return mask;
}