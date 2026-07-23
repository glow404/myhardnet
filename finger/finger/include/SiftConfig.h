#ifndef SIFT_CONFIG_H
#define SIFT_CONFIG_H

#include <opencv2/opencv.hpp>

// SIFT 常量（对�?OpenCV 源码�?
struct SIFTConstants {
    // ================= 描述�?(Descriptor) 结构参数 =================

    // 描述子网格的宽度�?
    // 标准 SIFT �?4x4 的网格�?
    // 影响�?
    // * 增大 (�?5)：描述子包含更多信息，区分度更高，但对变形和定位误差更敏感，计算更慢�?
    // * 减小 (�?3)：描述子更简单，鲁棒性更强，但区分度下降（容易匹配错）�?
    int   SIFT_DESCR_WIDTH = 4;

    // 每个网格内的方向直方图的柱数 (Bins)�?
    // 标准�?8 个方�?(�?45 度一个柱)�?
    // 最终描述子维度 = width * width * bins = 4 * 4 * 8 = 128 维�?
    // 影响�?
    // * 增大 (�?12)：方向记录更细致，但它对旋转噪声更敏感�?
    int   SIFT_DESCR_HIST_BINS = 8;

    // ================= 尺度空间参数 =================

    // 假设输入图像自带的初始高斯模糊半径�?
    // SIFT 假设相机拍出来的照片不是绝对清晰的，而是已经�?sigma=0.5 的模糊�?
    // 影响�?
    // * 如果你的指纹图非常清晰锐利，可以稍微改小（如 0.4）�?
    // * 如果改大，会导致金字塔第一层更加模糊，丢失微小细节�?
    float SIFT_INIT_SIGMA = 0.5f;

    // 图像边缘忽略的像素距离�?
    // 因为在边缘无法计算完整的梯度或插值，所以直接忽略�?
    // 影响�?
    // * 增大：图像边缘的特征点会被丢弃�?
    int   SIFT_IMG_BORDER = 5;

    // ================= 极值点精确定位参数 =================

    // 亚像素插值时的最大迭代次数�?
    // 也就是为了找 (10.2, 5.3) 这种精确坐标，牛顿迭代法最多跑几次�?
    // 影响�?
    // * 增大：定位更准，但特征提取速度变慢�?
    // * 减小：速度快，但坐标精度下降�?
    int   SIFT_MAX_INTERP_STEPS = 5;

    // ================= 主方向计算参�?(Orientation) =================

    // 计算关键点主方向时的直方图柱数�?
    // 36 表示�?360 度切�?36 份，每份 10 度�?
    // 影响�?
    // * 增大 (�?72)：角度分辨率更高，但受噪声影响大，主方向可能不稳定�?
    int   SIFT_ORI_HIST_BINS = 36;

    // 计算主方向时，高斯加权窗口的大小系数�?
    // 窗口半径 = SIFT_ORI_RADIUS * scale�?
    // 影响�?
    // * 增大：参考更大范围内的像素来决定主方向，更稳定，但容易受背景干扰�?
    float SIFT_ORI_SIG_FCTR = 1.0f;

    // 上面那个高斯窗口半径的硬性倍率�?
    // 实际上就�?3 * 1.5 = 4.5 sigma。根�?3-sigma 准则，覆盖绝大部分高斯能量�?
    float SIFT_ORI_RADIUS = 4.5f;

    // 辅方向的峰值比率阈值�?
    // 在计算方向时，如果发现另一个方向的能量达到了主方向能量�?80% (0.8)�?
    // SIFT 会在这个位置再生成一个具有该新方向的特征点�?
    // 影响�?
    // * 减小 (�?0.6)：会产生更多重叠的特征点（同一个点，两个方向）�?
    //   这对于指纹这种纹理重复的图像可能有好处，能增加匹配成功率，但会增加计算量�?
    // * 增大 (�?0.9)：特征点更少，只保留最强方向�?
    float SIFT_ORI_PEAK_RATIO = 0.7f;

    // ================= 描述子生成参�?=================

    // 描述子采样区域的大小系数�?
    // 决定了计�?128 维向量时，要看多大范围的像素�?
    // 影响�?
    // * 增大：描述子包含更多周边背景信息，不仅看特征点本身，还看它邻居�?
    //   对抗指纹变形（挤压）的能力会变弱�?
    float SIFT_DESCR_SCL_FCTR = 3.0f;

    // 描述子数值截断阈�?(光照不变�?�?
    // 为了防止某个方向的光照太强（比如反光）主导了整个特征向量�?
    // 所有超�?0.2 的分量都会被强制截断�?0.2�?
    // 影响�?
    // * 减小 (�?0.15)：对光照变化和反光更鲁棒（更不敏感）�?
    // * 增大：保留更真实的梯度强度差异，但在光照不均时容易匹配失败�?
    float SIFT_DESCR_MAG_THR = 0.2f;

    // 浮点数转整数的缩放因子�?
    // 512 是为了配合后面的归一化，使得最终数值落�?0-255 之间�?
    // 不要改�?
    float SIFT_INT_DESCR_FCTR = 512.0f;

    // 定点数缩放比例�?
    // 用于内部计算时把浮点坐标转整数，通常设为 1.0 即可�?
    float SIFT_FIXPT_SCALE = 1.0f;
};

// SIFT 可调参数
struct SIFTParams {
    int    nfeatures = 300;
    int    nOctaveLayers = 4;

    // 【修�?1】对比度阈值：�?0.04 改为 0.02 (抓取更微弱的纹理)
    double contrastThreshold = 0.03;

    // 【修�?2】边缘阈值：�?10.0 改为 20.0 (保留更多脊线特征)
    double edgeThreshold = 17.5;

    // 【修�?3】初始模糊：�?1.6 改为 1.6 (保持不变，OpenCV默认也是1.6)
    double sigma = 1.70;

    // 【修�?4】Ratio Test：从 0.75 改为 0.7 (稍微严格一点，或�?.75也可�?
    double ratio_thresh = 0.85f;

    double ransac_thresh = 5.0;

    // 【修�?5】仿射模型：建议改为 false (先用基础的单应性矩阵，OpenCV默认是false)
    bool   use_affine_model = true;

    bool   enable_distance_thresh = false;
    double distance_thresh = 250.0;

    // 【修�?6】双向验证：改为 false (为了追求匹配数量，先关掉严格双向)
    bool   strict_two_way = false;

    // 【修�?7】RootSIFT：改�?true 
    bool   enable_rootsift = true;

    bool   enable_clahe = true;

    // true: 使用原始 x2 上采样 SIFT 链路。
    // false: 不上采样，但自动切换到关闭上采样后的补偿参数。
    bool   enable_initial_upsample = true;

    // 为了兼容旧调用先保留这个字段，但核心逻辑已不再依赖它。
    bool   enable_no_upsample_compensation = false;
    double no_upsample_sigma =0.99;
    double no_upsample_contrast_threshold = 0.008;
    int    no_upsample_nOctaveLayers = 5;
    double no_upsample_edgeThreshold = 20.0;
    double no_upsample_clahe_clip = 4.0;
    double no_upsample_blur_sigma = 0.6;

    // 【修�?8】CLAHE强度：从 2.0 改为 4.0 (增强指纹对比�?
    double clahe_clip = 4.0;

    cv::Size clahe_tile = cv::Size(8, 8);
    bool   enable_blur = true;
    double blur_sigma = 0.8;


    // 【新增优化开关�?
    // 【新增优化开关�?
    bool enable_hamming_match = true;    // 开启汉明距离匹�?
    bool enable_max_min_filter = true;   // 开启极大极小值分类过?

    bool enable_orientation_verification = true;
    bool enable_orientation_peak_filter = true;
    float orientation_peak_bin_deg = 10.0f;
    float orientation_peak_band_deg = 15.0f;
    float orientation_peak_min_ratio = 0.40f;
    float orientation_sample_thresh_deg = 20.0f;
    float orientation_inlier_thresh_deg = 20.0f;
    bool enable_orientation_triplet_presample = true;
    bool enable_orientation_geometry_binding = true;
    int orientation_sample_size = 3;
    float orientation_triplet_side_ratio_thresh = 0.25f;
    double orientation_triplet_area_min = 25.0;
    int orientation_ransac_iterations = 600;
};
#endif
