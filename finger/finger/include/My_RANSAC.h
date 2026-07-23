#ifndef My_RANSAC_H
#define My_RANSAC_H

#include <vector>

// 定义基础坐标结构，脱离 OpenCV 依赖
struct Point2D {
    float x, y;
};

// 仿射变换模型结构 (2x3 矩阵)
struct AffineModel {
    double m[2][3];
};

class MyRANSAC {
public:
    /**
     * @brief 手动实现的仿射变换 RANSAC 算法
     * @param src 源点集（测试图特征点）
     * @param dst 目标点集（模板图特征点）
     * @param threshold 内点判定距离阈值（建议 3.0-5.0）
     * @param max_iters 最大迭代次数（建议 1000-2000）
     * @param best_model 输出：计算出的最优仿射矩阵
     * @param best_inliers 输出：内点在原始向量中的索引列表
     * @return 最终匹配的内点数量 (Score)
     */
    static int estimateAffineRANSAC(
        const std::vector<Point2D>& src,
        const std::vector<Point2D>& dst,
        float threshold,
        int max_iters,
        AffineModel& best_model,
        std::vector<int>& best_inliers);

private:
    // 内部函数：求解 3 对点的线性方程组
    static bool solveAffine3Points(Point2D s1, Point2D s2, Point2D s3,
        Point2D d1, Point2D d2, Point2D d3, AffineModel& out);

    // 内部函数：利用最小二乘法进行全局精度优化
    static void refineModelLeastSquares(const std::vector<Point2D>& src,
        const std::vector<Point2D>& dst,
        const std::vector<int>& inliers,
        AffineModel& out);
};

#endif 