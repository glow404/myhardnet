#include "My_RANSAC.h"
#include <random>
#include <cmath>
#include <algorithm>
/**
 * @brief 核心入口：手动实现 RANSAC 仿射变换
 */
int MyRANSAC::estimateAffineRANSAC(
    const std::vector<Point2D>& src,
    const std::vector<Point2D>& dst,
    float threshold,
    int max_iters,
    AffineModel& best_model,
    std::vector<int>& best_inliers)
{
    int n = (int)src.size();
    if (n < 3) return 0; // 仿射变换至少需要3对点

    best_inliers.clear();
    std::mt19937 rng(42); // 固定随机种子保证结果可复现
    float thresh_sq = threshold * threshold;

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. 随机抽取 3 对不重复的点索引
        int idx[3];
        for (int i = 0; i < 3; ++i) {
            bool unique;
            do {
                unique = true;
                idx[i] = rng() % n;
                for (int j = 0; j < i; ++j) if (idx[i] == idx[j]) unique = false;
            } while (!unique);
        }

        // 2. 求解当前 3 对点的精确仿射模型
        AffineModel current_model;
        if (!solveAffine3Points(src[idx[0]], src[idx[1]], src[idx[2]],
            dst[idx[0]], dst[idx[1]], dst[idx[2]], current_model)) {
            continue;
        }

        // 3. 统计符合该模型的内点 (Inliers)
        std::vector<int> current_inliers;
        for (int i = 0; i < n; ++i) {
            // 应用仿射矩阵变换：x' = ax + by + c
            double tx = current_model.m[0][0] * src[i].x + current_model.m[0][1] * src[i].y + current_model.m[0][2];
            double ty = current_model.m[1][0] * src[i].x + current_model.m[1][1] * src[i].y + current_model.m[1][2];

            // 计算欧氏距离平方
            double err_sq = (tx - dst[i].x) * (tx - dst[i].x) + (ty - dst[i].y) * (ty - dst[i].y);
            if (err_sq < thresh_sq) {
                current_inliers.push_back(i);
            }
        }

        // 4. 保存拥有最多内点的模型
        if (current_inliers.size() > best_inliers.size()) {
            best_inliers = current_inliers;
        }
    }

    // =====================================================================
    // 5. 【关键：精度补偿】利用所有内点进行最小二乘拟合 (与 OpenCV 逻辑一致)
    // =====================================================================
    if (best_inliers.size() >= 3) {
        refineModelLeastSquares(src, dst, best_inliers, best_model);
        return (int)best_inliers.size();
    }

    return 0;
}

/**
 * @brief 内部函数：通过 3 对点求解 2x3 仿射矩阵
 */
bool MyRANSAC::solveAffine3Points(Point2D s1, Point2D s2, Point2D s3,
    Point2D d1, Point2D d2, Point2D d3, AffineModel& out) {
    // 计算 3x3 矩阵行列式，检查三点是否共线
    double det = s1.x * (s2.y - s3.y) - s1.y * (s2.x - s3.x) + (s2.x * s3.y - s3.x * s2.y);
    if (std::abs(det) < 1e-10) return false;

    // 使用代数余子式法解线性方程组 (针对每一行独立求解)
    auto solveRow = [&](double r1, double r2, double r3, double* res) {
        res[0] = ((s2.y - s3.y) * r1 + (s3.y - s1.y) * r2 + (s1.y - s2.y) * r3) / det;
        res[1] = ((s3.x - s2.x) * r1 + (s1.x - s3.x) * r2 + (s2.x - s1.x) * r3) / det;
        res[2] = ((s2.x * s3.y - s3.x * s2.y) * r1 + (s3.x * s1.y - s1.x * s3.y) * r2 + (s1.x * s2.y - s2.x * s1.y) * r3) / det;
    };

    solveRow(d1.x, d2.x, d3.x, out.m[0]);
    solveRow(d1.y, d2.y, d3.y, out.m[1]);
    return true;
}

/**
 * @brief 内部函数：利用最小二乘法对所有内点进行全局优化拟合
 */
void MyRANSAC::refineModelLeastSquares(const std::vector<Point2D>& src, const std::vector<Point2D>& dst,
    const std::vector<int>& inliers, AffineModel& out) {
    double Sxx = 0, Syy = 0, Sxy = 0, Sx = 0, Sy = 0;
    double Sdx_x = 0, Sdx_y = 0, Sdx = 0, Sdy_x = 0, Sdy_y = 0, Sdy = 0;
    int n = (int)inliers.size();

    // 统计所有内点的矩 (Moments)
    for (int idx : inliers) {
        double x = src[idx].x, y = src[idx].y;
        double dx = dst[idx].x, dy = dst[idx].y;
        Sxx += x * x; Syy += y * y; Sxy += x * y; Sx += x; Sy += y;
        Sdx_x += dx * x; Sdx_y += dx * y; Sdx += dx;
        Sdy_x += dy * x; Sdy_y += dy * y; Sdy += dy;
    }

    // 构建并求解正规方程组 (Normal Equations)
    auto solveLS = [&](double b1, double b2, double b3, double* res) {
        double A[3][3] = { {Sxx, Sxy, Sx}, {Sxy, Syy, Sy}, {Sx, Sy, (double)n} };

        // 克莱姆法则求解 3x3 方程组
        double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
            A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
            A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

        if (std::abs(det) < 1e-12) return;

        res[0] = ((A[1][1] * A[2][2] - A[1][2] * A[2][1]) * b1 - (A[0][1] * A[2][2] - A[0][2] * A[2][1]) * b2 + (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * b3) / det;
        res[1] = (-(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * b1 + (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * b2 - (A[0][0] * A[1][2] - A[0][2] * A[1][0]) * b3) / det;
        res[2] = ((A[1][0] * A[2][1] - A[1][1] * A[2][0]) * b1 - (A[0][0] * A[2][1] - A[0][1] * A[2][0]) * b2 + (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * b3) / det;
    };

    solveLS(Sdx_x, Sdx_y, Sdx, out.m[0]);
    solveLS(Sdy_x, Sdy_y, Sdy, out.m[1]);
}