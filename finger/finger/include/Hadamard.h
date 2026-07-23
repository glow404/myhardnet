#ifndef HADAMARD_H
#define HADAMARD_H

#include <immintrin.h>
#include <stdint.h>

class Hadamard {
public:
    static void generateHadamard128();

     /**
     * @brief 核心：将 128位 SIFT 二进制描述子物理旋转 180 度
     * * 基于数学推理 ：
     * 1. 4x4 网格对调 (row, col) -> (3-row, 3-col)
     * 2. 8方向 Bin 偏移 (bin) -> (bin + 4) % 8
     * 这等效于原始 128 维索引进行 XOR 124 操作。
     * * 在 Sylvester 哈达玛空间中，这映射为特定比特位的取反。
     */
    static void applyHadamardProjection(const float* src_float_desc, uint64_t* dst_bin_desc);

    static inline void flip180(const uint64_t* src, uint64_t* dst) {
        const uint64_t MASK0 = 0x0FF0F00FF00F0FF0ULL;
        const uint64_t MASK1 = 0xF00F0FF00FF0F00FULL;

        dst[0] = src[0] ^ MASK0;
        dst[1] = src[1] ^ MASK1;
    }

    static inline int calculateHamming128(const uint64_t* a, const uint64_t* b) {
#ifdef _MSC_VER
        return static_cast<int>(__popcnt64(a[0] ^ b[0]) + __popcnt64(a[1] ^ b[1]));
#else
        return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]);
#endif
    }

    // Packed layout:
    // - word 0: stable64, unchanged under 180-degree flip
    // - word 1: flip64, inverted under 180-degree flip
    static inline void calculateForwardAndFlipHamming128(
        const uint64_t* a,
        const uint64_t* b,
        int& forward_dist,
        int& flip_dist) {
#ifdef _MSC_VER
        const int stable_diff = static_cast<int>(__popcnt64(a[0] ^ b[0]));
        const int flip_diff = static_cast<int>(__popcnt64(a[1] ^ b[1]));
#else
        const int stable_diff = __builtin_popcountll(a[0] ^ b[0]);
        const int flip_diff = __builtin_popcountll(a[1] ^ b[1]);
#endif
        forward_dist = stable_diff + flip_diff;
        flip_dist = stable_diff + (64 - flip_diff);
    }
};

#endif
