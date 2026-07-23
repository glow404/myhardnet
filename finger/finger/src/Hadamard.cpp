#include "Hadamard.h"

#include <cmath>
#include <cstdint>

namespace {
float G_Hadamard128[128][128];
bool G_IsInitialized = false;

constexpr uint64_t kFlipMaskRaw[2] = {
    0x0FF0F00FF00F0FF0ULL,
    0xF00F0FF00FF0F00FULL,
};

inline bool raw_flip_mask_bit(int raw_index) {
    const int word = raw_index / 64;
    const int bit = raw_index % 64;
    return ((kFlipMaskRaw[word] >> bit) & 1ULL) != 0;
}

void pack_descriptor_by_flip_symmetry(const uint64_t* raw_desc, uint64_t* packed_desc) {
    packed_desc[0] = 0;
    packed_desc[1] = 0;

    int stable_pos = 0;
    int flip_pos = 0;
    for (int raw_index = 0; raw_index < 128; ++raw_index) {
        const int raw_word = raw_index / 64;
        const int raw_bit = raw_index % 64;
        const uint64_t raw_value = (raw_desc[raw_word] >> raw_bit) & 1ULL;

        if (raw_flip_mask_bit(raw_index)) {
            packed_desc[1] |= (raw_value << flip_pos);
            ++flip_pos;
        }
        else {
            packed_desc[0] |= (raw_value << stable_pos);
            ++stable_pos;
        }
    }
}
}  // namespace

void Hadamard::generateHadamard128() {
    if (G_IsInitialized) {
        return;
    }

    G_Hadamard128[0][0] = 1.0f;
    for (int k = 1; k <= 7; ++k) {
        const int prev_size = 1 << (k - 1);
        for (int i = 0; i < prev_size; ++i) {
            for (int j = 0; j < prev_size; ++j) {
                const float value = G_Hadamard128[i][j];
                G_Hadamard128[i][j + prev_size] = value;
                G_Hadamard128[i + prev_size][j] = value;
                G_Hadamard128[i + prev_size][j + prev_size] = -value;
            }
        }
    }

    G_IsInitialized = true;
}

void Hadamard::applyHadamardProjection(const float* src_float_desc, uint64_t* dst_bin_desc) {
    if (!G_IsInitialized) {
        generateHadamard128();
    }

    uint64_t raw_desc[2] = {0, 0};
    for (int i = 0; i < 128; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < 128; ++j) {
            sum += G_Hadamard128[i][j] * src_float_desc[j];
        }

        if (sum >= 0.0f) {
            if (i < 64) {
                raw_desc[0] |= (1ULL << i);
            }
            else {
                raw_desc[1] |= (1ULL << (i - 64));
            }
        }
    }

    pack_descriptor_by_flip_symmetry(raw_desc, dst_bin_desc);
}
