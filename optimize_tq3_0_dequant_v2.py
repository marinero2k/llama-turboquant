#!/usr/bin/env python3
"""Optimize tq3_0 dequant tile loaders for speed.
Uses function-local LUT (lands in registers) instead of __device__ global.
Uses memcpy for uint32_t extraction to avoid alignment UB.

Run: python3 optimize_tq3_0_dequant_v2.py /home/mar/llama-turboquant/ggml/src/ggml-cuda/fattn-mma-tq3_0.cuh
Then rebuild: cd build && cmake --build . -j$(nproc)
"""
import sys, shutil

path = sys.argv[1]
shutil.copy2(path, path + ".opt2_bak")
print(f"Backup: {path}.opt2_bak")

with open(path) as f:
    src = f.read()

fixes = 0

# ============================================================
# Replace tq3_0_load_tile (K dequant) with optimized version
# ============================================================
OLD_K_FUNC = """// Dequantize one tq3_0 K batch into tile_K shared memory (f16 layout)
// Output layout matches flash_attn_ext_f16_load_tile: [token_i][stride_tile] half2
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq3_0_load_tile(
        const block_tq3_0 * const __restrict__ K_tq3,
        half2 * const __restrict__ tile_K,
        const int D,           // = DKQ (number of elements per token)
        const int stride_K,    // stride in block_tq3_0 per token
        const int i_sup) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int blocks_per_row = D / QK_TQ3_0;

    for (int idx = threadIdx.y * warp_size + threadIdx.x;
         idx < nbatch_fa * blocks_per_row;
         idx += nwarps * warp_size) {

        const int ti = idx / blocks_per_row;
        const int bj = idx % blocks_per_row;

        if (oob_check && ti >= i_sup) {
            const int base = ti * stride_tile + bj * (QK_TQ3_0/2);
#pragma unroll
            for (int j = 0; j < QK_TQ3_0/2; j++)
                tile_K[base + j] = make_half2(0.0f, 0.0f);
            continue;
        }

        const block_tq3_0 & blk = K_tq3[ti * stride_K + bj];
        const float d = __half2float(blk.gamma);

        float x[QK_TQ3_0];
#pragma unroll
        for (int j = 0; j < QK_TQ3_0; j++) {
            const int qi = (blk.qs[j/4] >> (2*(j%4))) & 0x3;
            x[j] = d * (qi == 0 ? -1.510f : qi == 1 ? -0.4528f : qi == 2 ? 0.4528f : 1.510f);
        }

        // Inverse WHT
#pragma unroll
        for (int step = 1; step < QK_TQ3_0; step <<= 1) {
#pragma unroll
            for (int i = 0; i < QK_TQ3_0; i += step*2) {
#pragma unroll
                for (int jj = i; jj < i+step; jj++) {
                    float a = x[jj], b = x[jj+step];
                    x[jj] = a + b; x[jj+step] = a - b;
                }
            }
        }
        const float inv_sqrt32 = 0.17677669529663688f;
#pragma unroll
        for (int j = 0; j < QK_TQ3_0; j++)
            x[j] *= inv_sqrt32 * tq3_mma_signs[j];

        const int base = ti * stride_tile + bj * (QK_TQ3_0/2);
#pragma unroll
        for (int j = 0; j < QK_TQ3_0/2; j++)
            tile_K[base + j] = make_half2(__float2half(x[j*2]), __float2half(x[j*2+1]));
    }
}"""

NEW_K_FUNC = """// Dequantize one tq3_0 K batch into tile_K shared memory (f16 layout)
// Optimized: local LUT, vectorized uint32 extraction, no branches
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq3_0_load_tile(
        const block_tq3_0 * const __restrict__ K_tq3,
        half2 * const __restrict__ tile_K,
        const int D,
        const int stride_K,
        const int i_sup) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int blocks_per_row = D / QK_TQ3_0;
    const float lut[4] = { -1.510f, -0.4528f, 0.4528f, 1.510f };

    for (int idx = threadIdx.y * warp_size + threadIdx.x;
         idx < nbatch_fa * blocks_per_row;
         idx += nwarps * warp_size) {

        const int ti = idx / blocks_per_row;
        const int bj = idx % blocks_per_row;
        const int base = ti * stride_tile + bj * (QK_TQ3_0/2);

        if (oob_check && ti >= i_sup) {
#pragma unroll
            for (int j = 0; j < QK_TQ3_0/2; j++)
                tile_K[base + j] = make_half2(0.0f, 0.0f);
            continue;
        }

        const block_tq3_0 & blk = K_tq3[ti * stride_K + bj];
        const float d = __half2float(blk.gamma);

        uint32_t qs_u32[2];
        memcpy(qs_u32, blk.qs, 8);

        float x[QK_TQ3_0];
#pragma unroll
        for (int j = 0; j < 16; j++) {
            x[j]    = d * lut[(qs_u32[0] >> (2*j)) & 0x3];
            x[j+16] = d * lut[(qs_u32[1] >> (2*j)) & 0x3];
        }

        // Inverse WHT — 5 butterfly stages
#pragma unroll
        for (int step = 1; step < QK_TQ3_0; step <<= 1) {
#pragma unroll
            for (int i = 0; i < QK_TQ3_0; i += step*2) {
#pragma unroll
                for (int jj = i; jj < i+step; jj++) {
                    const float a = x[jj], b = x[jj+step];
                    x[jj] = a + b; x[jj+step] = a - b;
                }
            }
        }

        constexpr float inv_sqrt32 = 0.17677669529663688f;
#pragma unroll
        for (int j = 0; j < QK_TQ3_0/2; j++) {
            tile_K[base + j] = make_half2(
                __float2half(x[j*2]   * (inv_sqrt32 * tq3_mma_signs[j*2])),
                __float2half(x[j*2+1] * (inv_sqrt32 * tq3_mma_signs[j*2+1])));
        }
    }
}"""

if OLD_K_FUNC in src:
    src = src.replace(OLD_K_FUNC, NEW_K_FUNC, 1)
    fixes += 1
    print("Replaced tq3_0_load_tile with optimized version ✓")
else:
    print("WARNING: tq3_0_load_tile pattern not found!")

# ============================================================
# Replace tq3_0v_load_tile (V dequant) with optimized version
# ============================================================
OLD_V_FUNC = """// Dequantize tq3_0v V tokens into tile_V shared memory (f16 layout)
// tq3_0v = same block layout as tq3_0 but stored WITHOUT WHT
// So dequant is just centroid lookup — no inverse WHT needed
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq3_0v_load_tile(
        const block_tq3_0 * const __restrict__ V_tq3v,
        half2 * const __restrict__ tile_V,
        const int D,
        const int stride_V,
        const int i_sup) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int blocks_per_row = D / QK_TQ3_0;

    for (int idx = threadIdx.y * warp_size + threadIdx.x;
         idx < nbatch_fa * blocks_per_row;
         idx += nwarps * warp_size) {

        const int ti = idx / blocks_per_row;
        const int bj = idx % blocks_per_row;

        if (oob_check && ti >= i_sup) {
            const int base = ti * stride_tile + bj * (QK_TQ3_0/2);
#pragma unroll
            for (int j = 0; j < QK_TQ3_0/2; j++)
                tile_V[base + j] = make_half2(0.0f, 0.0f);
            continue;
        }

        const block_tq3_0 & blk = V_tq3v[ti * stride_V + bj];
        const float d = __half2float(blk.gamma);

        // Centroid lookup only — no WHT for V (tq3_0v stores original space)
        const int base = ti * stride_tile + bj * (QK_TQ3_0/2);
#pragma unroll
        for (int j = 0; j < QK_TQ3_0/2; j++) {
            const int e0 = j*2;
            const int e1 = j*2 + 1;
            const int qi0 = (blk.qs[e0/4] >> (2*(e0%4))) & 0x3;
            const int qi1 = (blk.qs[e1/4] >> (2*(e1%4))) & 0x3;
            const float v0 = d * (qi0 == 0 ? -1.510f : qi0 == 1 ? -0.4528f : qi0 == 2 ? 0.4528f : 1.510f);
            const float v1 = d * (qi1 == 0 ? -1.510f : qi1 == 1 ? -0.4528f : qi1 == 2 ? 0.4528f : 1.510f);
            tile_V[base + j] = make_half2(__float2half(v0), __float2half(v1));
        }
    }
}"""

NEW_V_FUNC = """// Dequantize tq3_0v V tokens into tile_V shared memory (f16 layout)
// Optimized: local LUT, vectorized uint32 extraction, no branches
template<int stride_tile, int nwarps, int nbatch_fa, bool oob_check>
static __device__ __forceinline__ void tq3_0v_load_tile(
        const block_tq3_0 * const __restrict__ V_tq3v,
        half2 * const __restrict__ tile_V,
        const int D,
        const int stride_V,
        const int i_sup) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int blocks_per_row = D / QK_TQ3_0;
    const float lut[4] = { -1.510f, -0.4528f, 0.4528f, 1.510f };

    for (int idx = threadIdx.y * warp_size + threadIdx.x;
         idx < nbatch_fa * blocks_per_row;
         idx += nwarps * warp_size) {

        const int ti = idx / blocks_per_row;
        const int bj = idx % blocks_per_row;
        const int base = ti * stride_tile + bj * (QK_TQ3_0/2);

        if (oob_check && ti >= i_sup) {
#pragma unroll
            for (int j = 0; j < QK_TQ3_0/2; j++)
                tile_V[base + j] = make_half2(0.0f, 0.0f);
            continue;
        }

        const block_tq3_0 & blk = V_tq3v[ti * stride_V + bj];
        const float d = __half2float(blk.gamma);

        uint32_t qs_u32[2];
        memcpy(qs_u32, blk.qs, 8);

#pragma unroll
        for (int j = 0; j < 8; j++) {
            tile_V[base + j] = make_half2(
                __float2half(d * lut[(qs_u32[0] >> (4*j))     & 0x3]),
                __float2half(d * lut[(qs_u32[0] >> (4*j + 2)) & 0x3]));
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            tile_V[base + 8 + j] = make_half2(
                __float2half(d * lut[(qs_u32[1] >> (4*j))     & 0x3]),
                __float2half(d * lut[(qs_u32[1] >> (4*j + 2)) & 0x3]));
        }
    }
}"""

if OLD_V_FUNC in src:
    src = src.replace(OLD_V_FUNC, NEW_V_FUNC, 1)
    fixes += 1
    print("Replaced tq3_0v_load_tile with optimized version ✓")
else:
    print("WARNING: tq3_0v_load_tile pattern not found!")

with open(path, 'w') as f:
    f.write(src)

print(f"\n{'='*60}")
if fixes == 2:
    print(f"SUCCESS: Both optimizations applied.")
else:
    print(f"WARNING: Only {fixes}/2 applied.")
print(f"Rebuild: cd build && cmake --build . -j$(nproc)")
print(f"{'='*60}")
