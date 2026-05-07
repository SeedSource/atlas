// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path GEMV with on-the-fly MLX 8-bit dequantization:
//
//   y[n] = sum over k of (W_dequant[n, k] * x[k])
//
// One threadgroup per output row. Threads stride over K, accumulate
// in FP32, simdgroup-reduce, then a final cross-simdgroup reduction
// writes the row total. Avoids materializing dequantized W in memory
// — saves ~K * 2 bytes of memory traffic per output element on the
// hot decode path.
//
// Layout (matches `mlx_int8_dequant`):
//   packed : uint32  [N, K / 4]
//   scales : bfloat  [N, K / group_size]
//   biases : bfloat  [N, K / group_size]
//   x      : bfloat  [K]
//   y      : bfloat  [N]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void mlx_int8_gemv(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device const bfloat *x      [[buffer(6)]],
    device bfloat       *y      [[buffer(7)]],
    uint   row     [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }

    threadgroup float partial[MAX_SIMDGROUPS];

    uint groups_per_row = K / group_size;
    float acc = 0.0;
    for (uint k = tid; k < K; k += tg_size) {
        uint word = packed[row * (K / 4u) + (k >> 2)];
        uint byte = (word >> ((k & 3u) * 8u)) & 0xFFu;
        uint g    = k / group_size;
        float s   = float(scales[row * groups_per_row + g]);
        float b   = float(biases[row * groups_per_row + g]);
        float w   = float(byte) * s + b;
        acc += w * float(x[k]);
    }

    // simdgroup reduction → first lane writes to shared memory
    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-simdgroup reduction in simdgroup 0
    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v);
        }
    }
}
