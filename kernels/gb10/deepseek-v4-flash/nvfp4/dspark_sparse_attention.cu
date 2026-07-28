// SPDX-License-Identifier: AGPL-3.0-only

// DSpark K=1 drafter: sparse windowed MLA attention with an attention-sink
// normalizer. Net-new drafter kernel absent from Atlas's per-token MLA decode.
//
// Correctness-first port of the DeepSeek reference `dspark_sparse_attention`
// (dsv4_nvidia/dspark_kernels.py:677-713, the torch reference that the Triton
// kernel matches). NO mma.sync / tcgen05 — a scalar block-per-(b,q,h) kernel.
// The scalar-loop formulation encoded here was cross-checked bit-for-bit
// against the reference einsum offline (spark-bench
// scripts/dspark-k1/sparse_mla_math_crosscheck.py: cos=1.0, maxabs~1e-16 over
// random + empty-main + full-window + sink-dominates + mixed-length cases).
//
// MLA: a single shared latent KV per token (head_dim, NOT per-head). The KV is
// the sliding-window ring `main_kv_cache` (window_size slots) concatenated with
// the current block `draft_kv` (block_size slots):
//   kv = [ main_kv_cache[0..window) ; draft_kv[0..block) ]      (T = window + block)
// Validity: a main slot k<window is valid iff k < valid_main_lengths[b]; every
// draft slot is valid. The ring rotation is irrelevant here — attention is a
// permutation-invariant weighted sum, so reading slots [0,window) with a count
// mask matches the reference regardless of where store_main_kv wrote them.
//
// Math (per (b, q, h)):
//   scores[k] = (q . kv[k]) * softmax_scale          (invalid -> -inf)
//   norm      = max( max_k scores[k], attn_sink[h] )
//   w[k]      = exp(scores[k] - norm)                 (invalid -> 0)
//   denom     = sum_k w[k] + exp(attn_sink[h] - norm) (the sink term)
//   out[d]    = ( sum_k w[k] * kv[k][d] ) / denom
//
// attn_sink is FP32 in the checkpoint (see Atlas PR #341 sink-dtype fix); q /
// draft_kv / main_kv_cache / out are BF16. Accumulation is FP32.

#include <cuda_bf16.h>

#define DSPARK_ATTN_BLOCK 256
#define DSPARK_NEG_INF (-1e30f)

// Block-wide reduction helpers over red[0..DSPARK_ATTN_BLOCK).
__device__ __forceinline__ float dspark_block_max(float* red, unsigned int tid) {
    for (unsigned int s = DSPARK_ATTN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    return red[0];
}

__device__ __forceinline__ float dspark_block_sum(float* red, unsigned int tid) {
    for (unsigned int s = DSPARK_ATTN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// Grid: (B * block_size * num_heads, 1, 1).  Block: (256, 1, 1).
// Dynamic shared memory: (head_dim + window_size + block_size) floats
//   [ q_sh : head_dim ][ scores : window_size + block_size ].
extern "C" __global__ void dspark_sparse_attention(
    const __nv_bfloat16* __restrict__ q,          // [B, block, H, D]
    const __nv_bfloat16* __restrict__ draft_kv,   // [B, block, D]
    const __nv_bfloat16* __restrict__ main_kv,    // [B, window, D]  (sliding-window ring)
    const int* __restrict__ valid_main_lengths,   // [B]
    const float* __restrict__ attn_sink,          // [H]  (FP32)
    __nv_bfloat16* __restrict__ out,              // [B, block, H, D]
    const float softmax_scale,
    const unsigned int batch_size,
    const unsigned int block_size,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const unsigned int window_size
) {
    const unsigned int prog = blockIdx.x;               // (b, qi, h) flattened
    const unsigned int H = num_heads;
    const unsigned int D = head_dim;
    const unsigned int W = window_size;
    const unsigned int T = W + block_size;

    const unsigned int h = prog % H;
    const unsigned int qi = (prog / H) % block_size;
    const unsigned int b = prog / (H * block_size);
    if (b >= batch_size) return;

    const unsigned int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* q_sh = smem;                 // [D]
    float* scores = smem + D;           // [T]
    __shared__ float red[DSPARK_ATTN_BLOCK];

    // Load q[b, qi, h, :] into shared (FP32).
    const __nv_bfloat16* qp = q + (((size_t)b * block_size + qi) * H + h) * D;
    for (unsigned int d = tid; d < D; d += DSPARK_ATTN_BLOCK) {
        q_sh[d] = (float)qp[d];
    }
    __syncthreads();

    const int vlen = valid_main_lengths[b];
    const __nv_bfloat16* main_b = main_kv + (size_t)b * W * D;
    const __nv_bfloat16* draft_b = draft_kv + (size_t)b * block_size * D;

    // Phase 1: each thread owns a strided subset of kv tokens; full dot over D.
    for (unsigned int k = tid; k < T; k += DSPARK_ATTN_BLOCK) {
        const __nv_bfloat16* kvk;
        if (k < W) {
            if ((int)k >= vlen) { scores[k] = DSPARK_NEG_INF; continue; }  // masked main
            kvk = main_b + (size_t)k * D;
        } else {
            kvk = draft_b + (size_t)(k - W) * D;                            // draft always valid
        }
        float acc = 0.0f;
        for (unsigned int d = 0; d < D; ++d) acc += q_sh[d] * (float)kvk[d];
        scores[k] = acc * softmax_scale;
    }
    __syncthreads();

    // Phase 2: rowmax over scores, then norm = max(rowmax, sink[h]).
    float local_max = DSPARK_NEG_INF;
    for (unsigned int k = tid; k < T; k += DSPARK_ATTN_BLOCK) {
        local_max = fmaxf(local_max, scores[k]);
    }
    red[tid] = local_max;
    __syncthreads();
    float rowmax = dspark_block_max(red, tid);
    const float sink = attn_sink[h];
    const float norm = fmaxf(rowmax, sink);
    __syncthreads();

    // Phase 3: convert scores -> weights in place; block-sum for denom.
    float local_sum = 0.0f;
    for (unsigned int k = tid; k < T; k += DSPARK_ATTN_BLOCK) {
        float w = (scores[k] <= DSPARK_NEG_INF * 0.5f) ? 0.0f : __expf(scores[k] - norm);
        scores[k] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    float sumw = dspark_block_sum(red, tid);
    const float denom = sumw + __expf(sink - norm);   // sink term; always > 0
    __syncthreads();

    // Phase 4: out[d] = ( sum_k w[k] * kv[k][d] ) / denom.
    __nv_bfloat16* op = out + (((size_t)b * block_size + qi) * H + h) * D;
    const float inv_denom = 1.0f / denom;
    for (unsigned int d = tid; d < D; d += DSPARK_ATTN_BLOCK) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < T; ++k) {
            float w = scores[k];
            if (w == 0.0f) continue;
            const __nv_bfloat16* kvk = (k < W) ? (main_b + (size_t)k * D)
                                               : (draft_b + (size_t)(k - W) * D);
            acc += w * (float)kvk[d];
        }
        op[d] = (__nv_bfloat16)(acc * inv_denom);
    }
}
