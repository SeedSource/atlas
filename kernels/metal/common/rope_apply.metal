// SPDX-License-Identifier: AGPL-3.0-only
//
// Rotary position embedding (RoPE) — applied in-place to a Q or K
// tensor.
//
// Standard RoPE pairs adjacent dimensions (d, d+1) and rotates them
// by an angle proportional to the token's absolute position:
//
//   theta = pos * inv_freq[d/2]
//   q[d]   =  q_old[d]   * cos(theta) - q_old[d+1] * sin(theta)
//   q[d+1] =  q_old[d]   * sin(theta) + q_old[d+1] * cos(theta)
//
// `inv_freq` is precomputed (one entry per dimension pair) on the
// host: `inv_freq[i] = 1.0 / (rope_theta ^ (2i / head_dim))`.
//
// This kernel uses the GPT-NeoX layout (the Qwen and Llama lineage):
// pairs are `(d, d + head_dim/2)` for d in [0, head_dim/2). Adjust
// the index math for legacy `(2i, 2i+1)` paired layout if needed.
//
// Layout:
//   x         : bfloat [num_tokens, num_heads, head_dim]
//   inv_freq  : float  [head_dim / 2]
//   positions : uint32 [num_tokens]
//
// Grid: (head_dim/2 threads, num_heads, num_tokens)

#include <metal_stdlib>
using namespace metal;

kernel void rope_apply(
    constant uint  &num_tokens [[buffer(0)]],
    constant uint  &num_heads  [[buffer(1)]],
    constant uint  &head_dim   [[buffer(2)]],
    device const uint   *positions [[buffer(3)]],
    device const float  *inv_freq  [[buffer(4)]],
    device bfloat       *x         [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d  = gid.x;          // 0 .. head_dim/2
    uint h  = gid.y;          // head index
    uint tok = gid.z;         // token index
    uint half_dim = head_dim >> 1u;
    if (d >= half_dim || h >= num_heads || tok >= num_tokens) {
        return;
    }

    uint pos = positions[tok];
    float theta = float(pos) * inv_freq[d];
    float c = cos(theta);
    float s = sin(theta);

    uint base = (tok * num_heads + h) * head_dim;
    uint i_lo = base + d;
    uint i_hi = base + d + half_dim;
    float lo = float(x[i_lo]);
    float hi = float(x[i_hi]);

    x[i_lo] = bfloat(lo * c - hi * s);
    x[i_hi] = bfloat(lo * s + hi * c);
}
