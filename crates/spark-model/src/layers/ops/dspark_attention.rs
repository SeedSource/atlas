// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark K=1 drafter sparse windowed-MLA attention dispatch.
//!
//! Wraps the `dspark_sparse_attention` kernel (windowed MLA over the
//! sliding-window `main_kv_cache` ring + current-block `draft_kv`, with an
//! attention-sink softmax normalizer). Correctness-first; see the `.cu` for the
//! reference-matched math and the offline cross-check.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Run DSpark sparse-MLA attention.
///
/// Shapes (BF16 unless noted): `q [B, block, H, D]`, `draft_kv [B, block, D]`,
/// `main_kv [B, window, D]`, `valid_main_lengths [B]` (i32), `attn_sink [H]`
/// (FP32), `out [B, block, H, D]`. One CUDA block per `(b, qi, h)`; dynamic
/// shared memory holds `q_sh[D]` + `scores[window+block]`.
#[allow(clippy::too_many_arguments)]
pub fn dspark_sparse_attention(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    draft_kv: DevicePtr,
    main_kv: DevicePtr,
    valid_main_lengths: DevicePtr,
    attn_sink: DevicePtr,
    out: DevicePtr,
    softmax_scale: f32,
    batch_size: u32,
    block_size: u32,
    num_heads: u32,
    head_dim: u32,
    window_size: u32,
    stream: u64,
) -> Result<()> {
    let programs = batch_size * block_size * num_heads;
    let shared_bytes = (head_dim + window_size + block_size) * 4; // floats: q_sh[D] + scores[T]
    KernelLaunch::new(gpu, kernel)
        .grid([programs, 1, 1])
        .block([256, 1, 1])
        .shared_mem(shared_bytes)
        .arg_ptr(q)
        .arg_ptr(draft_kv)
        .arg_ptr(main_kv)
        .arg_ptr(valid_main_lengths)
        .arg_ptr(attn_sink)
        .arg_ptr(out)
        .arg_f32(softmax_scale)
        .arg_u32(batch_size)
        .arg_u32(block_size)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(window_size)
        .launch(stream)
}
