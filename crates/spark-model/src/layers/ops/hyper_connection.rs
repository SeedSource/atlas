// SPDX-License-Identifier: AGPL-3.0-only

//! Manifold-Constrained Hyper-Connections (mHC) kernel dispatch (DeepSeek-V4).
//!
//! Wraps the `hyper_connection` module kernels (`hc_pre`, `hc_post`,
//! `hc_head`). The hidden state is stored BF16 as `[T, hc_mult, H]`
//! (stream-major per token). HC parameters are float32 device buffers.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Broadcast a single hidden state into `hc_mult` identical streams:
/// `streams[t, i, d] = hidden[t, d]`. One block per token.
pub fn hc_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    streams: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(streams)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Collapse `hc_mult` streams to one (RMS-rescaled mix → sigmoid `pre`
/// weighted sum) and emit `post` / `comb` (Sinkhorn) for the matching
/// `hc_post`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    hc_fn: DevicePtr,
    hc_scale: DevicePtr,
    hc_base: DevicePtr,
    y_out: DevicePtr,
    post_out: DevicePtr,
    comb_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    sinkhorn_iters: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(hc_fn)
        .arg_ptr(hc_scale)
        .arg_ptr(hc_base)
        .arg_ptr(y_out)
        .arg_ptr(post_out)
        .arg_ptr(comb_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_u32(sinkhorn_iters)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Expand the sublayer output back into `hc_mult` streams, mixing the saved
/// residual streams through the doubly-stochastic `comb`. `out` may alias
/// `residual`. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    block_out: DevicePtr,
    residual: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(block_out)
        .arg_ptr(residual)
        .arg_ptr(post)
        .arg_ptr(comb)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Final collapse before the LM head: a single learned sigmoid-weighted sum
/// over the `hc_mult` streams. One block per token.
#[allow(clippy::too_many_arguments)]
pub fn hc_head(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    head_fn: DevicePtr,
    head_scale: DevicePtr,
    head_base: DevicePtr,
    y_out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    norm_eps: f32,
    hc_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(head_fn)
        .arg_ptr(head_scale)
        .arg_ptr(head_base)
        .arg_ptr(y_out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .arg_f32(norm_eps)
        .arg_f32(hc_eps)
        .launch(stream)
}

/// Plain arithmetic mean over the `hc_mult` streams of the post-layer residual:
/// `out[t, d] = mean_i streams[t, i, d]`. This is the native / vLLM DSpark
/// `main_hidden` reduction (`h.mean(dim=stream)`), an UNWEIGHTED mean — distinct
/// from the learned sigmoid-weighted [`hc_head`] collapse. Pure read over the
/// FP32 mHC `streams` highway; emits BF16 into a *separate* `out` buffer (the
/// DFlash capture stack) and never mutates `streams`. One block per token.
///
/// `streams` is `[T, hc_mult, H]` FP32 (stream-major per token); `out` is
/// `[T, H]` BF16. Callers pass pointers pre-offset to the token/slot of
/// interest with `num_tokens = 1` for the single-token decode capture.
pub fn hc_stream_mean(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    streams: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(streams)
        .arg_ptr(out)
        .arg_u32(hidden_size)
        .arg_u32(hc_mult)
        .launch(stream)
}

/// Host reference for the [`hc_stream_mean`] kernel: the exact math the device
/// kernel implements, used to lock the reduction semantics under unit test
/// (the CUDA kernel itself is validated at serve time). `streams` is the FP32
/// `[hc_mult, hidden]` residual for one token (stream-major); returns the
/// per-dim mean rounded to BF16 (truncating cast, matching `__float2bfloat16`
/// round-to-nearest is approximated here by returning the f32 mean — callers
/// compare with a tolerance).
#[cfg(test)]
pub(crate) fn stream_mean_ref(streams: &[f32], hidden: usize, hc_mult: usize) -> Vec<f32> {
    assert_eq!(streams.len(), hidden * hc_mult, "streams shape [hc_mult, hidden]");
    let inv = 1.0f32 / hc_mult as f32;
    (0..hidden)
        .map(|d| {
            let mut acc = 0.0f32;
            for i in 0..hc_mult {
                acc += streams[i * hidden + d];
            }
            acc * inv
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::stream_mean_ref;

    #[test]
    fn stream_mean_matches_manual_mean() {
        // Synthetic [hc_mult=4, hidden=3] stream-major tensor.
        let hidden = 3usize;
        let hc_mult = 4usize;
        // stream 0: [1,2,3], 1: [3,4,5], 2: [5,6,7], 3: [7,8,9]
        let streams = vec![
            1.0, 2.0, 3.0, // stream 0
            3.0, 4.0, 5.0, // stream 1
            5.0, 6.0, 7.0, // stream 2
            7.0, 8.0, 9.0, // stream 3
        ];
        let got = stream_mean_ref(&streams, hidden, hc_mult);
        // mean per dim: (1+3+5+7)/4=4, (2+4+6+8)/4=5, (3+5+7+9)/4=6
        assert_eq!(got, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn stream_mean_single_stream_is_identity() {
        let got = stream_mean_ref(&[2.5, -1.0, 8.0], 3, 1);
        assert_eq!(got, vec![2.5, -1.0, 8.0]);
    }
}
