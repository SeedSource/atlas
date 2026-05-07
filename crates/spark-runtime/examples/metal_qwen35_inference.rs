// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end Qwen3.5-4B-MLX-8bit inference on the metal backend.
//!
//! Tokenize a prompt → embed → run all 32 layers → final RMSNorm →
//! LM head (tied to embed_tokens) → argmax → decode → print.
//!
//! ⚠️  Linear-attention layers are currently identity-passthrough.
//! The model is hybrid (8 full_attention + 24 linear_attention via
//! GDN). The full_attention path is the parity-tested kernel chain
//! used by `metal_real_model_full_attention_block_layer3`. The
//! linear_attention path needs the GDN orchestration around the
//! existing `gated_delta_rule_decode` + `causal_conv1d_decode`
//! kernels — that's a follow-on. With identity passthrough, the
//! generated token won't match what the real model would produce
//! (75 % of the model's contribution is bypassed) — but every
//! other piece of the inference pipeline (tokenizer integration,
//! per-token KV-cache building, multi-layer chain, LM head, sampler)
//! exercises end-to-end on real Qwen3.5-4B-MLX-8bit weights.
//!
//! Run with:
//!
//!     ATLAS_TARGET_HW=metal \
//!     ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
//!     ATLAS_TARGET_QUANT=mlx_int8 \
//!     PROMPT="What is the capital of France?" \
//!     cargo run --release --example metal_qwen35_inference \
//!         --features metal --no-default-features

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::MlxInt8Weight;
use std::time::Instant;
use tokenizers::Tokenizer;

// Helpers kept available for inline edits during debugging.
#[allow(dead_code)]
fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(dead_code)]
fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

// ── Dims (Qwen3.5-4B from upstream config.json `text_config`) ───
const HIDDEN: u32 = 2560;
const NUM_HEADS: u32 = 16;
const NUM_KV_HEADS: u32 = 4;
const HEAD_DIM: u32 = 256;
const INTERMEDIATE: u32 = 9216;
const NUM_LAYERS: u32 = 32;
const RMS_EPS: f32 = 1e-6;
const GROUP_SIZE: u32 = 64;
const ROPE_THETA: f32 = 10_000_000.0;
const VOCAB: u32 = 248_320;
const Q_TOTAL: u32 = NUM_HEADS * HEAD_DIM * 2; // attn_output_gate
const Q_ONLY: u32 = NUM_HEADS * HEAD_DIM;
const KV_DIM: u32 = NUM_KV_HEADS * HEAD_DIM;

/// Per-layer weights for a `full_attention` layer.
struct FullAttentionLayer {
    input_ln: DevicePtr,
    q_norm: DevicePtr,
    k_norm: DevicePtr,
    post_ln: DevicePtr,
    q_proj: MlxInt8Weight,
    k_proj: MlxInt8Weight,
    v_proj: MlxInt8Weight,
    o_proj: MlxInt8Weight,
    gate_proj: MlxInt8Weight,
    up_proj: MlxInt8Weight,
    down_proj: MlxInt8Weight,
}

impl FullAttentionLayer {
    fn load(backend: &MetalGpuBackend, st: &SafeTensors, layer_idx: u32) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_bf16 = |name: &str| -> Result<DevicePtr> {
            let t = st
                .tensor(name)
                .with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        Ok(Self {
            input_ln: load_bf16(&format!("{prefix}.input_layernorm.weight"))?,
            q_norm: load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?,
            post_ln: load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.self_attn.q_proj"), GROUP_SIZE)?,
            k_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.self_attn.k_proj"), GROUP_SIZE)?,
            v_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.self_attn.v_proj"), GROUP_SIZE)?,
            o_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.self_attn.o_proj"), GROUP_SIZE)?,
            gate_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.mlp.gate_proj"), GROUP_SIZE)?,
            up_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.mlp.up_proj"), GROUP_SIZE)?,
            down_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.mlp.down_proj"), GROUP_SIZE)?,
        })
    }
}

// ── Linear-attention (GDN) dims ────────────────────────────────
const NUM_K_HEADS_LIN: u32 = 16;
const NUM_V_HEADS_LIN: u32 = 32;
const K_HEAD_DIM_LIN: u32 = 128;
const V_HEAD_DIM_LIN: u32 = 128;
const QKV_TOTAL_LIN: u32 = 8192; // = K_HEADS*K_DIM + K_HEADS*K_DIM + V_HEADS*V_DIM = 2048+2048+4096
const Z_DIM_LIN: u32 = 4096; // = NUM_V_HEADS_LIN * V_HEAD_DIM_LIN
const NUM_STATE_HEADS: u32 = 32;
const CONV_KERNEL_SIZE: u32 = 4;

/// Per-layer weights for a `linear_attention` (GDN) layer.
struct LinearAttentionLayer {
    input_ln: DevicePtr,
    a_log: DevicePtr,        // F32 [num_state_heads]
    dt_bias: DevicePtr,      // BF16 [num_state_heads]
    conv1d_weight: DevicePtr, // BF16 [QKV_TOTAL_LIN, kernel_size]
    in_proj_a: MlxInt8Weight,
    in_proj_b: MlxInt8Weight,
    in_proj_qkv: MlxInt8Weight,
    in_proj_z: MlxInt8Weight,
    norm_weight: DevicePtr,  // BF16 [V_HEAD_DIM_LIN]
    out_proj: MlxInt8Weight,
}

impl LinearAttentionLayer {
    fn load(backend: &MetalGpuBackend, st: &SafeTensors, layer_idx: u32) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_raw = |name: &str| -> Result<DevicePtr> {
            let t = st
                .tensor(name)
                .with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        Ok(Self {
            input_ln: load_raw(&format!("{prefix}.input_layernorm.weight"))?,
            a_log: load_raw(&format!("{prefix}.linear_attn.A_log"))?,
            dt_bias: load_raw(&format!("{prefix}.linear_attn.dt_bias"))?,
            conv1d_weight: load_raw(&format!("{prefix}.linear_attn.conv1d.weight"))?,
            in_proj_a: MlxInt8Weight::load(backend, st, &format!("{prefix}.linear_attn.in_proj_a"), GROUP_SIZE)?,
            in_proj_b: MlxInt8Weight::load(backend, st, &format!("{prefix}.linear_attn.in_proj_b"), GROUP_SIZE)?,
            in_proj_qkv: MlxInt8Weight::load(backend, st, &format!("{prefix}.linear_attn.in_proj_qkv"), GROUP_SIZE)?,
            in_proj_z: MlxInt8Weight::load(backend, st, &format!("{prefix}.linear_attn.in_proj_z"), GROUP_SIZE)?,
            norm_weight: load_raw(&format!("{prefix}.linear_attn.norm.weight"))?,
            out_proj: MlxInt8Weight::load(backend, st, &format!("{prefix}.linear_attn.out_proj"), GROUP_SIZE)?,
        })
    }
}

/// Per-layer SSM/conv state for a linear-attention layer.
struct LinearAttentionState {
    /// FP32 [QKV_TOTAL_LIN, d_conv]. Persists across tokens. The
    /// `causal_conv1d_update_l2norm` kernel matches the CUDA
    /// reference and uses FP32 state — prevents recurrent precision
    /// drift past 8K tokens that BF16 truncation introduces.
    conv1d_state: DevicePtr,
    /// FP32 [batch=1, num_v_heads, k_dim, v_dim]. Persists across tokens.
    gdn_state: DevicePtr,
}

impl LinearAttentionState {
    fn alloc(backend: &MetalGpuBackend) -> Result<Self> {
        // BF16 state sized for kernel_size - 1 — matches the simpler
        // `causal_conv1d_decode` kernel that the example currently
        // routes through. The fused FP32 + L2-norm variant requires
        // wider state and isn't wired in yet.
        let conv_state_bytes = (QKV_TOTAL_LIN * (CONV_KERNEL_SIZE - 1)) as usize * 2;
        let gdn_state_floats = (NUM_V_HEADS_LIN * K_HEAD_DIM_LIN * V_HEAD_DIM_LIN) as usize;
        let conv_ptr = backend.alloc(conv_state_bytes)?;
        let gdn_ptr = backend.alloc(gdn_state_floats * 4)?;
        backend.memset(conv_ptr, 0, conv_state_bytes)?;
        backend.memset(gdn_ptr, 0, gdn_state_floats * 4)?;
        Ok(Self {
            conv1d_state: conv_ptr,
            gdn_state: gdn_ptr,
        })
    }
}

/// Per-call scratch buffers for the linear-attention forward.
struct LinScratch {
    x_norm: DevicePtr,     // BF16 [HIDDEN]
    dt_raw: DevicePtr,     // BF16 [num_state_heads]
    b_raw: DevicePtr,      // BF16 [num_state_heads]
    qkv: DevicePtr,        // BF16 [QKV_TOTAL_LIN] pre-conv
    qkv_smooth: DevicePtr, // BF16 [QKV_TOTAL_LIN] post-conv
    z: DevicePtr,          // BF16 [Z_DIM_LIN]
    gate: DevicePtr,       // F32 [num_state_heads]
    beta: DevicePtr,       // F32 [num_state_heads]
    y: DevicePtr,          // BF16 [Z_DIM_LIN]
    y_norm: DevicePtr,     // BF16 [Z_DIM_LIN]
    z_silu: DevicePtr,     // BF16 [Z_DIM_LIN]
    y_gated: DevicePtr,    // BF16 [Z_DIM_LIN]
    out: DevicePtr,        // BF16 [HIDDEN]
    x_resid: DevicePtr,    // BF16 [HIDDEN]
}

fn alloc_lin_scratch(backend: &MetalGpuBackend) -> Result<LinScratch> {
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    let alloc_f32 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 4)?) };
    Ok(LinScratch {
        x_norm: alloc_bf16(HIDDEN)?,
        dt_raw: alloc_bf16(NUM_STATE_HEADS)?,
        b_raw: alloc_bf16(NUM_STATE_HEADS)?,
        qkv: alloc_bf16(QKV_TOTAL_LIN)?,
        qkv_smooth: alloc_bf16(QKV_TOTAL_LIN)?,
        z: alloc_bf16(Z_DIM_LIN)?,
        gate: alloc_f32(NUM_STATE_HEADS)?,
        beta: alloc_f32(NUM_STATE_HEADS)?,
        y: alloc_bf16(Z_DIM_LIN)?,
        y_norm: alloc_bf16(Z_DIM_LIN)?,
        z_silu: alloc_bf16(Z_DIM_LIN)?,
        y_gated: alloc_bf16(Z_DIM_LIN)?,
        out: alloc_bf16(HIDDEN)?,
        x_resid: alloc_bf16(HIDDEN)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn forward_linear_attention(
    backend: &MetalGpuBackend,
    layer: &LinearAttentionLayer,
    state: &LinearAttentionState,
    scratch: &LinScratch,
    rms: spark_runtime::gpu::KernelHandle,
    conv1d: spark_runtime::gpu::KernelHandle,
    gdn_gate: spark_runtime::gpu::KernelHandle,
    sigmoid: spark_runtime::gpu::KernelHandle,
    silu_op: spark_runtime::gpu::KernelHandle,
    mul: spark_runtime::gpu::KernelHandle,
    gdn_dec: spark_runtime::gpu::KernelHandle,
    add: spark_runtime::gpu::KernelHandle,
    x_in: DevicePtr,
    x_buf: DevicePtr,
    stream: u64,
) -> Result<DevicePtr> {
    // 1. norm
    backend.launch_typed(rms, [1, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(x_in), KernelArg::Buffer(layer.input_ln), KernelArg::Buffer(scratch.x_norm),
    ])?;
    // 2. projections
    layer.in_proj_a.gemv(backend, scratch.x_norm, scratch.dt_raw, stream)?;
    layer.in_proj_b.gemv(backend, scratch.x_norm, scratch.b_raw, stream)?;
    layer.in_proj_qkv.gemv(backend, scratch.x_norm, scratch.qkv, stream)?;
    layer.in_proj_z.gemv(backend, scratch.x_norm, scratch.z, stream)?;

    // 3. plain causal conv1d (no fused SiLU/L2-norm yet).
    backend.launch_typed(conv1d, [QKV_TOTAL_LIN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&QKV_TOTAL_LIN.to_le_bytes()),
        KernelArg::Bytes(&CONV_KERNEL_SIZE.to_le_bytes()),
        KernelArg::Buffer(layer.conv1d_weight),
        KernelArg::Buffer(scratch.qkv),
        KernelArg::Buffer(state.conv1d_state),
        KernelArg::Buffer(scratch.qkv_smooth),
    ])?;

    // 4. gate = exp(softplus(dt + dt_bias) * -exp(A_log))
    backend.launch_typed(gdn_gate, [NUM_STATE_HEADS.div_ceil(32), 1, 1], [32, 1, 1], 0, stream, &[
        KernelArg::Bytes(&NUM_STATE_HEADS.to_le_bytes()),
        KernelArg::Buffer(scratch.dt_raw),
        KernelArg::Buffer(layer.dt_bias),
        KernelArg::Buffer(layer.a_log),
        KernelArg::Buffer(scratch.gate),
    ])?;
    // 5. beta = sigmoid(b_raw) → FP32
    backend.launch_typed(sigmoid, [NUM_STATE_HEADS.div_ceil(32), 1, 1], [32, 1, 1], 0, stream, &[
        KernelArg::Bytes(&NUM_STATE_HEADS.to_le_bytes()),
        KernelArg::Buffer(scratch.b_raw),
        KernelArg::Buffer(scratch.beta),
    ])?;

    // 6. Split qkv_smooth: Q[2048] | K[2048] | V[4096] sequential.
    let q_offset = 0;
    let k_offset = (NUM_K_HEADS_LIN * K_HEAD_DIM_LIN) as usize * 2; // 2048 BF16 = 4096B
    let v_offset = (2 * NUM_K_HEADS_LIN * K_HEAD_DIM_LIN) as usize * 2; // 4096 BF16 = 8192B
    let q_view = scratch.qkv_smooth.offset(q_offset);
    let k_view = scratch.qkv_smooth.offset(k_offset);
    let v_view = scratch.qkv_smooth.offset(v_offset);

    // 7. gated_delta_rule_decode
    let batch_size = 1u32;
    let total_groups = NUM_V_HEADS_LIN * batch_size;
    backend.launch_typed(gdn_dec, [total_groups, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Buffer(state.gdn_state),
        KernelArg::Buffer(q_view),
        KernelArg::Buffer(k_view),
        KernelArg::Buffer(v_view),
        KernelArg::Buffer(scratch.gate),
        KernelArg::Buffer(scratch.beta),
        KernelArg::Buffer(scratch.y),
        KernelArg::Bytes(&batch_size.to_le_bytes()),
        KernelArg::Bytes(&NUM_K_HEADS_LIN.to_le_bytes()),
        KernelArg::Bytes(&NUM_V_HEADS_LIN.to_le_bytes()),
        KernelArg::Bytes(&K_HEAD_DIM_LIN.to_le_bytes()),
        KernelArg::Bytes(&V_HEAD_DIM_LIN.to_le_bytes()),
    ])?;

    // 8. per-head rms_norm at head_dim=128 over Z_DIM_LIN/V_HEAD_DIM_LIN = 32 tokens
    backend.launch_typed(rms, [NUM_V_HEADS_LIN, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&V_HEAD_DIM_LIN.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(scratch.y), KernelArg::Buffer(layer.norm_weight), KernelArg::Buffer(scratch.y_norm),
    ])?;

    // 9. silu(z)
    backend.launch_typed(silu_op, [Z_DIM_LIN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&Z_DIM_LIN.to_le_bytes()),
        KernelArg::Buffer(scratch.z), KernelArg::Buffer(scratch.z_silu),
    ])?;

    // 10. y_gated = silu(z) * y_norm
    backend.launch_typed(mul, [Z_DIM_LIN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&Z_DIM_LIN.to_le_bytes()),
        KernelArg::Buffer(scratch.z_silu), KernelArg::Buffer(scratch.y_norm), KernelArg::Buffer(scratch.y_gated),
    ])?;

    // 11. out_proj
    layer.out_proj.gemv(backend, scratch.y_gated, scratch.out, stream)?;

    // 12. residual
    backend.launch_typed(add, [HIDDEN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Buffer(x_in), KernelArg::Buffer(scratch.out), KernelArg::Buffer(scratch.x_resid),
    ])?;

    // Copy to caller's stable x_buf.
    backend.copy_d2d_async(scratch.x_resid, x_buf, HIDDEN as usize * 2, stream)?;
    Ok(x_buf)
}

/// Per-layer KV cache for a single attention layer (single-batch).
struct LayerKvCache {
    k: DevicePtr,
    v: DevicePtr,
    /// Capacity in tokens — caller pre-allocates `max_seq_len * KV_DIM`.
    #[allow(dead_code)]
    capacity: u32,
}

/// Per-layer scratch buffers reused across forward passes.
struct Scratch {
    x_norm: DevicePtr,
    q_full: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    q_norm_out: DevicePtr,
    k_norm_out: DevicePtr,
    attn_out: DevicePtr,
    gated_attn: DevicePtr,
    o: DevicePtr,
    x_resid: DevicePtr,
    x_norm2: DevicePtr,
    gate_act: DevicePtr,
    up_act: DevicePtr,
    ffn_act: DevicePtr,
    ffn_out: DevicePtr,
    x_out: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn forward_full_attention(
    backend: &MetalGpuBackend,
    layer: &FullAttentionLayer,
    scratch: &Scratch,
    kv: &LayerKvCache,
    rms: spark_runtime::gpu::KernelHandle,
    rope: spark_runtime::gpu::KernelHandle,
    kvap: spark_runtime::gpu::KernelHandle,
    attn: spark_runtime::gpu::KernelHandle,
    sg: spark_runtime::gpu::KernelHandle,
    add: spark_runtime::gpu::KernelHandle,
    silu: spark_runtime::gpu::KernelHandle,
    inv_freq_ptr: DevicePtr,
    positions_ptr: DevicePtr,
    x_in: DevicePtr,
    cache_pos: u32,
    seq_len_attn: u32,
    stream: u64,
) -> Result<DevicePtr> {
    // norm1
    backend.launch_typed(rms, [1, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(x_in), KernelArg::Buffer(layer.input_ln), KernelArg::Buffer(scratch.x_norm),
    ])?;
    layer.q_proj.gemv(backend, scratch.x_norm, scratch.q_full, stream)?;
    layer.k_proj.gemv(backend, scratch.x_norm, scratch.k, stream)?;
    layer.v_proj.gemv(backend, scratch.x_norm, scratch.v, stream)?;

    let q_view = scratch.q_full;
    let gate_view = scratch.q_full.offset(Q_ONLY as usize * 2);

    // per-head q/k norm (treat each head as a token)
    backend.launch_typed(rms, [NUM_HEADS, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(q_view), KernelArg::Buffer(layer.q_norm), KernelArg::Buffer(scratch.q_norm_out),
    ])?;
    backend.launch_typed(rms, [NUM_KV_HEADS, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(scratch.k), KernelArg::Buffer(layer.k_norm), KernelArg::Buffer(scratch.k_norm_out),
    ])?;
    backend.copy_d2d_async(scratch.q_norm_out, q_view, Q_ONLY as usize * 2, stream)?;
    backend.copy_d2d_async(scratch.k_norm_out, scratch.k, KV_DIM as usize * 2, stream)?;

    // RoPE — note `positions_ptr` must contain the current absolute pos.
    let half_dim = HEAD_DIM / 2;
    let n_tokens = 1u32;
    backend.launch_typed(rope, [half_dim, NUM_HEADS, 1], [1, 1, 1], 0, stream, &[
        KernelArg::Bytes(&n_tokens.to_le_bytes()),
        KernelArg::Bytes(&NUM_HEADS.to_le_bytes()),
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Buffer(positions_ptr), KernelArg::Buffer(inv_freq_ptr), KernelArg::Buffer(q_view),
    ])?;
    backend.launch_typed(rope, [half_dim, NUM_KV_HEADS, 1], [1, 1, 1], 0, stream, &[
        KernelArg::Bytes(&n_tokens.to_le_bytes()),
        KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Buffer(positions_ptr), KernelArg::Buffer(inv_freq_ptr), KernelArg::Buffer(scratch.k),
    ])?;

    // KV cache append at cache_pos
    backend.launch_typed(kvap, [HEAD_DIM, NUM_KV_HEADS, 1], [1, 1, 1], 0, stream, &[
        KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Bytes(&cache_pos.to_le_bytes()),
        KernelArg::Buffer(scratch.k), KernelArg::Buffer(scratch.v),
        KernelArg::Buffer(kv.k), KernelArg::Buffer(kv.v),
    ])?;

    // attention_decode with seq_len = seq_len_attn (= cache_pos + 1)
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    backend.launch_typed(attn, [NUM_HEADS, 1, 1], [32, 1, 1], 0, stream, &[
        KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
        KernelArg::Bytes(&NUM_HEADS.to_le_bytes()),
        KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
        KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
        KernelArg::Bytes(&scale.to_le_bytes()),
        KernelArg::Buffer(q_view), KernelArg::Buffer(kv.k),
        KernelArg::Buffer(kv.v), KernelArg::Buffer(scratch.attn_out),
    ])?;

    // sigmoid_gate(attn_gate, attn_out)
    backend.launch_typed(sg, [Q_ONLY.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&Q_ONLY.to_le_bytes()),
        KernelArg::Buffer(gate_view), KernelArg::Buffer(scratch.attn_out), KernelArg::Buffer(scratch.gated_attn),
    ])?;

    // o_proj
    layer.o_proj.gemv(backend, scratch.gated_attn, scratch.o, stream)?;

    // residual
    backend.launch_typed(add, [HIDDEN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Buffer(x_in), KernelArg::Buffer(scratch.o), KernelArg::Buffer(scratch.x_resid),
    ])?;

    // norm2 → FFN → residual
    backend.launch_typed(rms, [1, 1, 1], [128, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
        KernelArg::Buffer(scratch.x_resid), KernelArg::Buffer(layer.post_ln), KernelArg::Buffer(scratch.x_norm2),
    ])?;
    layer.gate_proj.gemv(backend, scratch.x_norm2, scratch.gate_act, stream)?;
    layer.up_proj.gemv(backend, scratch.x_norm2, scratch.up_act, stream)?;
    backend.launch_typed(silu, [INTERMEDIATE.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&INTERMEDIATE.to_le_bytes()),
        KernelArg::Buffer(scratch.gate_act), KernelArg::Buffer(scratch.up_act), KernelArg::Buffer(scratch.ffn_act),
    ])?;
    layer.down_proj.gemv(backend, scratch.ffn_act, scratch.ffn_out, stream)?;
    backend.launch_typed(add, [HIDDEN.div_ceil(64), 1, 1], [64, 1, 1], 0, stream, &[
        KernelArg::Bytes(&HIDDEN.to_le_bytes()),
        KernelArg::Buffer(scratch.x_resid), KernelArg::Buffer(scratch.ffn_out), KernelArg::Buffer(scratch.x_out),
    ])?;
    Ok(scratch.x_out)
}

fn alloc_scratch(backend: &MetalGpuBackend) -> Result<Scratch> {
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    Ok(Scratch {
        x_norm: alloc_bf16(HIDDEN)?,
        q_full: alloc_bf16(Q_TOTAL)?,
        k: alloc_bf16(KV_DIM)?,
        v: alloc_bf16(KV_DIM)?,
        q_norm_out: alloc_bf16(Q_ONLY)?,
        k_norm_out: alloc_bf16(KV_DIM)?,
        attn_out: alloc_bf16(Q_ONLY)?,
        gated_attn: alloc_bf16(Q_ONLY)?,
        o: alloc_bf16(HIDDEN)?,
        x_resid: alloc_bf16(HIDDEN)?,
        x_norm2: alloc_bf16(HIDDEN)?,
        gate_act: alloc_bf16(INTERMEDIATE)?,
        up_act: alloc_bf16(INTERMEDIATE)?,
        ffn_act: alloc_bf16(INTERMEDIATE)?,
        ffn_out: alloc_bf16(HIDDEN)?,
        x_out: alloc_bf16(HIDDEN)?,
    })
}

fn main() -> Result<()> {
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "What is the capital of France?".to_string());
    let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("$HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });

    println!("=== Atlas Metal · Qwen3.5-4B-MLX-8bit inference ===");
    println!("model dir: {model_dir}");
    println!("prompt:    {prompt:?}");
    println!();
    println!(
        "⚠️  Note: linear_attention layers are identity passthrough. \
         The next-token prediction is informed only by the 8 \
         full_attention layers (3, 7, 11, 15, 19, 23, 27, 31)."
    );
    println!();

    // Tokenizer.
    let tok_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let token_strs: Vec<String> = encoding
        .get_tokens()
        .iter()
        .map(|s| s.to_string())
        .collect();
    println!("tokenized to {} tokens: {:?}", token_ids.len(), token_strs);
    if token_ids.is_empty() {
        bail!("empty token list — tokenizer produced nothing for the prompt");
    }
    let prompt_len = token_ids.len() as u32;

    // Layer types from config.json.
    let cfg_text =
        std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json"))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;
    let layer_types: Vec<String> = cfg["text_config"]["layer_types"]
        .as_array()
        .context("layer_types missing")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    if layer_types.len() as u32 != NUM_LAYERS {
        bail!(
            "expected {NUM_LAYERS} layers, got {} in layer_types",
            layer_types.len()
        );
    }
    let full_attn_count = layer_types.iter().filter(|s| s.as_str() == "full_attention").count();
    let lin_attn_count = layer_types.iter().filter(|s| s.as_str() == "linear_attention").count();
    println!(
        "layer types: {} full_attention + {} linear_attention",
        full_attn_count, lin_attn_count
    );

    // Backend.
    let modules = atlas_kernels::metallib_modules();
    if modules.is_empty() {
        bail!(
            "metal kernel registry empty — re-build with \
             ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
             ATLAS_TARGET_QUANT=mlx_int8"
        );
    }
    let backend = MetalGpuBackend::new(0, &modules)?;
    println!("metal backend ready, {} kernel modules", modules.len());

    // mmap the safetensors.
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    let file = std::fs::File::open(&st_path).context("open safetensors")?;
    let mmap = unsafe { memmap2::Mmap::map(&file).context("mmap")? };
    let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;

    // Load embed_tokens (used both for input embedding and tied LM head).
    println!("loading embed_tokens (vocab=248320, hidden=2560)...");
    let t0 = Instant::now();
    let embed_tokens = MlxInt8Weight::load(
        &backend,
        &st,
        "language_model.model.embed_tokens",
        GROUP_SIZE,
    )?;
    println!("  → embed_tokens loaded in {:.2?}", t0.elapsed());

    // Load final norm.
    let t = st.tensor("language_model.model.norm.weight").unwrap();
    let final_norm = backend.alloc(t.data().len())?;
    backend.copy_h2d(t.data(), final_norm)?;

    // Load all layers (8 full_attention + 24 linear_attention).
    println!("loading all 32 layers...");
    let t0 = Instant::now();
    let mut full_layers: Vec<Option<FullAttentionLayer>> = (0..NUM_LAYERS).map(|_| None).collect();
    let mut lin_layers: Vec<Option<LinearAttentionLayer>> = (0..NUM_LAYERS).map(|_| None).collect();
    for (idx, ty) in layer_types.iter().enumerate() {
        if ty == "full_attention" {
            full_layers[idx] = Some(FullAttentionLayer::load(&backend, &st, idx as u32)?);
        } else if ty == "linear_attention" {
            lin_layers[idx] = Some(LinearAttentionLayer::load(&backend, &st, idx as u32)?);
        }
    }
    println!("  → all weights loaded in {:.2?}", t0.elapsed());

    // Allocate scratch + KV caches (one cache per full_attention layer).
    // Capacity covers prompt + decode budget; bump via $ATLAS_DECODE_TOKENS.
    let n_decode_budget: u32 = std::env::var("ATLAS_DECODE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let max_seq_len = prompt_len + n_decode_budget + 4;
    let scratch = alloc_scratch(&backend)?;
    let lin_scratch = alloc_lin_scratch(&backend)?;
    let kv_caches: Vec<LayerKvCache> = (0..full_attn_count)
        .map(|_| -> Result<LayerKvCache> {
            Ok(LayerKvCache {
                k: backend.alloc((max_seq_len * KV_DIM) as usize * 2)?,
                v: backend.alloc((max_seq_len * KV_DIM) as usize * 2)?,
                capacity: max_seq_len,
            })
        })
        .collect::<Result<_>>()?;
    // Map layer_idx → kv_cache slot.
    let mut full_kv_slot: Vec<Option<usize>> = (0..NUM_LAYERS).map(|_| None).collect();
    {
        let mut next_slot = 0;
        for (idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "full_attention" {
                full_kv_slot[idx] = Some(next_slot);
                next_slot += 1;
            }
        }
    }
    // Per-linear-attention-layer SSM/conv state.
    let lin_states: Vec<LinearAttentionState> = (0..lin_attn_count)
        .map(|_| LinearAttentionState::alloc(&backend))
        .collect::<Result<_>>()?;
    let mut lin_state_slot: Vec<Option<usize>> = (0..NUM_LAYERS).map(|_| None).collect();
    {
        let mut next_slot = 0;
        for (idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "linear_attention" {
                lin_state_slot[idx] = Some(next_slot);
                next_slot += 1;
            }
        }
    }

    // Per-layer working buffer for the residual stream (one BF16[hidden]).
    let x_buf = backend.alloc(HIDDEN as usize * 2)?;
    // The output of forward_full_attention writes to scratch.x_out — we
    // d2d-copy back into x_buf at end of each layer to keep the
    // residual-stream pointer stable across layers.

    // RoPE inv_freq table (precomputed).
    let half_dim = HEAD_DIM / 2;
    let inv_freq_bytes: Vec<u8> = (0..half_dim)
        .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .flat_map(|f: f32| f.to_le_bytes())
        .collect();
    let inv_freq_ptr = backend.alloc(inv_freq_bytes.len())?;
    backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr)?;

    // positions_ptr is rewritten per token (current absolute position).
    let positions_ptr = backend.alloc(4)?;

    // Pre-resolve every kernel handle.
    let stream = backend.default_stream();
    let rms = backend.kernel("rms_norm", "rms_norm")?;
    let rope = backend.kernel("rope_apply", "rope_apply")?;
    let kvap = backend.kernel("kv_cache_append", "kv_cache_append")?;
    let attn = backend.kernel("attention_decode", "attention_decode")?;
    let sg = backend.kernel("sigmoid_gate", "sigmoid_gate")?;
    let add = backend.kernel("bf16_add", "bf16_add")?;
    let silu = backend.kernel("silu_gate", "silu_gate")?;
    let embed = backend.kernel("embed_lookup", "embed_lookup")?;
    // Reverted to plain causal_conv1d_decode for now — the
    // SiLU+L2-norm fused variant degrades output further. Revisit
    // once we have a parity test for the fused kernel.
    let conv1d = backend.kernel("causal_conv1d_decode", "causal_conv1d_decode")?;
    // The four GDN helpers all live in `gdn_helpers.metal` so the
    // metallib module name is shared.
    let gdn_gate = backend.kernel("gdn_helpers", "gdn_compute_gate")?;
    let sigmoid = backend.kernel("gdn_helpers", "sigmoid_bf16_to_f32")?;
    let silu_op = backend.kernel("gdn_helpers", "silu_apply")?;
    let mul = backend.kernel("gdn_helpers", "bf16_mul")?;
    let gdn_dec = backend.kernel("gated_delta_rule_decode", "gated_delta_rule_decode")?;

    // ── Embed-then-feed loop: process every prompt token through
    //    every layer, building KV cache. The hidden after the LAST
    //    prompt token is what we sample from.
    println!();
    println!("running prefill: {prompt_len} tokens × {NUM_LAYERS} layers");
    let t_total = Instant::now();
    for (tok_idx, &token_id) in token_ids.iter().enumerate() {
        // Embedding lookup for this token: write the embedding into x_buf.
        // Use embed_lookup kernel against the dequantized embed_tokens.
        // We don't have the full dequantized embed_tokens in BF16 — it's
        // 248320 * 2560 * 2 bytes ≈ 1.2 GB. Instead, dequantize ONLY the
        // single row we need by allocating a small BF16 scratch row and
        // calling mlx_int8_dequant on a single-row slice — no, the
        // dequant kernel needs the packed buffer to be aligned to its
        // expected layout. Easier path: call mlx_int8_gemv with a
        // one-hot input vector to extract the row.
        //
        // Simplest: dequantize the entire embed_tokens once, cache it in
        // BF16 GPU memory (1.2 GB — fits in UMA on Apple Silicon), then
        // use embed_lookup. That's what the next block does (lazily on
        // first access).
        if tok_idx == 0 {
            // Lazy: allocate + dequantize embed_tokens to BF16 once.
            // This is the LM head's working buffer too.
            // (Performed inline so we only pay the cost when we know we
            // need it — saves 1.2 GB if the user runs only one of the
            // earlier examples.)
            // Note: in production, the embed lookup would walk the MLX
            // packed bytes directly per-row (saves 1.2 GB); this version
            // trades memory for kernel-reuse simplicity.
        }

        // Per-token prefill step: write token_id, lookup, then layer chain.
        backend.copy_h2d(&token_id.to_le_bytes(), positions_ptr)?;
        // We need the embedding for `token_id` in x_buf. Use a single-
        // token embed_lookup. Since dequantizing all 1.2 GB of
        // embed_tokens up front is expensive, we materialize per-token
        // by running mlx_int8_dequant on the row's worth of packed
        // bytes (HIDDEN cols, group_size=64 → 40 groups). The kernel
        // expects packed[N, K/4] etc; we slice to N=1 and offset into
        // the source packed buffer by token_id rows.
        //
        // Even simpler given UMA: copy the row's BF16-equivalent by
        // dequantizing inline via a tiny per-call gemv with a one-hot
        // vector. For the demo, we just call the existing dequant kernel
        // on the WHOLE embed_tokens once, lazily, and then use
        // embed_lookup against that. Allocate the dequant buffer here:
        // GROUP_SIZE is 64; matches MLX's standard group size.
        const _: [(); 1] = [(); (GROUP_SIZE == 64) as usize];
        // FAST PATH for this demo: emit embedding via the embed_tokens
        // gemv with a one-hot vector of length VOCAB. That's
        // 248320-element matmul per token, dominated by memory bandwidth.

        // Build one-hot input vector [VOCAB] BF16 (CPU-side since we copy
        // it h2d each iter). The result is embed_tokens @ one_hot[token_id]
        // = the token_id-th row of dequantized embed_tokens.
        // But embed_tokens.gemv expects in_features = HIDDEN (2560); the
        // weight is [VOCAB, HIDDEN/4 packed] so out_features = VOCAB and
        // in_features = HIDDEN. So gemv(x[2560]) → y[VOCAB] is the LM-head
        // direction, NOT the embed direction.
        //
        // To EMBED a token: pick row token_id of dequantized embed_tokens,
        // = HIDDEN BF16 values. The kernel that does this is
        // embed_lookup, but it needs a fully-dequantized BF16 table.
        // For this demo, build that table once on first iteration.
        if tok_idx == 0 {
            // Lazy-init: allocate + run mlx_int8_dequant on embed_tokens
            // to produce a BF16 [VOCAB, HIDDEN] table.
            // 248320 * 2560 * 2 = 1.27 GB. Fits in M-series UMA budget.
            println!("  (lazy) dequantizing embed_tokens to BF16 table (1.27 GB)...");
            let t_dq = Instant::now();
            let embed_table_bytes = (VOCAB * HIDDEN) as usize * 2;
            let embed_table = backend.alloc(embed_table_bytes)?;
            embed_tokens.dequantize_to(&backend, embed_table, stream)?;
            backend.synchronize(stream)?;
            println!("  → dequantized in {:.2?}", t_dq.elapsed());
            // Stash the table pointer in a Box leaked into the closure
            // below — for a one-shot example this is fine.
            EMBED_TABLE.store(embed_table.0, std::sync::atomic::Ordering::SeqCst);
        }
        let embed_table = DevicePtr(EMBED_TABLE.load(std::sync::atomic::Ordering::SeqCst));

        // embed_lookup expects token_ids[num_tokens], embed_table[vocab, hidden],
        // out[num_tokens, hidden]. We do one token at a time.
        let token_id_bytes = token_id.to_le_bytes();
        let token_buf = backend.alloc(4)?;
        backend.copy_h2d(&token_id_bytes, token_buf)?;
        let n_tokens = 1u32;
        backend.launch_typed(embed, [HIDDEN.div_ceil(8), n_tokens, 1], [8, 1, 1], 0, stream, &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&HIDDEN.to_le_bytes()),
            KernelArg::Bytes(&VOCAB.to_le_bytes()),
            KernelArg::Buffer(token_buf), KernelArg::Buffer(embed_table), KernelArg::Buffer(x_buf),
        ])?;
        backend.free(token_buf)?;

        // Set the position for RoPE = absolute index in the sequence.
        let pos_u32 = tok_idx as u32;
        backend.copy_h2d(&pos_u32.to_le_bytes(), positions_ptr)?;

        // Layer chain.
        let mut x = x_buf;
        for (layer_idx, ty) in layer_types.iter().enumerate() {
            if ty == "full_attention" {
                let layer = full_layers[layer_idx]
                    .as_ref()
                    .expect("full_attn layer not loaded");
                let kv = &kv_caches[full_kv_slot[layer_idx].unwrap()];
                let cache_pos = tok_idx as u32;
                let seq_len_attn = (tok_idx + 1) as u32;
                let out = forward_full_attention(
                    &backend, layer, &scratch, kv, rms, rope, kvap, attn, sg, add, silu,
                    inv_freq_ptr, positions_ptr,
                    x, cache_pos, seq_len_attn, stream,
                )?;
                // Copy out → x_buf so the next layer's input is stable.
                backend.copy_d2d_async(out, x_buf, HIDDEN as usize * 2, stream)?;
                x = x_buf;
            } else {
                // linear_attention: real GDN orchestration.
                let layer = lin_layers[layer_idx]
                    .as_ref()
                    .expect("linear_attn layer not loaded");
                let state = &lin_states[lin_state_slot[layer_idx].unwrap()];
                let out = forward_linear_attention(
                    &backend, layer, state, &lin_scratch,
                    rms, conv1d, gdn_gate, sigmoid, silu_op, mul, gdn_dec, add,
                    x, x_buf, stream,
                )?;
                x = out;
            }
        }
        backend.synchronize(stream)?;
    }
    let prefill_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!("prefill complete in {prefill_ms:.1} ms ({:.1} ms/tok)", prefill_ms / prompt_len as f64);

    // Allocate sample-time buffers + kernels.
    let x_final = backend.alloc(HIDDEN as usize * 2)?;
    let logits = backend.alloc(VOCAB as usize * 2)?;
    let argmax = backend.kernel("argmax_bf16", "argmax_bf16")?;
    let result_buf = backend.alloc(4)?;

    // Helper: run final_norm + LM head + argmax → token id.
    let sample_next = |x_in: DevicePtr| -> Result<u32> {
        backend.launch_typed(rms, [1, 1, 1], [128, 1, 1], 0, stream, &[
            KernelArg::Bytes(&HIDDEN.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(x_in), KernelArg::Buffer(final_norm), KernelArg::Buffer(x_final),
        ])?;
        embed_tokens.gemv(&backend, x_final, logits, stream)?;
        backend.launch_typed(argmax, [1, 1, 1], [128, 1, 1], 0, stream, &[
            KernelArg::Bytes(&VOCAB.to_le_bytes()),
            KernelArg::Buffer(logits), KernelArg::Buffer(result_buf),
        ])?;
        backend.synchronize(stream)?;
        let mut buf = [0u8; 4];
        backend.copy_d2h(result_buf, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    };

    // First sample after prefill.
    let next_token_id = sample_next(x_buf)?;
    let next_text = tokenizer
        .decode(&[next_token_id], false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    println!();
    println!("=== After prefill, first generated token ===");
    println!("  token_id: {next_token_id}");
    println!("  text:     {next_text:?}");

    // Continue greedy decoding for N more tokens to see a full response.
    let n_decode: usize = std::env::var("ATLAS_DECODE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    println!();
    println!("running greedy decode for {n_decode} more tokens...");
    let t_dec = Instant::now();
    let mut generated_ids = vec![next_token_id];
    let mut current_token = next_token_id;
    let mut cur_pos = prompt_len;
    let embed_table = DevicePtr(EMBED_TABLE.load(std::sync::atomic::Ordering::SeqCst));

    for _ in 0..n_decode {
        // Reallocate KV caches if we're about to exceed capacity. For
        // simplicity in this demo we don't grow — limit decode tokens
        // to fit the pre-allocated max_seq_len.
        if cur_pos >= max_seq_len {
            println!("  (reached pre-allocated KV capacity {max_seq_len}, stopping)");
            break;
        }

        // Embed current token.
        let token_buf = backend.alloc(4)?;
        backend.copy_h2d(&current_token.to_le_bytes(), token_buf)?;
        let n_tokens = 1u32;
        backend.launch_typed(embed, [HIDDEN.div_ceil(8), n_tokens, 1], [8, 1, 1], 0, stream, &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&HIDDEN.to_le_bytes()),
            KernelArg::Bytes(&VOCAB.to_le_bytes()),
            KernelArg::Buffer(token_buf), KernelArg::Buffer(embed_table), KernelArg::Buffer(x_buf),
        ])?;
        backend.free(token_buf)?;

        // Position for RoPE.
        backend.copy_h2d(&cur_pos.to_le_bytes(), positions_ptr)?;

        // Layer chain.
        let mut x = x_buf;
        for (layer_idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "full_attention" {
                let layer = full_layers[layer_idx].as_ref().unwrap();
                let kv = &kv_caches[full_kv_slot[layer_idx].unwrap()];
                let cache_pos = cur_pos;
                let seq_len_attn = cur_pos + 1;
                let out = forward_full_attention(
                    &backend, layer, &scratch, kv, rms, rope, kvap, attn, sg, add, silu,
                    inv_freq_ptr, positions_ptr,
                    x, cache_pos, seq_len_attn, stream,
                )?;
                backend.copy_d2d_async(out, x_buf, HIDDEN as usize * 2, stream)?;
                x = x_buf;
            } else {
                let layer = lin_layers[layer_idx].as_ref().unwrap();
                let state = &lin_states[lin_state_slot[layer_idx].unwrap()];
                let out = forward_linear_attention(
                    &backend, layer, state, &lin_scratch,
                    rms, conv1d, gdn_gate, sigmoid, silu_op, mul, gdn_dec, add,
                    x, x_buf, stream,
                )?;
                x = out;
            }
        }
        backend.synchronize(stream)?;

        // Sample.
        current_token = sample_next(x_buf)?;
        generated_ids.push(current_token);
        cur_pos += 1;

        // Bail on EOS to avoid runaway generation.
        if current_token == 248044 {
            // <|im_end|> per tokenizer_config.json
            println!("  (hit <|im_end|>)");
            break;
        }
    }
    let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

    let full_text = tokenizer
        .decode(&generated_ids, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!();
    println!("=== Full generation ({} tokens, {dec_ms:.1} ms, {:.1} tok/s) ===",
             generated_ids.len(), generated_ids.len() as f64 / (dec_ms / 1000.0));
    println!("  ids: {generated_ids:?}");
    println!("  text: {full_text:?}");
    println!();
    println!(
        "All 32 layers fired (8 full_attention + 24 linear_attention via \
         GDN). The GDN orchestration is best-effort — the kernel-level \
         math (gated_delta_rule_decode) matches the CUDA reference \
         exactly but the surrounding pre/post wiring (qkv split, gate \
         clamping, residual placement) may diverge from the upstream \
         Python reference in subtle ways. Token-level parity vs \
         mlx_lm.generate is the next verification step."
    );

    Ok(())
}

// Stash for lazy embed_table allocation (one-shot demo simplification).
static EMBED_TABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
