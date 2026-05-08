// SPDX-License-Identifier: AGPL-3.0-only
//! Full-attention decoder layer (struct, weight load, decode forward).

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::{MlxInt8Weight, gemv_gate_up};

use super::dims::*;

pub(crate) struct LayerKvCache {
    pub(crate) k: DevicePtr,
    pub(crate) v: DevicePtr,
    /// Capacity in tokens — caller pre-allocates `max_seq_len * KV_DIM`.
    #[allow(dead_code)]
    pub(crate) capacity: u32,
}

pub(crate) struct FullAttentionLayer {
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
    pub(crate) fn load(
        backend: &MetalGpuBackend,
        st: &SafeTensors,
        layer_idx: u32,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_bf16 = |name: &str| -> Result<DevicePtr> {
            let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        Ok(Self {
            input_ln: load_bf16(&format!("{prefix}.input_layernorm.weight"))?,
            q_norm: load_bf16(&format!("{prefix}.self_attn.q_norm.weight"))?,
            k_norm: load_bf16(&format!("{prefix}.self_attn.k_norm.weight"))?,
            post_ln: load_bf16(&format!("{prefix}.post_attention_layernorm.weight"))?,
            q_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.self_attn.q_proj"),
                GROUP_SIZE,
            )?,
            k_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.self_attn.k_proj"),
                GROUP_SIZE,
            )?,
            v_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.self_attn.v_proj"),
                GROUP_SIZE,
            )?,
            o_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.self_attn.o_proj"),
                GROUP_SIZE,
            )?,
            gate_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.mlp.gate_proj"),
                GROUP_SIZE,
            )?,
            up_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.mlp.up_proj"),
                GROUP_SIZE,
            )?,
            down_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.mlp.down_proj"),
                GROUP_SIZE,
            )?,
        })
    }
}

/// Per-layer scratch buffers reused across forward passes.
pub(crate) struct Scratch {
    x_norm: DevicePtr,
    q_full: DevicePtr,
    q_split: DevicePtr,    // [num_heads, head_dim] after deinterleave
    gate_split: DevicePtr, // [num_heads, head_dim] after deinterleave
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
    x_out: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_full_attention(
    backend: &MetalGpuBackend,
    layer: &FullAttentionLayer,
    scratch: &Scratch,
    kv: &LayerKvCache,
    rms: spark_runtime::gpu::KernelHandle,
    rope: spark_runtime::gpu::KernelHandle,
    kvap: spark_runtime::gpu::KernelHandle,
    attn: spark_runtime::gpu::KernelHandle,
    sg: spark_runtime::gpu::KernelHandle,
    _add: spark_runtime::gpu::KernelHandle,
    add_rms: spark_runtime::gpu::KernelHandle,
    _silu: spark_runtime::gpu::KernelHandle,
    qkv_split: spark_runtime::gpu::KernelHandle,
    inv_freq_ptr: DevicePtr,
    positions_ptr: DevicePtr,
    x_in: DevicePtr,
    cache_pos: u32,
    seq_len_attn: u32,
    stream: u64,
) -> Result<DevicePtr> {
    // norm1
    backend.launch_typed(
        rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&HIDDEN.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(layer.input_ln),
            KernelArg::Buffer(scratch.x_norm),
        ],
    )?;
    layer
        .q_proj
        .gemv(backend, scratch.x_norm, scratch.q_full, stream)?;
    // Fused k_proj + v_proj — both share x_norm and have identical
    // (N=KV_DIM, K=HIDDEN, group_size) shapes for Qwen3.5.
    gemv_gate_up(
        backend,
        &layer.k_proj,
        &layer.v_proj,
        scratch.x_norm,
        scratch.k,
        scratch.v,
        stream,
    )?;

    // Qwen3.5 q_proj output is [num_heads, head_dim * 2] interleaved
    // per head as [Q_h | gate_h]. Deinterleave into separate buffers
    // before normalization / RoPE / attention.
    backend.launch_typed(
        qkv_split,
        [HEAD_DIM, NUM_HEADS, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&NUM_HEADS.to_le_bytes()),
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Buffer(scratch.q_full),
            KernelArg::Buffer(scratch.q_split),
            KernelArg::Buffer(scratch.gate_split),
        ],
    )?;
    let q_view = scratch.q_split;
    let gate_view = scratch.gate_split;

    // per-head q/k norm (treat each head as a token)
    backend.launch_typed(
        rms,
        [NUM_HEADS, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(q_view),
            KernelArg::Buffer(layer.q_norm),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    backend.launch_typed(
        rms,
        [NUM_KV_HEADS, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(scratch.k),
            KernelArg::Buffer(layer.k_norm),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;
    // RoPE on the q_norm_out / k_norm_out buffers directly. Eliminates
    // the previous d2d copy back into q_view / scratch.k by routing
    // the post-norm tensors straight into RoPE → KV-append → attention.
    let half_dim = ROTARY_DIM / 2;
    let n_tokens = 1u32;
    backend.launch_typed(
        rope,
        [half_dim, NUM_HEADS, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&NUM_HEADS.to_le_bytes()),
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&ROTARY_DIM.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.q_norm_out),
        ],
    )?;
    backend.launch_typed(
        rope,
        [half_dim, NUM_KV_HEADS, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&ROTARY_DIM.to_le_bytes()),
            KernelArg::Buffer(positions_ptr),
            KernelArg::Buffer(inv_freq_ptr),
            KernelArg::Buffer(scratch.k_norm_out),
        ],
    )?;

    // KV cache append uses the post-RoPE k_norm_out (k still holds
    // the pre-norm projection — irrelevant for the cache).
    backend.launch_typed(
        kvap,
        [HEAD_DIM, NUM_KV_HEADS, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&cache_pos.to_le_bytes()),
            KernelArg::Buffer(scratch.k_norm_out),
            KernelArg::Buffer(scratch.v),
            KernelArg::Buffer(kv.k),
            KernelArg::Buffer(kv.v),
        ],
    )?;

    // attention_decode with seq_len = seq_len_attn (= cache_pos + 1).
    // Q comes from the post-norm/post-RoPE buffer; gate_view still
    // aliases the second half of q_full (untouched, holds the raw
    // attn-output gate produced by q_proj).
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    backend.launch_typed(
        attn,
        [NUM_HEADS, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&seq_len_attn.to_le_bytes()),
            KernelArg::Bytes(&NUM_HEADS.to_le_bytes()),
            KernelArg::Bytes(&NUM_KV_HEADS.to_le_bytes()),
            KernelArg::Bytes(&HEAD_DIM.to_le_bytes()),
            KernelArg::Bytes(&scale.to_le_bytes()),
            KernelArg::Buffer(scratch.q_norm_out),
            KernelArg::Buffer(kv.k),
            KernelArg::Buffer(kv.v),
            KernelArg::Buffer(scratch.attn_out),
        ],
    )?;
    let _ = q_view;

    // sigmoid_gate(attn_gate, attn_out)
    backend.launch_typed(
        sg,
        [Q_ONLY.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&Q_ONLY.to_le_bytes()),
            KernelArg::Buffer(gate_view),
            KernelArg::Buffer(scratch.attn_out),
            KernelArg::Buffer(scratch.gated_attn),
        ],
    )?;

    // o_proj
    layer
        .o_proj
        .gemv(backend, scratch.gated_attn, scratch.o, stream)?;

    // Fused residual + post-attention RMSNorm.
    backend.launch_typed(
        add_rms,
        [1, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&HIDDEN.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(x_in),
            KernelArg::Buffer(scratch.o),
            KernelArg::Buffer(layer.post_ln),
            KernelArg::Buffer(scratch.x_resid),
            KernelArg::Buffer(scratch.x_norm2),
        ],
    )?;
    // Fused dual-output GEMV: shares x_norm2 across gate_proj and up_proj.
    gemv_gate_up(
        backend,
        &layer.gate_proj,
        &layer.up_proj,
        scratch.x_norm2,
        scratch.gate_act,
        scratch.up_act,
        stream,
    )?;
    // Fused: x_out = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    layer.down_proj.gemv_silu_gate_resid(
        backend,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_out,
        stream,
    )?;
    Ok(scratch.x_out)
}

pub(crate) fn alloc_scratch(backend: &MetalGpuBackend) -> Result<Scratch> {
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    Ok(Scratch {
        x_norm: alloc_bf16(HIDDEN)?,
        q_full: alloc_bf16(Q_TOTAL)?,
        q_split: alloc_bf16(Q_ONLY)?,
        gate_split: alloc_bf16(Q_ONLY)?,
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
        x_out: alloc_bf16(HIDDEN)?,
    })
}
