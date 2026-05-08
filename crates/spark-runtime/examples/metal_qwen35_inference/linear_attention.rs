// SPDX-License-Identifier: AGPL-3.0-only
//! GDN (linear-attention) decoder layer (struct, weight load, state, decode forward).

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::{MlxInt8Weight, gemv_gate_up};

use super::dims::*;

pub(crate) struct LinearAttentionLayer {
    input_ln: DevicePtr,
    a_log: DevicePtr,         // F32 [num_state_heads]
    dt_bias: DevicePtr,       // BF16 [num_state_heads]
    conv1d_weight: DevicePtr, // BF16 [QKV_TOTAL_LIN, kernel_size]
    in_proj_a: MlxInt8Weight,
    in_proj_b: MlxInt8Weight,
    in_proj_qkv: MlxInt8Weight,
    in_proj_z: MlxInt8Weight,
    norm_weight: DevicePtr, // BF16 [V_HEAD_DIM_LIN]
    out_proj: MlxInt8Weight,
    // Post-attention MLP — Qwen3.5 decoder layer applies it for both
    // GDN and full-attention layers; missing this here was the root
    // cause of L00→L01 residual divergence.
    post_ln: DevicePtr,
    gate_proj: MlxInt8Weight,
    up_proj: MlxInt8Weight,
    down_proj: MlxInt8Weight,
}

impl LinearAttentionLayer {
    pub(crate) fn load(
        backend: &MetalGpuBackend,
        st: &SafeTensors,
        layer_idx: u32,
    ) -> Result<Self> {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let load_raw = |name: &str| -> Result<DevicePtr> {
            let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
            let p = backend.alloc(t.data().len())?;
            backend.copy_h2d(t.data(), p)?;
            Ok(p)
        };
        Ok(Self {
            input_ln: load_raw(&format!("{prefix}.input_layernorm.weight"))?,
            a_log: load_raw(&format!("{prefix}.linear_attn.A_log"))?,
            dt_bias: load_raw(&format!("{prefix}.linear_attn.dt_bias"))?,
            conv1d_weight: load_raw(&format!("{prefix}.linear_attn.conv1d.weight"))?,
            in_proj_a: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.linear_attn.in_proj_a"),
                GROUP_SIZE,
            )?,
            in_proj_b: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.linear_attn.in_proj_b"),
                GROUP_SIZE,
            )?,
            in_proj_qkv: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.linear_attn.in_proj_qkv"),
                GROUP_SIZE,
            )?,
            in_proj_z: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.linear_attn.in_proj_z"),
                GROUP_SIZE,
            )?,
            norm_weight: load_raw(&format!("{prefix}.linear_attn.norm.weight"))?,
            out_proj: MlxInt8Weight::load(
                backend,
                st,
                &format!("{prefix}.linear_attn.out_proj"),
                GROUP_SIZE,
            )?,
            post_ln: load_raw(&format!("{prefix}.post_attention_layernorm.weight"))?,
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

/// Per-layer SSM/conv state for a linear-attention layer.

pub(crate) struct LinearAttentionState {
    /// FP32 [QKV_TOTAL_LIN, d_conv]. Persists across tokens. The
    /// `causal_conv1d_update_l2norm` kernel matches the CUDA
    /// reference and uses FP32 state — prevents recurrent precision
    /// drift past 8K tokens that BF16 truncation introduces.
    conv1d_state: DevicePtr,
    /// FP32 [batch=1, num_v_heads, k_dim, v_dim]. Persists across tokens.
    gdn_state: DevicePtr,
}

impl LinearAttentionState {
    pub(crate) fn alloc(backend: &MetalGpuBackend) -> Result<Self> {
        // FP32 state sized for full d_conv (kernel writes new_input
        // into state[d_conv-1] after shifting; conv reads all d_conv).
        let conv_state_bytes = (QKV_TOTAL_LIN * CONV_KERNEL_SIZE) as usize * 4;
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
pub(crate) struct LinScratch {
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
    out: DevicePtr,        // BF16 [HIDDEN]
    x_resid: DevicePtr,    // BF16 [HIDDEN] = x_in + GDN_out
    // MLP scratch (post-attention).
    x_norm2: DevicePtr,  // BF16 [HIDDEN]
    gate_act: DevicePtr, // BF16 [INTERMEDIATE]
    up_act: DevicePtr,   // BF16 [INTERMEDIATE]
    x_final: DevicePtr,  // BF16 [HIDDEN] = x_resid + down_proj@(silu(g)*u)
}

pub(crate) fn alloc_lin_scratch(backend: &MetalGpuBackend) -> Result<LinScratch> {
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
        out: alloc_bf16(HIDDEN)?,
        x_resid: alloc_bf16(HIDDEN)?,
        x_norm2: alloc_bf16(HIDDEN)?,
        gate_act: alloc_bf16(INTERMEDIATE)?,
        up_act: alloc_bf16(INTERMEDIATE)?,
        x_final: alloc_bf16(HIDDEN)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_linear_attention(
    backend: &MetalGpuBackend,
    layer: &LinearAttentionLayer,
    state: &LinearAttentionState,
    scratch: &LinScratch,
    rms: spark_runtime::gpu::KernelHandle,
    conv1d: spark_runtime::gpu::KernelHandle,
    gdn_gate: spark_runtime::gpu::KernelHandle,
    sigmoid: spark_runtime::gpu::KernelHandle,
    _silu_op: spark_runtime::gpu::KernelHandle,
    _silu_swiglu: spark_runtime::gpu::KernelHandle,
    _mul: spark_runtime::gpu::KernelHandle,
    gdn_dec: spark_runtime::gpu::KernelHandle,
    _add: spark_runtime::gpu::KernelHandle,
    add_rms: spark_runtime::gpu::KernelHandle,
    x_in: DevicePtr,
    x_buf: DevicePtr,
    stream: u64,
    intra_dump: Option<&dyn Fn(&str, DevicePtr, u32) -> Result<()>>,
) -> Result<DevicePtr> {
    // 1. norm
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
    // 2. projections — fused in_proj_a/in_proj_b share `x_norm` (both
    // produce per-head [num_v_heads=32] vectors; same shape).
    gemv_gate_up(
        backend,
        &layer.in_proj_a,
        &layer.in_proj_b,
        scratch.x_norm,
        scratch.dt_raw,
        scratch.b_raw,
        stream,
    )?;
    layer
        .in_proj_qkv
        .gemv(backend, scratch.x_norm, scratch.qkv, stream)?;
    layer
        .in_proj_z
        .gemv(backend, scratch.x_norm, scratch.z, stream)?;

    // 3. fused causal_conv1d_update_l2norm: conv + SiLU + per-head
    // L2-norm on Q+K, SiLU only on V. Matches the CUDA reference.
    let batch_one: u32 = 1;
    let block_x: u32 = K_HEAD_DIM_LIN; // 128 — one head per block
    let blocks_per_batch = QKV_TOTAL_LIN.div_ceil(block_x);
    let qk_channels: u32 = 2 * NUM_K_HEADS_LIN * K_HEAD_DIM_LIN; // 4096
    let l2_eps: f32 = 1e-6;
    backend.launch_typed(
        conv1d,
        [blocks_per_batch * batch_one, 1, 1],
        [block_x, 1, 1],
        0,
        stream,
        &[
            KernelArg::Buffer(state.conv1d_state),
            KernelArg::Buffer(scratch.qkv),
            KernelArg::Buffer(layer.conv1d_weight),
            KernelArg::Buffer(scratch.qkv_smooth),
            KernelArg::Bytes(&batch_one.to_le_bytes()),
            KernelArg::Bytes(&QKV_TOTAL_LIN.to_le_bytes()),
            KernelArg::Bytes(&CONV_KERNEL_SIZE.to_le_bytes()),
            KernelArg::Bytes(&qk_channels.to_le_bytes()),
            KernelArg::Bytes(&K_HEAD_DIM_LIN.to_le_bytes()),
            KernelArg::Bytes(&l2_eps.to_le_bytes()),
        ],
    )?;
    // NOTE: no Q post-scaling step is required. MLX scales q by
    // (1/d) and k by (1/sqrt(d)) *after* its rms_norm; its GDN kernel
    // then produces output without scaling. Atlas takes a
    // mathematically equivalent path: the fused conv1d kernel above
    // produces unit-L2-per-head Q+K, the `gated_delta_rule_decode`
    // kernel applies the output `1/sqrt(d)` factor, and the
    // cumulative algebra collapses to the same y.

    // 4. gate = exp(softplus(dt + dt_bias) * -exp(A_log))
    backend.launch_typed(
        gdn_gate,
        [NUM_STATE_HEADS.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&NUM_STATE_HEADS.to_le_bytes()),
            KernelArg::Buffer(scratch.dt_raw),
            KernelArg::Buffer(layer.dt_bias),
            KernelArg::Buffer(layer.a_log),
            KernelArg::Buffer(scratch.gate),
        ],
    )?;
    // 5. beta = sigmoid(b_raw) → FP32
    backend.launch_typed(
        sigmoid,
        [NUM_STATE_HEADS.div_ceil(32), 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&NUM_STATE_HEADS.to_le_bytes()),
            KernelArg::Buffer(scratch.b_raw),
            KernelArg::Buffer(scratch.beta),
        ],
    )?;

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
    backend.launch_typed(
        gdn_dec,
        [total_groups, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
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
        ],
    )?;

    // 8. per-head rms_norm at head_dim=128 over Z_DIM_LIN/V_HEAD_DIM_LIN = 32 tokens
    backend.launch_typed(
        rms,
        [NUM_V_HEADS_LIN, 1, 1],
        [128, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&V_HEAD_DIM_LIN.to_le_bytes()),
            KernelArg::Bytes(&RMS_EPS.to_le_bytes()),
            KernelArg::Buffer(scratch.y),
            KernelArg::Buffer(layer.norm_weight),
            KernelArg::Buffer(scratch.y_norm),
        ],
    )?;

    // 9+10+11. Fused: out = out_proj @ (silu(z) ⊙ y_norm).
    // Replaces silu(z) → mul(z_silu, y_norm) → gemv(out_proj) — three
    // launches collapse to one, eliminates two Z_DIM_LIN-sized
    // staging buffers (z_silu, y_gated).
    layer
        .out_proj
        .gemv_silu_gate(backend, scratch.z, scratch.y_norm, scratch.out, stream)?;

    // 12+13. Fused residual + post-attention RMSNorm.
    //   h   = x + GDN(input_layernorm(x))     -- scratch.x_resid (intermediate)
    //   x_norm2 = rms_norm(h, post_ln, eps)   -- input to MLP
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
            KernelArg::Buffer(scratch.out),
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
    // Fused: x_final = x_resid + down_proj @ (silu(gate_act) ⊙ up_act).
    // Replaces silu_gate → down_proj.gemv → bf16_add (3 launches +
    // 2 HIDDEN/INTERMEDIATE staging buffers) with a single kernel.
    layer.down_proj.gemv_silu_gate_resid(
        backend,
        scratch.gate_act,
        scratch.up_act,
        scratch.x_resid,
        scratch.x_final,
        stream,
    )?;

    // Intra-layer dumps (gated externally via Option). Sync first so
    // the d→h reads see the kernel results.
    if let Some(dump) = intra_dump {
        backend.synchronize(stream)?;
        dump("gdn_x_norm", scratch.x_norm, HIDDEN)?;
        dump("gdn_qkv_pre", scratch.qkv, QKV_TOTAL_LIN)?;
        dump("gdn_qkv_smooth", scratch.qkv_smooth, QKV_TOTAL_LIN)?;
        dump("gdn_y", scratch.y, Z_DIM_LIN)?;
        dump("gdn_y_norm", scratch.y_norm, Z_DIM_LIN)?;
        dump("gdn_out", scratch.out, HIDDEN)?;
        dump("gdn_x_resid", scratch.x_resid, HIDDEN)?;
        dump("gdn_x_final", scratch.x_final, HIDDEN)?;
    }

    // Copy x_final (post-MLP-residual) to caller's stable buffer.
    backend.copy_d2d_async(scratch.x_final, x_buf, HIDDEN as usize * 2, stream)?;
    Ok(x_buf)
}
