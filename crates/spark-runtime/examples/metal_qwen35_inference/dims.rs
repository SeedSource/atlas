// SPDX-License-Identifier: AGPL-3.0-only
//! Compile-time dimensions for Qwen3.5-4B-MLX-8bit.

// ── Dims (Qwen3.5-4B from upstream config.json `text_config`) ───
pub(crate) const HIDDEN: u32 = 2560;
pub(crate) const NUM_HEADS: u32 = 16;
pub(crate) const NUM_KV_HEADS: u32 = 4;
pub(crate) const HEAD_DIM: u32 = 256;
pub(crate) const INTERMEDIATE: u32 = 9216;
pub(crate) const NUM_LAYERS: u32 = 32;
pub(crate) const RMS_EPS: f32 = 1e-6;
pub(crate) const GROUP_SIZE: u32 = 64;
pub(crate) const ROPE_THETA: f32 = 10_000_000.0;
// Qwen3.5-VL `partial_rotary_factor = 0.25` → only the first 64 of
// 256 head_dim elements are rotated. The remaining 192 pass through.
pub(crate) const ROTARY_DIM: u32 = HEAD_DIM / 4;
pub(crate) const VOCAB: u32 = 248_320;
pub(crate) const Q_TOTAL: u32 = NUM_HEADS * HEAD_DIM * 2; // attn_output_gate
pub(crate) const Q_ONLY: u32 = NUM_HEADS * HEAD_DIM;
pub(crate) const KV_DIM: u32 = NUM_KV_HEADS * HEAD_DIM;

// ── Linear-attention (GDN) dims ────────────────────────────────
pub(crate) const NUM_K_HEADS_LIN: u32 = 16;
pub(crate) const NUM_V_HEADS_LIN: u32 = 32;
pub(crate) const K_HEAD_DIM_LIN: u32 = 128;
pub(crate) const V_HEAD_DIM_LIN: u32 = 128;
pub(crate) const QKV_TOTAL_LIN: u32 = 8192; // = K_HEADS*K_DIM + K_HEADS*K_DIM + V_HEADS*V_DIM = 2048+2048+4096
pub(crate) const Z_DIM_LIN: u32 = 4096; // = NUM_V_HEADS_LIN * V_HEAD_DIM_LIN
pub(crate) const NUM_STATE_HEADS: u32 = 32;
pub(crate) const CONV_KERNEL_SIZE: u32 = 4;
