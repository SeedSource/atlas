// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash **native DSpark** draft proposer (K=1).
//!
//! Implements [`DraftProposer`] over the [`DeepseekV4DSparkModule`] loaded by
//! [`load_v4_dspark_module`]. Distinct from the NVIDIA-style single-module
//! [`crate::layers::DeepseekV4MtpHead`]: the native DSpark drafter is a
//! **three-stage** semi-autoregressive block (each stage a full reused V4 layer
//! — MLA + manifold-constrained hyper-connections (mHC) + 256-expert MXFP4 MoE)
//! topped by the DSpark heads (`main_proj`/`main_norm` block-input projection,
//! final norm, serial Markov token head, confidence head).
//!
//! ## Commit 4 (this file) — sliding-window main-KV + block-input projection
//!
//! Commit 3 landed the lifecycle scaffold with a shared [`PagedKvCache`] and an
//! empty `propose()`. Commit 4 makes the two structural corrections the Gate-3
//! audit named, and advances `propose()` as far as Atlas's existing kernels
//! allow:
//!
//! 1. **Sliding-window `main_kv_cache` ring buffer** (Gate-3 "biggest
//!    correction"). The scaffold's paged KV is replaced by Mia's per-stage
//!    sliding-window ring buffer ([`DsparkMainKvCache`], `dspark.py:282-292`):
//!    `[max_seqs, window, head_dim]`, absolute main-token position `p` → ring
//!    slot `p % window` (`dspark.py:421`), `valid_len = seg_len - rejected`
//!    catch-up on reject (`dspark.py:385/442`). The ring index contract is pure
//!    host math ([`ring_slot`] / [`ring_valid_len`]) and CPU-unit-tested; the
//!    device buffers are process-lifetime on the head (one per stage), the
//!    per-sequence write position lives on the state.
//!
//! 2. **`propose()` = Mia `draft()` call order** (`dspark.py:868-995`). The
//!    block-input projection [`DeepseekV4DSparkHead::project_main`] (Mia
//!    `project_main`, `dspark.py:708-711`: `main_proj` then `main_norm` over the
//!    `[40,41,42]` target-hidden stack) is implemented with the existing
//!    `dense_gemv` + `rms_norm` kernels and exercised by `propose()` — it is the
//!    `main_proj_out` / `main_norm_out` golden boundaries. The remaining
//!    semi-AR block forward (noise block → embed → hc_expand → 3× sparse-MLA
//!    stage `forward_dspark` → hc_head → base logits → serial-Markov argmax loop
//!    → confidence → emit pos-0) is a **net-new drafter kernel surface** — see
//!    the STOP note in `propose()` — and does not reuse the target's per-token
//!    MLA decode. It converges on GPU once those kernels land in the image
//!    build.
//!
//! Externally the proposer exposes only **K=1** (`num_drafts > 1` is warned and
//! ignored); the internal semi-AR block width (`dspark_block_size = 5`) is an
//! implementation detail.

use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::deepseek_v4::DeepseekV4DSparkModule;
use crate::weight_map::DenseWeight;

/// Ring slot for an absolute main-token position: Mia's `positions %
/// window_size` (`dspark.py:421`). Pure host math; unit-tested.
#[inline]
fn ring_slot(position: usize, window: usize) -> usize {
    debug_assert!(window > 0, "ring window must be non-zero");
    position % window
}

/// Number of leading rows of a stored segment that are *valid* (committed) when
/// `rejected` suffix tokens are trimmed: Mia's `valid_len = (seg_len -
/// rejected).clamp(min=1, max=seg_len)` (`dspark.py:385` / `:442`). Rows at
/// offsets `>= valid_len` keep the ring's previous values (the rejected drafts
/// never enter the sliding window). Pure host math; unit-tested.
#[inline]
#[allow(dead_code)] // encodes the reject contract the net-new store path applies (GPU lane).
fn ring_valid_len(seg_len: usize, rejected: usize) -> usize {
    if seg_len == 0 {
        return 0;
    }
    seg_len.saturating_sub(rejected).clamp(1, seg_len)
}

/// One drafter stage's sliding-window `main_kv_cache` ring buffer — Mia
/// `DeepSeekV4DSparkAttention.main_kv_cache` (`dspark.py:282-292`). Logical
/// shape `[max_seqs, window, head_dim]`, BF16, zero-initialized (`torch.zeros`).
///
/// The MLA-absorbed drafter KV row is `head_dim = kv_lora_rank +
/// qk_rope_head_dim` wide, matching the target's MLA cache row so the stage
/// projection lands at the right stride. For the K=1 single-stream serve
/// profile (`--max-num-seqs 1`) `max_seqs = 1`.
///
/// The device buffer is process-lifetime (owned by the head, shared across the
/// single stream); the per-sequence write position lives on the proposer state.
/// A stage attention feeds the ring via `store_main_kv` (the projected main KV,
/// scattered at `position % window`) — that projection is part of the net-new
/// drafter stage forward and is wired in the GPU-convergence lane.
#[allow(dead_code)] // `max_seqs`/`row_offset` consumed by the net-new stage store (GPU lane).
struct DsparkMainKvCache {
    /// `[max_seqs * window * head_dim]` BF16 device storage.
    buf: DevicePtr,
    /// Sliding-window depth (`dspark_window_size`, e.g. 128).
    window: usize,
    /// MLA-absorbed KV row width (`kv_lora_rank + qk_rope_head_dim`).
    head_dim: usize,
    /// Concurrent-sequence rows (1 for the K=1 single-stream profile).
    max_seqs: usize,
    /// Total byte length of `buf` (BF16).
    bytes: usize,
}

impl DsparkMainKvCache {
    /// Allocate + zero one stage's ring buffer.
    fn new(gpu: &dyn GpuBackend, window: usize, head_dim: usize, max_seqs: usize) -> Result<Self> {
        let bytes = max_seqs * window * head_dim * 2; // BF16
        let buf = gpu.alloc(bytes)?;
        gpu.memset(buf, 0, bytes)?; // torch.zeros init
        Ok(Self {
            buf,
            window,
            head_dim,
            max_seqs,
            bytes,
        })
    }

    /// Byte offset of `(seq_row, slot)` in the flat `[max_seqs, window,
    /// head_dim]` BF16 buffer.
    #[allow(dead_code)] // consumed by the net-new stage `store_main_kv` (GPU lane).
    fn row_offset(&self, seq_row: usize, slot: usize) -> usize {
        ((seq_row * self.window) + slot) * self.head_dim * 2
    }

    /// Zero the whole ring (clean sequence boundary for the single-stream
    /// profile — Mia keeps the buffer persistent and resets via positions; for
    /// one stream, a zero + position reset is an exact superset).
    fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.memset(self.buf, 0, self.bytes)?;
        Ok(())
    }
}

/// Per-sequence state for the DeepSeek-V4 native DSpark proposer.
#[allow(dead_code)]
pub struct DeepseekV4DSparkProposerState {
    /// Absolute number of committed main-KV tokens for this sequence. The ring
    /// slot of the next write is `main_kv_pos % window` ([`ring_slot`]); it
    /// advances by the accepted-token count each `after_verify` (rejected
    /// drafts never enter the window — [`ring_valid_len`]). Replaces the
    /// scaffold's per-stage paged block tables.
    pub main_kv_pos: usize,
    /// Drafts produced by the last `propose()` (for `after_verify` reject
    /// accounting).
    pub last_num_drafted: usize,
    /// Per-stage state for the reused V4 bodies (MLA layers use
    /// `EmptyLayerState`, but we allocate via each stage's own `alloc_state` so
    /// any future stateful body type is handled correctly).
    pub body_states: Vec<Box<dyn LayerState>>,
}

impl ProposerState for DeepseekV4DSparkProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// DeepSeek-V4 native DSpark draft proposer (K=1).
#[allow(dead_code)]
pub struct DeepseekV4DSparkHead {
    /// The loaded native DSpark module: 3 reused V4 stage bodies + DSpark heads
    /// (`main_proj`/`main_norm`, final norm, serial Markov head, confidence
    /// head, last-stage `hc_head`).
    module: DeepseekV4DSparkModule,
    /// Shared token embedding table (BF16), from the target model.
    embed_tokens: DenseWeight,
    /// Shared LM head (BF16), from the target model. Every draft is re-verified
    /// by the target's head, so the draft head only affects acceptance.
    lm_head: DenseWeight,
    /// Reduced vocab size for the draft LM-head GEMV (0 = full vocab).
    mtp_vocab_size: u32,
    /// Number of draft stages (`n_mtp_layers`, = `module.stages.len()`).
    num_stages: usize,
    /// Semi-AR internal block width (`dspark_block_size`, = 5). The drafter
    /// always computes the full block; K=1 emits position 0 only.
    block_size: usize,
    /// Noise/placeholder token id (`dspark_noise_token_id`, = 128799) filling
    /// draft-block positions `1..block_size`.
    noise_token_id: u32,
    /// Per-stage sliding-window `main_kv_cache` ring buffers (one per stage,
    /// Mia registers `main_kv_cache` per `DeepSeekV4DSparkAttention`). Shared,
    /// process-lifetime; the per-sequence write position is on the state.
    /// Empty when `dspark_window_size == 0` (mis-config guard).
    main_kv_caches: Vec<DsparkMainKvCache>,
    /// Fix B: full (un-sharded) drafter head count `nq` and O-projection group
    /// count `o_groups`. The drafter runs rank-0-complete (its MLA weights are
    /// loaded un-sharded via `drafter_stage_config`), so the stage forward uses
    /// the FULL counts (`local * tp`, = 64 / 8 at TP=2) — NOT the TP-local
    /// `config.num_attention_heads` / `config.o_groups` the target model uses —
    /// and issues no wo_b all-reduce.
    drafter_nq: u32,
    drafter_o_groups: u32,

    // Kernel handles (mirrors `DeepseekV4MtpHead`).
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
    // ── Net-new DSpark stage-forward kernel handles ──
    /// mHC middle-mixing collapse (`hc_pre`) — attn + ffn residual sites.
    hc_pre_k: KernelHandle,
    /// mHC middle-mixing expand (`hc_post`) — attn + ffn residual sites.
    hc_post_k: KernelHandle,
    /// Net-new sparse windowed-MLA attention (`dspark_sparse_attention`).
    #[allow(dead_code)] // consumed by the stage-attn helper (Increment B, same lane)
    dspark_attn_k: KernelHandle,
    /// Forward interleaved YaRN rope (q/kv) — reused V4 rope kernel.
    #[allow(dead_code)] // consumed by the stage-attn helper (Increment B, same lane)
    rope_fwd_k: KernelHandle,
    /// Inverse (conjugate) interleaved YaRN rope (attn out de-rotate).
    rope_inv_k: KernelHandle,
    /// Extract the trailing `rope` lanes of an MLA `[*, heads, head_dim]` tensor
    /// into a contiguous `[*, heads, rope]` buffer (for the rope kernels).
    mla_q_rope_extract_batched_k: KernelHandle,
    /// Write roped `[*, heads, rope]` lanes back into the full `[*, heads, head_dim]`.
    mla_q_rope_writeback_batched_k: KernelHandle,
    /// `[head_dim]` BF16 ones — non-affine per-head Q RMSNorm (`q*rsqrt(mean(q²)+eps)`).
    q_headnorm_ones: DenseWeight,

    /// Monotonic `propose()` call index, appended to boundary-dump filenames so
    /// successive calls do not overwrite (used only when `ATLAS_DSPARK_DUMP_DIR`
    /// is armed; lets the offline harness content-match the golden step).
    dump_call: std::sync::atomic::AtomicUsize,
    /// Chain confidence (position-0 sigmoid) of the most recent `propose`, f32
    /// bits. NaN sentinel = not yet computed (`last_confidence()` → None).
    last_conf: std::sync::atomic::AtomicU32,
}

impl DeepseekV4DSparkHead {
    /// Build the proposer from a loaded [`DeepseekV4DSparkModule`] and the
    /// shared embedding + BF16 LM head. Mirrors
    /// [`crate::layers::DeepseekV4MtpHead::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module: DeepseekV4DSparkModule,
        embed_tokens: DenseWeight,
        lm_head: DenseWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        _max_seq_len: usize,
    ) -> Result<Self> {
        let num_stages = module.stages.len();

        // Drafter main-KV row = MLA-absorbed shape (kv_lora_rank +
        // qk_rope_head_dim), same as the target MLA cache. BF16 (tiny; avoids
        // FP8 unit-scale collapse).
        // Drafter attention uses the DIRECT head_dim (qk_nope + qk_rope =
        // config.head_dim), NOT the absorbed kv_lora+rope. The reference
        // `_project_q_and_draft_kv` splits `head_dim` off `fused_wqa_wkv`; the
        // dumped ring is `[max_seqs, window, head_dim]` (= 512 for V4-Flash).
        let head_dim = config.head_dim;
        let window = config.dspark_window_size;
        // K=1 single-stream serve profile → one ring row. (A batched profile
        // would size this to max_num_seqs and index by the request→slot map;
        // deferred with the batched drafter, GATE2 item 17.)
        let max_seqs = 1usize;

        // One sliding-window ring buffer per stage. A zero `window` means the
        // checkpoint did not ship `window_size` (mis-config) — build with no
        // rings; `propose()` guards on this.
        let mut main_kv_caches = Vec::with_capacity(num_stages);
        if window > 0 {
            for _ in 0..num_stages {
                main_kv_caches.push(DsparkMainKvCache::new(gpu, window, head_dim, max_seqs)?);
            }
        } else {
            tracing::warn!(
                "DeepSeek-V4 native DSpark: dspark_window_size == 0 — sliding-window \
                 main_kv_cache not allocated (native DSpark drafter cannot run)"
            );
        }

        // Fix B: rank-0-complete drafter → full (un-sharded) head/group counts.
        let (drafter_nq, drafter_o_groups) =
            crate::weight_loader::deepseek_v4::dspark::drafter_head_counts(config);

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            num_stages,
            block_size: config.dspark_block_size.max(1),
            noise_token_id: config.dspark_noise_token_id,
            main_kv_caches,
            drafter_nq,
            drafter_o_groups,
            // V4 ships HF-vanilla norm weights (norms are loaded exactly) — the
            // offset-from-1 kernel would apply `1 + w`.
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            hc_pre_k: gpu.kernel("hyper_connection", "hc_pre")?,
            hc_post_k: gpu.kernel("hyper_connection", "hc_post")?,
            dspark_attn_k: gpu.kernel("dspark_sparse_attention", "dspark_sparse_attention")?,
            rope_fwd_k: gpu.kernel("rope", "rope_forward_yarn_interleaved")?,
            rope_inv_k: gpu.kernel("rope", "rope_forward_yarn_interleaved_inv")?,
            mla_q_rope_extract_batched_k: gpu
                .kernel("mla_absorbed", "mla_q_rope_extract_batched")?,
            mla_q_rope_writeback_batched_k: gpu
                .kernel("mla_absorbed", "mla_q_rope_writeback_batched")?,
            // Non-affine per-head Q RMSNorm: rms_norm with a [head_dim] BF16 ones
            // weight (BF16 1.0 = 0x3F80 LE → bytes 0x80,0x3F).
            q_headnorm_ones: {
                let bytes = head_dim * 2;
                let ptr = gpu.alloc(bytes)?;
                let ones: Vec<u8> = std::iter::repeat_with(|| [0x80u8, 0x3Fu8])
                    .take(head_dim)
                    .flatten()
                    .collect();
                gpu.copy_h2d(&ones, ptr)?;
                DenseWeight { weight: ptr }
            },
            dump_call: std::sync::atomic::AtomicUsize::new(0),
            last_conf: std::sync::atomic::AtomicU32::new(f32::NAN.to_bits()),
        })
    }

    /// Allocate per-sequence state. One body sub-state per draft stage; the ring
    /// write position starts at 0.
    pub fn alloc_state_inner(&self, gpu: &dyn GpuBackend) -> Result<DeepseekV4DSparkProposerState> {
        let mut body_states = Vec::with_capacity(self.num_stages);
        for stage in &self.module.stages {
            body_states.push(stage.alloc_state(gpu)?);
        }
        Ok(DeepseekV4DSparkProposerState {
            main_kv_pos: 0,
            last_num_drafted: 0,
            body_states,
        })
    }

    /// Mia `project_main` (`dspark.py:708-711`): the block-input projection.
    ///
    /// `main_hidden` is the `[40,41,42]` target-hidden stack (`3 * hidden_size`
    /// BF16, shallow-to-deep concat = the drafter's `main_hidden_in` golden).
    /// Applies `main_proj` (`[hidden, 3*hidden]`, dequantized BF16) then
    /// `main_norm` (vanilla RMSNorm). Writes the block-input `main_x`
    /// (`[hidden]` BF16 = the `main_norm_out` golden) into `out`. Uses only the
    /// existing `dense_gemv` + `rms_norm` kernels.
    fn project_main(
        &self,
        main_hidden: DevicePtr,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let stack_in = (h as usize * self.module.stages.len().max(1)) as u32; // 3 * hidden

        // One dump index per project_main invocation (shared by the 3 boundary
        // files so the harness groups them). Only read when dumping is armed.
        let call = self
            .dump_call
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Dump the [40,41,42] input stack (boundary #1 `main_hidden_in`) so a
        // `main_norm_out` divergence localizes to the upstream target capture
        // vs project_main itself.
        self.dump_boundary(ctx, main_hidden, "main_hidden_in", call, 1, stack_in, stream)?;

        // main_proj: [hidden, 3*hidden] · main_hidden[3*hidden] -> proj[hidden]
        // (the `main_proj_out` golden). Scratch: reuse the norm-output buffer.
        let proj = ctx.buffers.norm_output();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            main_hidden,
            &self.module.main_proj,
            proj,
            h,        // out_dim = hidden
            stack_in, // in_dim  = 3 * hidden
            stream,
        )?;
        // main_norm: vanilla RMSNorm -> main_x (the `main_norm_out` golden).
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            proj,
            &self.module.main_norm,
            out,
            1,
            h,
            eps,
            stream,
        )?;

        // Golden boundary dump (armed by `ATLAS_DSPARK_DUMP_DIR`): the
        // `main_proj_out` (`proj`) and `main_norm_out` (`out`) tensors, so the
        // offline harness can gate `project_main` against the Mia goldens
        // before any net-new kernel lands. No-op when the env is unset.
        self.dump_boundary(ctx, proj, "main_proj_out", call, 1, h, stream)?;
        self.dump_boundary(ctx, out, "main_norm_out", call, 1, h, stream)?;
        Ok(())
    }

    /// Increment A skeleton: the semi-AR block stage forward with the attention
    /// **stubbed** (identity passthrough), exercising the mHC (`hc_pre`/`hc_post`)
    /// + `rms_norm` + reused 256-expert `MoE` plumbing against the assembled
    /// stage bodies. Gated by `ATLAS_DSPARK_DUMP_DIR` (dev/gate only) so a normal
    /// serve stays byte-neutral until Increment B lands the net-new sparse-MLA
    /// attention. Dumps `stage_out` (FP32 hc-streams `[block, hc_mult, hidden]`)
    /// per stage for the golden gate. The FP32 highway is ping-ponged between two
    /// buffers so `hc_post` never aliases its own residual (it mixes multiple
    /// residual streams through the doubly-stochastic `comb`).
    /// Full drafter forward: 3 stage forwards + head path → K=1 proposal token.
    /// Returns `Some(token)` (block position-0 argmax) or `None` if the block is
    /// empty. Boundary dumps inside are env-gated (production-neutral).
    fn run_stage_forward_dev(
        &self,
        last_token: u32,
        position: usize,
        main_x: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<u32>> {
        use crate::layers::qwen3_attention::Qwen3AttentionLayer;
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size as u32;
        let block = self.block_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        // Env-gated intra-stage boundary dumps for first-divergence localization
        // (12 boundaries; disabled by default, and additionally no-op unless
        // ATLAS_DSPARK_DUMP_DIR is set). Pure device→host reads — neutral to the
        // computation, so `stage_out` is byte-identical with this on or off.
        let dump_bd = std::env::var("ATLAS_DSPARK_DUMP_STAGE_BOUNDARIES").is_ok();

        // Downcast the assembled stage bodies to read their mHC params + MoE.
        let bodies: Vec<&Qwen3AttentionLayer> = self
            .module
            .stages
            .iter()
            .map(|s| {
                s.as_any()
                    .and_then(|a| a.downcast_ref::<Qwen3AttentionLayer>())
                    .ok_or_else(|| anyhow::anyhow!("DSpark stage body is not Qwen3AttentionLayer"))
            })
            .collect::<Result<_>>()?;
        let hc0 = bodies[0]
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSpark stage has no mHC weights"))?;
        let hc_mult = hc0.hc_mult as u32;
        let sinkhorn = hc0.sinkhorn_iters as u32;
        let hc_eps = hc0.hc_eps;

        // Per-call scratch (correctness-first; a batched/stateful profile pools these).
        let hidden_bytes = (block * h) as usize * 2; // BF16 [block, hidden]
        let stream_bytes = (block * hc_mult * h) as usize * 4; // FP32 [block, hc_mult, hidden]
        let mut cur = gpu.alloc(stream_bytes)?; // FP32 highway (ping)
        let mut nxt = gpu.alloc(stream_bytes)?; // FP32 highway (pong)
        let x_embed = gpu.alloc(hidden_bytes)?;
        let y_out = gpu.alloc(hidden_bytes)?; // hc_pre collapsed (BF16)
        let post = gpu.alloc((block * hc_mult) as usize * 4)?; // FP32
        let comb = gpu.alloc((block * hc_mult * hc_mult) as usize * 4)?; // FP32
        let norm_out = gpu.alloc(hidden_bytes)?; // rms_norm out / MoE in-place (BF16)
        let sublayer = gpu.alloc(hidden_bytes)?; // attn output (BF16)

        // Noise block embed: [anchor=last_token, noise×(block-1)] → x_embed [block, hidden].
        // Diagnostic (default OFF): route-B pins the block ANCHOR (row 0) to the
        // matched reference token so b01/b02 row-0 lines up with the golden pair;
        // without it Atlas embeds its live `last_token` and row 0 diverges (rows
        // 1..block are the fixed noise token and already match). Neutral when unset.
        let anchor = std::env::var("ATLAS_DSPARK_INJECT_ANCHOR_TOKEN")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(last_token);
        let row_bytes = h as usize * 2;
        for t in 0..block as usize {
            let tok = if t == 0 { anchor } else { self.noise_token_id } as usize;
            let src = self.embed_tokens.weight.offset(tok * row_bytes);
            gpu.copy_d2d_async(src, x_embed.offset(t * row_bytes), row_bytes, stream)?;
        }
        // hc_expand: x_embed → cur [block, hc_mult, hidden] FP32.
        ops::hc_expand(gpu, self.hc_expand_k, x_embed, cur, block, h, hc_mult, stream)?;

        // Route-B: force the decode position to the injected step (so valid-len /
        // slot / block positions match the golden matched-pair). No-op unless set.
        let position = std::env::var("ATLAS_DSPARK_INJECT_POSITION")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(position);
        // Block draft positions = main_pos + [0..block); main-KV store slot/valid-len.
        let window = self.main_kv_caches[0].window;
        let main_slot = position % window;
        let valid_main_len = ((position + 1).min(window)) as i32;
        let block_pos_host: Vec<u8> = (0..block)
            .flat_map(|t| (position as u32 + t).to_le_bytes())
            .collect();
        let block_pos_dev = gpu.alloc(block as usize * 4)?;
        gpu.copy_h2d(&block_pos_host, block_pos_dev)?;
        let main_pos_dev = gpu.alloc(4)?;
        gpu.copy_h2d(&(position as u32).to_le_bytes(), main_pos_dev)?;

        for (i, l) in bodies.iter().enumerate() {
            let hc = l.hc.as_ref().unwrap();
            // ── attn residual site: hc_pre → attn_norm → [STUB attn] → hc_post ──
            ops::hc_pre(
                gpu, self.hc_pre_k, cur, hc.attn.hc_fn, hc.attn.hc_scale, hc.attn.hc_base, y_out,
                post, comb, block, h, hc_mult, sinkhorn, eps, hc_eps, stream,
            )?;
            // B01: hc_pre output (collapsed, BF16 [block, hidden]).
            if dump_bd {
                self.dump_boundary(ctx, y_out, "b01_hc_pre", i, block, h, stream)?;
            }
            ops::rms_norm(gpu, self.rms_norm_k, y_out, l.dspark_attn_norm(), norm_out, block, h, eps, stream)?;
            // B02: attn_norm output (BF16 [block, hidden]).
            if dump_bd {
                self.dump_boundary(ctx, norm_out, "b02_attn_norm", i, block, h, stream)?;
            }
            // Route-B: inject the golden post-store ring for this stage (→ skip the
            // in-kernel store so the injected ring is authoritative). No-op OFF.
            let skip_store = self.maybe_inject_ring(i, ctx)?;
            // Net-new sparse-MLA stage attention (Increment B). B03–B08 dumped inside.
            self.stage_attn(
                *l, norm_out, main_x, block_pos_dev, main_pos_dev, valid_main_len,
                &self.main_kv_caches[i], main_slot, skip_store, sublayer, i, dump_bd, ctx, stream,
            )?;
            ops::hc_post(gpu, self.hc_post_k, sublayer, cur, post, comb, nxt, block, h, hc_mult, stream)?;
            std::mem::swap(&mut cur, &mut nxt);
            // B09: attention hc_post output (FP32 highway [block, hc_mult, hidden]).
            if dump_bd {
                self.dump_boundary_f32(ctx, cur, "b09_attn_hcpost", i, block * hc_mult, h, stream)?;
            }
            // ── ffn residual site: hc_pre → ffn_norm → MoE → hc_post ──
            ops::hc_pre(
                gpu, self.hc_pre_k, cur, hc.ffn.hc_fn, hc.ffn.hc_scale, hc.ffn.hc_base, y_out, post,
                comb, block, h, hc_mult, sinkhorn, eps, hc_eps, stream,
            )?;
            ops::rms_norm(gpu, self.rms_norm_k, y_out, l.dspark_ffn_norm(), norm_out, block, h, eps, stream)?;
            // B10: ffn_norm output (BF16 [block, hidden]) — dump before in-place MoE.
            if dump_bd {
                self.dump_boundary(ctx, norm_out, "b10_ffn_norm", i, block, h, stream)?;
            }
            l.dspark_ffn().forward_prefill(norm_out, block as usize, ctx, stream)?;
            // `forward_prefill` READS its input buffer and WRITES the routed+shared
            // result to `ctx.buffers.moe_output()` — it does NOT update `norm_out`
            // in place (the shared-codebase contract; see moe/forward_prefill.rs).
            // Copy the real MoE output back into `norm_out` so the b11 dump and the
            // terminating `hc_post` consume it. Without this the entire MoE
            // contribution is orphaned in moe_output() and hc_post folds the stale
            // ffn-norm (b11 == b10 byte-identical, cos 0.067). [block, hidden] BF16.
            gpu.copy_d2d_async(ctx.buffers.moe_output(), norm_out, (block * h) as usize * 2, stream)?;
            // B11: MoE output (BF16 [block, hidden], now resident in norm_out).
            if dump_bd {
                self.dump_boundary(ctx, norm_out, "b11_moe_out", i, block, h, stream)?;
            }
            ops::hc_post(gpu, self.hc_post_k, norm_out, cur, post, comb, nxt, block, h, hc_mult, stream)?;
            std::mem::swap(&mut cur, &mut nxt);
            // stage_out = the post-ffn highway (FP32 [block, hc_mult, hidden]).
            self.dump_boundary_f32(ctx, cur, "stage_out", i, block * hc_mult, h, stream)?;
            // B12: same tensor as stage_out, uniform boundary-set name.
            if dump_bd {
                self.dump_boundary_f32(ctx, cur, "b12_stage_out", i, block * hc_mult, h, stream)?;
            }
        }

        // ── HEAD PATH (post-3-stage → K=1 proposal) ──
        // `cur` = final hc_streams highway [block, hc_mult, hidden] FP32. Mirrors
        // the reference drafter head (`dspark.py:781-995`): hc_head collapse →
        // mtp.2.norm → shared lm_head logits → serial-Markov argmax loop → emit
        // block position-0 token (K=1). Confidence (mtp.2.confidence_head) is
        // computed for `last_confidence()` but is off the emit critical path
        // (tau=0 default). The serial loop always runs all `block` positions;
        // K=1 emits position 0 (`draft_token_ids[:, :1]`).
        let vocab = ctx.config.vocab_size as u32;
        let rank = ctx.config.dspark_markov_rank as u32;
        let hc = self
            .module
            .hc_head
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSpark last stage has no hc_head weights"))?;

        let dense = gpu.alloc(hidden_bytes)?; // [block, hidden] BF16
        let normed = gpu.alloc(hidden_bytes)?; // [block, hidden] BF16
        let logits = gpu.alloc((block * vocab) as usize * 2)?; // [block, vocab] BF16
        let mk_embed = gpu.alloc(rank as usize * 2)?; // [rank] BF16 (current prev embed)
        let mk_embed0 = gpu.alloc(rank as usize * 2)?; // saved pos-0 embed for confidence
        let mk_logits = gpu.alloc(vocab as usize * 2)?; // [vocab] BF16
        let argmax_out = gpu.alloc(4)?; // u32 index
        let feats = gpu.alloc((h as usize + rank as usize) * 2)?; // [hidden+rank] BF16 (confidence)
        let conf_bf16 = gpu.alloc(2)?; // [1] BF16 pre-sigmoid

        // 1. hc_head collapse: cur [block, hc_mult, hidden] FP32 → dense [block, hidden] BF16.
        ops::hc_head(
            gpu, self.hc_head_k, cur, hc.hc_fn, hc.hc_scale, hc.hc_base, dense, block, h, hc_mult,
            eps, hc_eps, stream,
        )?;
        // 2. final norm (mtp.2.norm), per block token.
        ops::rms_norm(gpu, self.rms_norm_k, dense, &self.module.norm, normed, block, h, eps, stream)?;
        // 3. base logits: shared lm_head GEMV per block position → logits [block, vocab].
        for p in 0..block as usize {
            ops::dense_gemv(
                gpu, self.dense_gemv_k, normed.offset(p * h as usize * 2), &self.lm_head,
                logits.offset(p * vocab as usize * 2), vocab, h, stream,
            )?;
        }
        // 4. serial-Markov argmax loop. Seed prev = anchor (last_token). For each
        //    position p: markov_embed = markov_w1[prev]; logits[p] += markov_w2 @
        //    markov_embed; next = argmax(logits[p]). K=1 emits position 0.
        let mut prev = last_token;
        let mut emit_token: Option<u32> = None;
        for p in 0..block as usize {
            gpu.copy_d2d_async(
                self.module.markov_w1.weight.offset(prev as usize * rank as usize * 2),
                mk_embed, rank as usize * 2, stream,
            )?;
            if p == 0 {
                gpu.copy_d2d_async(mk_embed, mk_embed0, rank as usize * 2, stream)?;
            }
            ops::dense_gemv(gpu, self.dense_gemv_k, mk_embed, &self.module.markov_w2, mk_logits, vocab, rank, stream)?;
            ops::residual_add(gpu, self.residual_add_k, logits.offset(p * vocab as usize * 2), mk_logits, vocab, stream)?;
            ops::argmax_bf16(gpu, self.argmax_k, logits.offset(p * vocab as usize * 2), argmax_out, vocab, stream)?;
            gpu.synchronize(stream)?;
            let mut tb = [0u8; 4];
            gpu.copy_d2h(argmax_out, &mut tb)?;
            let tok = u32::from_le_bytes(tb);
            if p == 0 {
                emit_token = Some(tok);
            }
            prev = tok;
        }
        // 5. confidence for the emitted position 0: sigmoid(proj(cat(dense_0,
        //    markov_embed_0))). Reported by `last_confidence()` (tau-gating only;
        //    not on the emit path).
        gpu.copy_d2d_async(dense, feats, h as usize * 2, stream)?;
        gpu.copy_d2d_async(mk_embed0, feats.offset(h as usize * 2), rank as usize * 2, stream)?;
        ops::dense_gemv(gpu, self.dense_gemv_k, feats, &self.module.confidence_proj, conf_bf16, 1, h + rank, stream)?;
        gpu.synchronize(stream)?;
        let mut cb = [0u8; 2];
        gpu.copy_d2h(conf_bf16, &mut cb)?;
        let conf_logit = f32::from_bits((u16::from_le_bytes(cb) as u32) << 16);
        let conf = 1.0f32 / (1.0 + (-conf_logit).exp());
        self.last_conf.store(conf.to_bits(), std::sync::atomic::Ordering::Relaxed);

        for p in [
            cur, nxt, x_embed, y_out, post, comb, norm_out, sublayer, block_pos_dev, main_pos_dev,
            dense, normed, logits, mk_embed, mk_embed0, mk_logits, argmax_out, feats, conf_bf16,
        ] {
            gpu.free(p)?;
        }
        Ok(emit_token)
    }

    /// Net-new DSpark stage attention (Increment B). Replaces the identity stub:
    /// `_project_q_and_draft_kv` → `store_main_kv` → `dspark_sparse_attention`
    /// (windowed MLA + sink) → inverse-rope → grouped low-rank O-projection →
    /// TP all-reduce. Mirrors `attn.forward_dspark` (dspark.py:471-540) and
    /// reuses the V4 decode rope/o-proj machinery. `attn_in` is the already-
    /// collapsed, attn-normed `[block, hidden]` BF16 (hc_pre collapsed the mHC
    /// streams, so no ndim-3 mean is needed). Writes `[block, hidden]` into `out`.
    #[allow(clippy::too_many_arguments)]
    fn stage_attn(
        &self,
        l: &crate::layers::qwen3_attention::Qwen3AttentionLayer,
        attn_in: DevicePtr,
        main_x: DevicePtr,
        block_pos_dev: DevicePtr,
        main_pos_dev: DevicePtr,
        valid_main_len: i32,
        ring: &DsparkMainKvCache,
        main_slot: usize,
        skip_store: bool,
        out: DevicePtr,
        stage_idx: usize,
        dump_bd: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let mla = l
            .mla
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSpark stage body has no MLA weights"))?;
        let h = ctx.config.hidden_size as u32;
        let block = self.block_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        // Fix B: rank-0-complete drafter uses the FULL (un-sharded) head/group
        // counts (`local * tp`, = 64 / 8 at TP=2), because the drafter MLA
        // weights are loaded un-sharded on rank 0 (`drafter_stage_config`). The
        // target model's TP-local `ctx.config.num_attention_heads` (= 32) would
        // read only half the un-sharded wq_b → wrong Q. `drafter_nq` is set in
        // `new()` from `drafter_head_counts(config)`.
        let nq = self.drafter_nq; // full drafter head count (un-sharded)
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let rope = mla.rope as u32;
        let hd = ctx.config.head_dim as u32; // direct head_dim = qk_nope + qk_rope (= 512)
        let nope = mla.nope as u32; // qk_nope_head_dim (= 448); rope lanes are the trailing `rope`
        let window = ring.window as u32;
        let scale = (hd as f32).powf(-0.5);
        let o_groups = self.drafter_o_groups.max(1); // full drafter O-proj groups (un-sharded)
        let group_in = (nq * hd) / o_groups;
        let latent_dim = o_groups * o_lora;

        // ── scratch (per-call; correctness-first) ──
        let b = block as usize;
        let qra = gpu.alloc(b * q_lora as usize * 2)?;
        let qra_n = gpu.alloc(b * q_lora as usize * 2)?;
        let q = gpu.alloc(b * (nq * hd) as usize * 2)?;
        let q_n = gpu.alloc(b * (nq * hd) as usize * 2)?;
        let kv = gpu.alloc(b * hd as usize * 2)?; // draft_kv (per block token)
        let kv_n = gpu.alloc(b * hd as usize * 2)?;
        let rope_tmp = gpu.alloc(b * (nq * rope) as usize * 2)?; // ≥ block*rope for kv too
        let attn_out = gpu.alloc(b * (nq * hd) as usize * 2)?;
        let o_latent = gpu.alloc(b * latent_dim as usize * 2)?;
        let o_out = gpu.alloc(b * h as usize * 2)?;
        let mkv = gpu.alloc(hd as usize * 2)?; // main-KV projection
        let mkv_n = gpu.alloc(hd as usize * 2)?;
        let valid_dev = gpu.alloc(4)?;
        gpu.copy_h2d(&valid_main_len.to_le_bytes(), valid_dev)?;

        let row_h = h as usize * 2;
        // ── _project_q_and_draft_kv (down-proj is replicated / disable_tp) ──
        for t in 0..b {
            let x_t = attn_in.offset(t * row_h);
            ops::dense_gemv(gpu, self.dense_gemv_k, x_t, &mla.wq_a, qra.offset(t * q_lora as usize * 2), q_lora, h, stream)?;
            ops::dense_gemv(gpu, self.dense_gemv_k, x_t, &mla.wkv_a, kv.offset(t * hd as usize * 2), hd, h, stream)?;
        }
        // q: q_norm(qra) → wq_b → [nq,hd] → per-head rsqrt-norm
        ops::rms_norm(gpu, self.rms_norm_k, qra, &mla.q_a_norm, qra_n, block, q_lora, eps, stream)?;
        for t in 0..b {
            ops::dense_gemv(gpu, self.dense_gemv_k, qra_n.offset(t * q_lora as usize * 2), &mla.wq_b, q.offset(t * (nq * hd) as usize * 2), nq * hd, q_lora, stream)?;
        }
        ops::rms_norm(gpu, self.rms_norm_k, q, &self.q_headnorm_ones, q_n, block * nq, hd, eps, stream)?;
        // B03: Q projection (post per-head norm, PRE-rope) BF16 [block, nq, hd].
        if dump_bd {
            self.dump_boundary(ctx, q_n, "b03_q_proj", stage_idx, block * nq, hd, stream)?;
        }
        // kv: kv_norm(kv) over head_dim
        ops::rms_norm(gpu, self.rms_norm_k, kv, &mla.kv_a_norm, kv_n, block, hd, eps, stream)?;
        // B04: draft-KV projection (post kv_norm, PRE-rope) BF16 [block, hd].
        if dump_bd {
            self.dump_boundary(ctx, kv_n, "b04_kv_proj", stage_idx, block, hd, stream)?;
        }
        // rope q (trailing `rope` lanes): extract → rope_yarn(fwd) → writeback
        ops::mla_q_rope_extract_batched(gpu, self.mla_q_rope_extract_batched_k, q_n, rope_tmp, block, nq, hd, nope, rope, nq * hd, stream)?;
        ops::rope_yarn(gpu, self.rope_fwd_k, rope_tmp, rope_tmp, block_pos_dev, block, nq, 0, rope, rope, mla.main_inv_freq, 1.0, stream)?;
        ops::mla_q_rope_writeback_batched(gpu, self.mla_q_rope_writeback_batched_k, rope_tmp, q_n, block, nq, hd, nope, rope, nq * hd, stream)?;
        // rope kv (single latent head)
        ops::mla_q_rope_extract_batched(gpu, self.mla_q_rope_extract_batched_k, kv_n, rope_tmp, block, 1, hd, nope, rope, hd, stream)?;
        ops::rope_yarn(gpu, self.rope_fwd_k, rope_tmp, rope_tmp, block_pos_dev, block, 1, 0, rope, rope, mla.main_inv_freq, 1.0, stream)?;
        ops::mla_q_rope_writeback_batched(gpu, self.mla_q_rope_writeback_batched_k, rope_tmp, kv_n, block, 1, hd, nope, rope, hd, stream)?;

        // ── store_main_kv: project main_x → kv_norm → rope(main_pos) → ring slot ──
        // Skipped under route-B ring injection (the injected ring is the exact
        // reference post-store state; re-storing would overwrite slot main_slot).
        if !skip_store {
            ops::dense_gemv(gpu, self.dense_gemv_k, main_x, &mla.wkv_a, mkv, hd, h, stream)?;
            ops::rms_norm(gpu, self.rms_norm_k, mkv, &mla.kv_a_norm, mkv_n, 1, hd, eps, stream)?;
            ops::mla_q_rope_extract_batched(gpu, self.mla_q_rope_extract_batched_k, mkv_n, rope_tmp, 1, 1, hd, nope, rope, hd, stream)?;
            ops::rope_yarn(gpu, self.rope_fwd_k, rope_tmp, rope_tmp, main_pos_dev, 1, 1, 0, rope, rope, mla.main_inv_freq, 1.0, stream)?;
            ops::mla_q_rope_writeback_batched(gpu, self.mla_q_rope_writeback_batched_k, rope_tmp, mkv_n, 1, 1, hd, nope, rope, hd, stream)?;
            gpu.copy_d2d_async(mkv_n, ring.buf.offset(main_slot * hd as usize * 2), hd as usize * 2, stream)?;
        }

        // ── sparse windowed-MLA attention (net-new kernel) ──
        ops::dspark_sparse_attention(gpu, self.dspark_attn_k, q_n, kv_n, ring.buf, valid_dev, mla.attn_sink, attn_out, scale, 1, block, nq, hd, window, stream)?;
        // B05: sparse-MLA output, PRE inverse-rope BF16 [block, nq, hd].
        if dump_bd {
            self.dump_boundary(ctx, attn_out, "b05_smla_out", stage_idx, block * nq, hd, stream)?;
        }

        // ── inverse-rope the attention output (de-rotate by block position) ──
        ops::mla_q_rope_extract_batched(gpu, self.mla_q_rope_extract_batched_k, attn_out, rope_tmp, block, nq, hd, nope, rope, nq * hd, stream)?;
        ops::rope_yarn(gpu, self.rope_inv_k, rope_tmp, rope_tmp, block_pos_dev, block, nq, 0, rope, rope, mla.main_inv_freq, 1.0, stream)?;
        ops::mla_q_rope_writeback_batched(gpu, self.mla_q_rope_writeback_batched_k, rope_tmp, attn_out, block, nq, hd, nope, rope, nq * hd, stream)?;
        // B06: inverse-RoPE output BF16 [block, nq, hd].
        if dump_bd {
            self.dump_boundary(ctx, attn_out, "b06_invrope_out", stage_idx, block * nq, hd, stream)?;
        }

        // ── grouped low-rank O-projection (wo_a block-diagonal → wo_b), per block token ──
        for t in 0..b {
            let ao_t = attn_out.offset(t * (nq * hd) as usize * 2);
            let ol_t = o_latent.offset(t * latent_dim as usize * 2);
            for g in 0..o_groups {
                let in_g = ao_t.offset((g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla.wo_a.weight.offset((g * o_lora * group_in) as usize * 2),
                };
                ops::dense_gemv(gpu, self.dense_gemv_k, in_g, &w_g, ol_t.offset((g * o_lora) as usize * 2), o_lora, group_in, stream)?;
            }
            ops::dense_gemv(gpu, self.dense_gemv_k, ol_t, &mla.wo_b, o_out.offset(t * row_h), h, latent_dim, stream)?;
        }
        // B07: grouped O-projection = wo_a grouped output `o_latent`
        // [block, o_groups*o_lora], the natural PRE-reduce checkpoint that maps
        // to the reference `projected` (dspark.py:526-539; wo_b/RowParallel is
        // the reduce = B08). Not the wo_b local partial (reference never exposes
        // it, and it is TP-sharding-dependent / non-comparable).
        if dump_bd {
            self.dump_boundary(ctx, o_latent, "b07_oproj_wo_a", stage_idx, block, latent_dim, stream)?;
        }
        // ── Fix B: NO TP all-reduce ──
        // The drafter is rank-0-complete: its wo_b is loaded UN-SHARDED (full
        // o_latent input), so this per-token wo_b GEMV already produces the
        // COMPLETE O-projection on rank 0. There is no row-parallel partial to
        // reduce (and no peer rank running the drafter to reduce with). The
        // former `ctx.comm` all-reduce here was dead code on the `comm: None`
        // propose path and is removed — o_out below is the full result.
        // B08: complete O-projection output BF16 [block, hidden].
        if dump_bd {
            self.dump_boundary(ctx, o_out, "b08_oproj_postred", stage_idx, block, h, stream)?;
        }
        gpu.copy_d2d_async(o_out, out, b * row_h, stream)?;

        for p in [
            qra, qra_n, q, q_n, kv, kv_n, rope_tmp, attn_out, o_latent, o_out, mkv, mkv_n, valid_dev,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }

    /// FP32 variant of [`Self::dump_boundary`] for the hc-streams highway
    /// (`stage_out` goldens are F32). Same filename contract, `_f32` suffix.
    fn dump_boundary_f32(
        &self,
        ctx: &ForwardContext,
        src: DevicePtr,
        tag: &str,
        call: usize,
        rows: u32,
        cols: u32,
        stream: u64,
    ) -> Result<()> {
        let Ok(dir) = std::env::var("ATLAS_DSPARK_DUMP_DIR") else {
            return Ok(());
        };
        if dir.is_empty() {
            return Ok(());
        }
        let gpu = ctx.gpu;
        let n_bytes = rows as usize * cols as usize * 4; // FP32
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; n_bytes];
        gpu.copy_d2h(src, &mut buf)?;
        let path = format!("{dir}/{tag}__call{call:04}__r{rows}_c{cols}_f32.bin");
        match std::fs::write(&path, &buf) {
            Ok(()) => tracing::info!("DSPARK DUMP: wrote {path} ({rows}x{cols} FP32)"),
            Err(e) => tracing::warn!("DSPARK DUMP: write {path} failed: {e}"),
        }
        Ok(())
    }

    /// Dump a BF16 device tensor to `$ATLAS_DSPARK_DUMP_DIR/{tag}__r{rows}_c{cols}_bf16.bin`
    /// (raw little-endian BF16, row-major). Mirrors the DFLASH `block_dump_buf`
    /// idiom (sync + D2H + raw write); the shape is encoded in the filename so
    /// the compare harness needs no sidecar. No-op unless the env var is set.
    fn dump_boundary(
        &self,
        ctx: &ForwardContext,
        src: DevicePtr,
        tag: &str,
        call: usize,
        rows: u32,
        cols: u32,
        stream: u64,
    ) -> Result<()> {
        let Ok(dir) = std::env::var("ATLAS_DSPARK_DUMP_DIR") else {
            return Ok(());
        };
        if dir.is_empty() {
            return Ok(());
        }
        let gpu = ctx.gpu;
        let n_bytes = rows as usize * cols as usize * 2; // BF16
        gpu.synchronize(stream)?;
        let mut buf = vec![0u8; n_bytes];
        gpu.copy_d2h(src, &mut buf)?;
        let path = format!("{dir}/{tag}__call{call:04}__r{rows}_c{cols}_bf16.bin");
        match std::fs::write(&path, &buf) {
            Ok(()) => tracing::info!("DSPARK DUMP: wrote {path} ({rows}x{cols} BF16)"),
            Err(e) => tracing::warn!("DSPARK DUMP: write {path} failed: {e}"),
        }
        Ok(())
    }

    /// Route-B diagnostic injection (default OFF). When
    /// `ATLAS_DSPARK_INJECT_MAIN_HIDDEN` is a file path, load it as the
    /// block-input `main_hidden` (raw little-endian BF16, logical shape
    /// `[1, hidden_size * num_stages]`) so `project_main` can be gated against
    /// the golden in isolation, free of live generation/token drift. Validates
    /// the exact byte count (= shape × BF16) and **fails loudly** on any
    /// mismatch — wrong tensor/dtype/rank refuses to inject rather than
    /// silently transform garbage. Returns a freshly-allocated device buffer
    /// the caller must `free`; `None` when the env is unset/empty (production).
    fn maybe_inject_main_hidden(&self, ctx: &ForwardContext) -> Result<Option<DevicePtr>> {
        let Ok(path) = std::env::var("ATLAS_DSPARK_INJECT_MAIN_HIDDEN") else {
            return Ok(None);
        };
        if path.is_empty() {
            return Ok(None);
        }
        let stack_in = ctx.config.hidden_size * self.num_stages.max(1);
        let expected = stack_in * 2; // BF16 [1, hidden * num_stages]
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("DSPARK INJECT: cannot read {path}: {e}"))?;
        if bytes.len() != expected {
            anyhow::bail!(
                "DSPARK INJECT: {path} is {} bytes, expected {} ([1,{}] BF16, hidden={} × \
                 stages={}). Wrong tensor / dtype / rank — refusing to inject.",
                bytes.len(),
                expected,
                stack_in,
                ctx.config.hidden_size,
                self.num_stages
            );
        }
        let buf = ctx.gpu.alloc(expected)?;
        ctx.gpu.copy_h2d(&bytes, buf)?;
        tracing::warn!(
            "DSPARK INJECT ACTIVE (DIAGNOSTIC): main_hidden ← {path} ({} BF16 elems). Output is \
             NOT from the live target capture — must never be set in production.",
            stack_in
        );
        Ok(Some(buf))
    }

    /// Route-B ring injection (DIAGNOSTIC, default OFF). When
    /// `ATLAS_DSPARK_INJECT_RING_DIR` is a dir containing `r0_ring_stage{i}.bin`
    /// (raw little-endian BF16, logical `[max_seqs, window, head_dim]` = the
    /// reference post-store ring), load stage `i`'s ring into
    /// `main_kv_caches[i].buf` and return `true` so the caller SKIPS the in-kernel
    /// `store_main_kv` (the injected ring is the authoritative attended state).
    /// Validates the exact byte count; fails loudly on mismatch. `false` (no
    /// injection) unless the env is set.
    fn maybe_inject_ring(&self, stage_idx: usize, ctx: &ForwardContext) -> Result<bool> {
        let Ok(dir) = std::env::var("ATLAS_DSPARK_INJECT_RING_DIR") else {
            return Ok(false);
        };
        if dir.is_empty() {
            return Ok(false);
        }
        let ring = &self.main_kv_caches[stage_idx];
        let path = format!("{dir}/r0_ring_stage{stage_idx}.bin");
        let bytes =
            std::fs::read(&path).map_err(|e| anyhow::anyhow!("DSPARK RING INJECT: read {path}: {e}"))?;
        if bytes.len() != ring.bytes {
            anyhow::bail!(
                "DSPARK RING INJECT: {path} is {} bytes, expected {} ([{}, {}, {}] BF16) — \
                 wrong shape/dtype, refusing to inject.",
                bytes.len(),
                ring.bytes,
                ring.max_seqs,
                ring.window,
                ring.head_dim
            );
        }
        ctx.gpu.copy_h2d(&bytes, ring.buf)?;
        tracing::warn!(
            "DSPARK RING INJECT ACTIVE (DIAGNOSTIC) stage {stage_idx} ← {path} ({} bytes)",
            bytes.len()
        );
        Ok(true)
    }
}

impl DraftProposer for DeepseekV4DSparkHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    /// Chain confidence of the most recent `propose`. The DSpark confidence head
    /// runs inside the net-new stage forward; until that lands report `None` so
    /// callers do not gate.
    fn last_confidence(&self) -> Option<f32> {
        let c = f32::from_bits(self.last_conf.load(std::sync::atomic::Ordering::Relaxed));
        if c.is_nan() { None } else { Some(c) }
    }

    #[allow(clippy::too_many_arguments)]
    fn propose(
        &self,
        _last_token: u32,
        _target_hidden: DevicePtr,
        _position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let _dspark_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 DSpark proposer state"))?;

        // Externally K=1: the semi-AR block width is internal; a caller asking
        // for >1 draft gets pos-0 only (once the forward lands).
        if num_drafts > 1 {
            tracing::warn!(
                "V4 DSpark proposer is K=1; num_drafts={num_drafts} ignored (emits block pos-0)"
            );
        }

        // No sliding-window ring (mis-config) → cannot draft.
        if self.main_kv_caches.is_empty() {
            tracing::debug!("DSpark propose: no main_kv_cache ring (window==0) — drafting none");
            return Ok(Vec::new());
        }

        // ── Mia `draft()` step (a): block-input projection (dspark.py:884) ──
        // main_proj → main_norm over the [40,41,42] target-hidden stack. This
        // exercises the `main_hidden_in` → `main_proj_out` → `main_norm_out`
        // golden boundaries with existing kernels; the resulting `main_x` is
        // the block input the net-new stage forward consumes.
        // Route-B prologue isolation (DIAGNOSTIC, default OFF): when
        // `ATLAS_DSPARK_INJECT_MAIN_HIDDEN` points at a raw-BF16 golden
        // `main_hidden_in`, inject it as the block-input in place of the live
        // target capture — removes generation/token drift so `project_main`
        // (main_proj + main_norm) can be gated against the golden in isolation.
        // No-op unless the env is set; fails loudly on any shape/byte mismatch.
        let main_x = ctx.buffers.hidden_states();
        let injected = self.maybe_inject_main_hidden(ctx)?;
        let main_hidden_src = injected.or(target_hidden_stack);
        let have_main_x = main_hidden_src.is_some();
        if let Some(main_hidden) = main_hidden_src {
            self.project_main(main_hidden, main_x, ctx, stream)?;
            tracing::debug!("DSpark propose: project_main done (main_norm_out ready)");
        } else {
            tracing::debug!("DSpark propose: no target_hidden_stack — skipping project_main");
        }
        if let Some(p) = injected {
            ctx.gpu.free(p)?;
        }

        // ── STOP — net-new drafter kernel boundary (GATE3 correction #3) ──
        //
        // The remaining Mia `draft()` body (dspark.py:888-995) is the semi-AR
        // block forward and does NOT reuse the target's per-token MLA decode:
        //   • build block_size(=5) noise block (`[:,0]=accepted`), embed,
        //     expand to hc_mult streams (dspark.py:888-895);
        //   • 3× stage `forward_dspark` — each: mHC-pre → attn_norm →
        //     **sparse-MLA over the sliding-window main_kv_cache + block
        //     draft_kv with an attn_sink normalizer** (`dspark_sparse_attention`,
        //     dspark_kernels.py:716) → fp8-einsum o-proj
        //     (`deepseek_v4_fp8_einsum`) → hc_post → mHC-pre → ffn_norm → MoE
        //     (reuses the target MegaMoE) → hc_post (dspark.py:471-540/741-779);
        //   • hc_head collapse → final norm → base logits;
        //   • serial-Markov argmax loop (`logits[:,pos] += markov; argmax`,
        //     dspark.py:980-985); confidence sigmoid (dspark.py:987-990);
        //   • emit block pos-0 for K=1 (`draft_token_ids[:, :1]`, PROP:1032).
        //
        // The sparse windowed attention, the block KV projection
        // (`_project_q_and_draft_kv`), and the fp8-einsum o-projection are a
        // **net-new drafter kernel surface** absent from Atlas's per-token MLA
        // decode. They are authored + compiled + convergence-tested in the
        // image-build / GPU lane (a CUDA build this session cannot run). Until
        // then `propose()` drafts nothing rather than silently substituting the
        // wrong (per-token MLA) attention, which would diverge on the
        // `stage_out` goldens.
        // Full drafter forward (3 sparse-MLA stages + head) → K=1 proposal token.
        // Needs the block input `main_x` from `project_main` above. Boundary dumps
        // inside are env-gated (`ATLAS_DSPARK_DUMP_STAGE_BOUNDARIES`), so a normal
        // serve runs the forward + emits pos-0 with no dumps. A forward error
        // degrades to drafting nothing (verify then decodes serially).
        if !have_main_x {
            return Ok(Vec::new());
        }
        // DIAGNOSTIC (default OFF): emit a fixed valid token WITHOUT running the
        // drafter forward, to bisect an e2e verify crash — does the K2 verify
        // fault because drafting runs at all (verify-integration bug), or because
        // the drafter forward disturbs shared decode scratch the captured verify
        // graph replays? If the crash persists with this on, it is not the forward.
        if std::env::var("ATLAS_DSPARK_EMIT_DUMMY").is_ok() {
            return Ok(vec![_last_token]);
        }
        match self.run_stage_forward_dev(_last_token, _position, main_x, ctx, stream) {
            Ok(Some(tok)) => Ok(vec![tok]),
            Ok(None) => Ok(Vec::new()),
            Err(e) => {
                tracing::warn!("DSpark drafter forward failed: {e} — drafting none");
                Ok(Vec::new())
            }
        }
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let dspark_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 DSpark proposer state"))?;

        // Sliding-window ring advance (Mia items 18-20): the accepted tokens
        // become committed main context and advance the ring write position;
        // the rejected suffix never enters the window (`valid_len = seg_len -
        // rejected`, [`ring_valid_len`]). Position advances by accepted count.
        let num_drafted = dspark_state.last_num_drafted;
        let num_rejected = num_drafted.saturating_sub(num_accepted);
        let old_pos = dspark_state.main_kv_pos;
        dspark_state.main_kv_pos = dspark_state.main_kv_pos.saturating_add(num_accepted);
        tracing::debug!(
            "V4 DSpark after_verify: accepted={num_accepted} drafted={num_drafted} \
             rejected={num_rejected} main_kv_pos: {old_pos} → {} (slot {})",
            dspark_state.main_kv_pos,
            self.main_kv_caches
                .first()
                .map(|c| ring_slot(dspark_state.main_kv_pos, c.window))
                .unwrap_or(0),
        );
        Ok(())
    }

    fn free_state(&self, gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let dspark_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 DSpark proposer state"))?;
        // Single-stream profile: zero the shared rings + reset the write
        // position at the sequence boundary (Mia keeps buffers persistent and
        // resets via positions; a zero is a safe superset for one stream).
        for cache in &self.main_kv_caches {
            cache.reset(gpu)?;
        }
        dspark_state.main_kv_pos = 0;
        dspark_state.last_num_drafted = 0;
        Ok(())
    }
}

// The head owns the ring device buffers, released with the backend at teardown
// (process-lifetime, exactly like `DeepseekV4MtpHead`'s KV pool). `DevicePtr`
// has no `Drop`, so no per-head free is issued — consistent with the MTP head.

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sliding-window ring index contract (host math; no GPU) ──

    #[test]
    fn ring_slot_wraps_at_window() {
        let window = 128;
        assert_eq!(ring_slot(0, window), 0);
        assert_eq!(ring_slot(127, window), 127);
        assert_eq!(ring_slot(128, window), 0, "wraps at window");
        assert_eq!(ring_slot(129, window), 1);
        assert_eq!(ring_slot(2 * window + 5, window), 5, "second wrap");
    }

    #[test]
    fn ring_valid_len_matches_mia_clamp() {
        // Mia: valid_len = (seg_len - rejected).clamp(min=1, max=seg_len).
        assert_eq!(ring_valid_len(5, 0), 5, "no reject → all valid");
        assert_eq!(ring_valid_len(5, 2), 3, "3 of 5 committed");
        assert_eq!(ring_valid_len(5, 4), 1, "one always committed");
        assert_eq!(ring_valid_len(5, 5), 1, "clamp min=1 even at full reject");
        assert_eq!(ring_valid_len(5, 10), 1, "over-reject clamps to 1");
        assert_eq!(ring_valid_len(0, 3), 0, "empty segment stays empty");
    }

    #[test]
    fn ring_row_offset_layout() {
        // [max_seqs, window, head_dim] BF16, row-major.
        let c = DsparkMainKvCache {
            buf: DevicePtr::NULL,
            window: 128,
            head_dim: 576,
            max_seqs: 1,
            bytes: 128 * 576 * 2,
        };
        assert_eq!(c.row_offset(0, 0), 0);
        assert_eq!(
            c.row_offset(0, 1),
            576 * 2,
            "next slot = head_dim BF16 rows"
        );
        assert_eq!(c.row_offset(0, 127), 127 * 576 * 2);
    }

    // ── after_verify ring-advance semantics (accepted advances; rejected does
    // not enter the window) ──
    #[test]
    fn after_verify_advances_by_accepted_only() {
        // Simulate the position math after_verify performs.
        let advance = |pos: usize, drafted: usize, accepted: usize| -> (usize, usize) {
            let rejected = drafted.saturating_sub(accepted);
            (pos.saturating_add(accepted), rejected)
        };
        // Drafted K=1, accepted 1 → advance 1, reject 0.
        assert_eq!(advance(10, 1, 1), (11, 0));
        // Drafted 1, accepted 0 (reject) → no advance, reject 1.
        assert_eq!(advance(10, 1, 0), (10, 1));
        // Bonus-only step (drafted 0) → advance by the bonus (accepted).
        assert_eq!(advance(10, 0, 1), (11, 0));
    }

    #[test]
    fn dspark_state_downcast_roundtrip() {
        let mut state: Box<dyn ProposerState> = Box::new(DeepseekV4DSparkProposerState {
            main_kv_pos: 200,
            last_num_drafted: 1,
            body_states: Vec::new(),
        });
        let s = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .expect("downcast to DSpark state");
        // Ring slot of the current write position (window 128): 200 % 128 = 72.
        assert_eq!(ring_slot(s.main_kv_pos, 128), 72);
    }
}
