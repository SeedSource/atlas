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
    #[allow(dead_code)] // consumed by the stage-attn helper (Increment B, same lane)
    rope_inv_k: KernelHandle,

    /// Monotonic `propose()` call index, appended to boundary-dump filenames so
    /// successive calls do not overwrite (used only when `ATLAS_DSPARK_DUMP_DIR`
    /// is armed; lets the offline harness content-match the golden step).
    dump_call: std::sync::atomic::AtomicUsize,
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
        let head_dim = config.kv_lora_rank + config.qk_rope_head_dim;
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

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            num_stages,
            block_size: config.dspark_block_size.max(1),
            noise_token_id: config.dspark_noise_token_id,
            main_kv_caches,
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
            dump_call: std::sync::atomic::AtomicUsize::new(0),
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
    fn run_stage_forward_dev(&self, last_token: u32, ctx: &ForwardContext, stream: u64) -> Result<()> {
        use crate::layers::qwen3_attention::Qwen3AttentionLayer;
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size as u32;
        let block = self.block_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;

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
        let row_bytes = h as usize * 2;
        for t in 0..block as usize {
            let tok = if t == 0 { last_token } else { self.noise_token_id } as usize;
            let src = self.embed_tokens.weight.offset(tok * row_bytes);
            gpu.copy_d2d_async(src, x_embed.offset(t * row_bytes), row_bytes, stream)?;
        }
        // hc_expand: x_embed → cur [block, hc_mult, hidden] FP32.
        ops::hc_expand(gpu, self.hc_expand_k, x_embed, cur, block, h, hc_mult, stream)?;

        for (i, l) in bodies.iter().enumerate() {
            let hc = l.hc.as_ref().unwrap();
            // ── attn residual site: hc_pre → attn_norm → [STUB attn] → hc_post ──
            ops::hc_pre(
                gpu, self.hc_pre_k, cur, hc.attn.hc_fn, hc.attn.hc_scale, hc.attn.hc_base, y_out,
                post, comb, block, h, hc_mult, sinkhorn, eps, hc_eps, stream,
            )?;
            ops::rms_norm(gpu, self.rms_norm_k, y_out, l.dspark_attn_norm(), norm_out, block, h, eps, stream)?;
            // STUB: identity passthrough (Increment B replaces with sparse-MLA).
            gpu.copy_d2d_async(norm_out, sublayer, hidden_bytes, stream)?;
            ops::hc_post(gpu, self.hc_post_k, sublayer, cur, post, comb, nxt, block, h, hc_mult, stream)?;
            std::mem::swap(&mut cur, &mut nxt);
            // ── ffn residual site: hc_pre → ffn_norm → MoE → hc_post ──
            ops::hc_pre(
                gpu, self.hc_pre_k, cur, hc.ffn.hc_fn, hc.ffn.hc_scale, hc.ffn.hc_base, y_out, post,
                comb, block, h, hc_mult, sinkhorn, eps, hc_eps, stream,
            )?;
            ops::rms_norm(gpu, self.rms_norm_k, y_out, l.dspark_ffn_norm(), norm_out, block, h, eps, stream)?;
            l.dspark_ffn().forward_prefill(norm_out, block as usize, ctx, stream)?;
            ops::hc_post(gpu, self.hc_post_k, norm_out, cur, post, comb, nxt, block, h, hc_mult, stream)?;
            std::mem::swap(&mut cur, &mut nxt);
            // stage_out = the post-ffn highway (FP32 [block, hc_mult, hidden]).
            self.dump_boundary_f32(ctx, cur, "stage_out", i, block * hc_mult, h, stream)?;
        }

        for p in [cur, nxt, x_embed, y_out, post, comb, norm_out, sublayer] {
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
}

impl DraftProposer for DeepseekV4DSparkHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    /// Chain confidence of the most recent `propose`. The DSpark confidence head
    /// runs inside the net-new stage forward; until that lands report `None` so
    /// callers do not gate.
    fn last_confidence(&self) -> Option<f32> {
        None
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
        let injected = self.maybe_inject_main_hidden(ctx)?;
        let main_hidden_src = injected.or(target_hidden_stack);
        if let Some(main_hidden) = main_hidden_src {
            let main_x = ctx.buffers.hidden_states();
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
        // Increment A (dev/gate only): run the stage-forward plumbing skeleton
        // (mHC + norm + reused MoE; attention STUBBED) to dump `stage_out` for
        // the golden gate. Gated on `ATLAS_DSPARK_DUMP_DIR` so a normal serve is
        // byte-neutral (still drafts nothing). Increment B replaces the stub with
        // the net-new sparse-MLA attention and returns the K=1 draft token.
        if std::env::var("ATLAS_DSPARK_DUMP_DIR")
            .map(|d| !d.is_empty())
            .unwrap_or(false)
        {
            if let Err(e) = self.run_stage_forward_dev(_last_token, ctx, stream) {
                tracing::warn!("DSpark stage-forward dev skeleton failed: {e}");
            }
        }
        tracing::debug!(
            "DSpark propose: semi-AR block forward (sparse-MLA stages) is the net-new \
             kernel boundary — drafting none pending the GPU-convergence lane"
        );
        Ok(Vec::new())
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
