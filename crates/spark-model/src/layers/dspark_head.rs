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
//! ## Commit 3 scaffold (this file)
//!
//! This commit lands the proposer's **structure + lifecycle + production
//! install** only. The numerical forward — the real block/Markov/confidence
//! math — is deferred to the gated Commit 4. Concretely:
//!
//!   * `new()` / `alloc_state` / `after_verify` / `free_state` are REAL: the
//!     proposer owns its per-sequence state and its own KV cache(s), mirroring
//!     [`crate::layers::DeepseekV4MtpHead`].
//!   * `propose()` is an explicit SCAFFOLD: it validates state shape, logs, and
//!     returns `Ok(Vec::new())` (drafts nothing). No target hidden / stage
//!     forward runs yet.
//!   * `last_confidence()` returns `None` until the confidence head is wired.
//!
//! Externally the proposer exposes only **K=1** (`num_drafts > 1` is warned and
//! ignored); the semi-AR block width is an internal detail of the deferred
//! forward.
//!
//! ## KV cache sizing (scaffold decision)
//!
//! The 3 stage bodies were each assembled with `attn_layer_idx =
//! num_hidden_layers` (interior / no-compressor path), exactly as
//! [`crate::layers::DeepseekV4MtpHead`] builds its single stage. So — like the
//! MTP head — the KV cache pool must have `num_hidden_layers + 1` layer slots
//! for that index to be valid. A single shared [`PagedKvCache`] is allocated
//! with that layer count and blocks scaled by the stage count, and the
//! per-sequence state carries **one block table per stage** so the three stages
//! draw disjoint physical blocks (no aliasing at the shared layer index). The
//! exact stage↔cache wiring is exercised only by the Commit 4 forward.

use parking_lot::Mutex;
use std::any::Any;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use crate::layer::{ForwardContext, LayerState};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_loader::deepseek_v4::DeepseekV4DSparkModule;
use crate::weight_map::DenseWeight;

/// Per-sequence state for the DeepSeek-V4 native DSpark proposer.
///
/// Several fields (`body_states`, `last_num_drafted`) are written now but first
/// consumed by the Commit 4 forward; the lifecycle callbacks (`after_verify` /
/// `free_state`) already use `seq_len` + `block_tables`.
#[allow(dead_code)]
pub struct DeepseekV4DSparkProposerState {
    /// One block table per draft stage for the drafter's OWN KV cache. The
    /// stages share a single pool layer index but draw disjoint blocks, so a
    /// table per stage keeps their physical slots separate.
    pub block_tables: Vec<Vec<u32>>,
    /// Current sequence length in the drafter KV cache.
    pub seq_len: usize,
    /// Drafts produced by the last `propose()` (for `after_verify` trimming).
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
///
/// Fields carrying the DSpark heads, shared embedding / LM head, reduced draft
/// vocab, and kernel handles are populated now but first read by the Commit 4
/// forward; `module.stages` (state alloc) and `kv_cache` (trim/free) are the
/// only pieces the Commit 3 lifecycle exercises.
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
    /// Shared single MLA-shaped KV cache pool for the drafter attention. Sized
    /// `num_hidden_layers + 1` layers (the stages' assembled `attn_layer_idx`);
    /// per-stage block tables (in the state) keep the stages' blocks disjoint.
    kv_cache: Mutex<PagedKvCache>,

    // Kernel handles (mirrors `DeepseekV4MtpHead`; consumed by the Commit 4
    // forward).
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    residual_add_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_head_k: KernelHandle,
    argmax_k: KernelHandle,
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
        max_seq_len: usize,
    ) -> Result<Self> {
        let num_stages = module.stages.len();

        // Drafter KV cache: single MLA-absorbed attention shape (num_kv_heads =
        // 1, head_dim = kv_lora_rank + qk_rope_head_dim), matching the target's
        // MLA cache so the reused V4 body's `write_kv_cache` / `run_paged_decode`
        // land at the correct strides. BF16 (tiny cache; avoids FP8 unit-scale
        // collapse). Each stage was assembled with `attn_layer_idx =
        // num_hidden_layers`, so the pool must carry `num_hidden_layers + 1`
        // layer slots for that index to be valid (only the last is used).
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        let num_layers = config.num_hidden_layers + 1;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: mla_cache_dim,
            num_layers,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        // One sequence's worth of blocks PER stage (the stages draw disjoint
        // blocks from the shared pool).
        let per_stage_blocks = max_seq_len / kv_config.block_size + 1;
        let dspark_num_blocks = per_stage_blocks * num_stages.max(1);
        let kv_cache = PagedKvCache::new(kv_config, dspark_num_blocks, gpu)?;

        Ok(Self {
            module,
            embed_tokens,
            lm_head,
            mtp_vocab_size,
            num_stages,
            kv_cache: Mutex::new(kv_cache),
            // V4 ships HF-vanilla norm weights (norms are loaded exactly) — the
            // offset-from-1 kernel would apply `1 + w`.
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            hc_expand_k: gpu.kernel("hyper_connection", "hc_expand")?,
            hc_head_k: gpu.kernel("hyper_connection", "hc_head")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
        })
    }

    /// Allocate per-sequence state. Allocates one empty block table + one body
    /// sub-state per draft stage (mirrors each stage's own `alloc_state`).
    pub fn alloc_state_inner(
        &self,
        gpu: &dyn GpuBackend,
    ) -> Result<DeepseekV4DSparkProposerState> {
        let mut body_states = Vec::with_capacity(self.num_stages);
        for stage in &self.module.stages {
            body_states.push(stage.alloc_state(gpu)?);
        }
        Ok(DeepseekV4DSparkProposerState {
            block_tables: vec![Vec::new(); self.num_stages],
            seq_len: 0,
            last_num_drafted: 0,
            body_states,
        })
    }
}

impl DraftProposer for DeepseekV4DSparkHead {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(self.alloc_state_inner(gpu)?))
    }

    /// Chain confidence of the most recent `propose`. The DSpark confidence head
    /// is wired in Commit 4; until then report `None` so callers do not gate.
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
        _ctx: &ForwardContext,
        _stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        _grammar_bitmask: Option<&[i32]>,
        _target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        // Downcast to validate the state shape even though we draft nothing —
        // keeps the install path honest and mirrors the MTP head's contract.
        let _dspark_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 DSpark proposer state"))?;

        // Externally K=1: the semi-AR block width is internal to the deferred
        // forward; a caller asking for >1 draft gets none until Commit 4.
        if num_drafts > 1 {
            tracing::warn!(
                "V4 DSpark proposer is K=1; num_drafts={num_drafts} ignored (scaffold drafts none)"
            );
        }

        tracing::debug!("DSpark propose scaffold — forward convergence lands in Commit 4");
        // Commit 4: real forward (forward_embed → 3 stages → hc_head → base
        // logits → serial markov → confidence → slice[:1]).
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
        // Trim `drafted - accepted` rejected rows by rolling back `seq_len`
        // (the slots are overwritten on the next propose). Mirrors
        // `DeepseekV4MtpHead::after_verify`.
        let num_drafted = dspark_state.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(num_accepted);
        let old_sl = dspark_state.seq_len;
        if num_to_trim > 0 {
            dspark_state.seq_len = dspark_state.seq_len.saturating_sub(num_to_trim);
        }
        tracing::debug!(
            "V4 DSpark after_verify: accepted={num_accepted} drafted={num_drafted} \
             trim={num_to_trim} dspark_seq_len: {old_sl} → {}",
            dspark_state.seq_len,
        );
        Ok(())
    }

    fn free_state(&self, _gpu: &dyn GpuBackend, state: &mut dyn ProposerState) -> Result<()> {
        let dspark_state = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid V4 DSpark proposer state"))?;
        let mut kv_cache = self.kv_cache.lock();
        for block_table in &mut dspark_state.block_tables {
            if !block_table.is_empty() {
                kv_cache.free_blocks(block_table);
                block_table.clear();
            }
        }
        drop(kv_cache);
        dspark_state.seq_len = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The state alloc/free roundtrip proper needs a GPU (PagedKvCache); the
    // no-GPU seam is the `ProposerState` downcast + the `seq_len` trim math the
    // lifecycle callbacks perform. Build a bare state, box it as
    // `dyn ProposerState`, and exercise both.
    #[test]
    fn dspark_state_downcast_and_trim_roundtrip() {
        let mut state: Box<dyn ProposerState> = Box::new(DeepseekV4DSparkProposerState {
            block_tables: vec![Vec::new(); 3],
            seq_len: 10,
            last_num_drafted: 4,
            body_states: Vec::new(),
        });

        let s = state
            .as_any_mut()
            .downcast_mut::<DeepseekV4DSparkProposerState>()
            .expect("downcast to DSpark state");
        assert_eq!(s.block_tables.len(), 3, "one block table per stage");

        // Mirror `after_verify`: drafted=4, accepted=1 ⇒ trim 3 rows.
        let num_drafted = s.last_num_drafted.max(1);
        let num_to_trim = num_drafted.saturating_sub(1);
        s.seq_len = s.seq_len.saturating_sub(num_to_trim);
        assert_eq!(num_to_trim, 3);
        assert_eq!(s.seq_len, 7, "seq_len rolled back by drafted-accepted");
    }
}
