// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash **native DSpark** drafter checkpoint loader.
//!
//! `deepseek-ai/DeepSeek-V4-Flash-DSpark` ships a native DSpark drafter as
//! **three** sequential draft stages under the `mtp.0/1/2.*` prefixes (config
//! `n_mtp_layers = 3`). This is distinct from the NVIDIA-style single-module
//! MTP head (`nvidia/DeepSeek-V4-Flash-NVFP4`, one `mtp.0` combiner with
//! `enorm`/`hnorm`/`e_proj`/`h_proj`) handled by [`super::mtp`]. The two are
//! disambiguated at load time by the **detection key**:
//!
//!   * native DSpark → `mtp.0.main_proj.weight` present
//!   * NVIDIA-style MTP → `mtp.0.enorm.weight` present
//!
//! Each of the 3 stages is structurally a main V4 layer — MLA attention,
//! manifold-constrained hyper-connections (mHC), and a 256-expert MXFP4/E8M0
//! MoE, full attention (no compressor) — reusing [`super::assemble::assemble_layer`]
//! with the `mtp.{i}` prefix and `layer_idx = num_hidden_layers` (interior /
//! no-compressor path), exactly as [`super::mtp`] builds its single stage.
//! HIGH#3 (audit 2026-07-22): the routed experts reuse the target model's own
//! native-MXFP4 (E8M0) path transcode-free — no new kernel/quant lane.
//!
//! On top of the 3 stages sit the DSpark-specific heads:
//!
//!   * Stage 0 only: `mtp.0.main_proj` (`[4096,12288]` FP8) + `mtp.0.main_norm`
//!     — the block-input projection that folds the target hidden + accepted
//!     tokens into the drafter's first stage input.
//!   * Stage 2 only (last stage): `mtp.2.norm` (final RMSNorm),
//!     `mtp.2.markov_head.markov_w{1,2}` (`[vocab,markov_rank]` BF16 — the serial
//!     Markov token head), `mtp.2.confidence_head.proj` (`[1,4352]` — the
//!     accept/reject confidence head), and the stage's own head hyper-connection
//!     `mtp.2.hc_head_*` (only the last stage collapses `hc_streams → h_out`).
//!
//! The token embedding and lm_head are shared with the parent model and supplied
//! to the (forthcoming) proposer at build time (not duplicated under `mtp.*`).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::HcHeadWeights;
use crate::weight_map::{DenseWeight, dense_auto};

/// Number of native DSpark draft stages (`n_mtp_layers = 3`).
const NUM_DSPARK_STAGES: usize = 3;

/// A loaded DeepSeek-V4 **native DSpark** drafter: the 3 reused V4 transformer
/// stages plus the DSpark-specific input projection (`main_proj`/`main_norm`),
/// final norm, serial Markov head (`markov_w1`/`markov_w2`), and confidence head.
/// Embedding + lm_head are shared with the parent model and supplied at
/// proposer-build time.
//
// Consumed by the native-DSpark K=1 proposer (`crate::layers::DeepseekV4DSparkHead`).
// The struct-level `allow(dead_code)` stays because the head fields
// (`main_proj`/`main_norm`/`norm`/`markov_*`/`confidence_proj`/`hc_head`) are
// first READ by the Commit 4 forward; the Commit 3 lifecycle reads `stages`
// only. Mirrors `super::mtp::DeepseekV4MtpModule`.
#[allow(dead_code)]
pub struct DeepseekV4DSparkModule {
    /// The 3 reused V4 layer bodies (MLA + mHC + MoE), built from the
    /// `mtp.0`/`mtp.1`/`mtp.2` prefixes. Chained in draft order.
    pub stages: Vec<Box<dyn TransformerLayer>>,
    /// Stage-0 input projection `mtp.0.main_proj` (`[hidden, 3*hidden]` FP8).
    pub main_proj: DenseWeight,
    /// RMSNorm applied to the stage-0 projected input (`mtp.0.main_norm`).
    pub main_norm: DenseWeight,
    /// Final RMSNorm applied before the shared lm_head / heads (`mtp.2.norm`).
    pub norm: DenseWeight,
    /// Serial Markov token head, factor 1 (`mtp.2.markov_head.markov_w1`,
    /// `[vocab, markov_rank]` BF16).
    pub markov_w1: DenseWeight,
    /// Serial Markov token head, factor 2 (`mtp.2.markov_head.markov_w2`,
    /// `[vocab, markov_rank]` BF16).
    pub markov_w2: DenseWeight,
    /// Confidence (accept/reject) head projection (`mtp.2.confidence_head.proj`,
    /// `[1, 4352]`).
    pub confidence_proj: DenseWeight,
    /// The last stage's OWN head hyper-connection (`mtp.2.hc_head_*`). Only the
    /// final stage collapses `hc_streams → h_out`; stages 0/1 run the MIDDLE mHC
    /// mixing only (their bodies were built with `hc_head = None`). `None` when
    /// `hc_mult == 0` (no mHC). Surfaced here for the proposer to collapse the
    /// streams after `stages[2].decode`, mirroring `super::mtp`.
    pub hc_head: Option<HcHeadWeights>,
}

/// Detection + gating predicate for the native DSpark drafter. The loader
/// engages only when MTP is enabled (`num_mtp_modules != 0`) AND the checkpoint
/// ships the native `mtp.0.main_proj` tensor. Factored out so the gate is
/// unit-testable without a GPU / `WeightStore`.
fn dspark_present(num_mtp_modules: usize, has_main_proj: bool) -> bool {
    num_mtp_modules != 0 && has_main_proj
}

/// Loads the DeepSeek-V4 **native DSpark** drafter if the checkpoint contains it.
///
/// Returns `Ok(None)` (no-op) when MTP is disabled (`num_mtp_modules == 0`) or
/// the checkpoint ships no native DSpark tensors (detection key
/// `mtp.0.main_proj.weight` absent — e.g. the NVIDIA-style single-module MTP
/// checkpoint, which `super::mtp` handles instead). Safe to call unconditionally
/// on the V4 load path.
//
// Wired into the V4 load path (`factory::build`) and consumed by the
// native-DSpark K=1 proposer (`crate::layers::DeepseekV4DSparkHead`), mirroring
// `super::mtp::load_v4_mtp_module`.
pub fn load_v4_dspark_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Option<DeepseekV4DSparkModule>> {
    let has_main_proj = store.contains("mtp.0.main_proj.weight");
    if !dspark_present(config.num_mtp_modules, has_main_proj) {
        if config.num_mtp_modules != 0 {
            tracing::info!(
                "DeepSeek-V4: num_mtp_modules={} but no mtp.0.main_proj tensor — \
                 native DSpark drafter disabled (may be NVIDIA-style MTP instead)",
                config.num_mtp_modules
            );
        }
        return Ok(None);
    }

    let mut yarn_inv_freq = DevicePtr::NULL;
    yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

    // ── Build the 3 stage bodies (MLA + mHC + 256-expert MoE) ──
    // Each stage reuses `assemble_layer` with the `mtp.{i}` prefix, exactly as
    // `super::mtp` builds its single stage. `layer_idx = num_hidden_layers` makes
    // compress_ratios / hash-layer / kv-dtype fall to safe defaults (no
    // compressor, no hash routing, bf16 KV). `force_all_experts = true` because
    // the drafter runs no-EP on rank 0 and needs every expert present.
    let mut stages: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(NUM_DSPARK_STAGES);
    let mut last_hc_head: Option<HcHeadWeights> = None;

    for i in 0..NUM_DSPARK_STAGES {
        let prefix = format!("mtp.{i}");
        let ap = format!("{prefix}.attn");
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };

        // ── Body attn pre-loads (V4-Flash: direct KV + grouped low-rank O) ──
        let input_norm = dense_auto(store, &format!("{prefix}.attn_norm.weight"), gpu)?;
        let post_attn_norm = dense_auto(store, &format!("{prefix}.ffn_norm.weight"), gpu)?;
        let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
        let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
        let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;
        let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
        let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;
        let wo_a = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
        let wo_b = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;

        // Head hyper-connection: the native DSpark checkpoint ships `hc_head_*`
        // ONLY on the LAST stage (`mtp.2`), which collapses `hc_streams → h_out`.
        // Stages 0/1 carry no head HC — their bodies run middle-mixing only.
        let is_last = i == NUM_DSPARK_STAGES - 1;
        let hc_head = if config.hc_mult > 0 && is_last {
            let hc = config.hc_mult;
            let hc_dim = hc * config.hidden_size;
            let head_fn = super::assemble::load_hc_f32(
                store,
                &[format!("{prefix}.hc_head_fn")],
                hc * hc_dim,
                gpu,
            )?;
            let head_base =
                super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_base")], hc, gpu)?;
            let head_scale =
                super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_scale")], 1, gpu)?;
            Some(HcHeadWeights {
                hc_fn: head_fn,
                hc_base: head_base,
                hc_scale: head_scale,
            })
        } else {
            None
        };

        let body = super::assemble::assemble_layer(
            config.num_hidden_layers,
            &prefix,
            true, // force_all_experts — drafter runs no-EP on rank 0
            input_norm,
            post_attn_norm,
            wq_a,
            None, // wq_a_nvfp4 — V4 attn is FP8/BF16, not NVFP4
            wq_b,
            None, // wq_b_nvfp4
            q_a_norm,
            wkv_a,
            None, // wkv_a_nvfp4
            null, // wkv_b — unused for V4-Flash
            kv_a_norm,
            wo_b, // o_dense
            None, // o_nvfp4
            null, // w_uk_t
            null, // w_uv
            null, // wq_b_rope
            null, // w_qk_absorbed
            null, // w_uk_block_diag
            null, // w_uv_block_diag
            yarn_inv_freq,
            wo_a,
            hc_head.clone(),
            store,
            config,
            gpu,
            layer_kv_dtypes,
        )?;
        stages.push(body);
        if is_last {
            last_hc_head = hc_head;
        }
    }

    // ── DSpark-specific heads ──
    // main_proj/main_norm live on stage 0 (block-input projection); norm/markov/
    // confidence live on the last stage (stage 2). dense_auto dequants the FP8
    // main_proj; the BF16 norms + markov/confidence load exactly.
    let main_proj = dense_auto(store, "mtp.0.main_proj.weight", gpu)?;
    let main_norm = dense_auto(store, "mtp.0.main_norm.weight", gpu)?;
    let norm = dense_auto(store, "mtp.2.norm.weight", gpu)?;
    let markov_w1 = dense_auto(store, "mtp.2.markov_head.markov_w1.weight", gpu)?;
    let markov_w2 = dense_auto(store, "mtp.2.markov_head.markov_w2.weight", gpu)?;
    let confidence_proj = dense_auto(store, "mtp.2.confidence_head.proj.weight", gpu)?;

    tracing::info!(
        "DeepSeek-V4 native DSpark drafter loaded: {} stages (MLA + mHC + 256-expert MoE) \
         + main_proj/main_norm (stage 0) + final norm + serial Markov head + confidence head",
        NUM_DSPARK_STAGES
    );

    Ok(Some(DeepseekV4DSparkModule {
        stages,
        main_proj,
        main_norm,
        norm,
        markov_w1,
        markov_w2,
        confidence_proj,
        hc_head: last_hc_head,
    }))
}

#[cfg(test)]
mod tests {
    use super::dspark_present;

    // Detection key: native DSpark loads iff MTP is enabled AND the checkpoint
    // ships `mtp.0.main_proj` (distinct from the NVIDIA-style `mtp.0.enorm`).
    #[test]
    fn detection_key_requires_mtp_enabled_and_main_proj() {
        // main_proj present + MTP enabled => load the native DSpark drafter.
        assert!(dspark_present(1, true), "num_mtp=1, main_proj present => load");
        assert!(dspark_present(3, true), "num_mtp=3, main_proj present => load");
        // main_proj absent => skip (NVIDIA-style MTP or no drafter).
        assert!(
            !dspark_present(1, false),
            "num_mtp=1 but no main_proj => skip (NVIDIA-style MTP)"
        );
        // MTP disabled => skip regardless of tensor presence.
        assert!(!dspark_present(0, true), "num_mtp=0 => skip even if main_proj present");
        assert!(!dspark_present(0, false), "num_mtp=0, no main_proj => skip");
    }
}
