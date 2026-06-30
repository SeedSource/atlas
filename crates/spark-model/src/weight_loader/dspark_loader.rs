// SPDX-License-Identifier: AGPL-3.0-only

//! DSpark drafter weight loader.
//!
//! Loads `deepseek-ai/dspark_qwen3_*` / our `Qwen3DSparkModel`-style drafter
//! checkpoints into the typed [`DSparkWeights`] structure consumed by
//! [`crate::layers::DSparkDraftHead`]. DSpark = DFlash backbone PLUS three
//! additions (DSpark paper §3, eqs 2–10):
//!
//!  * **Serial Markov head** — a low-rank rank-`markov_rank` bias
//!    `B_k(x_{k-1}, v) = markov_w1[x_{k-1}] · markov_w2[:, v]` re-factorizes the
//!    parallel block draft autoregressively. Tensors `markov_head.markov_w1`,
//!    `markov_head.markov_w2`, both `[vocab, markov_rank]` BF16.
//!  * **Confidence head** — per-position survival prob
//!    `c_k = σ(proj·[h_k ; markov_w1[x_{k-1}]])`. Tensor `confidence_head.proj`
//!    (`weight [1, hidden + markov_rank]`, `bias [1]`).
//!  * **Hardware-aware scheduler** — no weights (CPU greedy + the SPS curve).
//!
//! Shared backbone with DFlash: `fc` (EAGLE3 target-context projection),
//! `hidden_norm`, `norm`, and `len(target_layer_ids)`-deep target-hidden capture.
//!
//! Unlike DFlash, our DSpark drafter checkpoint **ships its own**
//! `embed_tokens` and `lm_head` (frozen copies of the target's). They are loaded
//! when present; absent → the runtime shares the target's at construction
//! (mirrors the DFlash flow).
//!
//! DSpark-specific config lives at the **top level** of the drafter's
//! `config.json` (NOT in a nested object as DFlash uses) — `block_size`,
//! `markov_rank`, `mask_token_id`, `target_layer_ids`, the confidence flags.
//!
//! Under TP the drafter is **not sharded** — it's small (~9 GB BF16 incl. its
//! own embed/lm_head), every rank loads the full set (mirrors DFlash/MTP).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

/// Drafter HF `config.json` (subset Atlas consumes). Field names mirror our
/// `Qwen3DSparkModel` `config.json` verbatim so `serde_json::from_str` works
/// directly on the raw file. DSpark config is FLAT (top-level), not nested.
#[derive(Debug, Clone, Deserialize)]
pub struct DSparkConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Block size γ. Our 27B DSpark draft ships `block_size: 7`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// Low-rank dimension of the Markov bias factorization. `256`.
    #[serde(default = "default_markov_rank")]
    pub markov_rank: usize,
    /// `"vanilla"` (low-rank matmul) vs a GRU variant. We support vanilla.
    #[serde(default)]
    pub markov_head_type: Option<String>,
    /// Whether a trained confidence head is present (`confidence_head.proj`).
    #[serde(default)]
    pub enable_confidence_head: bool,
    /// Whether the confidence head concatenates the Markov lookup with `h_k`
    /// (input dim = `hidden + markov_rank`) vs `hidden` alone.
    #[serde(default)]
    pub confidence_head_with_markov: bool,

    /// Token id used to fill the γ "to-be-predicted" positions. `248077` for
    /// our 27B DSpark draft.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[3, 19, 35, 51, 59]` for our 27B draft. Order matters: shallow-to-deep
    /// concatenation is what `fc` expects.
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,

    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub max_position_embeddings: usize,
    /// Qwen3-next style output gate on attention (our draft: `true`).
    #[serde(default)]
    pub attn_output_gate: bool,
    /// RoPE parameters — kept as raw JSON; the head parses theta/scaling.
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

fn default_block_size() -> usize {
    7
}
fn default_markov_rank() -> usize {
    256
}
fn default_rms_eps() -> f32 {
    1e-6
}

/// Raw weight bundle for the DSpark drafter, post-load.
///
/// Verified against our `Qwen3DSparkModel` 27B draft (`step_1000`, June 2026):
/// 64 BF16 tensors — `fc`, `hidden_norm`, `norm`, `embed_tokens`, `lm_head`,
/// the two `markov_head.markov_w{1,2}`, `confidence_head.proj.{weight,bias}`,
/// plus 11 weights per drafter layer × 5 layers.
#[allow(dead_code)]
pub struct DSparkWeights {
    pub config: DSparkConfig,

    /// `[draft_hidden, len(target_layer_ids) * target_hidden]`.
    /// Our 27B draft: `[5120, 25600]` (5 × 5120).
    pub fc: DenseWeight,
    /// `[draft_hidden]` — RMSNorm on projected target context.
    pub hidden_norm: DenseWeight,
    /// `[draft_hidden]` — final RMSNorm before LM head.
    pub norm: DenseWeight,

    /// Drafter's own embedding / LM head (frozen copies of target's). `Some`
    /// when shipped in the checkpoint (our 27B draft), `None` → share target's.
    pub embed_tokens: Option<DenseWeight>,
    pub lm_head: Option<DenseWeight>,

    pub layers: Vec<DSparkLayerWeights>,

    /// Serial Markov head, both `[vocab, markov_rank]` BF16.
    pub markov_w1: DenseWeight,
    pub markov_w2: DenseWeight,

    /// Confidence head: `proj.weight [1, hidden + markov_rank]`, `proj.bias [1]`.
    /// `Some` iff `enable_confidence_head`.
    pub confidence_proj_weight: Option<DenseWeight>,
    pub confidence_proj_bias: Option<DenseWeight>,
}

/// Per-drafter-layer raw weights (BF16). Same shape across all 5 layers.
/// Note vs DFlash: `head_dim=256`, 24:4 GQA, and `attn_output_gate` — the
/// attention shape differs, but the *tensor set* is the same Qwen3 layout.
#[allow(dead_code)]
pub struct DSparkLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Probe a [`WeightStore`] for DSpark drafter weights. The `markov_head.markov_w1`
/// tensor is unique to DSpark (DFlash has `fc` but no Markov head), so it's the
/// disambiguating detection key. Accepts bare and `model.`-prefixed layouts.
pub fn store_has_dspark_weights(store: &WeightStore) -> bool {
    store.contains("markov_head.markov_w1.weight")
        || store.contains("model.markov_head.markov_w1.weight")
}

/// Parse a DSpark drafter's `config.json` into a [`DSparkConfig`].
pub fn parse_dspark_config(json: &str) -> Result<DSparkConfig> {
    serde_json::from_str(json).context("Parsing DSpark drafter config.json")
}

/// Load DSpark drafter weights from a [`WeightStore`] pointing at the drafter
/// checkpoint. Bare-key layout (no `model.` prefix), Qwen3 naming plus the
/// Markov/confidence heads. `embed_tokens`/`lm_head` loaded when present.
pub fn load_dspark_weights(
    drafter_store: &WeightStore,
    drafter_config: &DSparkConfig,
    _gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DSparkWeights>> {
    if !store_has_dspark_weights(drafter_store) {
        tracing::debug!("DSpark drafter store has no `markov_head.markov_w1` — skipping");
        return Ok(None);
    }

    let prefix = if drafter_store.contains("model.markov_head.markov_w1.weight") {
        "model."
    } else {
        ""
    };

    let fc = dense(drafter_store, &format!("{prefix}fc.weight"))
        .context("DSpark drafter: load fc.weight")?;
    let hidden_norm = dense(drafter_store, &format!("{prefix}hidden_norm.weight"))
        .context("DSpark drafter: load hidden_norm.weight")?;
    let norm = dense(drafter_store, &format!("{prefix}norm.weight"))
        .context("DSpark drafter: load norm.weight")?;

    // Drafter ships its own embed/lm_head (frozen target copies). Optional:
    // absent → share the target's at construction.
    let embed_tokens = if drafter_store.contains(&format!("{prefix}embed_tokens.weight")) {
        Some(dense(drafter_store, &format!("{prefix}embed_tokens.weight"))?)
    } else {
        None
    };
    let lm_head = if drafter_store.contains(&format!("{prefix}lm_head.weight")) {
        Some(dense(drafter_store, &format!("{prefix}lm_head.weight"))?)
    } else {
        None
    };

    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        layers.push(DSparkLayerWeights {
            input_layernorm: dense(drafter_store, &format!("{lp}.input_layernorm.weight"))?,
            post_attention_layernorm: dense(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
            )?,
            q_proj: dense(drafter_store, &format!("{lp}.self_attn.q_proj.weight"))?,
            k_proj: dense(drafter_store, &format!("{lp}.self_attn.k_proj.weight"))?,
            v_proj: dense(drafter_store, &format!("{lp}.self_attn.v_proj.weight"))?,
            o_proj: dense(drafter_store, &format!("{lp}.self_attn.o_proj.weight"))?,
            q_norm: dense(drafter_store, &format!("{lp}.self_attn.q_norm.weight"))?,
            k_norm: dense(drafter_store, &format!("{lp}.self_attn.k_norm.weight"))?,
            gate_proj: dense(drafter_store, &format!("{lp}.mlp.gate_proj.weight"))?,
            up_proj: dense(drafter_store, &format!("{lp}.mlp.up_proj.weight"))?,
            down_proj: dense(drafter_store, &format!("{lp}.mlp.down_proj.weight"))?,
        });
    }

    // The two Markov factors (always present — they're the detection key).
    let markov_w1 = dense(drafter_store, &format!("{prefix}markov_head.markov_w1.weight"))
        .context("DSpark drafter: load markov_head.markov_w1.weight")?;
    let markov_w2 = dense(drafter_store, &format!("{prefix}markov_head.markov_w2.weight"))
        .context("DSpark drafter: load markov_head.markov_w2.weight")?;

    // Confidence head — present iff enabled. Both weight and bias.
    let (confidence_proj_weight, confidence_proj_bias) = if drafter_config.enable_confidence_head
        && drafter_store.contains(&format!("{prefix}confidence_head.proj.weight"))
    {
        (
            Some(dense(
                drafter_store,
                &format!("{prefix}confidence_head.proj.weight"),
            )?),
            // bias [1]; tolerate a bias-less export.
            if drafter_store.contains(&format!("{prefix}confidence_head.proj.bias")) {
                Some(dense(
                    drafter_store,
                    &format!("{prefix}confidence_head.proj.bias"),
                )?)
            } else {
                None
            },
        )
    } else {
        (None, None)
    };

    tracing::info!(
        "DSpark drafter loaded: {} layers, hidden={}, vocab={}, γ={}, markov_rank={}, conf_head={}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.block_size,
        drafter_config.markov_rank,
        confidence_proj_weight.is_some(),
        drafter_config.target_layer_ids,
    );

    Ok(Some(DSparkWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        embed_tokens,
        lm_head,
        layers,
        markov_w1,
        markov_w2,
        confidence_proj_weight,
        confidence_proj_bias,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse our 27B DSpark drafter `config.json` from the local run dir.
    /// Skipped when absent — keeps CI hermetic. Asserts the locked dims:
    /// 5 layers, hidden=5120, head_dim=256, 24:4 GQA, γ=7, markov_rank=256,
    /// mask=248077, target_layers=[3,19,35,51,59], confidence head on.
    #[test]
    fn parse_qwen3_27b_dspark_config() {
        const SNAP: &str =
            "/home/msi1/runs/sm121-acceleration/h200-draft-27b-full/step_1000/config.json";
        let json = match std::fs::read_to_string(SNAP) {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Skipping: DSpark drafter config not present");
                return;
            }
        };
        let c = parse_dspark_config(&json).expect("parse DSpark drafter config");
        assert_eq!(c.num_hidden_layers, 5);
        assert_eq!(c.hidden_size, 5120);
        assert_eq!(c.intermediate_size, 17408);
        assert_eq!(c.num_attention_heads, 24);
        assert_eq!(c.num_key_value_heads, 4);
        assert_eq!(c.head_dim, 256);
        assert_eq!(c.vocab_size, 248320);
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.block_size, 7);
        assert_eq!(c.markov_rank, 256);
        assert!(c.enable_confidence_head);
        assert!(c.confidence_head_with_markov);
        assert_eq!(c.mask_token_id, 248077);
        assert_eq!(c.target_layer_ids, vec![3, 19, 35, 51, 59]);
        assert!(c.attn_output_gate);
    }
}
