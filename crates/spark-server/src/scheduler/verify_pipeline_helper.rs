// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-time pre-sample LogitsProcessor pipeline (Phase C-2 wiring).
//!
//! The MTP / speculative-decode verify paths used to consume the raw
//! GPU `argmax_bf16` ID at every verify position, completely bypassing
//! the 8-stage [`crate::scheduler::logit_processors`] pipeline that the
//! non-MTP path runs on every sampled token. Result: tokens emitted
//! through verify (the dominant decode path when MTP is enabled —
//! every accepted/bonus token came from `decode_verify_graphed`) never
//! saw mid-word `</think>` defer, post-close think mask, tool-during-
//! think mask, forced think-end injection, pin-to-tool-call, forced-
//! token fast-path, or grammar bitmask. This is the root cause of
//! grammar desync, malformed tool calls, mid-word `</think>` cuts and
//! stray `<think>` re-entry observed on Qwen3.6-FP8 (opencode-session
//! transcripts, 2026-05-24).
//!
//! This module replays the same dequant + pipeline on a host-side copy
//! of the verify logits buffer (`[K, vocab]` BF16, written by
//! `decode_verify_graphed_*` into `model.logits_buffer_ptr()`), then
//! picks the resulting argmax. Cost: ~0.8 ms per verify position for a
//! ~256k vocab on host, mirroring the non-MTP `process_seq_logits` path
//! in `decode_logits_seq.rs`. The CUDA-graphed `argmax_bf16` saving of
//! ~0.5 ms/step is preserved for the **draft** path (drafts already go
//! through a separate grammar-bitmask path in MTP propose); only the
//! **verify-time** argmax is replaced.
//!
//! Per-position semantics: the pipeline is applied independently to
//! each verify position 0..K. For position 0 the `ActiveSeq` state is
//! exactly the post-`last_token` state, identical to the non-MTP
//! decode site. For positions ≥ 1, the driver SPECULATIVELY ADVANCES
//! the xgrammar matcher via `gs.accept_token(pick_{i-1})` between
//! positions, so each position's bitmask reflects the matcher state
//! that will actually exist at `emit_token` time on the accept path.
//! Speculative advances are rolled back via `gs.rollback(n)` once all
//! K positions have been picked; the real `emit_token` calls then
//! re-advance the matcher normally for the verified tokens that
//! actually get emitted.
//!
//! **DO NOT remove the speculative advance.** Prior versions emitted
//! position-1 argmax against position-0 bitmask, which desynced
//! xgrammar on the accept path and tripped the non-silent
//! `accept_token` kill switch (observed live on
//! opencode-realfix.jsonl 2026-05-24: every response ended with
//! `length` + `tok=198 output_len=30-60` because the bonus token was
//! masked at position 0's state — a `\n` legal at JSON-value-start
//! is not legal at JSON-comma-or-closebrace).
//!
//! Other state-dependent masks (mid-word lookback, last_token reads)
//! still see slightly stale `output_tokens` for positions ≥ 1 —
//! best-effort, mirrors greedy unroll.

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::bf16_to_f32;
use crate::scheduler::logit_processors::{LogitsContext, run_pipeline};
use spark_model::traits::Model;

/// Per-position verify logits, dequantised + processed through the full
/// pre-sample pipeline. Returns the chosen token: either the forced
/// token from a [`crate::scheduler::logit_processors::forced_token::ForcedTokenFastPath`]
/// short-circuit, or the post-pipeline argmax.
///
/// `logits_bytes`: byte slice for ONE verify position; length
/// `vocab_size * 2` (BF16) or `vocab_size * 4` (FP32).
/// `is_fp32`: true when the model emits FP32 logits (Gemma-4 dense).
/// `a`: the active sequence; the pipeline mutates seq state in place
/// (F2 confidence arm, sentence_defer_count, etc.).
/// `ctx`: tokenizer special-token IDs used by the pipeline.
///
/// Mirrors the host-side path of `decode_logits_seq::process_seq_logits`
/// for byte-identical pipeline semantics.
pub fn verify_pick_with_pipeline(
    logits_bytes: &[u8],
    is_fp32: bool,
    vocab_size: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> u32 {
    // 1. Dequant per the same scheme as `process_seq_logits`.
    let mut f32_logits: Vec<f32> = if is_fp32 {
        (0..vocab_size)
            .map(|j| {
                let off = j * 4;
                f32::from_le_bytes([
                    logits_bytes[off],
                    logits_bytes[off + 1],
                    logits_bytes[off + 2],
                    logits_bytes[off + 3],
                ])
            })
            .collect()
    } else {
        (0..vocab_size)
            .map(|j| {
                let lo = logits_bytes[j * 2];
                let hi = logits_bytes[j * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };

    // 2. Run the canonical pipeline. Short-circuit returns the forced
    //    token directly — no argmax scan needed.
    if let Some(forced) = run_pipeline(&mut f32_logits, a, ctx) {
        return forced;
    }

    // 3. Argmax over the (now-masked) vector. `f32::partial_cmp` with
    //    NaN-safe fallback to `Equal` matches the sampler's argmax
    //    branch behaviour.
    let mut best_id: u32 = 0;
    let mut best_val: f32 = f32::NEG_INFINITY;
    for (i, &v) in f32_logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}

/// Convenience: copy the full `[K, vocab]` verify logits buffer to
/// host and apply [`verify_pick_with_pipeline`] to every position,
/// returning the K processed token IDs. Falls back to the raw argmax
/// IDs if the D2H copy fails (matches `verify_resample` and
/// `extract_verify_logprobs` failure semantics).
///
/// `argmax_ids` is the GPU-graphed argmax already returned by
/// `decode_verify_graphed*`; used as the fallback for the failure
/// path and as the array length source.
pub fn verify_pick_all_with_pipeline(
    model: &dyn Model,
    argmax_ids: &[u32],
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let k = argmax_ids.len();
    if k == 0 {
        return Vec::new();
    }
    let vocab = model.vocab_size();
    // BF16 always for verify path: `decode_verify_graphed_*` writes BF16
    // to `logits_buffer()`. The FP32-lm_head path (Gemma-4 dense) does
    // not go through verify (no MTP for dense Gemma).
    let elem_bytes = 2usize;
    let total = k * vocab * elem_bytes;
    let mut buf = vec![0u8; total];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return argmax_ids.to_vec();
    }

    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // Speculative-advance counter — we rollback this many tokens off
    // the xgrammar matcher at the end so the real `emit_token` calls
    // (which run after this helper returns) re-advance with the
    // verified-or-rejected tokens from a clean state.
    let mut grammar_advances: usize = 0;

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx);
        picks.push(pick);

        // Speculatively advance the matcher with `pick[i]` so the next
        // position's bitmask reflects post-emit state. Skip on the last
        // position (no next position to mask) and when the seq has no
        // grammar (nothing to advance).
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // Matcher advance can fail if `pick` is not in the current
            // bitmask. If our pipeline correctly applied the bitmask,
            // pick is the argmax over masked logits → MUST be in the
            // bitmask → advance MUST succeed. The defensive check
            // exists for forced-token fast-path returns where the
            // grammar may have terminated; those legitimately can't
            // advance further.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
            grammar_advances += 1;
        }
    }

    // Roll back all speculative advances so the matcher returns to its
    // pre-call state. `emit_token` will then re-advance it normally for
    // the tokens that actually get accepted by the scheduler.
    if grammar_advances > 0
        && let Some(ref mut gs) = a.grammar_state
    {
        gs.rollback(grammar_advances);
    }

    picks
}
