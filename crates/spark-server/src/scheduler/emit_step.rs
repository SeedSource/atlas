// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    // Cooperative cancellation from the streaming pipeline. The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup,
    // F44 perm-fail, loop-watchdog) flip this flag when they decide
    // the response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the
    // model keep emitting tokens that just get suppressed.
    if let Some(ref f) = a.cancel_flag
        && f.load(std::sync::atomic::Ordering::Acquire)
    {
        a.finished = true;
        return;
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if let Some(ims) = im_start_hard_stop()
        && tok == ims
    {
        // Push the hard-stop token to output_tokens so lifecycle.rs reports
        // `finish_reason="stop"` (because `<|im_start|>` is registered in
        // `eos_tokens` at startup — see tokenizer_runtime.rs::im_start_id).
        // Without this push, `last_tok = output_tokens.last()` is the prior
        // content token, lifecycle's `is_eos` check fails, and the response
        // is mis-reported as `finish_reason="length"` (Bug 3 from OpenClaw
        // 2026-05-08 session: "Done: 13 tokens (length) despite max_tokens=
        // 8192" — clients then misinterpret the truncation as a real
        // length-limit hit and either retry or surface a wrong error).
        // The streamed-text path strips stop tokens server-side, so the
        // client never sees the literal `<|im_start|>` bytes.
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = Some(a.spontaneous_think_budget);
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        return; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Track CURRENT tool-body phase (P3.1, 2026-04-25). Set on the
    // open token, clear on the close. The flag drives sampler
    // scoping: when true, the main decode path zeroes
    // repetition/presence/frequency/DRY so legitimate JSON
    // micro-repetition (`":"`, `","`, key names) is not penalised.
    //
    // Stuck-in-tool-body watchdog (2026-05-24, NVFP4 doom-loop fix):
    // some models emit `<tool_call>` opener and never reach the close
    // — burns to max_tokens with the sanitizer suppressing as orphan.
    // `tool_body_streak_tokens` counts consecutive tokens spent inside
    // a tool body; when it exceeds `MAX_TOOL_BODY_TOKENS` (1024) we
    // end the response cleanly. Resets to 0 on close.
    const MAX_TOOL_BODY_TOKENS: u32 = 1024;
    if !a.inside_thinking {
        if a.tool_call_start_token == Some(tok) {
            a.inside_tool_body = true;
            a.tool_body_streak_tokens = 0;
        } else if a.tool_call_end_token == Some(tok) {
            a.inside_tool_body = false;
            a.tool_body_streak_tokens = 0;
            a.inside_parameter_body = false;
            a.param_body_chars_emitted = 0;
        } else if a.inside_tool_body {
            a.tool_body_streak_tokens = a.tool_body_streak_tokens.saturating_add(1);
            if a.tool_body_streak_tokens > MAX_TOOL_BODY_TOKENS {
                tracing::warn!(
                    streak = a.tool_body_streak_tokens,
                    "Stuck in tool body for {MAX_TOOL_BODY_TOKENS}+ tokens with no </tool_call>; ending response (model never closed the envelope — would otherwise burn to max_tokens). Sanitizer will salvage what it can."
                );
                a.finished = true;
            }

            // Tier-1 (Epoch 1) parameter-body state tracking. Token IDs
            // for Qwen3.6 byte-level BPE tokenizer (verified live via
            // /tokenize endpoint 2026-05-25):
            //   `<`         = 27
            //   `parameter` = 15704
            //   `=`         = 28
            //   `>`         = 29
            //   `</`        = 510  ← masked while body is empty
            // Opener `<parameter=KEY>` ends at `>` (29). The KEY span
            // between `=` and `>` is variable (typically 1-3 tokens
            // like `command`, `filePath`, `content`). To robustly
            // detect the opener, look back through the last 8 tokens
            // when `>` is emitted and check for the `<parameter=`
            // signature [27, 15704, 28] without an intervening
            // `</parameter>` close.
            //
            // Closer `</parameter>` starts at `</` (510). When emitted
            // inside the body, clear the flag.
            //
            // While `inside_parameter_body && param_body_chars_emitted
            // == 0`, decode_logits_seq.rs masks token 510 with bias
            // -8.0 — forcing the model to emit at least one non-close
            // token before the close-tag's first byte can be sampled.
            const TOK_LT: u32 = 27;
            const TOK_PARAMETER: u32 = 15704;
            const TOK_EQ: u32 = 28;
            const TOK_GT: u32 = 29;
            const TOK_LT_SLASH: u32 = 510;
            if a.inside_parameter_body {
                if tok == TOK_LT_SLASH {
                    // Start of `</parameter>` close-tag — exit body.
                    a.inside_parameter_body = false;
                    a.param_body_chars_emitted = 0;
                } else {
                    // Epoch-2 fix: don't count whitespace-only tokens
                    // toward the chars counter (otherwise the model
                    // can emit ` `/`\n`/etc to satisfy the gate while
                    // the parser's `.trim()` strips it to empty).
                    // Common Qwen3.6 whitespace tokens (verified via
                    // /tokenize endpoint): 220, 198, 197, 256, 271.
                    let is_whitespace_token = matches!(
                        tok,
                        220 | 198 | 197 | 256 | 271
                    );
                    if !is_whitespace_token {
                        a.param_body_chars_emitted =
                            a.param_body_chars_emitted.saturating_add(1);
                    }
                }
            } else if tok == TOK_GT {
                // Possible end of `<parameter=KEY>` opener. Look back
                // through last 8 tokens for the [27, 15704, 28]
                // signature WITHOUT an intervening 510 (`</`) or 29
                // (`>` — would be a previous param close).
                let n = a.output_tokens.len();
                if n >= 4 {
                    let start = n.saturating_sub(8);
                    let window = &a.output_tokens[start..n];
                    // Find the signature; skip if 510 appears between
                    // the signature and the current `>` (would mean
                    // the parameter already closed before this `>`).
                    let mut sig_idx: Option<usize> = None;
                    for i in 0..window.len().saturating_sub(2) {
                        if window[i] == TOK_LT
                            && window[i + 1] == TOK_PARAMETER
                            && window[i + 2] == TOK_EQ
                        {
                            sig_idx = Some(i + 3);
                        }
                    }
                    if let Some(after_eq) = sig_idx {
                        // Check no 510 / 29 between after_eq and the
                        // end of the window (the `>` is the LAST
                        // emitted token, but it's already in
                        // output_tokens at this point? Actually emit
                        // happens here so check window[after_eq..]).
                        let body_segment = &window[after_eq..];
                        let intervening_close = body_segment.iter().any(|&t| {
                            t == TOK_LT_SLASH || t == TOK_GT
                        });
                        if !intervening_close {
                            a.inside_parameter_body = true;
                            a.param_body_chars_emitted = 0;
                        }
                    }
                }
            }
        }
    }

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced {
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — xgrammar NPDA refused the emitted token; matcher is now desynced from the stream. Ending response to prevent cascading grammar-mask corruption."
            );
            a.finished = true;
        }
    }

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    a.output_tokens.push(tok);

    // Thinking tokens are "free" (don't decrement remaining).
    // Detect </think> transition. Track thinking token count for budget enforcement.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.think_ended = true;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring to next sentence boundary"
                );
            }
        }
    } else {
        a.remaining -= 1;
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
        // Content-phase loop watchdog. Mirrored from
        // `handle_content_token` (decode_logits_content.rs) because
        // that handler is only invoked on the non-MTP decode path
        // (`process_decode_logits`). MTP speculative decode
        // (`verify_k2_step`) reaches every token through this
        // `emit_token` instead — without this mirror, the
        // content-loop watchdog never fires while MTP is enabled, and
        // the model can burn the full `max_tokens` budget on a
        // period-N attractor. Observed live 2026-05-24 on
        // opencode-hotfix2b.jsonl seq=13: 8193 content tokens of
        // `[29, 198, 510, 15704, …]` period-4 loop (the
        // `parameter>\n` attractor) with no watchdog fire,
        // finish=length.
        //
        // 2026-05-24 sweep #3: Re-introduced the `!a.inside_tool_body`
        // gate (mirrors the handle_content_token policy). The previous
        // inside-body false-positives turned out to be triggered by a
        // separate MTP-pipeline gap (see bench/hotfix3-debug/
        // SYNTHESIS.md). With the pipeline correctly applied to MTP
        // verify, JSON structural repetition is bounded by the
        // grammar's terminal state. The `parameter>\n` real-loop case
        // is still caught one tick AFTER the model exits the tool
        // body — its emission outside the body forms a tight period-N
        // tail.
        //
        // Skip rollback here — `emit_token` doesn't take `&dyn Model`
        // (the SSM rewind requires it) and plumbing it through every
        // call site would balloon the diff. Instead set `a.finished`
        // and let the lifecycle close the response. The non-MTP path
        // retains rollback via `handle_content_token`.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_with,
            detect_content_token_loop_normalized_with, disable_watchdogs, enable_loop_watchdog,
            numeric_token_mask,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        if !disable_watchdogs()
            && enable_loop_watchdog()
            && !a.inside_tool_body
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop_with(&a.output_tokens, a.repetition_detection)
                || numeric_token_mask().as_deref().is_some_and(|m| {
                    detect_content_token_loop_normalized_with(
                        &a.output_tokens,
                        m,
                        a.repetition_detection,
                    )
                }))
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            a.finished = true;
        }
    }

    // EOS handling: grammar-based, legacy, or min_tokens suppression.
    let grammar_suppresses_eos = a
        .grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated());
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    let suppress_eos = grammar_suppresses_eos || legacy_suppresses_eos || min_tokens_suppresses;

    if a.eos_tokens.contains(&tok) && !suppress_eos {
        a.finished = true;
        return;
    }
    if a.eos_tokens.contains(&tok) && suppress_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // or min_tokens not reached. Don't stop — let the model continue generating.
        return;
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    Some(state)
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}
