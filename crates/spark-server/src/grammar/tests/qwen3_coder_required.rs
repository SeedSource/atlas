// SPDX-License-Identifier: AGPL-3.0-only

//! Regression tests for the qwen3_coder grammar's envelope shape.
//!
//! As of the body-format fix (2026-05-25): the grammar uses
//! `any_text` for the body inside `<tool_call>\n<function=NAME>\n…
//! \n</function>\n</tool_call>` to match the model's native XML
//! `<parameter=KEY>VALUE</parameter>` wire format. The body is
//! intentionally unconstrained at the grammar level — required-
//! parameter enforcement now happens host-side via
//! `validate_single_tool_call` and `backfill_required_params`
//! after `parse_one_call` extracts the XML/JSON args. See
//! `compile_tools.rs::compile_qwen3_coder_tool_grammar` and
//! `tool_handlers.rs:46` for the layered validation path.
//!
//! These tests pin the **envelope-shape** properties:
//! - Canonical bodies (XML or JSON) are ACCEPTED.
//! - Malformed envelopes (missing open/close tag) are REJECTED.
//!
//! The previous tests in this file asserted that the grammar
//! itself rejected empty-body tool calls — a property of the
//! prior `json_schema` body content type. Required-field
//! rejection is now the validator's responsibility, covered by
//! validator-side tests in `tool_parser/tests/`.

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn exec_tool_def() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "exec".to_string(),
            description: Some("Run a shell command".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            })),
        },
    }]
}

/// Mirror of `tests/minimax.rs::grammar_accepts` — fresh matcher,
/// feed bytes, accept iff every byte parses AND the grammar reaches
/// an accepting (terminated) state.
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

/// Positive baseline: the grammar must accept the canonical native
/// qwen3_coder XML body — the format the model is actually trained
/// to emit. Pins the wire-format envelope so a future grammar
/// rework cannot regress to forcing JSON-shape output (the exact
/// regression that caused interior-byte corruption and JSON
/// delimiter cascades against opencode multi-turn sessions in
/// 2026-05-25 sessions).
#[test]
fn qwen3_coder_grammar_accepts_canonical_xml_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let canonical_xml =
        "<tool_call>\n<function=exec>\n<parameter=command>ls /tmp</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, canonical_xml),
        "canonical native-XML qwen3_coder body must be accepted; input: {canonical_xml:?}"
    );
}

/// The legacy JSON body shape is still accepted by `any_text` — the
/// parser's JSON fallback at `parse_single_b.rs:137-148` keeps
/// JSON-shaped tool calls working. Pins that the grammar loosening
/// did not narrow either supported shape.
#[test]
fn qwen3_coder_grammar_accepts_legacy_json_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let json_body =
        "<tool_call>\n<function=exec>\n{\"command\": \"ls /tmp\"}\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, json_body),
        "legacy JSON-shaped body must still be accepted; input: {json_body:?}"
    );
}

/// Multi-parameter native-XML body — pins that consecutive
/// `<parameter=KEY>VALUE</parameter>` blocks are accepted without
/// the FSM clipping the closing `</parameter>` boundary (the exact
/// pruning failure that the 2026-05-23 sweep originally tried to
/// dodge by switching to JSON body).
#[test]
fn qwen3_coder_grammar_accepts_multi_xml_params() {
    let tools = vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "write".to_string(),
            description: Some("Write to a file".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filePath", "content"]
            })),
        },
    }];

    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let multi_param = "<tool_call>\n<function=write>\n<parameter=filePath>/tmp/test-rust-axum-v42/Cargo.toml</parameter>\n<parameter=content>[package]\nname = \"test-rust-axum-v42\"</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, multi_param),
        "multi-param native XML body must be accepted with full byte fidelity \
         (path tokens like `axum-v42` and content tokens with newlines/quotes). \
         Input: {multi_param:?}"
    );
}

/// Tier-0 non-empty enforcement (2026-05-25 evening): the qwen3_coder
/// grammar's regex content type must REJECT a parameter body that is
/// empty or whitespace-only. This is the Atlas-internal version of
/// llama.cpp#20164's "empty-parameter under long context" failure mode.
/// Without this, the model's in-tool sampler (which intentionally zeros
/// rep/DRY/freq/presence penalties because XGrammar usually shapes the
/// output) can pick `</parameter>` as its very next token after the
/// opener — burning opencode tool-call turns on empty bash commands and
/// empty file paths.
#[test]
fn qwen3_coder_grammar_rejects_empty_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let empty_body =
        "<tool_call>\n<function=exec>\n<parameter=command></parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, empty_body),
        "empty parameter body must be REJECTED by Tier-0 regex. Input: {empty_body:?}"
    );
}

#[test]
fn qwen3_coder_grammar_rejects_whitespace_only_parameter_body() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = exec_tool_def();
    let compiled = engine
        .compile_qwen3_coder_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let whitespace_body =
        "<tool_call>\n<function=exec>\n<parameter=command>   \n  </parameter>\n</function>\n</tool_call>";
    assert!(
        !grammar_accepts(&compiled, whitespace_body),
        "whitespace-only parameter body must be REJECTED. Input: {whitespace_body:?}"
    );
}
