// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end Qwen3.5-4B-MLX-8bit inference on the metal backend.
//!
//! Tokenize a prompt → embed → run all 32 layers → final RMSNorm →
//! LM head (tied to embed_tokens) → argmax → decode → print.
//!
//! ⚠️  Linear-attention layers are currently identity-passthrough.
//! The model is hybrid (8 full_attention + 24 linear_attention via
//! GDN). The full_attention path is the parity-tested kernel chain
//! used by `metal_real_model_full_attention_block_layer3`. The
//! linear_attention path needs the GDN orchestration around the
//! existing `gated_delta_rule_decode` + `causal_conv1d_decode`
//! kernels — that's a follow-on. With identity passthrough, the
//! generated token won't match what the real model would produce
//! (75 % of the model's contribution is bypassed) — but every
//! other piece of the inference pipeline (tokenizer integration,
//! per-token KV-cache building, multi-layer chain, LM head, sampler)
//! exercises end-to-end on real Qwen3.5-4B-MLX-8bit weights.
//!
//! Run with:
//!
//!     ATLAS_TARGET_HW=metal \
//!     ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
//!     ATLAS_TARGET_QUANT=mlx_int8 \
//!     PROMPT="What is the capital of France?" \
//!     cargo run --release --example metal_qwen35_inference \
//!         --features metal --no-default-features

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::MlxInt8Weight;
use std::time::Instant;
use tokenizers::Tokenizer;

// Helpers kept available for inline edits during debugging.
#[allow(dead_code)]
fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(dead_code)]
fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

mod dims;
mod full_attention;
mod linear_attention;

use dims::*;
use full_attention::*;
use linear_attention::*;

fn main() -> Result<()> {
    let prompt =
        std::env::var("PROMPT").unwrap_or_else(|_| "What is the capital of France?".to_string());
    let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("$HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });

    println!("=== Atlas Metal · Qwen3.5-4B-MLX-8bit inference ===");
    println!("model dir: {model_dir}");
    println!("prompt:    {prompt:?}");
    println!();
    println!(
        "⚠️  Note: linear_attention layers are identity passthrough. \
         The next-token prediction is informed only by the 8 \
         full_attention layers (3, 7, 11, 15, 19, 23, 27, 31)."
    );
    println!();

    // Tokenizer.
    let tok_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer =
        Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let token_strs: Vec<String> = encoding
        .get_tokens()
        .iter()
        .map(|s| s.to_string())
        .collect();
    println!("tokenized to {} tokens: {:?}", token_ids.len(), token_strs);
    if token_ids.is_empty() {
        bail!("empty token list — tokenizer produced nothing for the prompt");
    }
    let prompt_len = token_ids.len() as u32;

    // Layer types from config.json.
    let cfg_text = std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json"))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)?;
    let layer_types: Vec<String> = cfg["text_config"]["layer_types"]
        .as_array()
        .context("layer_types missing")?
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    if layer_types.len() as u32 != NUM_LAYERS {
        bail!(
            "expected {NUM_LAYERS} layers, got {} in layer_types",
            layer_types.len()
        );
    }
    let full_attn_count = layer_types
        .iter()
        .filter(|s| s.as_str() == "full_attention")
        .count();
    let lin_attn_count = layer_types
        .iter()
        .filter(|s| s.as_str() == "linear_attention")
        .count();
    println!(
        "layer types: {} full_attention + {} linear_attention",
        full_attn_count, lin_attn_count
    );

    // Backend.
    let modules = atlas_kernels::metallib_modules();
    if modules.is_empty() {
        bail!(
            "metal kernel registry empty — re-build with \
             ATLAS_TARGET_HW=metal ATLAS_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
             ATLAS_TARGET_QUANT=mlx_int8"
        );
    }
    let backend = MetalGpuBackend::new(0, &modules)?;
    println!("metal backend ready, {} kernel modules", modules.len());

    // mmap the safetensors.
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    let file = std::fs::File::open(&st_path).context("open safetensors")?;
    let mmap = unsafe { memmap2::Mmap::map(&file).context("mmap")? };
    let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;

    // Load embed_tokens (used both for input embedding and tied LM head).
    println!("loading embed_tokens (vocab=248320, hidden=2560)...");
    let t0 = Instant::now();
    let embed_tokens = MlxInt8Weight::load(
        &backend,
        &st,
        "language_model.model.embed_tokens",
        GROUP_SIZE,
    )?;
    println!("  → embed_tokens loaded in {:.2?}", t0.elapsed());

    // Load final norm.
    let t = st.tensor("language_model.model.norm.weight").unwrap();
    let final_norm = backend.alloc(t.data().len())?;
    backend.copy_h2d(t.data(), final_norm)?;

    // Load all layers (8 full_attention + 24 linear_attention).
    println!("loading all 32 layers...");
    let t0 = Instant::now();
    let mut full_layers: Vec<Option<FullAttentionLayer>> = (0..NUM_LAYERS).map(|_| None).collect();
    let mut lin_layers: Vec<Option<LinearAttentionLayer>> = (0..NUM_LAYERS).map(|_| None).collect();
    for (idx, ty) in layer_types.iter().enumerate() {
        if ty == "full_attention" {
            full_layers[idx] = Some(FullAttentionLayer::load(&backend, &st, idx as u32)?);
        } else if ty == "linear_attention" {
            lin_layers[idx] = Some(LinearAttentionLayer::load(&backend, &st, idx as u32)?);
        }
    }
    println!("  → all weights loaded in {:.2?}", t0.elapsed());

    // Allocate scratch + KV caches (one cache per full_attention layer).
    // Capacity covers prompt + decode budget; bump via $ATLAS_DECODE_TOKENS.
    let n_decode_budget: u32 = std::env::var("ATLAS_DECODE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let max_seq_len = prompt_len + n_decode_budget + 4;
    let scratch = alloc_scratch(&backend)?;
    let lin_scratch = alloc_lin_scratch(&backend)?;
    let kv_caches: Vec<LayerKvCache> = (0..full_attn_count)
        .map(|_| -> Result<LayerKvCache> {
            Ok(LayerKvCache {
                k: backend.alloc((max_seq_len * KV_DIM) as usize * 2)?,
                v: backend.alloc((max_seq_len * KV_DIM) as usize * 2)?,
                capacity: max_seq_len,
            })
        })
        .collect::<Result<_>>()?;
    // Map layer_idx → kv_cache slot.
    let mut full_kv_slot: Vec<Option<usize>> = (0..NUM_LAYERS).map(|_| None).collect();
    {
        let mut next_slot = 0;
        for (idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "full_attention" {
                full_kv_slot[idx] = Some(next_slot);
                next_slot += 1;
            }
        }
    }
    // Per-linear-attention-layer SSM/conv state.
    let lin_states: Vec<LinearAttentionState> = (0..lin_attn_count)
        .map(|_| LinearAttentionState::alloc(&backend))
        .collect::<Result<_>>()?;
    let mut lin_state_slot: Vec<Option<usize>> = (0..NUM_LAYERS).map(|_| None).collect();
    {
        let mut next_slot = 0;
        for (idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "linear_attention" {
                lin_state_slot[idx] = Some(next_slot);
                next_slot += 1;
            }
        }
    }

    // Per-layer working buffer for the residual stream (one BF16[hidden]).
    let x_buf = backend.alloc(HIDDEN as usize * 2)?;
    // The output of forward_full_attention writes to scratch.x_out — we
    // d2d-copy back into x_buf at end of each layer to keep the
    // residual-stream pointer stable across layers.

    // RoPE inv_freq table (precomputed). Partial RoPE: only the first
    // `ROTARY_DIM` elements of each head are rotated, so the table
    // has `ROTARY_DIM/2` entries indexed by 1/(theta^(2i/ROTARY_DIM)).
    let half_dim = ROTARY_DIM / 2;
    let inv_freq_bytes: Vec<u8> = (0..half_dim)
        .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f32 / ROTARY_DIM as f32))
        .flat_map(|f: f32| f.to_le_bytes())
        .collect();
    let inv_freq_ptr = backend.alloc(inv_freq_bytes.len())?;
    backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr)?;

    // positions_ptr is rewritten per token (current absolute position).
    let positions_ptr = backend.alloc(4)?;

    // Pre-resolve every kernel handle.
    let stream = backend.default_stream();
    let rms = backend.kernel("rms_norm", "rms_norm")?;
    let rope = backend.kernel("rope_apply", "rope_apply")?;
    let kvap = backend.kernel("kv_cache_append", "kv_cache_append")?;
    let attn = backend.kernel("attention_decode", "attention_decode")?;
    let sg = backend.kernel("sigmoid_gate", "sigmoid_gate")?;
    let add = backend.kernel("bf16_add", "bf16_add")?;
    let add_rms = backend.kernel("add_rms_norm", "add_rms_norm")?;
    let silu = backend.kernel("silu_gate", "silu_gate")?;
    let qkv_split = backend.kernel("qwen35_qkv_split", "qwen35_qkv_split")?;
    let embed = backend.kernel("embed_lookup", "embed_lookup")?;
    let conv1d = backend.kernel("causal_conv1d_update_l2norm", "causal_conv1d_update_l2norm")?;
    // The four GDN helpers all live in `gdn_helpers.metal` so the
    // metallib module name is shared.
    let gdn_gate = backend.kernel("gdn_helpers", "gdn_compute_gate")?;
    let sigmoid = backend.kernel("gdn_helpers", "sigmoid_bf16_to_f32")?;
    let silu_op = backend.kernel("gdn_helpers", "silu_apply")?;
    let mul = backend.kernel("gdn_helpers", "bf16_mul")?;
    let gdn_dec = backend.kernel("gated_delta_rule_decode", "gated_delta_rule_decode")?;

    // ── Embed-then-feed loop: process every prompt token through
    //    every layer, building KV cache. The hidden after the LAST
    //    prompt token is what we sample from.
    println!();
    let dump_dir = std::env::var("ATLAS_RESIDUAL_DUMP_DIR")
        .ok()
        .map(std::path::PathBuf::from);
    if let Some(d) = &dump_dir {
        std::fs::create_dir_all(d)?;
        println!("residual dumps: {}", d.display());
    }
    let dump_resid = |label: &str, ptr: DevicePtr| -> Result<()> {
        if let Some(d) = &dump_dir {
            let mut buf = vec![0u8; HIDDEN as usize * 2];
            backend.copy_d2h(ptr, &mut buf)?;
            // Convert BF16 → FP32 to match MLX dump.
            let f32_bytes: Vec<u8> = buf
                .chunks_exact(2)
                .flat_map(|c| {
                    half::bf16::from_le_bytes([c[0], c[1]])
                        .to_f32()
                        .to_le_bytes()
                })
                .collect();
            std::fs::write(d.join(format!("atlas_{label}.bin")), &f32_bytes)?;
        }
        Ok(())
    };
    let dump_bf16_n = |label: &str, ptr: DevicePtr, n: u32| -> Result<()> {
        if let Some(d) = &dump_dir {
            let mut buf = vec![0u8; n as usize * 2];
            backend.copy_d2h(ptr, &mut buf)?;
            let f32_bytes: Vec<u8> = buf
                .chunks_exact(2)
                .flat_map(|c| {
                    half::bf16::from_le_bytes([c[0], c[1]])
                        .to_f32()
                        .to_le_bytes()
                })
                .collect();
            std::fs::write(d.join(format!("atlas_{label}.bin")), &f32_bytes)?;
        }
        Ok(())
    };

    println!("running prefill: {prompt_len} tokens × {NUM_LAYERS} layers");
    let t_total = Instant::now();
    for (tok_idx, &token_id) in token_ids.iter().enumerate() {
        // Embedding lookup for this token: write the embedding into x_buf.
        // Use embed_lookup kernel against the dequantized embed_tokens.
        // We don't have the full dequantized embed_tokens in BF16 — it's
        // 248320 * 2560 * 2 bytes ≈ 1.2 GB. Instead, dequantize ONLY the
        // single row we need by allocating a small BF16 scratch row and
        // calling mlx_int8_dequant on a single-row slice — no, the
        // dequant kernel needs the packed buffer to be aligned to its
        // expected layout. Easier path: call mlx_int8_gemv with a
        // one-hot input vector to extract the row.
        //
        // Simplest: dequantize the entire embed_tokens once, cache it in
        // BF16 GPU memory (1.2 GB — fits in UMA on Apple Silicon), then
        // use embed_lookup. That's what the next block does (lazily on
        // first access).
        if tok_idx == 0 {
            // Lazy: allocate + dequantize embed_tokens to BF16 once.
            // This is the LM head's working buffer too.
            // (Performed inline so we only pay the cost when we know we
            // need it — saves 1.2 GB if the user runs only one of the
            // earlier examples.)
            // Note: in production, the embed lookup would walk the MLX
            // packed bytes directly per-row (saves 1.2 GB); this version
            // trades memory for kernel-reuse simplicity.
        }

        // Per-token prefill step: write token_id, lookup, then layer chain.
        backend.copy_h2d(&token_id.to_le_bytes(), positions_ptr)?;
        // We need the embedding for `token_id` in x_buf. Use a single-
        // token embed_lookup. Since dequantizing all 1.2 GB of
        // embed_tokens up front is expensive, we materialize per-token
        // by running mlx_int8_dequant on the row's worth of packed
        // bytes (HIDDEN cols, group_size=64 → 40 groups). The kernel
        // expects packed[N, K/4] etc; we slice to N=1 and offset into
        // the source packed buffer by token_id rows.
        //
        // Even simpler given UMA: copy the row's BF16-equivalent by
        // dequantizing inline via a tiny per-call gemv with a one-hot
        // vector. For the demo, we just call the existing dequant kernel
        // on the WHOLE embed_tokens once, lazily, and then use
        // embed_lookup against that. Allocate the dequant buffer here:
        // GROUP_SIZE is 64; matches MLX's standard group size.
        const _: [(); 1] = [(); (GROUP_SIZE == 64) as usize];
        // FAST PATH for this demo: emit embedding via the embed_tokens
        // gemv with a one-hot vector of length VOCAB. That's
        // 248320-element matmul per token, dominated by memory bandwidth.

        // Build one-hot input vector [VOCAB] BF16 (CPU-side since we copy
        // it h2d each iter). The result is embed_tokens @ one_hot[token_id]
        // = the token_id-th row of dequantized embed_tokens.
        // But embed_tokens.gemv expects in_features = HIDDEN (2560); the
        // weight is [VOCAB, HIDDEN/4 packed] so out_features = VOCAB and
        // in_features = HIDDEN. So gemv(x[2560]) → y[VOCAB] is the LM-head
        // direction, NOT the embed direction.
        //
        // To EMBED a token: pick row token_id of dequantized embed_tokens,
        // = HIDDEN BF16 values. The kernel that does this is
        // embed_lookup, but it needs a fully-dequantized BF16 table.
        // For this demo, build that table once on first iteration.
        if tok_idx == 0 {
            // Lazy-init: allocate + run mlx_int8_dequant on embed_tokens
            // to produce a BF16 [VOCAB, HIDDEN] table.
            // 248320 * 2560 * 2 = 1.27 GB. Fits in M-series UMA budget.
            println!("  (lazy) dequantizing embed_tokens to BF16 table (1.27 GB)...");
            let t_dq = Instant::now();
            let embed_table_bytes = (VOCAB * HIDDEN) as usize * 2;
            let embed_table = backend.alloc(embed_table_bytes)?;
            embed_tokens.dequantize_to(&backend, embed_table, stream)?;
            backend.synchronize(stream)?;
            println!("  → dequantized in {:.2?}", t_dq.elapsed());
            // Stash the table pointer in a Box leaked into the closure
            // below — for a one-shot example this is fine.
            EMBED_TABLE.store(embed_table.0, std::sync::atomic::Ordering::SeqCst);
        }
        let embed_table = DevicePtr(EMBED_TABLE.load(std::sync::atomic::Ordering::SeqCst));

        // embed_lookup expects token_ids[num_tokens], embed_table[vocab, hidden],
        // out[num_tokens, hidden]. We do one token at a time.
        let token_id_bytes = token_id.to_le_bytes();
        let token_buf = backend.alloc(4)?;
        backend.copy_h2d(&token_id_bytes, token_buf)?;
        let n_tokens = 1u32;
        backend.launch_typed(
            embed,
            [HIDDEN.div_ceil(8), n_tokens, 1],
            [8, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&HIDDEN.to_le_bytes()),
                KernelArg::Bytes(&VOCAB.to_le_bytes()),
                KernelArg::Buffer(token_buf),
                KernelArg::Buffer(embed_table),
                KernelArg::Buffer(x_buf),
            ],
        )?;
        backend.free(token_buf)?;

        // Set the position for RoPE = absolute index in the sequence.
        let pos_u32 = tok_idx as u32;
        backend.copy_h2d(&pos_u32.to_le_bytes(), positions_ptr)?;

        // Layer chain.
        let mut x = x_buf;
        // Dump after embedding (only for last prompt token to match MLX).
        if tok_idx == token_ids.len() - 1 {
            backend.synchronize(stream)?;
            dump_resid("resid_after_embed", x)?;
        }
        for (layer_idx, ty) in layer_types.iter().enumerate() {
            if ty == "full_attention" {
                let layer = full_layers[layer_idx]
                    .as_ref()
                    .expect("full_attn layer not loaded");
                let kv = &kv_caches[full_kv_slot[layer_idx].unwrap()];
                let cache_pos = tok_idx as u32;
                let seq_len_attn = (tok_idx + 1) as u32;
                let out = forward_full_attention(
                    &backend,
                    layer,
                    &scratch,
                    kv,
                    rms,
                    rope,
                    kvap,
                    attn,
                    sg,
                    add,
                    add_rms,
                    silu,
                    qkv_split,
                    inv_freq_ptr,
                    positions_ptr,
                    x,
                    cache_pos,
                    seq_len_attn,
                    stream,
                )?;
                // Copy out → x_buf so the next layer's input is stable.
                backend.copy_d2d_async(out, x_buf, HIDDEN as usize * 2, stream)?;
                x = x_buf;
            } else {
                // linear_attention: real GDN orchestration.
                let layer = lin_layers[layer_idx]
                    .as_ref()
                    .expect("linear_attn layer not loaded");
                let state = &lin_states[lin_state_slot[layer_idx].unwrap()];
                // For layer 1 last token: dump intra-GDN intermediates so
                // we can localize the residual divergence vs MLX.
                let intra: Option<&dyn Fn(&str, DevicePtr, u32) -> Result<()>> =
                    if tok_idx == token_ids.len() - 1 && layer_idx == 0 {
                        Some(&dump_bf16_n)
                    } else {
                        None
                    };
                let out = forward_linear_attention(
                    &backend,
                    layer,
                    state,
                    &lin_scratch,
                    rms,
                    conv1d,
                    gdn_gate,
                    sigmoid,
                    silu_op,
                    silu,
                    mul,
                    gdn_dec,
                    add,
                    add_rms,
                    x,
                    x_buf,
                    stream,
                    intra,
                )?;
                x = out;
            }
            // Dump after each layer for last prompt token.
            if tok_idx == token_ids.len() - 1 {
                backend.synchronize(stream)?;
                dump_resid(&format!("resid_after_layer_{layer_idx:02}"), x)?;
            }
        }
        backend.synchronize(stream)?;
    }
    let prefill_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    println!(
        "prefill complete in {prefill_ms:.1} ms ({:.1} ms/tok)",
        prefill_ms / prompt_len as f64
    );

    // Allocate sample-time buffers + kernels.
    let x_final = backend.alloc(HIDDEN as usize * 2)?;
    let logits = backend.alloc(VOCAB as usize * 2)?;
    let argmax = backend.kernel("argmax_bf16", "argmax_bf16")?;
    let result_buf = backend.alloc(4)?;

    // Helper: run final_norm + LM head + argmax → token id.
    let sample_next = |x_in: DevicePtr| -> Result<u32> {
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
                KernelArg::Buffer(final_norm),
                KernelArg::Buffer(x_final),
            ],
        )?;
        embed_tokens.gemv(&backend, x_final, logits, stream)?;
        backend.launch_typed(
            argmax,
            [1, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&VOCAB.to_le_bytes()),
                KernelArg::Buffer(logits),
                KernelArg::Buffer(result_buf),
            ],
        )?;
        backend.synchronize(stream)?;
        let mut buf = [0u8; 4];
        backend.copy_d2h(result_buf, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    };

    // First sample after prefill.
    let next_token_id = sample_next(x_buf)?;
    let next_text = tokenizer
        .decode(&[next_token_id], false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    println!();
    println!("=== After prefill, first generated token ===");
    println!("  token_id: {next_token_id}");
    println!("  text:     {next_text:?}");

    // Continue greedy decoding for N more tokens to see a full response.
    let n_decode: usize = std::env::var("ATLAS_DECODE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    println!();
    println!("running greedy decode for {n_decode} more tokens...");
    let t_dec = Instant::now();
    let mut generated_ids = vec![next_token_id];
    let mut current_token = next_token_id;
    let mut cur_pos = prompt_len;
    let embed_table = DevicePtr(EMBED_TABLE.load(std::sync::atomic::Ordering::SeqCst));

    for _ in 0..n_decode {
        // Reallocate KV caches if we're about to exceed capacity. For
        // simplicity in this demo we don't grow — limit decode tokens
        // to fit the pre-allocated max_seq_len.
        if cur_pos >= max_seq_len {
            println!("  (reached pre-allocated KV capacity {max_seq_len}, stopping)");
            break;
        }

        // Embed current token.
        let token_buf = backend.alloc(4)?;
        backend.copy_h2d(&current_token.to_le_bytes(), token_buf)?;
        let n_tokens = 1u32;
        backend.launch_typed(
            embed,
            [HIDDEN.div_ceil(8), n_tokens, 1],
            [8, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&HIDDEN.to_le_bytes()),
                KernelArg::Bytes(&VOCAB.to_le_bytes()),
                KernelArg::Buffer(token_buf),
                KernelArg::Buffer(embed_table),
                KernelArg::Buffer(x_buf),
            ],
        )?;
        backend.free(token_buf)?;

        // Position for RoPE.
        backend.copy_h2d(&cur_pos.to_le_bytes(), positions_ptr)?;

        // Layer chain.
        let mut x = x_buf;
        for (layer_idx, ty) in layer_types.iter().enumerate() {
            if ty.as_str() == "full_attention" {
                let layer = full_layers[layer_idx].as_ref().unwrap();
                let kv = &kv_caches[full_kv_slot[layer_idx].unwrap()];
                let cache_pos = cur_pos;
                let seq_len_attn = cur_pos + 1;
                let out = forward_full_attention(
                    &backend,
                    layer,
                    &scratch,
                    kv,
                    rms,
                    rope,
                    kvap,
                    attn,
                    sg,
                    add,
                    add_rms,
                    silu,
                    qkv_split,
                    inv_freq_ptr,
                    positions_ptr,
                    x,
                    cache_pos,
                    seq_len_attn,
                    stream,
                )?;
                backend.copy_d2d_async(out, x_buf, HIDDEN as usize * 2, stream)?;
                x = x_buf;
            } else {
                let layer = lin_layers[layer_idx].as_ref().unwrap();
                let state = &lin_states[lin_state_slot[layer_idx].unwrap()];
                let out = forward_linear_attention(
                    &backend,
                    layer,
                    state,
                    &lin_scratch,
                    rms,
                    conv1d,
                    gdn_gate,
                    sigmoid,
                    silu_op,
                    silu,
                    mul,
                    gdn_dec,
                    add,
                    add_rms,
                    x,
                    x_buf,
                    stream,
                    None,
                )?;
                x = out;
            }
        }
        backend.synchronize(stream)?;

        // Sample.
        current_token = sample_next(x_buf)?;
        generated_ids.push(current_token);
        cur_pos += 1;

        // Bail on EOS to avoid runaway generation.
        if current_token == 248044 {
            // <|im_end|> per tokenizer_config.json
            println!("  (hit <|im_end|>)");
            break;
        }
    }
    let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;

    let full_text = tokenizer
        .decode(&generated_ids, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!();
    println!(
        "=== Full generation ({} tokens, {dec_ms:.1} ms, {:.1} tok/s) ===",
        generated_ids.len(),
        generated_ids.len() as f64 / (dec_ms / 1000.0)
    );
    println!("  ids: {generated_ids:?}");
    println!("  text: {full_text:?}");
    println!();
    println!(
        "All 32 layers fired (8 full_attention + 24 linear_attention via \
         GDN). The GDN orchestration is best-effort — the kernel-level \
         math (gated_delta_rule_decode) matches the CUDA reference \
         exactly but the surrounding pre/post wiring (qkv split, gate \
         clamping, residual placement) may diverge from the upstream \
         Python reference in subtle ways. Token-level parity vs \
         mlx_lm.generate is the next verification step."
    );

    Ok(())
}

// Stash for lazy embed_table allocation (one-shot demo simplification).
static EMBED_TABLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
