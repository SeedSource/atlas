// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers used by every `metal_backend::tests` submodule —
//! byte-conversion utilities and a synthetic MLX-int8 fixture builder.

// ── Byte-conversion helpers (bytemuck-free) ──────────────────

pub(super) fn u32_slice_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub(super) fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub(super) fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

/// Build an MLX-int8 fixture (synthetic packed weights, scales, biases,
/// + the dequantised reference). Returned as raw little-endian byte
/// blobs so callers can `copy_h2d` them straight into `MTLBuffer`s,
/// plus an FP32-friendly `Vec<bf16>` for CPU-side reference math.
///
/// Returned tuple: `(packed_bytes_le, scales_bytes_le, biases_bytes_le, w_bf16_dequant)`.
pub(super) fn build_mlx_fixture(
    n_rows: usize,
    n_cols: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<half::bf16>) {
    assert!(n_cols % 4 == 0 && n_cols % group_size == 0);
    let groups_per_row = n_cols / group_size;

    let mut bytes_flat: Vec<u8> = Vec::with_capacity(n_rows * n_cols);
    for r in 0..n_rows {
        for c in 0..n_cols {
            bytes_flat.push(((r * 13 + c * 5 + 17) % 256) as u8);
        }
    }
    let mut packed: Vec<u32> = Vec::with_capacity(n_rows * n_cols / 4);
    for r in 0..n_rows {
        for c in (0..n_cols).step_by(4) {
            let base = r * n_cols + c;
            let word = (bytes_flat[base] as u32)
                | ((bytes_flat[base + 1] as u32) << 8)
                | ((bytes_flat[base + 2] as u32) << 16)
                | ((bytes_flat[base + 3] as u32) << 24);
            packed.push(word);
        }
    }
    let mut scales: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    let mut biases: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    for r in 0..n_rows {
        for g in 0..groups_per_row {
            scales.push(half::bf16::from_f32(
                0.001 + 0.0005 * r as f32 + 0.0007 * g as f32,
            ));
            biases.push(half::bf16::from_f32(
                -0.05 + 0.01 * r as f32 + 0.005 * g as f32,
            ));
        }
    }

    let mut w_dequant: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let byte = bytes_flat[r * n_cols + c] as f32;
            let g = c / group_size;
            let s = scales[r * groups_per_row + g].to_f32();
            let b = biases[r * groups_per_row + g].to_f32();
            w_dequant[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
        }
    }
    (
        u32_slice_to_bytes(&packed),
        bf16_slice_to_bytes(&scales),
        bf16_slice_to_bytes(&biases),
        w_dequant,
    )
}
