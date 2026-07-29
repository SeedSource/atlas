//! Source contract: every batched shared-expert blend kernel must tolerate a
//! NULL `gate_weight`.
//!
//! DeepSeek-V4-Flash has no `shared_expert_gate` tensor, so the weight loader
//! assembles it as `DevicePtr::NULL` (`weight_loader/deepseek_v4/assemble.rs`)
//! and `forward_k2`/`forward_k3` pass that null pointer straight into the
//! blend kernel. A kernel that dereferences it unconditionally is a hard
//! CUDA_ERROR_ILLEGAL_ADDRESS (700) on the first MTP verify step.
//!
//! This happened twice: `batch2` was fixed, `batch3` was not, so K=2 worked and
//! K=3 crashed. The defect class is "guard added at one verify width only", so
//! the test scans *every* `moe_weighted_sum_blend_batchN` kernel rather than
//! naming widths individually — a future `batch4` is covered on arrival.
//!
//! Source-level, not GPU-level, on purpose: the fault is a missing branch in
//! CUDA source, and a text contract catches it in `cargo test` with no device.

use std::fs;
use std::path::PathBuf;

fn kernels_common_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../kernels/gb10/common")
        .canonicalize()
        .expect("kernels/gb10/common must exist relative to spark-model")
}

/// Body of each `moe_weighted_sum_blend_batchN` kernel in `src`, with its name.
fn blend_kernels(src: &str) -> Vec<(String, String)> {
    const MARKER: &str = "__global__ void moe_weighted_sum_blend_batch";
    let mut out = Vec::new();
    for (start, _) in src.match_indices(MARKER) {
        let after = &src[start + MARKER.len()..];
        let width: String = after.chars().take_while(char::is_ascii_digit).collect();
        // Kernel body ends at the next kernel definition, or end of file.
        let rest = &src[start + MARKER.len()..];
        let end = rest.find("__global__ void").map_or(src.len(), |o| start + MARKER.len() + o);
        out.push((format!("moe_weighted_sum_blend_batch{width}"), src[start..end].to_string()));
    }
    out
}

#[test]
fn every_blend_kernel_guards_null_shared_gate() {
    let dir = kernels_common_dir();
    let mut checked = 0usize;

    for entry in fs::read_dir(&dir).expect("read kernels/gb10/common") {
        let path = entry.expect("dir entry").path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if !(name.starts_with("moe_shared_expert_fused_batch") && name.ends_with(".cu")) {
            continue;
        }
        let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));

        for (kernel, body) in blend_kernels(&src) {
            let deref = body
                .find("(const uint4*)gate_weight")
                .unwrap_or_else(|| panic!("{name}: {kernel} has no gate_weight load to guard"));
            let guard = body.find("gate_weight == 0").unwrap_or_else(|| {
                panic!(
                    "{name}: {kernel} dereferences gate_weight with no `gate_weight == 0` guard. \
                     DeepSeek-V4 passes DevicePtr::NULL here -> CUDA-700 on the first MTP verify \
                     step at this width. Mirror moe_shared_expert_fused_batch2.cu."
                )
            });
            assert!(
                guard < deref,
                "{name}: {kernel} guards gate_weight only *after* dereferencing it"
            );
            assert!(
                body[guard..deref].contains("sigmoid_val = 1.0f"),
                "{name}: {kernel} null-gate branch must set sigmoid_val = 1.0f \
                 (ungated shared expert), matching batch2"
            );
            checked += 1;
        }
    }

    assert!(
        checked >= 2,
        "expected at least the batch2 and batch3 blend kernels, found {checked}"
    );
}
