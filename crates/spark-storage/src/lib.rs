// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

// Atlas spark-storage: high-speed NVMe-backed KV cache offload.
//
// Phase 0 of `--high-speed-swap` (see plan at
// /workspace/.claude/plans/i-want-to-ensure-valiant-bunny.md): runtime probe
// that decides whether the production backend should be cuFile/GDS or
// io_uring + pinned-host bounce. Later phases add the predictor, scratch
// pool, eviction, and I/O thread.
//
// Feature gating: every module that touches the CUDA driver (raw FFI in
// `cuda_min`, the module/event helpers in `cuda_module`, anything that
// holds a `DeviceBuffer`) is gated behind the `cuda` feature so the
// crate compiles on Apple Silicon (`--no-default-features --features
// metal`) where the high-speed-swap path won't be reachable anyway.

#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_min;
#[cfg(feature = "cuda")]
pub mod cuda_module;

// Re-export the module/event/launch helpers from their new home so existing
// `use spark_storage::cuda_min::{CudaModule, CudaEvent, launch_kernel}` paths
// keep working.
#[cfg(feature = "cuda")]
pub use cuda_module::{CudaEvent, CudaModule, launch_kernel};

// Pure CPU-side modules — types, configs, references. Always compiled.
pub mod attention_ref;
pub mod config;
pub mod eviction;
pub mod group;
pub mod model_dims;
pub mod predictor_ref;
pub mod projection;

// `ModelDims` is a plain POD struct (no GPU state) that
// `spark-model`'s public surface threads through every layer's
// forward signature; it must stay reachable on metal builds even
// though the high-speed-swap orchestrator that consumes it is
// CUDA-gated.
pub use model_dims::ModelDims;

// `layout` opens disk files with `O_DIRECT` and pre-allocates via
// `posix_fallocate` — both Linux-specific. Only the cuda-side modules
// (high_speed_swap, backend/io_uring, backend/posix) consume it, so
// gating it on the cuda feature is sufficient.
#[cfg(feature = "cuda")]
pub mod layout;

// CUDA-only modules: each holds raw `cu*` FFI calls or a `DeviceBuffer`,
// or transitively imports from the cuda_* modules above. Gated together
// because separating them would just smear the boundary.
#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(feature = "cuda")]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod high_speed_swap;
#[cfg(feature = "cuda")]
pub mod predictor;
#[cfg(feature = "cuda")]
pub mod probe;
#[cfg(feature = "cuda")]
pub mod scratch_pool;
#[cfg(feature = "cuda")]
pub mod tiled_attention;

#[cfg(feature = "cuda")]
pub use backend::{IoUringBackend, PosixBackend, ReadRequest, StorageBackend};
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
#[cfg(feature = "cuda")]
pub use high_speed_swap::{HighSpeedSwap, install_local, local_installed, with_local};

#[cfg(feature = "cuda")]
pub use predictor::{Predictor, PredictorDims};
#[cfg(feature = "cuda")]
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
#[cfg(feature = "cuda")]
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
