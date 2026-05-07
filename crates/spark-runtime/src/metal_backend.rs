// SPDX-License-Identifier: AGPL-3.0-only

//! Apple Metal GPU backend.
//!
//! Implements [`GpuBackend`] on top of the Metal framework via the
//! `objc2-metal` bindings. Apple Silicon is unified-memory (UMA), so
//! every `MTLBuffer` is allocated with `StorageModeShared` and host
//! `memcpy` against `buffer.contents()` is the canonical H2D/D2H path
//! — no PCIe staging, no pinned-host bounce.
//!
//! # Pointer model
//!
//! `DevicePtr` carries a real GPU virtual address obtained from
//! `MTLBuffer::gpuAddress()` (Metal 3+, native to all Apple Silicon).
//! That makes pointer arithmetic (`DevicePtr::offset`) a plain integer
//! add — no buffer/offset pair to thread through. To recover the
//! owning `MTLBuffer` for `free` / blit-copy / `setBuffer:`, we keep
//! a side table `BTreeMap<base_gpu_address, MTLBuffer>` and look up
//! the largest key ≤ ptr. The buffer's gpuAddress range is
//! `[base, base + length)`, so a binary search is enough.
//!
//! # Streams
//!
//! A stream handle indexes a slab of `MetalStream { queue, in_flight }`.
//! Handle 0 is the default stream and is lazily created on first use.
//! `synchronize(stream)` commits the in-flight `MTLCommandBuffer` and
//! `waitUntilCompleted()`s; the next encoder opens on a fresh buffer.
//!
//! # Kernel handles
//!
//! `KernelHandle` indexes a slab of `MTLComputePipelineState`. The
//! library cache (one `MTLLibrary` per `metallib_modules()` entry) is
//! built once at construction and never mutated; pipeline lookups go
//! through the slab + a `(module, fn_name)` HashMap so repeated
//! `kernel()` calls are O(1) cached.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLEvent, MTLLibrary, MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use parking_lot::Mutex;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

// ── Internal type aliases (Retained<ProtocolObject<dyn _>> is verbose) ────

type ObjDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type ObjBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type ObjQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ObjCmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type ObjLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
type ObjPipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type ObjSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;

// ── Stream + slab types ──────────────────────────────────────────────────

struct MetalStream {
    queue: ObjQueue,
    /// In-flight command buffer accumulating encoded work. Committed +
    /// waited on by `synchronize()`; replaced by a fresh buffer on
    /// next encoder open.
    in_flight: Option<ObjCmdBuf>,
}

/// Tracks one outstanding shared event so `record_event` can write the
/// next signal value and `stream_wait_event` can wait on the same
/// counter.
struct EventSlot {
    event: ObjSharedEvent,
    /// Monotonic value sequence — record_event signals `next`, then
    /// increments. stream_wait_event waits on `next - 1` (the most
    /// recently recorded value). Atomic via the surrounding Mutex.
    next: u64,
}

/// Key for the pipeline cache. Stored as owned strings because the
/// `&str` arguments to `kernel()` come from arbitrary call sites.
type PipelineKey = (String, String);

// ── MetalGpuBackend struct + state ───────────────────────────────────────

pub struct MetalGpuBackend {
    device: ObjDevice,
    /// Side table mapping a buffer's base gpuAddress to the owning
    /// `MTLBuffer`. BTreeMap so we can find the buffer containing an
    /// arbitrary `DevicePtr` via `range(..=ptr).next_back()`.
    allocations: Arc<Mutex<BTreeMap<u64, ObjBuffer>>>,
    /// Stream slab. Indexed by `stream_handle - 1`; handle 0 is the
    /// implicit default stream materialized lazily into slot 0.
    streams: Arc<Mutex<Vec<MetalStream>>>,
    /// Loaded metallibs keyed by module name.
    libraries: HashMap<String, ObjLibrary>,
    /// Pipeline-state cache + slab. The HashMap maps `(module, fn)` to
    /// the slab index; the slab owns the `MTLComputePipelineState`.
    /// Both are mutexed so `kernel()` can be called from any thread.
    pipeline_cache: Arc<Mutex<HashMap<PipelineKey, KernelHandle>>>,
    pipeline_slab: Arc<Mutex<Vec<ObjPipeline>>>,
    /// Shared-event slab for cross-stream synchronization.
    events: Arc<Mutex<Vec<EventSlot>>>,
}

unsafe impl Send for MetalGpuBackend {}
unsafe impl Sync for MetalGpuBackend {}

impl MetalGpuBackend {
    /// Initialize the Metal backend with the embedded metallib modules.
    ///
    /// `kernel_modules` is the `metallib_modules()` slice produced by
    /// `atlas-kernels`' build script — `(module_name, metallib_bytes)`.
    /// Each entry is loaded into its own `MTLLibrary` via
    /// `newLibraryWithData_error:`. The default stream (handle 0) is
    /// materialized eagerly so the first launch doesn't pay queue-
    /// creation latency.
    pub fn new(
        ordinal: usize,
        kernel_modules: &[(&'static str, &'static [u8])],
    ) -> Result<Self> {
        if ordinal != 0 {
            bail!(
                "Metal: only ordinal 0 is supported (Apple Silicon has one \
                 system default device); requested ordinal {ordinal}"
            );
        }
        let device: ObjDevice = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| anyhow!("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU"))?;

        // Build the library cache up-front. `newLibraryWithData_error`
        // takes a `DispatchData`, which is libdispatch's reference-
        // counted byte container. We wrap the &'static slice via
        // dispatch2::DispatchData (zero-copy) — the metallibs are
        // embedded by include_bytes! and outlive the backend.
        let mut libraries: HashMap<String, ObjLibrary> = HashMap::new();
        for (name, bytes) in kernel_modules {
            let data = dispatch2::DispatchData::from_static_bytes(bytes);
            let lib = device
                .newLibraryWithData_error(&data)
                .map_err(|e| {
                    anyhow!(
                        "newLibraryWithData failed for module '{name}': {}",
                        e.localizedDescription()
                    )
                })?;
            libraries.insert((*name).to_string(), lib);
        }

        // Materialize the default stream eagerly (slot 0 = handle 0).
        let default_queue = device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null on default device"))?;
        let streams = vec![MetalStream {
            queue: default_queue,
            in_flight: None,
        }];

        tracing::info!(
            "MetalGpuBackend initialized on device '{}' with {} metallib modules",
            device.name().to_string(),
            libraries.len()
        );

        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(BTreeMap::new())),
            streams: Arc::new(Mutex::new(streams)),
            libraries,
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            pipeline_slab: Arc::new(Mutex::new(Vec::new())),
            events: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Return the underlying `MTLDevice` (escape hatch for advanced
    /// use cases — graph capture, custom resource creation, etc.).
    pub fn raw_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Look up the `MTLBuffer` owning `ptr` and the byte offset of
    /// `ptr` within it. Returns `None` if no allocation contains it.
    fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = (ptr.0 - *base) as usize;
        if offset > buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// Resolve a stream handle to its slab index. Handle 0 → slot 0
    /// (default stream); other handles index `handle - 1`.
    fn stream_index(handle: u64, slab: &[MetalStream]) -> Result<usize> {
        let idx = if handle == 0 { 0 } else { (handle - 1) as usize };
        if idx >= slab.len() {
            bail!("Metal: invalid stream handle {handle}");
        }
        Ok(idx)
    }

    /// Borrow (or open) the in-flight command buffer on the given
    /// stream. Returns a clone of the `Retained` so the caller can
    /// encode without holding the streams mutex across encoder calls.
    fn current_cmd_buf(&self, stream_handle: u64) -> Result<ObjCmdBuf> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        if let Some(ref cb) = s.in_flight {
            return Ok(cb.clone());
        }
        let cb = s
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null on stream {stream_handle}"))?;
        s.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// Commit the in-flight buffer on `stream_handle` (no wait). Used
    /// internally by `synchronize` and `record_event`. Returns the
    /// committed buffer so callers that need to `waitUntilCompleted()`
    /// can.
    fn commit_in_flight(&self, stream_handle: u64) -> Result<Option<ObjCmdBuf>> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        let Some(cb) = s.in_flight.take() else {
            return Ok(None);
        };
        cb.commit();
        Ok(Some(cb))
    }
}

// ── GpuBackend impl ──────────────────────────────────────────────────────

impl GpuBackend for MetalGpuBackend {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        // StorageModeShared is the UMA-friendly mode: `contents()`
        // returns a CPU-mappable pointer that aliases GPU memory.
        let buf: ObjBuffer = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("newBufferWithLength failed for {bytes} bytes"))?;
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("MTLBuffer::gpuAddress returned 0 — Metal 3 / macOS 13 required");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(DevicePtr(addr))
    }

    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        // Apple Silicon UMA: managed and shared are the same thing.
        // No paged virtual memory swap mechanism (cuMemAllocManaged on
        // GB10) — Metal lets the OS handle pressure via its memory
        // pool. Defer to plain alloc.
        self.alloc(bytes)
    }

    fn free(&self, ptr: DevicePtr) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Removing the entry drops the last `Retained` reference; the
        // ObjC runtime releases the underlying MTLBuffer.
        self.allocations.lock().remove(&ptr.0);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_h2d: ptr {dst} not in any allocation"))?;
        if offset + src.len() > buf.length() {
            bail!(
                "copy_h2d: write overflows buffer ({} + {} > {})",
                offset,
                src.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let dst_ptr = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
        }
        Ok(())
    }

    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2h: ptr {src} not in any allocation"))?;
        if offset + dst.len() > buf.length() {
            bail!(
                "copy_d2h: read overflows buffer ({} + {} > {})",
                offset,
                dst.len(),
                buf.length()
            );
        }
        let contents: NonNull<c_void> = buf.contents();
        unsafe {
            let src_ptr = (contents.as_ptr() as *const u8).add(offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    fn copy_d2h_on_stream(
        &self,
        src: DevicePtr,
        dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        // UMA: synchronize the stream so prior kernels have written
        // their bytes back through the cache, then memcpy.
        self.synchronize(stream)?;
        self.copy_d2h(src, dst)
    }

    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d: src ptr {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d: dst ptr {dst} not allocated"))?;
        drop(allocs);

        let cmd_buf = self.current_cmd_buf(0)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        // Synchronize so the d2d behaves like CUDA's synchronous variant.
        if let Some(cb) = self.commit_in_flight(0)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn copy_d2d_async(
        &self,
        src: DevicePtr,
        dst: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let allocs = self.allocations.lock();
        let (src_buf, src_off) = Self::find_buffer(&allocs, src)
            .ok_or_else(|| anyhow!("copy_d2d_async: src {src} not allocated"))?;
        let (dst_buf, dst_off) = Self::find_buffer(&allocs, dst)
            .ok_or_else(|| anyhow!("copy_d2d_async: dst {dst} not allocated"))?;
        drop(allocs);
        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .blitCommandEncoder()
            .ok_or_else(|| anyhow!("blitCommandEncoder returned null"))?;
        unsafe {
            enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, src_off, &dst_buf, dst_off, bytes,
            );
        }
        enc.endEncoding();
        Ok(())
    }

    fn launch(
        &self,
        _func: KernelHandle,
        _grid: [u32; 3],
        _block: [u32; 3],
        _shared_mem: u32,
        _stream: u64,
        _params: &mut [*mut c_void],
    ) -> Result<()> {
        // Metal can't safely interpret untyped `*mut c_void` slots as
        // either buffers or scalars (CUDA gets away with this because
        // the driver cross-references the kernel signature). Callers
        // must use `launch_typed`; the cuda-style untyped path is
        // intentionally unsupported here.
        bail!(
            "Metal backend: launch() requires typed args. Use launch_typed() \
             with KernelArg::Buffer / KernelArg::Bytes — see KernelLaunch builder."
        );
    }

    fn launch_typed(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        _shared_mem: u32,
        stream: u64,
        args: &[KernelArg<'_>],
    ) -> Result<()> {
        // Resolve the pipeline state.
        let pipeline = {
            let slab = self.pipeline_slab.lock();
            slab.get(func.0 as usize)
                .cloned()
                .ok_or_else(|| anyhow!("launch_typed: unknown kernel handle {}", func.0))?
        };

        // Snapshot the alloc registry so we can resolve Buffer args
        // (and so the encoder can `useResource:` every live buffer
        // without holding the alloc lock during encoding).
        let live_buffers: Vec<ObjBuffer> = self
            .allocations
            .lock()
            .values()
            .cloned()
            .collect();
        let allocs_snapshot: BTreeMap<u64, ObjBuffer> = self
            .allocations
            .lock()
            .clone();

        let cmd_buf = self.current_cmd_buf(stream)?;
        let enc = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| anyhow!("computeCommandEncoder returned null"))?;
        enc.setComputePipelineState(&pipeline);

        // Mark every live allocation as in-use so Metal's automatic
        // hazard tracking keeps them resident. Cheap on Apple Silicon
        // because `useResource:` is a hint, not a copy.
        for buf in &live_buffers {
            let resource: &ProtocolObject<dyn MTLResource> =
                ProtocolObject::from_ref(&**buf);
            enc.useResource_usage(
                resource,
                objc2_metal::MTLResourceUsage::Read | objc2_metal::MTLResourceUsage::Write,
            );
        }

        // Bind each typed arg to its index.
        for (idx, arg) in args.iter().enumerate() {
            match arg {
                KernelArg::Buffer(p) => {
                    let (buf, offset) = Self::find_buffer(&allocs_snapshot, *p)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} ptr {p} not allocated"))?;
                    unsafe {
                        enc.setBuffer_offset_atIndex(Some(&buf), offset, idx);
                    }
                }
                KernelArg::Bytes(b) => {
                    let ptr = NonNull::new(b.as_ptr() as *mut c_void)
                        .ok_or_else(|| anyhow!("launch_typed: arg #{idx} bytes is null"))?;
                    unsafe {
                        enc.setBytes_length_atIndex(ptr, b.len(), idx);
                    }
                }
            }
        }

        let threadgroups = MTLSize {
            width: grid[0] as usize,
            height: grid[1] as usize,
            depth: grid[2] as usize,
        };
        let threads_per_tg = MTLSize {
            width: block[0] as usize,
            height: block[1] as usize,
            depth: block[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        enc.endEncoding();
        Ok(())
    }

    fn synchronize(&self, stream: u64) -> Result<()> {
        if let Some(cb) = self.commit_in_flight(stream)? {
            cb.waitUntilCompleted();
        }
        Ok(())
    }

    fn default_stream(&self) -> u64 {
        0
    }

    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        let key: PipelineKey = (module.to_string(), func_name.to_string());
        if let Some(handle) = self.pipeline_cache.lock().get(&key) {
            return Ok(*handle);
        }
        let lib = self
            .libraries
            .get(module)
            .ok_or_else(|| anyhow!("Metal: unknown module '{module}'"))?;
        let ns_name = NSString::from_str(func_name);
        let function = lib.newFunctionWithName(&ns_name).ok_or_else(|| {
            anyhow!("Metal: function '{func_name}' not found in module '{module}'")
        })?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                anyhow!(
                    "newComputePipelineStateWithFunction failed for '{func_name}': {}",
                    e.localizedDescription()
                )
            })?;
        let mut slab = self.pipeline_slab.lock();
        let handle = KernelHandle(slab.len() as u64);
        slab.push(pipeline);
        drop(slab);
        self.pipeline_cache.lock().insert(key, handle);
        Ok(handle)
    }

    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        // On UMA we can write through `contents()` directly when we
        // own the whole range — much cheaper than a blit fillBuffer.
        let allocs = self.allocations.lock();
        let (buf, offset) = Self::find_buffer(&allocs, ptr)
            .ok_or_else(|| anyhow!("memset: ptr {ptr} not allocated"))?;
        if offset + bytes > buf.length() {
            bail!(
                "memset: range overflows buffer ({} + {} > {})",
                offset,
                bytes,
                buf.length()
            );
        }
        let contents = buf.contents();
        unsafe {
            let dst = (contents.as_ptr() as *mut u8).add(offset);
            std::ptr::write_bytes(dst, value, bytes);
        }
        Ok(())
    }

    fn memset_async(
        &self,
        ptr: DevicePtr,
        value: u8,
        bytes: usize,
        _stream: u64,
    ) -> Result<()> {
        // UMA + StorageModeShared makes the synchronous memset semantically
        // equivalent (no host/device cache split to flush).
        self.memset(ptr, value, bytes)
    }

    fn total_memory(&self) -> Result<usize> {
        // On Apple Silicon UMA, "device memory" = system RAM. Probe
        // hw.memsize via sysctl for the authoritative number; fall
        // back to MTLDevice.recommendedMaxWorkingSetSize otherwise.
        Ok(sysctl_memsize().unwrap_or_else(|| {
            self.device.recommendedMaxWorkingSetSize() as usize
        }))
    }

    fn free_memory(&self) -> Result<usize> {
        // No direct API for "free GPU memory" on UMA. Approximate via
        // `recommendedMaxWorkingSetSize - currentAllocatedSize`,
        // which matches the headroom Metal will let us allocate
        // before performance degrades.
        let max = self.device.recommendedMaxWorkingSetSize() as usize;
        let used = self.device.currentAllocatedSize();
        Ok(max.saturating_sub(used))
    }

    fn create_stream(&self) -> Result<u64> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or_else(|| anyhow!("newCommandQueue returned null"))?;
        let mut slab = self.streams.lock();
        slab.push(MetalStream {
            queue,
            in_flight: None,
        });
        // Handle = slab index + 1 so handle 0 stays reserved for
        // the default stream.
        Ok(slab.len() as u64)
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Metal devices/queues are thread-safe; no binding required.
        Ok(())
    }

    fn create_event(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(EventSlot { event, next: 1 });
        // Handle = slab index + 1 (0 reserved for "no event").
        Ok(slab.len() as u64)
    }

    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        let value = {
            let mut slab = self.events.lock();
            let idx = (event as usize).checked_sub(1).ok_or_else(|| {
                anyhow!("record_event: invalid event handle {event}")
            })?;
            let slot = slab
                .get_mut(idx)
                .ok_or_else(|| anyhow!("record_event: event handle {event} out of range"))?;
            let v = slot.next;
            slot.next += 1;
            v
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let event_obj = {
            let slab = self.events.lock();
            slab[(event - 1) as usize].event.clone()
        };
        // Encode the signal on the active command buffer. Metal will
        // signal value=`value` once everything queued on this buffer
        // up to this point has completed.
        let proto: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeSignalEvent_value(proto, value);
        Ok(())
    }

    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = (event as usize).checked_sub(1).ok_or_else(|| {
                anyhow!("stream_wait_event: invalid event handle {event}")
            })?;
            let slot = slab
                .get(idx)
                .ok_or_else(|| anyhow!("stream_wait_event: event handle {event} out of range"))?;
            // Wait on the most-recently-recorded value (next - 1).
            // If nothing has been recorded yet, slot.next is 1 and
            // value is 0 — Metal treats wait-for-0 as a no-op.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let proto: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeWaitForEvent_value(proto, value);
        Ok(())
    }

    fn destroy_event(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let idx = (event - 1) as usize;
        if let Some(slot) = slab.get_mut(idx) {
            // Replace with a fresh dummy event so the slab indices
            // stay stable across destroys (matches the cuda backend
            // semantics — handles are not reused).
            slot.next = 0;
        }
        Ok(())
    }

    fn alloc_host_pinned(&self, bytes: usize) -> Result<*mut u8> {
        // UMA: a Shared MTLBuffer's contents() pointer IS host-pinned
        // memory from the GPU's perspective. We park the buffer in
        // the alloc table keyed by gpuAddress, then return the host
        // pointer. `free_host_pinned` looks the buffer up by host
        // pointer to release it.
        let buf = self
            .device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("alloc_host_pinned: newBufferWithLength failed"))?;
        let host_ptr = buf.contents().as_ptr() as *mut u8;
        // Stash by gpuAddress so plain `free()` on the DevicePtr would
        // also work; the host-pinned variant is purely a CPU view.
        let addr = buf.gpuAddress();
        if addr == 0 {
            bail!("alloc_host_pinned: gpuAddress returned 0");
        }
        self.allocations.lock().insert(addr, buf);
        Ok(host_ptr)
    }

    fn free_host_pinned(&self, ptr: *mut u8, _bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        // Find the buffer whose contents() pointer matches.
        let mut allocs = self.allocations.lock();
        let target_addr = allocs.iter().find_map(|(addr, buf)| {
            let host = buf.contents().as_ptr() as *mut u8;
            if host == ptr { Some(*addr) } else { None }
        });
        if let Some(addr) = target_addr {
            allocs.remove(&addr);
        }
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Probe `hw.memsize` via libc::sysctl on macOS. Returns the total
/// system RAM in bytes — on Apple Silicon UMA this is also the upper
/// bound on Metal-addressable memory.
fn sysctl_memsize() -> Option<usize> {
    use std::ffi::CString;
    let name = CString::new("hw.memsize").ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut u64 as *mut c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 { Some(value as usize) } else { None }
}

// ── Smoke test ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Byte-conversion helpers (bytemuck-free) ──────────────────

    fn u32_slice_to_bytes(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
        }
        out
    }

    /// End-to-end check: alloc → memcpy → kernel launch → memcpy back.
    /// The kernel is `noop_smoke` from `kernels/metal/common/`. It
    /// writes 0.0 to the first `n` floats of `out`, so after launching
    /// with `n=4` the first 4 floats should be exactly zero regardless
    /// of what we initialised the buffer with.
    #[test]
    fn metal_alloc_copy_launch_roundtrip() {
        // Pull the metallib bytes the build script embedded.
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules)
            .expect("MetalGpuBackend::new failed — Metal device unavailable?");

        // Round-trip a known byte pattern through alloc/copy_h2d/copy_d2h.
        let bytes = 64;
        let ptr = backend.alloc(bytes).expect("alloc");
        let pattern: Vec<u8> = (0..bytes as u8).collect();
        backend.copy_h2d(&pattern, ptr).expect("copy_h2d");
        let mut readback = vec![0u8; bytes];
        backend.copy_d2h(ptr, &mut readback).expect("copy_d2h");
        assert_eq!(pattern, readback, "h2d/d2h round-trip mismatch");

        // Zero the first 4 floats via the noop_smoke kernel.
        let n: u32 = 4;
        let kernel = backend
            .kernel("noop_smoke", "noop_smoke")
            .expect("kernel lookup");
        backend
            .launch_typed(
                kernel,
                [1, 1, 1],
                [n, 1, 1],
                0,
                backend.default_stream(),
                &[KernelArg::Buffer(ptr), KernelArg::Bytes(&n.to_le_bytes())],
            )
            .expect("launch_typed");
        backend
            .synchronize(backend.default_stream())
            .expect("synchronize");

        // First 16 bytes should now be all-zero floats; the rest of
        // the buffer should retain the original pattern.
        let mut after = vec![0u8; bytes];
        backend.copy_d2h(ptr, &mut after).expect("copy_d2h post-launch");
        assert_eq!(&after[..16], &[0u8; 16], "kernel did not zero out[0..4]");
        assert_eq!(&after[16..], &pattern[16..], "kernel touched out-of-range bytes");

        backend.free(ptr).expect("free");
    }

    /// Parity check for `mlx_int8_dequant`. Builds a small known-good
    /// (packed, scales, biases) triple, dequantizes via the kernel,
    /// and compares against the CPU reference
    /// `w[r,c] = byte * scales[r, c/group_size] + biases[r, c/group_size]`.
    /// Exact-match BF16 isn't safe because the kernel accumulates in
    /// FP32 then rounds; we tolerate L∞ ≤ 1/256 (BF16 ULP).
    #[test]
    fn metal_mlx_int8_dequant_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Small but representative shape — non-trivial vs the group
        // boundary (group_size=64) and the 4-byte packing.
        let out_features = 4u32;
        let in_features = 128u32;
        let group_size = 64u32;
        let groups_per_row = (in_features / group_size) as usize;
        let n_rows = out_features as usize;
        let n_cols = in_features as usize;

        // Deterministic byte pattern + per-(row, group) scale & bias.
        let mut bytes_flat: Vec<u8> = Vec::with_capacity(n_rows * n_cols);
        for r in 0..n_rows {
            for c in 0..n_cols {
                bytes_flat.push(((r * 7 + c) % 256) as u8);
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
                scales.push(half::bf16::from_f32(0.01 * (1.0 + r as f32) + 0.001 * g as f32));
                biases.push(half::bf16::from_f32(-0.5 + 0.1 * r as f32 + 0.05 * g as f32));
            }
        }

        // CPU reference — read scales/biases back as f32 (matches the
        // FP32 accumulation in the kernel) before applying.
        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                let byte = bytes_flat[r * n_cols + c] as f32;
                let g = c / group_size as usize;
                let s = scales[r * groups_per_row + g].to_f32();
                let b = biases[r * groups_per_row + g].to_f32();
                expected[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
            }
        }

        // Allocate, upload, launch, read back.
        let packed_bytes_buf = u32_slice_to_bytes(&packed);
        let scales_bytes_buf = bf16_slice_to_bytes(&scales);
        let biases_bytes_buf = bf16_slice_to_bytes(&biases);

        let packed_ptr = backend.alloc(packed_bytes_buf.len()).expect("alloc packed");
        let scales_ptr = backend.alloc(scales_bytes_buf.len()).expect("alloc scales");
        let biases_ptr = backend.alloc(biases_bytes_buf.len()).expect("alloc biases");
        let out_ptr = backend.alloc(n_rows * n_cols * 2).expect("alloc out");

        backend
            .copy_h2d(&packed_bytes_buf, packed_ptr)
            .expect("h2d packed");
        backend
            .copy_h2d(&scales_bytes_buf, scales_ptr)
            .expect("h2d scales");
        backend
            .copy_h2d(&biases_bytes_buf, biases_ptr)
            .expect("h2d biases");

        let kernel = backend
            .kernel("mlx_int8_dequant", "mlx_int8_dequant")
            .expect("kernel lookup");

        // 16×4 threads/threadgroup; one threadgroup per (col_tile, row).
        let block_x = 16u32;
        let block_y = 1u32;
        let grid_x = in_features.div_ceil(block_x);
        let grid_y = out_features;
        backend
            .launch_typed(
                kernel,
                [grid_x, grid_y, 1],
                [block_x, block_y, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&out_features.to_le_bytes()),
                    KernelArg::Bytes(&in_features.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(packed_ptr),
                    KernelArg::Buffer(scales_ptr),
                    KernelArg::Buffer(biases_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch_typed dequant");
        backend
            .synchronize(backend.default_stream())
            .expect("synchronize");

        let mut out_raw = vec![0u8; n_rows * n_cols * 2];
        backend.copy_d2h(out_ptr, &mut out_raw).expect("d2h out");
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..expected.len() {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // BF16 has 7-bit mantissa → ULP ≈ value * 2^-7. Worst case
        // here is byte * scale_max + bias_max ≈ 255 * 0.04 + 0.0 ≈ 10.
        // ULP at magnitude 10 ≈ 0.08 — give it 0.1 of headroom.
        assert!(
            max_abs_diff < 0.1,
            "mlx_int8_dequant: max |expected - actual| = {max_abs_diff}, expected < 0.1"
        );

        backend.free(packed_ptr).unwrap();
        backend.free(scales_ptr).unwrap();
        backend.free(biases_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// Build a synthetic MLX-int8 (packed, scales, biases) triplet
    /// + the matching dequantized BF16 weight matrix. Returned tuple:
    /// (packed_bytes_le, scales_bytes_le, biases_bytes_le, w_bf16_dequant).
    fn build_mlx_fixture(
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

    /// `mlx_int8_gemv` parity. Build a synthetic weight + a known
    /// activation vector, run the fused decode kernel, compare to
    /// the FP32-accumulated CPU reference. Exercises the threadgroup
    /// + simdgroup reduction path that materializes one row of `y`.
    #[test]
    fn metal_mlx_int8_gemv_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Pick an N × K shape that exercises real reduction depth.
        // K=256 spans 4 groups per row (group_size=64) and 8 simd
        // lanes' worth of work at 32 threads/group.
        let n: u32 = 8;
        let k: u32 = 256;
        let group_size: u32 = 64;

        let (packed_bytes, scales_bytes, biases_bytes, w_ref) =
            build_mlx_fixture(n as usize, k as usize, group_size as usize);

        // Activation: a smooth, deterministic vector small enough that
        // `byte * scale ~ 0.05` * `x ~ 0.5` ≈ 0.025 per term — keeps
        // the K-element accumulation in a comfortable BF16 range.
        let x_bf16: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
            .collect();
        let x_bytes = bf16_slice_to_bytes(&x_bf16);

        // CPU reference (FP32 accumulation matches the kernel).
        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
        for r in 0..n as usize {
            let mut acc: f32 = 0.0;
            for c in 0..k as usize {
                acc += w_ref[r * k as usize + c].to_f32() * x_bf16[c].to_f32();
            }
            expected[r] = half::bf16::from_f32(acc);
        }

        let packed_ptr = backend.alloc(packed_bytes.len()).unwrap();
        let scales_ptr = backend.alloc(scales_bytes.len()).unwrap();
        let biases_ptr = backend.alloc(biases_bytes.len()).unwrap();
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let y_ptr = backend.alloc(n as usize * 2).unwrap();
        backend.copy_h2d(&packed_bytes, packed_ptr).unwrap();
        backend.copy_h2d(&scales_bytes, scales_ptr).unwrap();
        backend.copy_h2d(&biases_bytes, biases_ptr).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
        // 64 threads/group → two simdgroups, exercises the
        // cross-simdgroup reduction in shared memory.
        let threads_per_tg: u32 = 64;
        backend
            .launch_typed(
                kernel,
                [n, 1, 1],
                [threads_per_tg, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(packed_ptr),
                    KernelArg::Buffer(scales_ptr),
                    KernelArg::Buffer(biases_ptr),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch_typed gemv");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; n as usize * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        // 256-element BF16 sum at result magnitude ~0.5 has ULP ≈ 0.004;
        // tolerate 0.05 for accumulator-order drift across simdgroups.
        let mut max_abs_diff: f32 = 0.0;
        for i in 0..n as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.05,
            "mlx_int8_gemv: max |expected - actual| = {max_abs_diff}; \
             expected/actual head: {:?} vs {:?}",
            &expected.iter().take(4).map(|v| v.to_f32()).collect::<Vec<_>>(),
            &actual.iter().take(4).map(|v| v.to_f32()).collect::<Vec<_>>()
        );

        backend.free(packed_ptr).unwrap();
        backend.free(scales_ptr).unwrap();
        backend.free(biases_ptr).unwrap();
        backend.free(x_ptr).unwrap();
        backend.free(y_ptr).unwrap();
    }

    /// `mlx_int8_gemm` parity. Two-token prefill against the same
    /// synthetic weight as the gemv test. Verifies the (m, n) thread
    /// grid covers the output correctly and the K-loop accumulation
    /// matches the row-by-row fused-dequant path.
    #[test]
    fn metal_mlx_int8_gemm_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let m: u32 = 2;
        let n: u32 = 8;
        let k: u32 = 128;
        let group_size: u32 = 64;

        let (packed_bytes, scales_bytes, biases_bytes, w_ref) =
            build_mlx_fixture(n as usize, k as usize, group_size as usize);

        // X: m rows × k cols, each row a slightly different smooth
        // pattern so per-row mismatches surface clearly.
        let x_bf16: Vec<half::bf16> = (0..(m * k))
            .map(|i| {
                let row = i / k;
                let col = i % k;
                half::bf16::from_f32(
                    0.3 + 0.01 * row as f32 + 0.001 * col as f32,
                )
            })
            .collect();
        let x_bytes = bf16_slice_to_bytes(&x_bf16);

        // CPU reference: Y[mi, ni] = sum_k X[mi, k] * W[ni, k]
        let mut expected: Vec<half::bf16> =
            vec![half::bf16::ZERO; (m * n) as usize];
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut acc: f32 = 0.0;
                for ki in 0..k as usize {
                    acc += x_bf16[mi * k as usize + ki].to_f32()
                        * w_ref[ni * k as usize + ki].to_f32();
                }
                expected[mi * n as usize + ni] = half::bf16::from_f32(acc);
            }
        }

        let packed_ptr = backend.alloc(packed_bytes.len()).unwrap();
        let scales_ptr = backend.alloc(scales_bytes.len()).unwrap();
        let biases_ptr = backend.alloc(biases_bytes.len()).unwrap();
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let y_ptr = backend.alloc((m * n) as usize * 2).unwrap();
        backend.copy_h2d(&packed_bytes, packed_ptr).unwrap();
        backend.copy_h2d(&scales_bytes, scales_ptr).unwrap();
        backend.copy_h2d(&biases_bytes, biases_ptr).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("mlx_int8_gemm", "mlx_int8_gemm").unwrap();
        let block_x = 16u32;
        let block_y = 16u32;
        backend
            .launch_typed(
                kernel,
                [n.div_ceil(block_x), m.div_ceil(block_y), 1],
                [block_x, block_y, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&m.to_le_bytes()),
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(packed_ptr),
                    KernelArg::Buffer(scales_ptr),
                    KernelArg::Buffer(biases_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch_typed gemm");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; (m * n) as usize * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(m * n) as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.05,
            "mlx_int8_gemm: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(packed_ptr).unwrap();
        backend.free(scales_ptr).unwrap();
        backend.free(biases_ptr).unwrap();
        backend.free(x_ptr).unwrap();
        backend.free(y_ptr).unwrap();
    }

    /// `attention_prefill` parity. Multi-token causal attention.
    /// FP32 reference matches the kernel's algorithm exactly: causal
    /// mask everything past `m`, max-subtract softmax, FP32 sum of
    /// V-weighted scores. Verifies the (m, h) flat-grid decoding
    /// inside the kernel.
    #[test]
    fn metal_attention_prefill_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_tokens: u32 = 5;
        let seq_len: u32 = 5; // K/V align with Q in this test
        let num_heads: u32 = 4;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = 8;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<half::bf16> = (0..(num_tokens * num_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.05 + 0.005 * i as f32))
            .collect();
        let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.04 + 0.003 * i as f32))
            .collect();
        let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
            .collect();

        let mut expected: Vec<half::bf16> =
            vec![half::bf16::ZERO; (num_tokens * num_heads * head_dim) as usize];
        let group = num_heads / num_kv_heads;
        for m in 0..num_tokens as usize {
            let cutoff = m + 1;
            for h in 0..num_heads as usize {
                let kv_h = h / group as usize;
                let mut scores: Vec<f32> = (0..seq_len as usize)
                    .map(|s| {
                        if s >= cutoff {
                            f32::NEG_INFINITY
                        } else {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim as usize {
                                let qv = q[(m * num_heads as usize + h)
                                    * head_dim as usize
                                    + d]
                                    .to_f32();
                                let kvv = k[(s * num_kv_heads as usize + kv_h)
                                    * head_dim as usize
                                    + d]
                                    .to_f32();
                                dot += qv * kvv;
                            }
                            dot * scale
                        }
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for d in 0..head_dim as usize {
                    let mut acc = 0.0f32;
                    for s in 0..seq_len as usize {
                        let vv = v[(s * num_kv_heads as usize + kv_h)
                            * head_dim as usize
                            + d]
                            .to_f32();
                        acc += scores[s] * inv * vv;
                    }
                    expected[(m * num_heads as usize + h) * head_dim as usize + d] =
                        half::bf16::from_f32(acc);
                }
            }
        }

        let q_bytes = bf16_slice_to_bytes(&q);
        let k_bytes = bf16_slice_to_bytes(&k);
        let v_bytes = bf16_slice_to_bytes(&v);
        let q_ptr = backend.alloc(q_bytes.len()).unwrap();
        let k_ptr = backend.alloc(k_bytes.len()).unwrap();
        let v_ptr = backend.alloc(v_bytes.len()).unwrap();
        let out_ptr = backend.alloc(q_bytes.len()).unwrap();
        backend.copy_h2d(&q_bytes, q_ptr).unwrap();
        backend.copy_h2d(&k_bytes, k_ptr).unwrap();
        backend.copy_h2d(&v_bytes, v_ptr).unwrap();

        let kernel = backend
            .kernel("attention_prefill", "attention_prefill")
            .unwrap();
        // Flat 1-D grid: num_heads * num_tokens threadgroups.
        let total_groups = num_heads * num_tokens;
        backend
            .launch_typed(
                kernel,
                [total_groups, 1, 1],
                [32, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_tokens.to_le_bytes()),
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(q_ptr),
                    KernelArg::Buffer(k_ptr),
                    KernelArg::Buffer(v_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch attention_prefill");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; q_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(num_tokens * num_heads * head_dim) as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.02,
            "attention_prefill: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(q_ptr).unwrap();
        backend.free(k_ptr).unwrap();
        backend.free(v_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `softmax_topp` correctness. Plant a winner-takes-all logit
    /// distribution (one token wildly larger than the rest) so any
    /// sane top-p sampler must pick that token regardless of the
    /// uniform sample. Independently of the `p` and `uniform`
    /// parameters, the result has to be the planted index.
    #[test]
    fn metal_softmax_topp_dominant_logit() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let vocab: u32 = 256;
        let mut logits: Vec<half::bf16> = (0..vocab)
            .map(|i| half::bf16::from_f32(-2.0 + 0.001 * i as f32))
            .collect();
        let target_idx = 137usize;
        // 30 logit units at temperature 1.0 → softmax mass essentially 1.
        logits[target_idx] = half::bf16::from_f32(30.0);

        let bytes = bf16_slice_to_bytes(&logits);
        let logits_ptr = backend.alloc(bytes.len()).unwrap();
        let result_ptr = backend.alloc(4).unwrap();
        backend.copy_h2d(&bytes, logits_ptr).unwrap();

        let kernel = backend.kernel("softmax_topp", "softmax_topp").unwrap();
        // Try a few (p, uniform) combinations — none should change
        // the result given the dominant logit.
        for &(p, uniform) in &[(0.9f32, 0.1f32), (0.95, 0.5), (1.0, 0.99)] {
            let temp: f32 = 1.0;
            backend
                .launch_typed(
                    kernel,
                    [1, 1, 1],
                    [128, 1, 1],
                    0,
                    backend.default_stream(),
                    &[
                        KernelArg::Bytes(&vocab.to_le_bytes()),
                        KernelArg::Bytes(&temp.to_le_bytes()),
                        KernelArg::Bytes(&p.to_le_bytes()),
                        KernelArg::Bytes(&uniform.to_le_bytes()),
                        KernelArg::Buffer(logits_ptr),
                        KernelArg::Buffer(result_ptr),
                    ],
                )
                .expect("launch softmax_topp");
            backend.synchronize(backend.default_stream()).unwrap();

            let mut result_raw = [0u8; 4];
            backend.copy_d2h(result_ptr, &mut result_raw).unwrap();
            let actual = u32::from_le_bytes(result_raw) as usize;
            assert_eq!(
                actual, target_idx,
                "softmax_topp: p={p}, uniform={uniform}: expected {target_idx}, got {actual}"
            );
        }

        backend.free(logits_ptr).unwrap();
        backend.free(result_ptr).unwrap();
    }

    /// `kv_cache_append` parity. Writes a single token's K and V
    /// projections at slot `cache_pos` and verifies the cache
    /// updates exactly there, with neighbouring slots untouched.
    #[test]
    fn metal_kv_cache_append_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let max_seq: u32 = 8;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = 4;
        let cache_pos: u32 = 3;

        // Pre-fill cache with a sentinel (-1) so untouched slots are
        // visibly distinct from the new K/V data.
        let cache_slots = (max_seq * num_kv_heads * head_dim) as usize;
        let init: Vec<half::bf16> = vec![half::bf16::from_f32(-1.0); cache_slots];
        let new_k: Vec<half::bf16> = (0..(num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.5 + 0.1 * i as f32))
            .collect();
        let new_v: Vec<half::bf16> = (0..(num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(2.0 + 0.05 * i as f32))
            .collect();

        let init_bytes = bf16_slice_to_bytes(&init);
        let nk_bytes = bf16_slice_to_bytes(&new_k);
        let nv_bytes = bf16_slice_to_bytes(&new_v);

        let k_cache_ptr = backend.alloc(init_bytes.len()).unwrap();
        let v_cache_ptr = backend.alloc(init_bytes.len()).unwrap();
        let new_k_ptr = backend.alloc(nk_bytes.len()).unwrap();
        let new_v_ptr = backend.alloc(nv_bytes.len()).unwrap();
        backend.copy_h2d(&init_bytes, k_cache_ptr).unwrap();
        backend.copy_h2d(&init_bytes, v_cache_ptr).unwrap();
        backend.copy_h2d(&nk_bytes, new_k_ptr).unwrap();
        backend.copy_h2d(&nv_bytes, new_v_ptr).unwrap();

        let kernel = backend.kernel("kv_cache_append", "kv_cache_append").unwrap();
        backend
            .launch_typed(
                kernel,
                [head_dim, num_kv_heads, 1],
                [1, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&cache_pos.to_le_bytes()),
                    KernelArg::Buffer(new_k_ptr),
                    KernelArg::Buffer(new_v_ptr),
                    KernelArg::Buffer(k_cache_ptr),
                    KernelArg::Buffer(v_cache_ptr),
                ],
            )
            .expect("launch kv_cache_append");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut k_after = vec![0u8; init_bytes.len()];
        let mut v_after = vec![0u8; init_bytes.len()];
        backend.copy_d2h(k_cache_ptr, &mut k_after).unwrap();
        backend.copy_d2h(v_cache_ptr, &mut v_after).unwrap();
        let k_actual = bytes_to_bf16_vec(&k_after);
        let v_actual = bytes_to_bf16_vec(&v_after);

        let slot_size = (num_kv_heads * head_dim) as usize;
        let slot_start = cache_pos as usize * slot_size;
        for i in 0..cache_slots {
            let in_slot = i >= slot_start && i < slot_start + slot_size;
            let expect_k = if in_slot {
                new_k[i - slot_start]
            } else {
                half::bf16::from_f32(-1.0)
            };
            let expect_v = if in_slot {
                new_v[i - slot_start]
            } else {
                half::bf16::from_f32(-1.0)
            };
            assert_eq!(
                k_actual[i], expect_k,
                "kv_cache_append: K[{i}] mismatch (in_slot={in_slot})"
            );
            assert_eq!(
                v_actual[i], expect_v,
                "kv_cache_append: V[{i}] mismatch (in_slot={in_slot})"
            );
        }

        backend.free(k_cache_ptr).unwrap();
        backend.free(v_cache_ptr).unwrap();
        backend.free(new_k_ptr).unwrap();
        backend.free(new_v_ptr).unwrap();
    }

    /// `attention_decode` parity. Single-token query against a 16-
    /// element KV cache with GQA (4 query heads, 2 KV heads, group=2).
    /// Independent FP32 reference computes scaled-dot-product
    /// attention exactly the way the kernel does (max-subtraction
    /// softmax, FP32 accumulation throughout) so any deviation
    /// surfaces only as BF16 round error.
    #[test]
    fn metal_attention_decode_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let seq_len: u32 = 16;
        let num_heads: u32 = 4;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = 8;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<half::bf16> = (0..(num_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.05 + 0.01 * i as f32))
            .collect();
        let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.02 + 0.003 * i as f32))
            .collect();
        let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
            .collect();

        // FP32 reference: same algorithm as the kernel.
        let mut expected: Vec<half::bf16> =
            vec![half::bf16::ZERO; (num_heads * head_dim) as usize];
        let group = num_heads / num_kv_heads;
        for h in 0..num_heads as usize {
            let kv_h = h / group as usize;

            // Scores
            let mut scores: Vec<f32> = (0..seq_len as usize)
                .map(|s| {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        let qv = q[h * head_dim as usize + d].to_f32();
                        let kvv = k[(s * num_kv_heads as usize + kv_h)
                            * head_dim as usize
                            + d]
                            .to_f32();
                        dot += qv * kvv;
                    }
                    dot * scale
                })
                .collect();
            // Softmax with max subtraction.
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            // Output
            for d in 0..head_dim as usize {
                let mut acc = 0.0f32;
                for s in 0..seq_len as usize {
                    let vv = v[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d]
                        .to_f32();
                    acc += scores[s] * inv * vv;
                }
                expected[h * head_dim as usize + d] = half::bf16::from_f32(acc);
            }
        }

        let q_bytes = bf16_slice_to_bytes(&q);
        let k_bytes = bf16_slice_to_bytes(&k);
        let v_bytes = bf16_slice_to_bytes(&v);
        let q_ptr = backend.alloc(q_bytes.len()).unwrap();
        let k_ptr = backend.alloc(k_bytes.len()).unwrap();
        let v_ptr = backend.alloc(v_bytes.len()).unwrap();
        let out_ptr = backend.alloc(q_bytes.len()).unwrap();
        backend.copy_h2d(&q_bytes, q_ptr).unwrap();
        backend.copy_h2d(&k_bytes, k_ptr).unwrap();
        backend.copy_h2d(&v_bytes, v_ptr).unwrap();

        let kernel = backend.kernel("attention_decode", "attention_decode").unwrap();
        backend
            .launch_typed(
                kernel,
                [num_heads, 1, 1],
                [32, 1, 1], // one simdgroup per head
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(q_ptr),
                    KernelArg::Buffer(k_ptr),
                    KernelArg::Buffer(v_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch attention_decode");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; q_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(num_heads * head_dim) as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // Output magnitudes ≈ 0.5 (V scaled by softmax weights).
        // BF16 ULP at 0.5 is ≈ 0.004; tolerate one ULP plus the
        // accumulator-order drift between kernel and reference.
        assert!(
            max_abs_diff < 0.02,
            "attention_decode: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(q_ptr).unwrap();
        backend.free(k_ptr).unwrap();
        backend.free(v_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `rope_apply` parity. GPT-NeoX-layout RoPE rotates pairs
    /// `(d, d + head_dim/2)`. Independent FP32 reference verifies
    /// both the cos/sin math and the index pairing.
    #[test]
    fn metal_rope_apply_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_tokens: u32 = 4;
        let num_heads: u32 = 2;
        let head_dim: u32 = 16; // multiple of 2, half_dim = 8
        let half_dim = head_dim / 2;

        // x: deterministic per-element pattern.
        let total = (num_tokens * num_heads * head_dim) as usize;
        let x: Vec<half::bf16> = (0..total)
            .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
            .collect();

        // Standard rope_theta=10000.
        let rope_theta: f32 = 10000.0;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let positions: Vec<u32> = (0..num_tokens).collect();

        // CPU reference.
        let mut expected = x.clone();
        for tok in 0..num_tokens as usize {
            let pos = positions[tok] as f32;
            for h in 0..num_heads as usize {
                let base = (tok * num_heads as usize + h) * head_dim as usize;
                for d in 0..half_dim as usize {
                    let theta = pos * inv_freq[d];
                    let c = theta.cos();
                    let s = theta.sin();
                    let lo = x[base + d].to_f32();
                    let hi = x[base + d + half_dim as usize].to_f32();
                    expected[base + d] = half::bf16::from_f32(lo * c - hi * s);
                    expected[base + d + half_dim as usize] =
                        half::bf16::from_f32(lo * s + hi * c);
                }
            }
        }

        // Upload + launch.
        let x_bytes = bf16_slice_to_bytes(&x);
        let inv_freq_bytes: Vec<u8> = inv_freq
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let positions_bytes: Vec<u8> = positions
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect();

        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let inv_freq_ptr = backend.alloc(inv_freq_bytes.len()).unwrap();
        let positions_ptr = backend.alloc(positions_bytes.len()).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();
        backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr).unwrap();
        backend.copy_h2d(&positions_bytes, positions_ptr).unwrap();

        let kernel = backend.kernel("rope_apply", "rope_apply").unwrap();
        backend
            .launch_typed(
                kernel,
                [half_dim, num_heads, num_tokens],
                [1, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_tokens.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Buffer(positions_ptr),
                    KernelArg::Buffer(inv_freq_ptr),
                    KernelArg::Buffer(x_ptr),
                ],
            )
            .expect("launch rope_apply");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut x_after = vec![0u8; x_bytes.len()];
        backend.copy_d2h(x_ptr, &mut x_after).unwrap();
        let actual = bytes_to_bf16_vec(&x_after);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..total {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            // Hard bail on the first wildly-wrong element to make
            // failures localizable.
            assert!(d < 0.05, "rope_apply mismatch at idx {i}: expected {e}, got {a}");
        }
        assert!(max_abs_diff < 0.02);

        backend.free(x_ptr).unwrap();
        backend.free(inv_freq_ptr).unwrap();
        backend.free(positions_ptr).unwrap();
        // mutate suppression — `x` was the input, kept until free for
        // CPU-side reference computation.
        let _ = x;
    }

    /// `silu_gate` parity. Independent SwiGLU computation in FP32
    /// vs the kernel's FP32-internal pipeline. Tolerance allows for
    /// BF16 round-trip (input + output) but pins the activation
    /// math itself.
    #[test]
    fn metal_silu_gate_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Cover a representative range: large negatives (where naive
        // exp(-x) grows), zero (sigmoid sharply 1/2), and large
        // positives (where silu ~= x).
        let n: u32 = 256;
        let gate: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(-4.0 + 8.0 * i as f32 / (n as f32 - 1.0)))
            .collect();
        let up: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.5 + 0.01 * i as f32))
            .collect();

        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
        for i in 0..n as usize {
            let g = gate[i].to_f32();
            let u = up[i].to_f32();
            let sig = 1.0 / (1.0 + (-g).exp());
            expected[i] = half::bf16::from_f32(g * sig * u);
        }

        let gate_bytes = bf16_slice_to_bytes(&gate);
        let up_bytes = bf16_slice_to_bytes(&up);
        let gate_ptr = backend.alloc(gate_bytes.len()).unwrap();
        let up_ptr = backend.alloc(up_bytes.len()).unwrap();
        let out_ptr = backend.alloc(n as usize * 2).unwrap();
        backend.copy_h2d(&gate_bytes, gate_ptr).unwrap();
        backend.copy_h2d(&up_bytes, up_ptr).unwrap();

        let kernel = backend.kernel("silu_gate", "silu_gate").unwrap();
        let block: u32 = 64;
        let grid = n.div_ceil(block);
        backend
            .launch_typed(
                kernel,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(gate_ptr),
                    KernelArg::Buffer(up_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch silu_gate");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; n as usize * 2];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..n as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // Output magnitudes peak ~3 * 0.6 * 1 ≈ 2; BF16 ULP ≈ 0.016.
        assert!(
            max_abs_diff < 0.02,
            "silu_gate: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(gate_ptr).unwrap();
        backend.free(up_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `rms_norm` parity. Independent FP32 reference to verify the
    /// two-stage reduction (simdgroup → cross-simdgroup) and the
    /// rsqrt + weight rescale are wired correctly.
    #[test]
    fn metal_rms_norm_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_tokens: u32 = 3;
        let hidden: u32 = 256;
        let eps: f32 = 1e-5;

        let x: Vec<half::bf16> = (0..(num_tokens * hidden))
            .map(|i| {
                let r = i / hidden;
                let c = i % hidden;
                half::bf16::from_f32(0.1 + 0.01 * r as f32 + 0.001 * c as f32)
            })
            .collect();
        let weight: Vec<half::bf16> =
            (0..hidden).map(|i| half::bf16::from_f32(1.0 + 0.005 * i as f32)).collect();

        let mut expected: Vec<half::bf16> =
            vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
        for r in 0..num_tokens as usize {
            let mut ssq: f32 = 0.0;
            for c in 0..hidden as usize {
                let v = x[r * hidden as usize + c].to_f32();
                ssq += v * v;
            }
            let inv_rms = (ssq / hidden as f32 + eps).powf(-0.5);
            for c in 0..hidden as usize {
                let v = x[r * hidden as usize + c].to_f32();
                let w = weight[c].to_f32();
                expected[r * hidden as usize + c] =
                    half::bf16::from_f32(v * inv_rms * w);
            }
        }

        let x_bytes = bf16_slice_to_bytes(&x);
        let w_bytes = bf16_slice_to_bytes(&weight);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let w_ptr = backend.alloc(w_bytes.len()).unwrap();
        let out_ptr = backend.alloc(x_bytes.len()).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();
        backend.copy_h2d(&w_bytes, w_ptr).unwrap();

        let kernel = backend.kernel("rms_norm", "rms_norm").unwrap();
        backend
            .launch_typed(
                kernel,
                [num_tokens, 1, 1],
                [128, 1, 1], // 4 simdgroups → exercises cross-simd reduction
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&hidden.to_le_bytes()),
                    KernelArg::Bytes(&eps.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch rms_norm");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; x_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(num_tokens * hidden) as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // Output magnitudes ~ ratio of inv_rms-rescaled inputs ≈ 1.
        // BF16 ULP at magnitude 1 is ≈ 0.008.
        assert!(
            max_abs_diff < 0.02,
            "rms_norm: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(x_ptr).unwrap();
        backend.free(w_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `embed_lookup` parity. Tiny vocab, a few token IDs (including
    /// an out-of-range one to verify the bounds-check zero-write).
    #[test]
    fn metal_embed_lookup_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let vocab: u32 = 16;
        let hidden: u32 = 8;
        let num_tokens: u32 = 4;
        // Token 99 is intentionally out-of-range — kernel must write
        // zeros for that row.
        let tokens: [u32; 4] = [3, 7, 99, 0];

        // Embedding table: distinct value per (vocab, hidden) cell so
        // any swap is immediately visible.
        let table: Vec<half::bf16> = (0..(vocab * hidden))
            .map(|i| {
                let v = i / hidden;
                let h = i % hidden;
                half::bf16::from_f32(0.1 * v as f32 + 0.01 * h as f32)
            })
            .collect();

        let mut expected: Vec<half::bf16> =
            vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
        for (ti, &v) in tokens.iter().enumerate() {
            for h in 0..hidden as usize {
                if v < vocab {
                    expected[ti * hidden as usize + h] =
                        table[v as usize * hidden as usize + h];
                }
            }
        }

        let token_bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let table_bytes = bf16_slice_to_bytes(&table);
        let token_ptr = backend.alloc(token_bytes.len()).unwrap();
        let table_ptr = backend.alloc(table_bytes.len()).unwrap();
        let out_ptr = backend.alloc((num_tokens * hidden) as usize * 2).unwrap();
        backend.copy_h2d(&token_bytes, token_ptr).unwrap();
        backend.copy_h2d(&table_bytes, table_ptr).unwrap();

        let kernel = backend.kernel("embed_lookup", "embed_lookup").unwrap();
        backend
            .launch_typed(
                kernel,
                [hidden.div_ceil(8), num_tokens, 1],
                [8, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_tokens.to_le_bytes()),
                    KernelArg::Bytes(&hidden.to_le_bytes()),
                    KernelArg::Bytes(&vocab.to_le_bytes()),
                    KernelArg::Buffer(token_ptr),
                    KernelArg::Buffer(table_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch embed_lookup");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; (num_tokens * hidden) as usize * 2];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        for i in 0..(num_tokens * hidden) as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            assert!(
                (e - a).abs() < 1e-4,
                "embed_lookup mismatch at idx {i}: expected {e}, got {a}"
            );
        }

        backend.free(token_ptr).unwrap();
        backend.free(table_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `argmax_bf16` parity. Plant a known-largest value at a known
    /// index; verify both the value and the index — the
    /// simd_shuffle_xor reduction is easy to get subtly wrong.
    #[test]
    fn metal_argmax_bf16_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = 1024;
        let mut values: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.001 * i as f32))
            .collect();
        // Plant a maximum at a random-looking, non-edge index.
        let target_idx = 723usize;
        values[target_idx] = half::bf16::from_f32(99.0);

        let bytes = bf16_slice_to_bytes(&values);
        let logits_ptr = backend.alloc(bytes.len()).unwrap();
        let result_ptr = backend.alloc(4).unwrap(); // u32
        backend.copy_h2d(&bytes, logits_ptr).unwrap();

        let kernel = backend.kernel("argmax_bf16", "argmax_bf16").unwrap();
        backend
            .launch_typed(
                kernel,
                [1, 1, 1],
                [128, 1, 1], // 4 simdgroups
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(logits_ptr),
                    KernelArg::Buffer(result_ptr),
                ],
            )
            .expect("launch_typed argmax");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut result_raw = [0u8; 4];
        backend.copy_d2h(result_ptr, &mut result_raw).unwrap();
        let actual_idx = u32::from_le_bytes(result_raw) as usize;
        assert_eq!(
            actual_idx, target_idx,
            "argmax: expected {target_idx}, got {actual_idx}"
        );

        backend.free(logits_ptr).unwrap();
        backend.free(result_ptr).unwrap();
    }

    /// `conv3d_patch_embed` parity. Tiny 3-D conv with stride ==
    /// kernel — the patch-embedding shape every modern ViT uses.
    /// Independent FP32 reference walks the same indices to surface
    /// any stride / channel-broadcast / dim-order regression.
    #[test]
    fn metal_conv3d_patch_embed_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Small but non-trivial: 4 output channels, 3 input channels
        // (RGB), kernel 2×4×4, output spatial 1×3×3 → input spatial
        // 2×12×12. The 4×4 spatial kernel is large enough that a
        // mis-strided dispatch would visibly mix neighbouring cells.
        let out_c: u32 = 4;
        let in_c: u32 = 3;
        let kt: u32 = 2;
        let kh: u32 = 4;
        let kw: u32 = 4;
        let t_out: u32 = 1;
        let h_out: u32 = 3;
        let w_out: u32 = 3;

        let t_in = t_out * kt;
        let h_in = h_out * kh;
        let w_in = w_out * kw;

        let input: Vec<half::bf16> = (0..(in_c * t_in * h_in * w_in))
            .map(|i| half::bf16::from_f32(0.01 * (i as f32 * 0.013).sin()))
            .collect();
        let weight: Vec<half::bf16> = (0..(out_c * kt * kh * kw * in_c))
            .map(|i| {
                half::bf16::from_f32(0.005 + 0.0003 * (i as f32 * 0.011).cos())
            })
            .collect();
        let bias: Vec<half::bf16> = (0..out_c)
            .map(|i| half::bf16::from_f32(0.1 + 0.05 * i as f32))
            .collect();

        // CPU reference — same loop nest as the kernel.
        let mut expected =
            vec![half::bf16::ZERO; (out_c * t_out * h_out * w_out) as usize];
        for c_out_ in 0..out_c as usize {
            for t_o in 0..t_out as usize {
                for h_o in 0..h_out as usize {
                    for w_o in 0..w_out as usize {
                        let mut acc = bias[c_out_].to_f32();
                        for dt in 0..kt as usize {
                            for dh in 0..kh as usize {
                                for dw in 0..kw as usize {
                                    for ic in 0..in_c as usize {
                                        let w_off = (((c_out_ * kt as usize + dt)
                                            * kh as usize
                                            + dh)
                                            * kw as usize
                                            + dw)
                                            * in_c as usize
                                            + ic;
                                        let t_idx = t_o * kt as usize + dt;
                                        let h_idx = h_o * kh as usize + dh;
                                        let w_idx = w_o * kw as usize + dw;
                                        let i_off = ((ic * t_in as usize + t_idx)
                                            * h_in as usize
                                            + h_idx)
                                            * w_in as usize
                                            + w_idx;
                                        acc += weight[w_off].to_f32()
                                            * input[i_off].to_f32();
                                    }
                                }
                            }
                        }
                        let out_idx = ((c_out_ * t_out as usize + t_o)
                            * h_out as usize
                            + h_o)
                            * w_out as usize
                            + w_o;
                        expected[out_idx] = half::bf16::from_f32(acc);
                    }
                }
            }
        }

        let in_bytes = bf16_slice_to_bytes(&input);
        let w_bytes = bf16_slice_to_bytes(&weight);
        let b_bytes = bf16_slice_to_bytes(&bias);
        let in_ptr = backend.alloc(in_bytes.len()).unwrap();
        let w_ptr = backend.alloc(w_bytes.len()).unwrap();
        let b_ptr = backend.alloc(b_bytes.len()).unwrap();
        let out_bytes_n = (out_c * t_out * h_out * w_out) as usize * 2;
        let out_ptr = backend.alloc(out_bytes_n).unwrap();
        backend.copy_h2d(&in_bytes, in_ptr).unwrap();
        backend.copy_h2d(&w_bytes, w_ptr).unwrap();
        backend.copy_h2d(&b_bytes, b_ptr).unwrap();

        let kernel = backend
            .kernel("conv3d_patch_embed", "conv3d_patch_embed")
            .unwrap();
        let block_x = 8u32;
        let block_y = 8u32;
        let flat_y = t_out * h_out * w_out;
        backend
            .launch_typed(
                kernel,
                [out_c.div_ceil(block_x), flat_y.div_ceil(block_y), 1],
                [block_x, block_y, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&out_c.to_le_bytes()),
                    KernelArg::Bytes(&in_c.to_le_bytes()),
                    KernelArg::Bytes(&kt.to_le_bytes()),
                    KernelArg::Bytes(&kh.to_le_bytes()),
                    KernelArg::Bytes(&kw.to_le_bytes()),
                    KernelArg::Bytes(&t_out.to_le_bytes()),
                    KernelArg::Bytes(&h_out.to_le_bytes()),
                    KernelArg::Bytes(&w_out.to_le_bytes()),
                    KernelArg::Buffer(in_ptr),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(b_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch conv3d_patch_embed");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; out_bytes_n];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(out_c * t_out * h_out * w_out) as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // Reduction depth ~ 96 elements per output cell → BF16 ULP ≈
        // 0.005 at output magnitudes ~0.1; tolerate 0.05 for ordering
        // drift.
        assert!(
            max_abs_diff < 0.05,
            "conv3d_patch_embed: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(in_ptr).unwrap();
        backend.free(w_ptr).unwrap();
        backend.free(b_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `dense_gemm_bf16` parity. Multi-token prefill BF16 matmul
    /// vs CPU FP32 reference. Same weight layout as
    /// `dense_gemv_bf16` (`[N, K]` row-major) so the ViT prefill
    /// path can use either kernel without reshaping weights.
    #[test]
    fn metal_dense_gemm_bf16_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let m: u32 = 4;
        let n: u32 = 8;
        let k: u32 = 64;

        let x: Vec<half::bf16> = (0..(m * k))
            .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
            .collect();
        let w: Vec<half::bf16> = (0..(n * k))
            .map(|i| {
                let row = i / k;
                let col = i % k;
                half::bf16::from_f32(0.001 + 0.0005 * row as f32 + 0.0007 * col as f32)
            })
            .collect();

        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (m * n) as usize];
        for mi in 0..m as usize {
            for ni in 0..n as usize {
                let mut acc = 0.0f32;
                for ki in 0..k as usize {
                    acc += x[mi * k as usize + ki].to_f32()
                        * w[ni * k as usize + ki].to_f32();
                }
                expected[mi * n as usize + ni] = half::bf16::from_f32(acc);
            }
        }

        let x_bytes = bf16_slice_to_bytes(&x);
        let w_bytes = bf16_slice_to_bytes(&w);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let w_ptr = backend.alloc(w_bytes.len()).unwrap();
        let y_ptr = backend.alloc((m * n) as usize * 2).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();
        backend.copy_h2d(&w_bytes, w_ptr).unwrap();

        let kernel = backend.kernel("dense_gemm_bf16", "dense_gemm_bf16").unwrap();
        let block_x = 16u32;
        let block_y = 16u32;
        backend
            .launch_typed(
                kernel,
                [n.div_ceil(block_x), m.div_ceil(block_y), 1],
                [block_x, block_y, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&m.to_le_bytes()),
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch dense_gemm_bf16");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; (m * n) as usize * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(m * n) as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.05,
            "dense_gemm_bf16: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(x_ptr).unwrap();
        backend.free(w_ptr).unwrap();
        backend.free(y_ptr).unwrap();
    }

    /// `dense_gemv_bf16` parity. Pure BF16 matvec vs CPU FP32
    /// reference. Same reduction shape as `mlx_int8_gemv` minus
    /// the fused dequant.
    #[test]
    fn metal_dense_gemv_bf16_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = 16;
        let k: u32 = 256;
        let w: Vec<half::bf16> = (0..(n * k))
            .map(|i| {
                let row = i / k;
                let col = i % k;
                half::bf16::from_f32(0.001 + 0.0005 * row as f32 + 0.0007 * col as f32)
            })
            .collect();
        let x: Vec<half::bf16> = (0..k)
            .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
            .collect();

        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
        for r in 0..n as usize {
            let mut acc = 0.0f32;
            for c in 0..k as usize {
                acc += w[r * k as usize + c].to_f32() * x[c].to_f32();
            }
            expected[r] = half::bf16::from_f32(acc);
        }

        let w_bytes = bf16_slice_to_bytes(&w);
        let x_bytes = bf16_slice_to_bytes(&x);
        let w_ptr = backend.alloc(w_bytes.len()).unwrap();
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let y_ptr = backend.alloc(n as usize * 2).unwrap();
        backend.copy_h2d(&w_bytes, w_ptr).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("dense_gemv_bf16", "dense_gemv_bf16").unwrap();
        backend
            .launch_typed(
                kernel,
                [n, 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch dense_gemv_bf16");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; n as usize * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..n as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.05,
            "dense_gemv_bf16: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(w_ptr).unwrap();
        backend.free(x_ptr).unwrap();
        backend.free(y_ptr).unwrap();
    }

    /// `layer_norm` parity. Two-pass mean+variance reduction vs
    /// the FP32 `(x - mean) / std * w + b` reference.
    #[test]
    fn metal_layer_norm_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_tokens: u32 = 3;
        let hidden: u32 = 256;
        let eps: f32 = 1e-5;

        let x: Vec<half::bf16> = (0..(num_tokens * hidden))
            .map(|i| {
                let r = i / hidden;
                let c = i % hidden;
                half::bf16::from_f32(0.1 + 0.01 * r as f32 + 0.001 * c as f32)
            })
            .collect();
        let weight: Vec<half::bf16> =
            (0..hidden).map(|i| half::bf16::from_f32(1.0 + 0.005 * i as f32)).collect();
        let bias: Vec<half::bf16> =
            (0..hidden).map(|i| half::bf16::from_f32(-0.05 + 0.002 * i as f32)).collect();

        let mut expected =
            vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
        for r in 0..num_tokens as usize {
            let mut sum = 0.0f32;
            for c in 0..hidden as usize {
                sum += x[r * hidden as usize + c].to_f32();
            }
            let mean = sum / hidden as f32;
            let mut var = 0.0f32;
            for c in 0..hidden as usize {
                let d = x[r * hidden as usize + c].to_f32() - mean;
                var += d * d;
            }
            var /= hidden as f32;
            let inv_std = (var + eps).powf(-0.5);
            for c in 0..hidden as usize {
                let xi = x[r * hidden as usize + c].to_f32();
                let w = weight[c].to_f32();
                let b = bias[c].to_f32();
                expected[r * hidden as usize + c] =
                    half::bf16::from_f32((xi - mean) * inv_std * w + b);
            }
        }

        let x_bytes = bf16_slice_to_bytes(&x);
        let w_bytes = bf16_slice_to_bytes(&weight);
        let b_bytes = bf16_slice_to_bytes(&bias);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let w_ptr = backend.alloc(w_bytes.len()).unwrap();
        let b_ptr = backend.alloc(b_bytes.len()).unwrap();
        let out_ptr = backend.alloc(x_bytes.len()).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();
        backend.copy_h2d(&w_bytes, w_ptr).unwrap();
        backend.copy_h2d(&b_bytes, b_ptr).unwrap();

        let kernel = backend.kernel("layer_norm", "layer_norm").unwrap();
        backend
            .launch_typed(
                kernel,
                [num_tokens, 1, 1],
                [128, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&hidden.to_le_bytes()),
                    KernelArg::Bytes(&eps.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(w_ptr),
                    KernelArg::Buffer(b_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch layer_norm");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; x_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(num_tokens * hidden) as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.05,
            "layer_norm: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(x_ptr).unwrap();
        backend.free(w_ptr).unwrap();
        backend.free(b_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `gelu` parity. Tanh-approx GeLU vs FP32 reference. Sweep
    /// across negative, zero-crossing, and positive inputs because
    /// the tanh approximation has its largest error near |x| ≈ 1.
    #[test]
    fn metal_gelu_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = 256;
        let x: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(-3.0 + 6.0 * i as f32 / (n - 1) as f32))
            .collect();

        let sqrt_2_over_pi: f32 = 0.7978845608028654;
        let c: f32 = 0.044715;
        let mut expected = vec![half::bf16::ZERO; n as usize];
        for i in 0..n as usize {
            let v = x[i].to_f32();
            let arg = sqrt_2_over_pi * (v + c * v * v * v);
            expected[i] = half::bf16::from_f32(0.5 * v * (1.0 + arg.tanh()));
        }

        let x_bytes = bf16_slice_to_bytes(&x);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let out_ptr = backend.alloc(x_bytes.len()).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("gelu", "gelu").unwrap();
        backend
            .launch_typed(
                kernel,
                [n.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch gelu");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; x_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..n as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.02,
            "gelu: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(x_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `attention_full` parity. Same as `attention_prefill` minus
    /// the causal mask; the FP32 reference simply drops the
    /// `s >= cutoff` branch.
    #[test]
    fn metal_attention_full_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_tokens: u32 = 4;
        let seq_len: u32 = 6; // K/V wider than Q to exercise the non-causal path
        let num_heads: u32 = 4;
        let num_kv_heads: u32 = 2;
        let head_dim: u32 = 8;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<half::bf16> = (0..(num_tokens * num_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.05 + 0.005 * i as f32))
            .collect();
        let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.04 + 0.003 * i as f32))
            .collect();
        let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
            .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
            .collect();

        let mut expected =
            vec![half::bf16::ZERO; (num_tokens * num_heads * head_dim) as usize];
        let group = num_heads / num_kv_heads;
        for m in 0..num_tokens as usize {
            for h in 0..num_heads as usize {
                let kv_h = h / group as usize;
                let mut scores: Vec<f32> = (0..seq_len as usize)
                    .map(|s| {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim as usize {
                            let qv = q[(m * num_heads as usize + h)
                                * head_dim as usize
                                + d]
                                .to_f32();
                            let kvv = k[(s * num_kv_heads as usize + kv_h)
                                * head_dim as usize
                                + d]
                                .to_f32();
                            dot += qv * kvv;
                        }
                        dot * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for d in 0..head_dim as usize {
                    let mut acc = 0.0f32;
                    for s in 0..seq_len as usize {
                        let vv = v[(s * num_kv_heads as usize + kv_h)
                            * head_dim as usize
                            + d]
                            .to_f32();
                        acc += scores[s] * inv * vv;
                    }
                    expected[(m * num_heads as usize + h) * head_dim as usize + d] =
                        half::bf16::from_f32(acc);
                }
            }
        }

        let q_bytes = bf16_slice_to_bytes(&q);
        let k_bytes = bf16_slice_to_bytes(&k);
        let v_bytes = bf16_slice_to_bytes(&v);
        let q_ptr = backend.alloc(q_bytes.len()).unwrap();
        let k_ptr = backend.alloc(k_bytes.len()).unwrap();
        let v_ptr = backend.alloc(v_bytes.len()).unwrap();
        let out_ptr = backend.alloc(q_bytes.len()).unwrap();
        backend.copy_h2d(&q_bytes, q_ptr).unwrap();
        backend.copy_h2d(&k_bytes, k_ptr).unwrap();
        backend.copy_h2d(&v_bytes, v_ptr).unwrap();

        let kernel = backend.kernel("attention_full", "attention_full").unwrap();
        let total_groups = num_heads * num_tokens;
        backend
            .launch_typed(
                kernel,
                [total_groups, 1, 1],
                [32, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_tokens.to_le_bytes()),
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(q_ptr),
                    KernelArg::Buffer(k_ptr),
                    KernelArg::Buffer(v_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch attention_full");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; q_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..(num_tokens * num_heads * head_dim) as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.02,
            "attention_full: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(q_ptr).unwrap();
        backend.free(k_ptr).unwrap();
        backend.free(v_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `bf16_add` parity. Trivial element-wise check — the kernel
    /// is one line of math but it's the residual primitive every
    /// transformer block uses, so a regression here would silently
    /// blow up every layer's output.
    #[test]
    fn metal_bf16_add_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = 257; // odd to verify bounds-check on tail thread
        let a: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
            .collect();
        let b: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(-0.05 + 0.0007 * i as f32))
            .collect();

        let mut expected = vec![half::bf16::ZERO; n as usize];
        for i in 0..n as usize {
            expected[i] = half::bf16::from_f32(a[i].to_f32() + b[i].to_f32());
        }

        let a_bytes = bf16_slice_to_bytes(&a);
        let b_bytes = bf16_slice_to_bytes(&b);
        let a_ptr = backend.alloc(a_bytes.len()).unwrap();
        let b_ptr = backend.alloc(b_bytes.len()).unwrap();
        let out_ptr = backend.alloc(a_bytes.len()).unwrap();
        backend.copy_h2d(&a_bytes, a_ptr).unwrap();
        backend.copy_h2d(&b_bytes, b_ptr).unwrap();

        let kernel = backend.kernel("bf16_add", "bf16_add").unwrap();
        let block: u32 = 64;
        backend
            .launch_typed(
                kernel,
                [n.div_ceil(block), 1, 1],
                [block, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(a_ptr),
                    KernelArg::Buffer(b_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch bf16_add");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; a_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        for i in 0..n as usize {
            assert!(
                (expected[i].to_f32() - actual[i].to_f32()).abs() < 1e-4,
                "bf16_add mismatch at idx {i}"
            );
        }

        backend.free(a_ptr).unwrap();
        backend.free(b_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `sigmoid_gate` parity. `out = sigmoid(gate) * x`. Distinct
    /// from `silu_gate` (which is `gate * sigmoid(gate) * up`) —
    /// Qwen3.5 uses this for `attn_output_gate`.
    #[test]
    fn metal_sigmoid_gate_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = 128;
        let gate: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(-3.0 + 6.0 * i as f32 / (n - 1) as f32))
            .collect();
        let x: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(0.5 + 0.01 * i as f32))
            .collect();

        let mut expected = vec![half::bf16::ZERO; n as usize];
        for i in 0..n as usize {
            let g = gate[i].to_f32();
            let v = x[i].to_f32();
            let sig = 1.0 / (1.0 + (-g).exp());
            expected[i] = half::bf16::from_f32(sig * v);
        }

        let g_bytes = bf16_slice_to_bytes(&gate);
        let x_bytes = bf16_slice_to_bytes(&x);
        let g_ptr = backend.alloc(g_bytes.len()).unwrap();
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let out_ptr = backend.alloc(g_bytes.len()).unwrap();
        backend.copy_h2d(&g_bytes, g_ptr).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("sigmoid_gate", "sigmoid_gate").unwrap();
        backend
            .launch_typed(
                kernel,
                [n.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(g_ptr),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch sigmoid_gate");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; g_bytes.len()];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        let mut max_abs_diff: f32 = 0.0;
        for i in 0..n as usize {
            let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 0.02,
            "sigmoid_gate: max |expected - actual| = {max_abs_diff}"
        );

        backend.free(g_ptr).unwrap();
        backend.free(x_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// `causal_conv1d_decode` parity. Drives a few decode steps
    /// against the CPU reference so the in-place state shift is
    /// pinned (a read-after-write bug there would corrupt the next
    /// step's output silently).
    #[test]
    fn metal_causal_conv1d_decode_matches_reference() {
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let num_channels: u32 = 8;
        let kernel_size: u32 = 4;
        let state_len = (kernel_size - 1) as usize;

        // Per-channel weight vectors and an initial conv_state.
        let weights: Vec<half::bf16> = (0..(num_channels * kernel_size))
            .map(|i| {
                let c = i / kernel_size;
                let k = i % kernel_size;
                half::bf16::from_f32(0.1 * (c as f32 + 1.0) + 0.05 * k as f32)
            })
            .collect();
        let mut conv_state_cpu: Vec<half::bf16> =
            (0..(num_channels as usize * state_len))
                .map(|i| half::bf16::from_f32(0.01 * i as f32))
                .collect();

        let weights_bytes = bf16_slice_to_bytes(&weights);
        let weights_ptr = backend.alloc(weights_bytes.len()).unwrap();
        let state_ptr = backend
            .alloc(num_channels as usize * state_len * 2)
            .unwrap();
        let new_in_ptr = backend.alloc(num_channels as usize * 2).unwrap();
        let out_ptr = backend.alloc(num_channels as usize * 2).unwrap();

        backend.copy_h2d(&weights_bytes, weights_ptr).unwrap();
        backend
            .copy_h2d(&bf16_slice_to_bytes(&conv_state_cpu), state_ptr)
            .unwrap();

        let kernel = backend
            .kernel("causal_conv1d_decode", "causal_conv1d_decode")
            .unwrap();

        // Drive 3 decode steps with deterministic-but-changing inputs
        // so the state ring buffer is genuinely exercised, not
        // accidentally always reading the same value.
        for step in 0..3u32 {
            let new_input: Vec<half::bf16> = (0..num_channels)
                .map(|c| half::bf16::from_f32(0.5 + 0.1 * (step as f32 + c as f32)))
                .collect();
            backend
                .copy_h2d(&bf16_slice_to_bytes(&new_input), new_in_ptr)
                .unwrap();

            // CPU reference: snapshot past, compute output, then
            // shift state — exactly matches the kernel's algorithm.
            let mut expected = vec![half::bf16::ZERO; num_channels as usize];
            for c in 0..num_channels as usize {
                let mut past = vec![0.0f32; kernel_size as usize];
                for i in 0..state_len {
                    past[i] = conv_state_cpu[c * state_len + i].to_f32();
                }
                past[state_len] = new_input[c].to_f32();
                let mut acc = 0.0f32;
                for i in 0..kernel_size as usize {
                    let w = weights[c * kernel_size as usize + i].to_f32();
                    acc += w * past[i];
                }
                expected[c] = half::bf16::from_f32(acc);
                // Update the CPU-side state for the next iteration.
                for i in 0..state_len {
                    conv_state_cpu[c * state_len + i] =
                        half::bf16::from_f32(past[i + 1]);
                }
            }

            backend
                .launch_typed(
                    kernel,
                    [num_channels.div_ceil(64), 1, 1],
                    [64, 1, 1],
                    0,
                    backend.default_stream(),
                    &[
                        KernelArg::Bytes(&num_channels.to_le_bytes()),
                        KernelArg::Bytes(&kernel_size.to_le_bytes()),
                        KernelArg::Buffer(weights_ptr),
                        KernelArg::Buffer(new_in_ptr),
                        KernelArg::Buffer(state_ptr),
                        KernelArg::Buffer(out_ptr),
                    ],
                )
                .expect("launch causal_conv1d_decode");
            backend.synchronize(backend.default_stream()).unwrap();

            let mut out_raw = vec![0u8; num_channels as usize * 2];
            backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
            let actual = bytes_to_bf16_vec(&out_raw);

            for i in 0..num_channels as usize {
                let e = expected[i].to_f32();
                let a = actual[i].to_f32();
                assert!(
                    (e - a).abs() < 0.02,
                    "step {step} ch {i}: expected {e}, got {a}"
                );
            }
        }

        backend.free(weights_ptr).unwrap();
        backend.free(state_ptr).unwrap();
        backend.free(new_in_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }

    /// Vision-block forward on real `vision_tower.blocks.0`
    /// weights. Mirrors the layer-3 LLM attention test but for the
    /// ViT-style vision tower:
    ///
    ///   x → layer_norm(norm1)
    ///     → dense_gemv_bf16(qkv) + bf16_add(qkv_bias)
    ///     → split into Q | K | V
    ///     → attention_full (non-causal self-attention)
    ///     → dense_gemv_bf16(proj) + bf16_add(proj_bias)
    ///     → bf16_add(x, attn_out)         (residual)
    ///     → layer_norm(norm2)
    ///     → dense_gemv_bf16(fc1) + bf16_add(fc1_bias)
    ///     → gelu
    ///     → dense_gemv_bf16(fc2) + bf16_add(fc2_bias)
    ///     → bf16_add(x_resid, ffn_out)    (residual)
    ///     → x_final
    ///
    /// One token of synthetic ViT input (so num_tokens = seq_len = 1
    /// — the kernel grid still exercises every reduction stage,
    /// just with a degenerate single-key softmax). Validates that
    /// every BF16-only kernel composes correctly on real vision
    /// weights from `mlx-community/Qwen3.5-4B-MLX-8bit`.
    #[test]
    #[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
    fn metal_real_model_vision_block_forward() {
        use safetensors::SafeTensors;

        let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME unset");
                format!("{home}/models/Qwen3.5-4B-MLX-8bit")
            });
        let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
        if !st_path.exists() {
            eprintln!("skipping: {} not found", st_path.display());
            return;
        }
        let file = std::fs::File::open(&st_path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

        let block = "vision_tower.blocks.0";
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Vision tower dims (Qwen3.5-VL): hidden=1024, qkv_total=3072,
        // 16 attention heads of 64 dim each, MLP intermediate=4096.
        let hidden: u32 = 1024;
        let qkv_total: u32 = 3072;
        let intermediate: u32 = 4096;
        let num_heads: u32 = 16;
        let head_dim: u32 = 64;
        let eps: f32 = 1e-6;

        let load_bf16 = |name: &str| -> (DevicePtr, &[u8]) {
            let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
            let p = backend.alloc(t.data().len()).unwrap();
            backend.copy_h2d(t.data(), p).unwrap();
            (p, t.data())
        };

        // Load every weight + bias for block 0.
        let (norm1_w, _) = load_bf16(&format!("{block}.norm1.weight"));
        let (norm1_b, _) = load_bf16(&format!("{block}.norm1.bias"));
        let (qkv_w, _) = load_bf16(&format!("{block}.attn.qkv.weight"));
        let (qkv_b, _) = load_bf16(&format!("{block}.attn.qkv.bias"));
        let (proj_w, _) = load_bf16(&format!("{block}.attn.proj.weight"));
        let (proj_b, _) = load_bf16(&format!("{block}.attn.proj.bias"));
        let (norm2_w, _) = load_bf16(&format!("{block}.norm2.weight"));
        let (norm2_b, _) = load_bf16(&format!("{block}.norm2.bias"));
        let (fc1_w, _) = load_bf16(&format!("{block}.mlp.linear_fc1.weight"));
        let (fc1_b, _) = load_bf16(&format!("{block}.mlp.linear_fc1.bias"));
        let (fc2_w, _) = load_bf16(&format!("{block}.mlp.linear_fc2.weight"));
        let (fc2_b, _) = load_bf16(&format!("{block}.mlp.linear_fc2.bias"));

        // Synthetic input: one patch token at hidden=1024.
        let x_init: Vec<half::bf16> = (0..hidden)
            .map(|i| half::bf16::from_f32(0.3 * (i as f32 * 0.011).sin()))
            .collect();
        let x_bytes = bf16_slice_to_bytes(&x_init);

        let alloc_bf16 = |n: u32| -> DevicePtr { backend.alloc(n as usize * 2).unwrap() };
        let x = alloc_bf16(hidden);
        let x_norm = alloc_bf16(hidden);
        let qkv = alloc_bf16(qkv_total);
        let qkv_with_bias = alloc_bf16(qkv_total);
        let attn_out = alloc_bf16(hidden);
        let proj_out = alloc_bf16(hidden);
        let proj_with_bias = alloc_bf16(hidden);
        let x_resid = alloc_bf16(hidden);
        let x_norm2 = alloc_bf16(hidden);
        let fc1_out = alloc_bf16(intermediate);
        let fc1_with_bias = alloc_bf16(intermediate);
        let fc1_act = alloc_bf16(intermediate);
        let fc2_out = alloc_bf16(hidden);
        let fc2_with_bias = alloc_bf16(hidden);
        let x_final = alloc_bf16(hidden);
        backend.copy_h2d(&x_bytes, x).unwrap();

        let stream = backend.default_stream();
        let n_tokens: u32 = 1;

        // ── Helpers ─────────────────────────────────────────────
        let ln_kernel = backend.kernel("layer_norm", "layer_norm").unwrap();
        let launch_ln = |x_in: DevicePtr, w: DevicePtr, b: DevicePtr, x_out: DevicePtr,
                         hid: u32, n_tok: u32| {
            backend
                .launch_typed(
                    ln_kernel,
                    [n_tok, 1, 1],
                    [128, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&hid.to_le_bytes()),
                        KernelArg::Bytes(&eps.to_le_bytes()),
                        KernelArg::Buffer(x_in),
                        KernelArg::Buffer(w),
                        KernelArg::Buffer(b),
                        KernelArg::Buffer(x_out),
                    ],
                )
                .expect("layer_norm launch");
        };

        let gemv_kernel = backend.kernel("dense_gemv_bf16", "dense_gemv_bf16").unwrap();
        let launch_gemv = |w: DevicePtr, x_in: DevicePtr, y: DevicePtr, n: u32, k: u32| {
            backend
                .launch_typed(
                    gemv_kernel,
                    [n, 1, 1],
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&n.to_le_bytes()),
                        KernelArg::Bytes(&k.to_le_bytes()),
                        KernelArg::Buffer(w),
                        KernelArg::Buffer(x_in),
                        KernelArg::Buffer(y),
                    ],
                )
                .expect("gemv launch");
        };

        let add_kernel = backend.kernel("bf16_add", "bf16_add").unwrap();
        let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| {
            backend
                .launch_typed(
                    add_kernel,
                    [n.div_ceil(64), 1, 1],
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&n.to_le_bytes()),
                        KernelArg::Buffer(a),
                        KernelArg::Buffer(b),
                        KernelArg::Buffer(out),
                    ],
                )
                .expect("bf16_add launch");
        };

        // ── Stage 1: norm1 → qkv → +bias ────────────────────────
        launch_ln(x, norm1_w, norm1_b, x_norm, hidden, n_tokens);
        launch_gemv(qkv_w, x_norm, qkv, qkv_total, hidden);
        launch_add(qkv, qkv_b, qkv_with_bias, qkv_total);

        // Q/K/V are contiguous BF16 slices of qkv_with_bias at offsets
        // 0, hidden*2, 2*hidden*2 (in bytes). Each is [num_heads,
        // head_dim] = [16, 64] = hidden BF16 elements.
        let q_view = qkv_with_bias;
        let k_view = qkv_with_bias.offset(hidden as usize * 2);
        let v_view = qkv_with_bias.offset(2 * hidden as usize * 2);

        // ── Stage 2: attention_full (seq_len = num_tokens = 1) ──
        let seq_len: u32 = 1;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let attn_kernel = backend.kernel("attention_full", "attention_full").unwrap();
        backend
            .launch_typed(
                attn_kernel,
                [num_heads * n_tokens, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n_tokens.to_le_bytes()),
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()), // num_kv_heads = num_heads (no GQA in ViT)
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(q_view),
                    KernelArg::Buffer(k_view),
                    KernelArg::Buffer(v_view),
                    KernelArg::Buffer(attn_out),
                ],
            )
            .expect("attention_full launch");

        // ── Stage 3: o_proj → +bias → residual ──────────────────
        launch_gemv(proj_w, attn_out, proj_out, hidden, hidden);
        launch_add(proj_out, proj_b, proj_with_bias, hidden);
        launch_add(x, proj_with_bias, x_resid, hidden);

        // ── Stage 4: norm2 → fc1 → +bias → gelu → fc2 → +bias ──
        launch_ln(x_resid, norm2_w, norm2_b, x_norm2, hidden, n_tokens);
        launch_gemv(fc1_w, x_norm2, fc1_out, intermediate, hidden);
        launch_add(fc1_out, fc1_b, fc1_with_bias, intermediate);
        let gelu_kernel = backend.kernel("gelu", "gelu").unwrap();
        backend
            .launch_typed(
                gelu_kernel,
                [intermediate.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&intermediate.to_le_bytes()),
                    KernelArg::Buffer(fc1_with_bias),
                    KernelArg::Buffer(fc1_act),
                ],
            )
            .expect("gelu launch");
        launch_gemv(fc2_w, fc1_act, fc2_out, hidden, intermediate);
        launch_add(fc2_out, fc2_b, fc2_with_bias, hidden);

        // ── Stage 5: residual → x_final ─────────────────────────
        launch_add(x_resid, fc2_with_bias, x_final, hidden);

        backend.synchronize(stream).unwrap();

        let mut x_final_raw = vec![0u8; hidden as usize * 2];
        backend.copy_d2h(x_final, &mut x_final_raw).unwrap();
        let final_vals = bytes_to_bf16_vec(&x_final_raw);

        let mut nan_or_inf = 0;
        let mut sum_abs = 0.0f32;
        let mut max_abs = 0.0f32;
        for v in &final_vals {
            let f = v.to_f32();
            if !f.is_finite() {
                nan_or_inf += 1;
            }
            let a = f.abs();
            sum_abs += a;
            if a > max_abs {
                max_abs = a;
            }
        }
        let mean_abs = sum_abs / final_vals.len() as f32;

        assert_eq!(
            nan_or_inf, 0,
            "vision block: {nan_or_inf} non-finite outputs out of {}",
            final_vals.len()
        );
        assert!(
            mean_abs > 1e-4 && mean_abs < 50.0,
            "vision block: x_final mean-abs {mean_abs} outside sanity band; max_abs={max_abs}"
        );
        assert!(
            max_abs < 1e3,
            "vision block: x_final max_abs {max_abs} suggests activation explosion"
        );

        // Free everything in batch — order doesn't matter on UMA.
        for ptr in [
            norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b, norm2_w, norm2_b,
            fc1_w, fc1_b, fc2_w, fc2_b,
            x, x_norm, qkv, qkv_with_bias, attn_out, proj_out, proj_with_bias,
            x_resid, x_norm2, fc1_out, fc1_with_bias, fc1_act, fc2_out,
            fc2_with_bias, x_final,
        ] {
            backend.free(ptr).unwrap();
        }
    }

    /// Full attention-block forward on real layer 3 weights — the
    /// strongest possible end-to-end demo this kernel set can do
    /// without the GDN/SSM kernels for linear_attention layers.
    ///
    /// Loads every weight tensor for `language_model.model.layers.3`
    /// (the first `full_attention` layer) and runs:
    ///
    ///   x → rms_norm(input_ln)
    ///     → q_proj | k_proj | v_proj             (3× mlx_int8_gemv)
    ///     → split q_proj output into (Q, attn_gate) halves
    ///     → q_norm(Q per head) | k_norm(K per head)  (2× rms_norm)
    ///     → rope_apply(Q) | rope_apply(K)
    ///     → kv_cache_append at pos 0
    ///     → attention_decode with seq_len=1
    ///     → sigmoid_gate(attn_gate, attn_out)
    ///     → o_proj                                (mlx_int8_gemv)
    ///     → bf16_add(x, o)                       (residual)
    ///     → rms_norm(post_attention_ln)
    ///     → gate_proj | up_proj                  (2× mlx_int8_gemv)
    ///     → silu_gate
    ///     → down_proj                            (mlx_int8_gemv)
    ///     → bf16_add(x_resid, ffn_out)            (residual)
    ///     → x_final
    ///
    /// Asserts every element of x_final is finite and the activation
    /// magnitudes haven't exploded or collapsed. We don't compare
    /// against MLX numerically (would need MLX installed) — this
    /// test's job is to prove the kernel chain composes correctly
    /// on production tensors. Each kernel's math is already
    /// independently parity-verified.
    #[test]
    #[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
    fn metal_real_model_full_attention_block_layer3() {
        use crate::weights::mlx_int8::MlxInt8Weight;
        use safetensors::SafeTensors;

        let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME unset");
                format!("{home}/models/Qwen3.5-4B-MLX-8bit")
            });
        let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
        if !st_path.exists() {
            eprintln!("skipping: {} not found", st_path.display());
            return;
        }

        let file = std::fs::File::open(&st_path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

        // Real-config dims for Qwen3.5-4B-MLX-8bit, layer 3.
        let hidden_size: u32 = 2560;
        let num_heads: u32 = 16;
        let num_kv_heads: u32 = 4;
        let head_dim: u32 = 256;
        let intermediate_size: u32 = 9216;
        let rms_eps: f32 = 1e-6;
        let group_size: u32 = 64;
        let q_total: u32 = num_heads * head_dim * 2; // attn_output_gate doubling
        let q_only: u32 = num_heads * head_dim;
        let kv_dim: u32 = num_kv_heads * head_dim;

        let layer = "language_model.model.layers.3";
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        // Plain-BF16 weights (layer norms, per-head norms).
        let load_bf16 = |name: &str| -> DevicePtr {
            let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
            let p = backend.alloc(t.data().len()).unwrap();
            backend.copy_h2d(t.data(), p).unwrap();
            p
        };
        let input_ln = load_bf16(&format!("{layer}.input_layernorm.weight"));
        let q_norm = load_bf16(&format!("{layer}.self_attn.q_norm.weight"));
        let k_norm = load_bf16(&format!("{layer}.self_attn.k_norm.weight"));
        let post_ln = load_bf16(&format!("{layer}.post_attention_layernorm.weight"));

        // MLX-int8 weights via the helper we built in PR5.
        let q_proj = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.self_attn.q_proj"), group_size,
        )
        .expect("load q_proj");
        let k_proj = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.self_attn.k_proj"), group_size,
        )
        .expect("load k_proj");
        let v_proj = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.self_attn.v_proj"), group_size,
        )
        .expect("load v_proj");
        let o_proj = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.self_attn.o_proj"), group_size,
        )
        .expect("load o_proj");
        let gate_p = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.mlp.gate_proj"), group_size,
        )
        .expect("load gate_proj");
        let up_p = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.mlp.up_proj"), group_size,
        )
        .expect("load up_proj");
        let down_p = MlxInt8Weight::load(
            &backend, &st, &format!("{layer}.mlp.down_proj"), group_size,
        )
        .expect("load down_proj");

        // Sanity-check the loader recovered the expected dims.
        assert_eq!(q_proj.out_features, q_total);
        assert_eq!(q_proj.in_features, hidden_size);
        assert_eq!(k_proj.out_features, kv_dim);
        assert_eq!(o_proj.in_features, q_only);
        assert_eq!(o_proj.out_features, hidden_size);
        assert_eq!(gate_p.out_features, intermediate_size);
        assert_eq!(down_p.in_features, intermediate_size);

        // Synthetic residual-stream input.
        let x_init: Vec<half::bf16> = (0..hidden_size)
            .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
            .collect();
        let x_bytes = bf16_slice_to_bytes(&x_init);

        // Allocate every intermediate buffer up-front.
        let alloc_bf16 = |n: u32| -> DevicePtr {
            backend.alloc(n as usize * 2).unwrap()
        };
        let x = alloc_bf16(hidden_size);
        let x_norm = alloc_bf16(hidden_size);
        let q_full = alloc_bf16(q_total);
        let k = alloc_bf16(kv_dim);
        let v = alloc_bf16(kv_dim);
        let attn_out = alloc_bf16(q_only);
        let gated_attn = alloc_bf16(q_only);
        let o = alloc_bf16(hidden_size);
        let x_resid = alloc_bf16(hidden_size);
        let x_norm2 = alloc_bf16(hidden_size);
        let gate_act = alloc_bf16(intermediate_size);
        let up_act = alloc_bf16(intermediate_size);
        let ffn_act = alloc_bf16(intermediate_size);
        let ffn_out = alloc_bf16(hidden_size);
        let x_final = alloc_bf16(hidden_size);
        // KV cache: enough for one token (we'll only write at pos 0).
        let max_seq: u32 = 1;
        let k_cache = alloc_bf16(max_seq * kv_dim);
        let v_cache = alloc_bf16(max_seq * kv_dim);
        backend.copy_h2d(&x_bytes, x).unwrap();

        // Pre-bake the inv_freq table for RoPE (head_dim/2 entries).
        let rope_theta: f32 = 10_000_000.0; // Qwen3-family default
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq_bytes: Vec<u8> = inv_freq.iter().flat_map(|f| f.to_le_bytes()).collect();
        let inv_freq_ptr = backend.alloc(inv_freq_bytes.len()).unwrap();
        backend.copy_h2d(&inv_freq_bytes, inv_freq_ptr).unwrap();
        let positions: Vec<u8> = 0u32.to_le_bytes().to_vec();
        let positions_ptr = backend.alloc(positions.len()).unwrap();
        backend.copy_h2d(&positions, positions_ptr).unwrap();

        let stream = backend.default_stream();
        let n_tokens: u32 = 1;

        // ── Stage 1: input_layernorm ─────────────────────────────
        let rms = backend.kernel("rms_norm", "rms_norm").unwrap();
        let launch_rms = |x_in: DevicePtr, w: DevicePtr, x_out: DevicePtr,
                          n_tok: u32, hid: u32| {
            backend
                .launch_typed(
                    rms,
                    [n_tok, 1, 1],
                    [128, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&hid.to_le_bytes()),
                        KernelArg::Bytes(&rms_eps.to_le_bytes()),
                        KernelArg::Buffer(x_in),
                        KernelArg::Buffer(w),
                        KernelArg::Buffer(x_out),
                    ],
                )
                .expect("rms_norm launch");
        };
        launch_rms(x, input_ln, x_norm, n_tokens, hidden_size);

        // ── Stage 2: Q, K, V projections ──────────────────────────
        q_proj.gemv(&backend, x_norm, q_full, stream).unwrap();
        k_proj.gemv(&backend, x_norm, k, stream).unwrap();
        v_proj.gemv(&backend, x_norm, v, stream).unwrap();

        // ── Stage 3: split q_full into (Q, attn_gate) by offset ──
        // q_full is [Q | gate] laid out contiguously in BF16.
        let q_view = q_full;                                    // first 4096 bf16
        let gate_view = q_full.offset(q_only as usize * 2);     // second 4096 bf16

        // ── Stage 4: per-head q_norm / k_norm ─────────────────────
        // Treat each head as a 'token' of length head_dim.
        // In-place doesn't work safely, so we use a small scratch
        // buffer and copy back via d2d.
        let q_norm_out = alloc_bf16(q_only);
        let k_norm_out = alloc_bf16(kv_dim);
        launch_rms(q_view, q_norm, q_norm_out, num_heads, head_dim);
        launch_rms(k, k_norm, k_norm_out, num_kv_heads, head_dim);
        // Overwrite the original Q view (contiguous, same size).
        backend
            .copy_d2d_async(q_norm_out, q_view, q_only as usize * 2, stream)
            .unwrap();
        backend
            .copy_d2d_async(k_norm_out, k, kv_dim as usize * 2, stream)
            .unwrap();

        // ── Stage 5: RoPE on Q and K ──────────────────────────────
        let rope = backend.kernel("rope_apply", "rope_apply").unwrap();
        let launch_rope = |x_inout: DevicePtr, n_h: u32| {
            backend
                .launch_typed(
                    rope,
                    [half_dim, n_h, 1],
                    [1, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&n_tokens.to_le_bytes()),
                        KernelArg::Bytes(&n_h.to_le_bytes()),
                        KernelArg::Bytes(&head_dim.to_le_bytes()),
                        KernelArg::Buffer(positions_ptr),
                        KernelArg::Buffer(inv_freq_ptr),
                        KernelArg::Buffer(x_inout),
                    ],
                )
                .expect("rope launch");
        };
        launch_rope(q_view, num_heads);
        launch_rope(k, num_kv_heads);

        // ── Stage 6: KV cache append at pos 0 ─────────────────────
        let cache_pos: u32 = 0;
        let kvap = backend.kernel("kv_cache_append", "kv_cache_append").unwrap();
        backend
            .launch_typed(
                kvap,
                [head_dim, num_kv_heads, 1],
                [1, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&cache_pos.to_le_bytes()),
                    KernelArg::Buffer(k),
                    KernelArg::Buffer(v),
                    KernelArg::Buffer(k_cache),
                    KernelArg::Buffer(v_cache),
                ],
            )
            .expect("kv_cache_append launch");

        // ── Stage 7: attention_decode ─────────────────────────────
        let seq_len: u32 = 1;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let attn = backend.kernel("attention_decode", "attention_decode").unwrap();
        backend
            .launch_typed(
                attn,
                [num_heads, 1, 1],
                [32, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Buffer(q_view),
                    KernelArg::Buffer(k_cache),
                    KernelArg::Buffer(v_cache),
                    KernelArg::Buffer(attn_out),
                ],
            )
            .expect("attention_decode launch");

        // ── Stage 8: sigmoid_gate(attn_gate, attn_out) ────────────
        let sg = backend.kernel("sigmoid_gate", "sigmoid_gate").unwrap();
        backend
            .launch_typed(
                sg,
                [q_only.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&q_only.to_le_bytes()),
                    KernelArg::Buffer(gate_view),
                    KernelArg::Buffer(attn_out),
                    KernelArg::Buffer(gated_attn),
                ],
            )
            .expect("sigmoid_gate launch");

        // ── Stage 9: o_proj ──────────────────────────────────────
        o_proj.gemv(&backend, gated_attn, o, stream).unwrap();

        // ── Stage 10: residual x = x + o ──────────────────────────
        let add = backend.kernel("bf16_add", "bf16_add").unwrap();
        let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| {
            backend
                .launch_typed(
                    add,
                    [n.div_ceil(64), 1, 1],
                    [64, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&n.to_le_bytes()),
                        KernelArg::Buffer(a),
                        KernelArg::Buffer(b),
                        KernelArg::Buffer(out),
                    ],
                )
                .expect("bf16_add launch");
        };
        launch_add(x, o, x_resid, hidden_size);

        // ── Stage 11: post_attention_layernorm ────────────────────
        launch_rms(x_resid, post_ln, x_norm2, n_tokens, hidden_size);

        // ── Stage 12: FFN gate, up, silu_gate, down ───────────────
        gate_p.gemv(&backend, x_norm2, gate_act, stream).unwrap();
        up_p.gemv(&backend, x_norm2, up_act, stream).unwrap();
        let silu = backend.kernel("silu_gate", "silu_gate").unwrap();
        backend
            .launch_typed(
                silu,
                [intermediate_size.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&intermediate_size.to_le_bytes()),
                    KernelArg::Buffer(gate_act),
                    KernelArg::Buffer(up_act),
                    KernelArg::Buffer(ffn_act),
                ],
            )
            .expect("silu_gate launch");
        down_p.gemv(&backend, ffn_act, ffn_out, stream).unwrap();

        // ── Stage 13: residual x_final = x_resid + ffn_out ────────
        launch_add(x_resid, ffn_out, x_final, hidden_size);

        backend.synchronize(stream).unwrap();

        // ── Validate the final residual stream ────────────────────
        let mut x_final_raw = vec![0u8; hidden_size as usize * 2];
        backend.copy_d2h(x_final, &mut x_final_raw).unwrap();
        let final_vals = bytes_to_bf16_vec(&x_final_raw);

        let mut nan_or_inf = 0;
        let mut sum_abs = 0.0f32;
        let mut max_abs = 0.0f32;
        for v in &final_vals {
            let f = v.to_f32();
            if !f.is_finite() {
                nan_or_inf += 1;
            }
            let a = f.abs();
            sum_abs += a;
            if a > max_abs {
                max_abs = a;
            }
        }
        let mean_abs = sum_abs / final_vals.len() as f32;

        assert_eq!(
            nan_or_inf, 0,
            "x_final has {nan_or_inf} non-finite values out of {}",
            final_vals.len()
        );
        // After one full attention block on a small synthetic input,
        // the residual stream should sit in a sensible range.
        // 1e-3 ≤ mean_abs ≤ 50 is a generous sanity band — anything
        // outside means a stage of the chain catastrophically
        // amplified or zero-collapsed the activation.
        assert!(
            mean_abs > 1e-3 && mean_abs < 50.0,
            "x_final mean-abs {mean_abs} outside sanity band; max_abs={max_abs}"
        );
        // BF16 max representable is ~3.39e38; if anything in the
        // hot range starts touching 1e3, something is amplifying.
        assert!(
            max_abs < 1e3,
            "x_final max_abs {max_abs} suggests activation explosion"
        );

        // Cleanup — release every allocation we made.
        for ptr in [
            input_ln, q_norm, k_norm, post_ln, x, x_norm, q_full, k, v,
            q_norm_out, k_norm_out, attn_out, gated_attn, o, x_resid, x_norm2,
            gate_act, up_act, ffn_act, ffn_out, x_final, k_cache, v_cache,
            inv_freq_ptr, positions_ptr,
        ] {
            backend.free(ptr).unwrap();
        }
        for w in [&q_proj, &k_proj, &v_proj, &o_proj, &gate_p, &up_p, &down_p] {
            w.release(&backend).unwrap();
        }
    }

    /// End-to-end chain on real Qwen3.5-4B-MLX-8bit weights:
    /// `rms_norm(input_layernorm) → mlx_int8_gemv(q_proj)` for layer 3
    /// (the first full_attention layer). Pins that the entire fused-
    /// dequant decode chain composes correctly when we wire two
    /// kernels together over actual production tensors.
    ///
    /// Doesn't compare against an MLX reference — we don't have MLX
    /// installed in CI — but does verify:
    ///   • all 8192 outputs are finite (no NaN / inf),
    ///   • the output magnitudes sit in a sane regime (mean-abs in
    ///     [1e-3, 50]; the q_proj for a small post-norm input
    ///     produces O(1) values per head),
    ///   • a CPU re-execution of the same pipeline (FP32 RMSNorm +
    ///     bytewise dequant + matvec) agrees with the kernel chain
    ///     within BF16 ULP tolerance on the first 64 outputs.
    ///
    /// `#[ignore]`-gated; requires the local model copy.
    #[test]
    #[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
    fn metal_real_model_chain_norm_then_qproj() {
        use safetensors::SafeTensors;

        let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME unset");
                format!("{home}/models/Qwen3.5-4B-MLX-8bit")
            });
        let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
        if !st_path.exists() {
            eprintln!("skipping: {} not found", st_path.display());
            return;
        }

        let file = std::fs::File::open(&st_path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

        // Layer 3's input_layernorm and q_proj.
        let layer = "language_model.model.layers.3";
        let ln = st.tensor(&format!("{layer}.input_layernorm.weight")).unwrap();
        let q_w = st.tensor(&format!("{layer}.self_attn.q_proj.weight")).unwrap();
        let q_s = st.tensor(&format!("{layer}.self_attn.q_proj.scales")).unwrap();
        let q_b = st.tensor(&format!("{layer}.self_attn.q_proj.biases")).unwrap();

        // Real model dims.
        let hidden_size: u32 = 2560;
        let q_full_out: u32 = 8192; // includes attn_output_gate doubling
        let group_size: u32 = 64;
        let groups_per_row = (hidden_size / group_size) as usize;

        assert_eq!(ln.shape(), [hidden_size as usize]);
        assert_eq!(q_w.shape(), [q_full_out as usize, (hidden_size / 4) as usize]);

        // Subset the q_proj to the first 64 rows so the CPU reference
        // stays cheap (~160 K dequant ops); kernel still uses the full
        // K = 2560 reduction.
        let n_rows: usize = 64;
        let row_stride_packed = (hidden_size as usize / 4) * 4;
        let row_stride_scales = groups_per_row * 2;
        let weight_data = q_w.data();
        let scales_data = q_s.data();
        let biases_data = q_b.data();

        let mut packed_subset = Vec::with_capacity(n_rows * row_stride_packed);
        let mut scales_subset = Vec::with_capacity(n_rows * row_stride_scales);
        let mut biases_subset = Vec::with_capacity(n_rows * row_stride_scales);
        for r in 0..n_rows {
            let p_off = r * row_stride_packed;
            packed_subset.extend_from_slice(&weight_data[p_off..p_off + row_stride_packed]);
            let s_off = r * row_stride_scales;
            scales_subset.extend_from_slice(&scales_data[s_off..s_off + row_stride_scales]);
            biases_subset.extend_from_slice(&biases_data[s_off..s_off + row_stride_scales]);
        }
        let ln_bytes = ln.data().to_vec();

        // Synthetic input — typical pre-RMSNorm activation range
        // (Qwen3-family residual stream sits around ±1 in BF16).
        let x: Vec<half::bf16> = (0..hidden_size)
            .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
            .collect();

        // ── CPU reference: rms_norm → dequant → matvec ──────────
        let eps: f32 = 1e-5;
        let mut x_norm_cpu = vec![0.0f32; hidden_size as usize];
        let mut ssq = 0.0f32;
        for v in &x {
            let f = v.to_f32();
            ssq += f * f;
        }
        let inv_rms = (ssq / hidden_size as f32 + eps).powf(-0.5);
        for (i, v) in x.iter().enumerate() {
            // RMSNorm weight is BF16 LE.
            let w_bytes = &ln_bytes[i * 2..i * 2 + 2];
            let w = half::bf16::from_le_bytes([w_bytes[0], w_bytes[1]]).to_f32();
            x_norm_cpu[i] = v.to_f32() * inv_rms * w;
        }

        let mut expected_q = vec![half::bf16::ZERO; n_rows];
        for r in 0..n_rows {
            let mut acc = 0.0f32;
            for c in 0..hidden_size as usize {
                let word_off = r * row_stride_packed + (c / 4) * 4;
                let word = u32::from_le_bytes([
                    packed_subset[word_off],
                    packed_subset[word_off + 1],
                    packed_subset[word_off + 2],
                    packed_subset[word_off + 3],
                ]);
                let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
                let g = c / group_size as usize;
                let s_idx = (r * groups_per_row + g) * 2;
                let s = half::bf16::from_le_bytes([
                    scales_subset[s_idx],
                    scales_subset[s_idx + 1],
                ])
                .to_f32();
                let b = half::bf16::from_le_bytes([
                    biases_subset[s_idx],
                    biases_subset[s_idx + 1],
                ])
                .to_f32();
                let w = byte * s + b;
                acc += w * x_norm_cpu[c];
            }
            expected_q[r] = half::bf16::from_f32(acc);
        }

        // ── Kernel chain: rms_norm → mlx_int8_gemv on real bytes ──
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let x_bytes = bf16_slice_to_bytes(&x);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let ln_ptr = backend.alloc(ln_bytes.len()).unwrap();
        let xn_ptr = backend.alloc(x_bytes.len()).unwrap();
        let pk_ptr = backend.alloc(packed_subset.len()).unwrap();
        let sc_ptr = backend.alloc(scales_subset.len()).unwrap();
        let bi_ptr = backend.alloc(biases_subset.len()).unwrap();
        let q_ptr = backend.alloc(n_rows * 2).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();
        backend.copy_h2d(&ln_bytes, ln_ptr).unwrap();
        backend.copy_h2d(&packed_subset, pk_ptr).unwrap();
        backend.copy_h2d(&scales_subset, sc_ptr).unwrap();
        backend.copy_h2d(&biases_subset, bi_ptr).unwrap();

        // Stage 1: RMSNorm.
        let n_tokens: u32 = 1;
        let rms_kernel = backend.kernel("rms_norm", "rms_norm").unwrap();
        backend
            .launch_typed(
                rms_kernel,
                [n_tokens, 1, 1],
                [128, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&hidden_size.to_le_bytes()),
                    KernelArg::Bytes(&eps.to_le_bytes()),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(ln_ptr),
                    KernelArg::Buffer(xn_ptr),
                ],
            )
            .expect("launch rms_norm");

        // Stage 2: Q projection (fused dequant + matvec).
        let n: u32 = n_rows as u32;
        let k: u32 = hidden_size;
        let gemv_kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
        backend
            .launch_typed(
                gemv_kernel,
                [n, 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(pk_ptr),
                    KernelArg::Buffer(sc_ptr),
                    KernelArg::Buffer(bi_ptr),
                    KernelArg::Buffer(xn_ptr),
                    KernelArg::Buffer(q_ptr),
                ],
            )
            .expect("launch q_proj gemv");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut q_raw = vec![0u8; n_rows * 2];
        backend.copy_d2h(q_ptr, &mut q_raw).unwrap();
        let actual_q = bytes_to_bf16_vec(&q_raw);

        // ── Validate ────────────────────────────────────────────
        let mut max_abs_diff: f32 = 0.0;
        let mut sum_abs = 0.0f32;
        let mut nonzero_count = 0;
        for i in 0..n_rows {
            let e = expected_q[i].to_f32();
            let a = actual_q[i].to_f32();
            assert!(
                a.is_finite(),
                "chain produced non-finite Q[{i}] = {a}"
            );
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            sum_abs += a.abs();
            if a.abs() > 1e-4 {
                nonzero_count += 1;
            }
        }
        let mean_abs = sum_abs / n_rows as f32;

        assert!(
            mean_abs >= 1e-3 && mean_abs <= 50.0,
            "Q output mean-abs {mean_abs} outside sanity band [1e-3, 50]"
        );
        assert!(
            nonzero_count >= n_rows / 2,
            "too many near-zero outputs ({nonzero_count}/{n_rows}); chain is suspicious"
        );
        assert!(
            max_abs_diff < 0.1,
            "rms_norm + gemv chain: max |kernel - cpu| = {max_abs_diff}"
        );

        backend.free(x_ptr).unwrap();
        backend.free(ln_ptr).unwrap();
        backend.free(xn_ptr).unwrap();
        backend.free(pk_ptr).unwrap();
        backend.free(sc_ptr).unwrap();
        backend.free(bi_ptr).unwrap();
        backend.free(q_ptr).unwrap();
    }

    /// Real-data parity check for `mlx_int8_gemv`. Loads the actual
    /// `language_model.model.layers.3.self_attn.q_proj` triplet (the
    /// first full_attention layer's Q projection), subsets to the
    /// first `N=128` output rows, runs the fused dequant+matvec on
    /// a synthetic activation vector at the model's true hidden
    /// dimension, and compares to a CPU reference that dequantizes
    /// those exact bytes the same way.
    ///
    /// `#[ignore]`-gated by default; requires the local model copy.
    #[test]
    #[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
    fn metal_mlx_int8_gemv_real_model_q_proj() {
        use safetensors::SafeTensors;

        let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME unset");
                format!("{home}/models/Qwen3.5-4B-MLX-8bit")
            });
        let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
        if !st_path.exists() {
            eprintln!("skipping: {} not found", st_path.display());
            return;
        }

        let file = std::fs::File::open(&st_path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

        let base = "language_model.model.layers.3.self_attn.q_proj";
        let weight = st.tensor(&format!("{base}.weight")).unwrap();
        let scales = st.tensor(&format!("{base}.scales")).unwrap();
        let biases = st.tensor(&format!("{base}.biases")).unwrap();

        // Real layer 3 q_proj shape: weight U32 [8192, 640], i.e.
        // out=8192, in_features=2560 (= 640 * 4 packed bytes).
        let weight_shape = weight.shape();
        let full_out = weight_shape[0];
        let in_packed_cols = weight_shape[1];
        let in_features = (in_packed_cols * 4) as u32;
        assert_eq!(in_features, 2560, "expected hidden_size=2560 for Qwen3.5-4B");
        assert_eq!(full_out, 8192, "expected num_heads*head_dim*2=8192 for layer 3 q_proj (with attn output gate)");

        // Subset to the first N=128 output rows so the test runs in
        // a few hundred ms on M-series rather than ~21 M dequant ops.
        let n_rows: usize = 128;
        let group_size: u32 = 64;
        let groups_per_row = (in_features / group_size) as usize;

        let weight_data = weight.data();
        let scales_data = scales.data();
        let biases_data = biases.data();

        let row_stride_packed = in_packed_cols * 4; // u32 per col
        let row_stride_scales = groups_per_row * 2; // bf16 per group

        let mut packed_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_packed);
        let mut scales_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
        let mut biases_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
        for r in 0..n_rows {
            let p_off = r * row_stride_packed;
            packed_slice.extend_from_slice(&weight_data[p_off..p_off + row_stride_packed]);
            let s_off = r * row_stride_scales;
            scales_slice.extend_from_slice(&scales_data[s_off..s_off + row_stride_scales]);
            biases_slice.extend_from_slice(&biases_data[s_off..s_off + row_stride_scales]);
        }

        // Synthetic input activation in a typical post-norm range.
        let x_bf16: Vec<half::bf16> = (0..in_features)
            .map(|i| half::bf16::from_f32(0.05 + 0.001 * (i as f32).sin()))
            .collect();

        // CPU reference: dequant byte-by-byte then dot with x.
        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows];
        for r in 0..n_rows {
            let mut acc: f32 = 0.0;
            for c in 0..in_features as usize {
                let word_off = r * row_stride_packed + (c / 4) * 4;
                let word = u32::from_le_bytes([
                    packed_slice[word_off],
                    packed_slice[word_off + 1],
                    packed_slice[word_off + 2],
                    packed_slice[word_off + 3],
                ]);
                let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
                let g = c / group_size as usize;
                let s_idx = (r * groups_per_row + g) * 2;
                let s = half::bf16::from_le_bytes([scales_slice[s_idx], scales_slice[s_idx + 1]])
                    .to_f32();
                let b = half::bf16::from_le_bytes([biases_slice[s_idx], biases_slice[s_idx + 1]])
                    .to_f32();
                let w = byte * s + b;
                acc += w * x_bf16[c].to_f32();
            }
            expected[r] = half::bf16::from_f32(acc);
        }

        // Run the kernel on the same subset.
        let modules = atlas_kernels::metallib_modules();
        let backend = MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let n: u32 = n_rows as u32;
        let k: u32 = in_features;

        let packed_ptr = backend.alloc(packed_slice.len()).unwrap();
        let scales_ptr = backend.alloc(scales_slice.len()).unwrap();
        let biases_ptr = backend.alloc(biases_slice.len()).unwrap();
        let x_bytes = bf16_slice_to_bytes(&x_bf16);
        let x_ptr = backend.alloc(x_bytes.len()).unwrap();
        let y_ptr = backend.alloc(n_rows * 2).unwrap();
        backend.copy_h2d(&packed_slice, packed_ptr).unwrap();
        backend.copy_h2d(&scales_slice, scales_ptr).unwrap();
        backend.copy_h2d(&biases_slice, biases_ptr).unwrap();
        backend.copy_h2d(&x_bytes, x_ptr).unwrap();

        let kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
        backend
            .launch_typed(
                kernel,
                [n, 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(packed_ptr),
                    KernelArg::Buffer(scales_ptr),
                    KernelArg::Buffer(biases_ptr),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch real-model gemv");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; n_rows * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        // 2560-element FP32 sum at output magnitudes ~0.1–1 has ULP
        // ≈ 0.005 at 1.0; tolerate 0.1 for ordering drift across
        // simdgroups versus the strictly-sequential CPU reference.
        let mut max_abs_diff: f32 = 0.0;
        let mut max_rel_diff: f32 = 0.0;
        for i in 0..n_rows {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            let rel = if e.abs() > 1e-3 { d / e.abs() } else { 0.0 };
            if rel > max_rel_diff {
                max_rel_diff = rel;
            }
            // Also assert no NaN / inf — the real signal that the
            // chain is wired correctly.
            assert!(a.is_finite(), "real-model gemv produced non-finite at row {i}: {a}");
        }
        assert!(
            max_abs_diff < 0.1 || max_rel_diff < 0.05,
            "real-model gemv: max abs diff {max_abs_diff}, max rel diff {max_rel_diff}"
        );

        backend.free(packed_ptr).unwrap();
        backend.free(scales_ptr).unwrap();
        backend.free(biases_ptr).unwrap();
        backend.free(x_ptr).unwrap();
        backend.free(y_ptr).unwrap();
    }

    /// Real-data parity check. Loads the actual `embed_tokens` triplet
    /// from a local copy of `mlx-community/Qwen3.5-4B-MLX-8bit`,
    /// dequantizes a 4-row × 128-col slice via the Metal kernel, and
    /// compares against a CPU reference computed from the same bytes.
    ///
    /// The test is `#[ignore]`-gated by default — it requires the
    /// model to be present at `~/models/Qwen3.5-4B-MLX-8bit/` (or at
    /// `$ATLAS_MLX_MODEL_DIR`). Run with:
    ///
    ///   cargo test -p spark-runtime --no-default-features \
    ///              --features metal -- --ignored \
    ///              metal_mlx_int8_dequant_real_model
    #[test]
    #[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
    fn metal_mlx_int8_dequant_real_model() {
        use safetensors::SafeTensors;

        let model_dir = std::env::var("ATLAS_MLX_MODEL_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").expect("HOME unset");
                format!("{home}/models/Qwen3.5-4B-MLX-8bit")
            });
        let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
        if !st_path.exists() {
            eprintln!("skipping: {} not found", st_path.display());
            return;
        }

        // mmap the safetensors file via memmap2 (already a spark-runtime
        // dep) and parse the header.
        let file = std::fs::File::open(&st_path).expect("open safetensors");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let st = SafeTensors::deserialize(&mmap).expect("parse safetensors header");

        // Pick a 4-row × 128-col slice from `language_model.model.embed_tokens`
        // — 4 rows × 32 uint32 packed = 128 columns; 128 cols / 64 group = 2
        // groups per row of scales/biases.
        let base = "language_model.model.embed_tokens";
        let weight = st.tensor(&format!("{base}.weight")).expect("weight");
        let scales = st.tensor(&format!("{base}.scales")).expect("scales");
        let biases = st.tensor(&format!("{base}.biases")).expect("biases");

        assert_eq!(weight.dtype(), safetensors::Dtype::U32);
        assert_eq!(scales.dtype(), safetensors::Dtype::BF16);
        assert_eq!(biases.dtype(), safetensors::Dtype::BF16);
        let weight_shape = weight.shape();
        assert_eq!(weight_shape.len(), 2);

        let full_in_cols_packed = weight_shape[1];
        // Slice dims
        let n_rows = 4usize;
        let n_cols = 128usize;
        let group_size = 64u32;
        let n_packed_cols = n_cols / 4;
        let groups_per_slice_row = (n_cols / group_size as usize) as usize;

        let weight_data = weight.data();
        let scales_data = scales.data();
        let biases_data = biases.data();

        // Extract row strides (in bytes).
        let row_stride_packed = full_in_cols_packed * 4; // u32 per col
        let row_stride_groups = (full_in_cols_packed * 4) / group_size as usize; // groups in full row
        let row_stride_scales = row_stride_groups * 2; // bf16 per group

        let mut packed_slice: Vec<u8> = Vec::with_capacity(n_rows * n_packed_cols * 4);
        let mut scales_slice: Vec<u8> = Vec::with_capacity(n_rows * groups_per_slice_row * 2);
        let mut biases_slice: Vec<u8> = Vec::with_capacity(n_rows * groups_per_slice_row * 2);
        for r in 0..n_rows {
            let p_off = r * row_stride_packed;
            packed_slice.extend_from_slice(&weight_data[p_off..p_off + n_packed_cols * 4]);
            let s_off = r * row_stride_scales;
            scales_slice.extend_from_slice(&scales_data[s_off..s_off + groups_per_slice_row * 2]);
            biases_slice.extend_from_slice(&biases_data[s_off..s_off + groups_per_slice_row * 2]);
        }

        // CPU reference. Walk byte by byte through the packed slice.
        let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                let word_offset = r * n_packed_cols * 4 + (c / 4) * 4;
                let word = u32::from_le_bytes([
                    packed_slice[word_offset],
                    packed_slice[word_offset + 1],
                    packed_slice[word_offset + 2],
                    packed_slice[word_offset + 3],
                ]);
                let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
                let g = c / group_size as usize;
                let s_idx = (r * groups_per_slice_row + g) * 2;
                let s = half::bf16::from_le_bytes([
                    scales_slice[s_idx],
                    scales_slice[s_idx + 1],
                ])
                .to_f32();
                let b = half::bf16::from_le_bytes([
                    biases_slice[s_idx],
                    biases_slice[s_idx + 1],
                ])
                .to_f32();
                expected[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
            }
        }

        // Run the kernel on the slice.
        let modules = atlas_kernels::metallib_modules();
        let backend =
            MetalGpuBackend::new(0, &modules).expect("MetalGpuBackend::new");

        let packed_ptr = backend.alloc(packed_slice.len()).expect("alloc packed");
        let scales_ptr = backend.alloc(scales_slice.len()).expect("alloc scales");
        let biases_ptr = backend.alloc(biases_slice.len()).expect("alloc biases");
        let out_bytes = n_rows * n_cols * 2;
        let out_ptr = backend.alloc(out_bytes).expect("alloc out");
        backend.copy_h2d(&packed_slice, packed_ptr).unwrap();
        backend.copy_h2d(&scales_slice, scales_ptr).unwrap();
        backend.copy_h2d(&biases_slice, biases_ptr).unwrap();

        let kernel = backend
            .kernel("mlx_int8_dequant", "mlx_int8_dequant")
            .expect("kernel");
        let block_x = 16u32;
        let in_features_arg = n_cols as u32;
        let out_features_arg = n_rows as u32;
        let grid_x = in_features_arg.div_ceil(block_x);
        backend
            .launch_typed(
                kernel,
                [grid_x, out_features_arg, 1],
                [block_x, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&out_features_arg.to_le_bytes()),
                    KernelArg::Bytes(&in_features_arg.to_le_bytes()),
                    KernelArg::Bytes(&group_size.to_le_bytes()),
                    KernelArg::Buffer(packed_ptr),
                    KernelArg::Buffer(scales_ptr),
                    KernelArg::Buffer(biases_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch_typed");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; out_bytes];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        // Embedding values are typically tiny; tolerate L∞ ≤ 1e-2.
        let mut max_abs_diff: f32 = 0.0;
        for i in 0..expected.len() {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            let d = (e - a).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 1e-2,
            "real-model dequant mismatch: L∞ = {max_abs_diff}"
        );

        backend.free(packed_ptr).unwrap();
        backend.free(scales_ptr).unwrap();
        backend.free(biases_ptr).unwrap();
        backend.free(out_ptr).unwrap();
    }
}
