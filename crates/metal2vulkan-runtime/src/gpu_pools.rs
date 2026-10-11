//! A command buffer's GPU pools: the index pool its index bounds and index rewrite passes reserve
//! indices in (`index_bounds.rs`, `index_rewrite.rs`), and the records pool its scanned emulated
//! mesh draws reserve records in (`render/mesh.rs`). The GPU decides how much of each a command
//! buffer takes (an indirect draw's count, an object stage's grids), after the draws that read
//! them were recorded, so a pool is one block bound for the whole command buffer.
//!
//! - **Sized from the device's memory** ([`pool_budget`]): a sixty-fourth of the device's largest
//!   device-local heap, within one storage binding (`maxStorageBufferRange`, less the index
//!   pool's header) and one allocation (`maxMemoryAllocationSize`), in whole words; the index
//!   pool's within a `u32` of bytes, its counter's. Not a constant of the runtime's: a device of
//!   twice the memory has pools of twice the bytes. Before, the index pool was 1 MiB on every
//!   device.
//! - **Leased per command buffer, made lazily.** The first pass of a command buffer that needs a
//!   pool takes the device's spare block of its kind, or makes one; the command buffer holds the
//!   [`PoolLease`] until it completes, or until it is dropped unsubmitted, and then the block is
//!   given back. No other command buffer gets a block while it is leased, so command buffers in
//!   flight at once each have their own. The device keeps at most one spare of each kind (the
//!   blocks are all one size, so the spare is any given back while none is idle); a block given
//!   back beside a spare is released. A command buffer that reaches no pass of a pool takes none.
//! - **Raw.** A block is a `VkBuffer` bound to private memory, outside the Metal address space,
//!   not a `Buffer`, which would hold the device from the device's own spare (as the mipmap
//!   scratch's, `mipmap_scratch.rs`). The device releases its spares as it is destroyed.
//! - **Not cleared.** Nothing reads a pool's bytes before a pass of the command buffer wrote them:
//!   the index pool's counter lies in the command buffer's upload arena (zero as it is uploaded),
//!   and its block holds only the indices the fills and writes place (the block's first
//!   `IndexBoundsPass::POOL_HEADER_BYTES` are never read); the records pool's cursor is a status
//!   word.
//! - **What still fails.** A draw whose indices or records the rest of its command buffer's pool
//!   cannot hold draws nothing and fails the command buffer (`IndexPoolOverflow`,
//!   `MeshRecordsOverflow`): the GPU learns the demand after the draw was recorded against the
//!   block, and no block can be made between. `Exactness::ExactWithinPool` says so.

use super::device::lock;
use super::error::Error;
use super::memory::Memory;
use super::mipmap_scratch::{Block, Spare, Take};
use super::upload::Upload;
use super::Device;
use ash::vk;
use metal2vulkan::bind::IndexBoundsPass;

/// The share of the largest device-local heap a pool takes: one part in this many.
const POOL_SHARE: u64 = 64;

/// The bytes of a pool of a device with `properties`' heaps, whose storage bindings reach
/// `max_range` bytes and whose allocations `max_allocation`, behind `header` bytes the block
/// holds before them: a sixty-fourth of the largest device-local heap, within one binding and
/// one allocation of the block, in whole words (the module docs).
pub(crate) fn pool_budget(
    properties: &vk::PhysicalDeviceMemoryProperties,
    max_range: u32,
    max_allocation: u64,
    header: u64,
) -> u64 {
    let count = (properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS);
    let heap = properties.memory_heaps[..count]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    (heap / POOL_SHARE)
        .min(u64::from(max_range).saturating_sub(header))
        .min(max_allocation.saturating_sub(header))
        / 4
        * 4
}

/// Which of a command buffer's pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PoolKind {
    /// The index pool: indices behind `IndexBoundsPass::POOL_HEADER_BYTES` the block does not
    /// use, a storage buffer and an index buffer.
    Index,
    /// The mesh records pool: records from the block's start, a storage buffer.
    MeshRecords,
}

impl PoolKind {
    /// The usage of the kind's blocks.
    fn usage(self) -> vk::BufferUsageFlags {
        match self {
            PoolKind::Index => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER
            }
            PoolKind::MeshRecords => vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    }
}

/// A device's pools: each kind's bytes and its spare block (the module docs).
#[derive(Default)]
pub(super) struct GpuPools {
    /// The index pool's bytes of indices, behind its header.
    pub(super) index_bytes: u64,
    /// The mesh records pool's bytes.
    pub(super) mesh_bytes: u64,
    index: std::sync::Mutex<Spare<Block>>,
    mesh: std::sync::Mutex<Spare<Block>>,
}

impl GpuPools {
    /// The pools of a device whose kinds hold `index_bytes` of indices and `mesh_bytes` of
    /// records.
    pub(super) fn new(index_bytes: u64, mesh_bytes: u64) -> GpuPools {
        GpuPools {
            index_bytes,
            mesh_bytes,
            ..GpuPools::default()
        }
    }

    fn spare(&self, kind: PoolKind) -> &std::sync::Mutex<Spare<Block>> {
        match kind {
            PoolKind::Index => &self.index,
            PoolKind::MeshRecords => &self.mesh,
        }
    }

    /// The bytes of a block of `kind`: the index pool's header and indices, or the records.
    pub(super) fn block_bytes(&self, kind: PoolKind) -> u64 {
        match kind {
            PoolKind::Index => IndexBoundsPass::POOL_HEADER_BYTES + self.index_bytes,
            PoolKind::MeshRecords => self.mesh_bytes,
        }
    }

    /// Release the spares as the device is destroyed.
    pub(super) fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        for kind in [PoolKind::Index, PoolKind::MeshRecords] {
            lock(self.spare(kind)).destroy(device, memory);
        }
    }
}

/// A command buffer's lease of a block of a pool, given back to its device as it drops (the
/// module docs).
pub(super) struct PoolLease {
    device: Device,
    kind: PoolKind,
    block: Option<Block>,
    bytes: u64,
}

impl PoolLease {
    /// The leased block's buffer.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.block.as_ref().map_or(vk::Buffer::null(), Block::raw)
    }

    /// The block's bytes.
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl std::fmt::Debug for PoolLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolLease")
            .field("kind", &self.kind)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        let shared = self.device.shared();
        let released = lock(shared.gpu_pools.spare(self.kind)).give(block, self.bytes);
        if let Some(block) = released {
            // SAFETY: a lease ends when its command buffer completes or is dropped unsubmitted,
            // and the spare is leased to none, so no command uses either block.
            unsafe { block.release(shared.vk(), &mut lock(&shared.memory)) };
        }
    }
}

/// A command buffer's index pool as its passes bind it (`State::index_pool`).
#[derive(Clone, Debug)]
pub(super) struct IndexPool {
    /// The counter: [`IndexBoundsPass::POOL_HEADER_BYTES`] of the command buffer's arena, zero
    /// before its first pass; the binding of the pool an index bounds plan and an index rewrite's
    /// place dispatch read and write.
    pub(super) counter: Upload,
    /// The block's `VkBuffer`: from its byte 0, the binding of the pool an index bounds fill and
    /// an index rewrite's write dispatch write, whose indices lie from
    /// [`IndexBoundsPass::POOL_HEADER_BYTES`] on, where the draws bind it as their index buffer.
    pub(super) block: vk::Buffer,
    /// The block's bytes, its unused header's included.
    pub(super) bytes: u64,
}

impl IndexPool {
    /// The pool of `counter` and the leased `block`.
    pub(super) fn of(counter: &Upload, block: &PoolLease) -> IndexPool {
        IndexPool {
            counter: counter.clone(),
            block: block.raw(),
            bytes: block.bytes(),
        }
    }

    /// The bytes of its indices: the block's past its header.
    pub(super) fn index_bytes(&self) -> u64 {
        self.bytes - IndexBoundsPass::POOL_HEADER_BYTES
    }
}

impl Device {
    /// A lease of a block of `kind` for a command buffer: the device's spare, or a new block
    /// (the module docs). Refused as memory is.
    pub(super) fn lease_pool(&self, kind: PoolKind) -> Result<PoolLease, Error> {
        let shared = self.shared();
        let bytes = shared.gpu_pools.block_bytes(kind);
        let taken = lock(shared.gpu_pools.spare(kind)).take(bytes);
        let block = match taken {
            Take::Reuse(block, _) => block,
            Take::Make(stale) => {
                if let Some(block) = stale {
                    // SAFETY: the spare was leased to none, so no command uses it.
                    unsafe { block.release(shared.vk(), &mut lock(&shared.memory)) };
                }
                self.make_block(bytes, kind.usage())?
            }
        };
        Ok(PoolLease {
            device: self.clone(),
            kind,
            block: Some(block),
            bytes,
        })
    }
}
