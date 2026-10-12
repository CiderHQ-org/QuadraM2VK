//! The runtime's own memory sub-allocator: blocks of `VkDeviceMemory` per memory type, each with a
//! free list of placed ranges.
//!
//! - **One mechanism.** A block with explicit placement serves the runtime's pools for resources
//!   made outside a heap, and later a Metal heap (step 7: a heap is one block, its resources placed
//!   at offsets).
//! - **Pooling.** Resources share blocks of [`BLOCK_BYTES`], so `maxMemoryAllocationCount` (4096
//!   on many devices) stays out of reach. A request larger than half a block takes a dedicated
//!   block of its own size, freed with it. A pooled block is freed when it empties, except one
//!   empty block per memory type that is kept for the next request.
//! - **Entry point.** [`allocate`] with the resource's memory requirements and its
//!   [`MemoryClass`]; [`Memory::free`] with the placement it gave. Buffers use it in step 1;
//!   textures and heaps add their classes and callers in their own steps. Imported host memory
//!   has its own entry, [`Memory::import_host`]: one dedicated block per import, never
//!   pooled, freed through the same [`Memory::free`]. A dma-buf's import
//!   (`Memory::import_dma_buf`) is in no block: the placement in it keeps it, and frees it as it
//!   goes (`dma_buf.rs`'s `DmaBufPages`).
//!
//! - **Addresses** ([`allocate_addressed`]). A buffer placed for a device address gets it from its
//!   block: a block that holds such a buffer has one buffer over its whole memory with the
//!   device-address usage (`BlockAddresses`), and a placed buffer's address is that buffer's plus
//!   its offset. The placed buffers themselves are made without the usage (`buffer.rs`), so the
//!   device's count of addressable buffers is the count of such blocks, not of the program's
//!   buffers: MoltenVK makes each addressable buffer resident in every encoder whose pipeline reads
//!   pointers, a cost each command buffer paid per live buffer (2.5 ms a commit at 32768 live
//!   buffers on an Apple M4, EV CPF-rset1-mvk4; MoltenVK's
//!   `MVKDevice::encodeGPUAddressableBuffers`).
//!
//! - **Images apart**. An image the runtime places in memory of its own takes a block that holds
//!   images alone ([`Holds`], [`allocate_image`]): no buffer, a block's buffer of addresses among
//!   them, is bound into such a block, nor such an image into a buffers' block. An image over a
//!   buffer's or a heap's bytes lies in that buffer's memory, at the offset Metal's layout gives
//!   it. Images are placed at a multiple of `bufferImageGranularity` and rounded up to it
//!   (`texture.rs`'s `ImagePlacement`).

use super::buffer::usage;
use super::device::lock;
#[cfg(unix)]
use super::dma_buf::DmaBufPages;
use super::error::{internal, vulkan, Error, MemoryKind};
use super::range::RangeAllocator;
use super::sharing::Sharing;
use super::work_counts::Counters;
use crate::release;
use ash::vk;
use std::ptr::NonNull;
#[cfg(unix)]
use std::sync::Arc;
use std::sync::Mutex;

/// The size of a pooled block, at most the device's `maxMemoryAllocationSize`.
pub(super) const BLOCK_BYTES: u64 = 64 << 20;

/// What memory a resource needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MemoryClass {
    /// Host-visible and coherent memory the host reads as well as writes (`Shared` storage of
    /// the default CPU cache mode, staging for the host's reads and writes): host-cached where the
    /// device has such, else device-local where it has such ([`candidates`]). Without unified
    /// memory that is system memory, as a shared resource's on a Mac with a discrete GPU.
    Shared,
    /// Host-visible and coherent memory the GPU uses and the host writes, reading little or
    /// nothing of it (`Shared` storage of the write-combined CPU cache mode, the upload arena, a
    /// function table's words): without unified memory device-local where the device has such
    /// (resizable BAR), else as `Shared`; with unified memory as `Shared` ([`candidates`]).
    WriteCombined,
    /// Device-local (`Private` storage), host-invisible where the device has a choice.
    Private,
    /// Host-visible and coherent (`Managed` storage), host-cached where the device has such
    /// memory: a managed resource is there for the CPU to read back. On a device with unified
    /// memory the GPU uses it too, and coherent memory needs no flush or invalidate, so
    /// `did_modify_range` and `synchronize_resource` record nothing beyond their order. On
    /// one without, it holds a managed buffer's host copy, and the buffer's own bytes are `Private`
    /// (`buffer.rs`'s `Mirror`, `managed.rs`).
    Managed,
    /// Device-local and lazily allocated where the device has such memory (`Memoryless`
    /// storage: a transient attachment's image), else as `Private`.
    Memoryless,
}

/// What a block of the runtime's holds: buffers (a heap's bytes and every buffer of addresses
/// among them), or images, never both. On an RTX 5080 a new texture's image bound into the block
/// that held the running work's buffers, 1024 bytes past a buffer's, waited in
/// `vkBindImageMemory` for that work to end (493 ms and 502 ms behind a 0.5 s kernel, EVs
/// STL-ps8-nv5080 and STL-p8-nv5080), where images bound into a block of images, or a new one,
/// took microseconds (EVs STL-p6-nv5080, STL-p8-nv5080, STL-p9-nv5080). So images take blocks
/// of their own, one pool per memory type beside the buffers'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Holds {
    Buffers,
    Images,
}

/// Whether a placement may share a pooled block (`Pool::place`), or takes a block of its own
/// whatever its size: memory no submission has used, so its bind waits for none. On an RTX 5080
/// a neutral image bound into a block of images, 1024 bytes past another neutral image that a
/// queued command buffer sampled, waited in `vkBindImageMemory` (NVIDIA's ioctl) until that work
/// ended (EV STL-au3-nvgdb; 490 ms behind a 0.5 s kernel in EV STL-au2-nv5080). On lavapipe the
/// same case waits for its queue lock instead, in making the image's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fit {
    Pooled,
    Apart,
}

/// Why [`Memory::import_host`] imported nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImportFailure {
    /// No memory type of the class is among the allowed ones.
    NoMemoryType,
    /// The allocation of the last type tried failed.
    Vulkan(vk::Result),
}

/// Whether the runtime may allocate memory of a type with `flags`: any type but the
/// device-coherent and device-uncached ones of `VK_AMD_device_coherent_memory`, which only a
/// device with its `deviceCoherentMemory` feature enabled may allocate
/// (`VUID-vkAllocateMemory-deviceCoherentMemory-02790`), and the runtime enables no such feature.
/// RADV lists them whether or not it is enabled (types 13 to 16 on a Radeon 780M, EV PERF-types2),
/// with the properties of its other types beside those bits.
pub(super) fn allocatable(flags: vk::MemoryPropertyFlags) -> bool {
    !flags.intersects(
        vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
    )
}

/// The memory types among `types` the host caches, as bits: those a host image copy
/// (`texture/host_copy.rs`) reads at the host's speed. The host reads uncached memory slowly, and
/// a host copy reads through the host's mapping of the image's memory: on an RTX 3090
/// `vkCopyImageToMemoryEXT` read a 1920 x 1080 `BGRA8Unorm` image's 8 MiB in 758 ms from
/// device-local host-visible memory (resizable BAR), in 127 ms from uncached system memory and in
/// 2.9 ms from host-cached system memory; on a Radeon 780M in 144 ms from RADV's device-local
/// host-visible type and in 4.6 ms from its host-cached one (EVs XSH-hrp2-3090, XSH-hrp4-radv).
///
/// Only coherent ones: the runtime flushes and invalidates no mapped range, as Metal's `Shared`
/// and `Managed` textures need no flush, so a cached type that is not coherent leaves the image's
/// texels to the driver's own cache upkeep around each host copy. ANV (Mesa 26.2.4) on Arrow
/// Lake-S's Xe-LPG (`deviceID` 0x7d67), which has no last-level cache shared with the GPU, lists
/// such a type (`HOST_CACHED` without `HOST_COHERENT`): with images there, the GPU read stale
/// 64-byte lines of the consumer's macOS guest's textures (4x4 texels of a Tile4 32-bit image; a
/// third of the boot logo's interior), and none once they took its coherent type.
pub(super) fn host_cached(types: &[vk::MemoryType]) -> u32 {
    use vk::MemoryPropertyFlags as P;
    let cached = P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED;
    (types.iter().enumerate())
        .filter(|(_, ty)| ty.property_flags.contains(cached))
        .fold(0, |bits, (index, _)| bits | 1 << index)
}

/// The memory types among `types` the host maps without coherence (`HOST_VISIBLE` without
/// `HOST_COHERENT`), as bits: those no image with host-transfer usage takes, cached or not, as
/// the runtime flushes no mapped range ([`host_cached`]).
pub(super) fn host_incoherent(types: &[vk::MemoryType]) -> u32 {
    use vk::MemoryPropertyFlags as P;
    (types.iter().enumerate())
        .filter(|(_, ty)| {
            ty.property_flags.contains(P::HOST_VISIBLE)
                && !ty.property_flags.contains(P::HOST_COHERENT)
        })
        .fold(0, |bits, (index, _)| bits | 1 << index)
}

/// The memory class an image of `class` with host-transfer usage takes in memory the host caches,
/// and its memory types among `allowed`, as bits: the host-cached types ([`host_cached`]) of
/// `class` where it has one among them; else, on a device with unified memory (`unified`), those
/// of `Shared` memory where it has one among them, as Metal's `Shared` and `Managed` textures on
/// unified memory lie in memory the CPU reads at its speed (on a Radeon 780M, which has no
/// device-local type the host caches, in system memory the host caches: the GPU samples, renders
/// into, clears and copies a 1080p image there as fast as in its device-local memory or faster,
/// EV XSH-cost2-radv). `None` where it has none: the image takes `class` and every allowed type,
/// as any image of the class does. A discrete GPU with no host-cached memory of the class (NVIDIA
/// on an RTX 3090) so still copies on the host, slowly: a copy through a submission would queue
/// behind the program's work, and its GPU reaches system memory across the bus.
pub(super) fn host_cached_placement(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Option<(MemoryClass, u32)> {
    let cached = allowed & host_cached(types);
    let has = |class: MemoryClass| !candidates(types, cached, class, unified).is_empty();
    let class = if has(class) {
        class
    } else if unified && has(MemoryClass::Shared) {
        MemoryClass::Shared
    } else {
        return None;
    };
    Some((class, cached))
}

/// The memory types `class` may use among `types` (the bits of `allowed`), best first, on a
/// device with unified memory (`unified`, `DeviceProperties::has_unified_memory`) or without.
/// Only `Memoryless` uses a lazily allocated type, which serves transient attachments alone, and
/// no class uses a type the runtime may not allocate ([`allocatable`]).
///
/// `Shared` and `WriteCombined` memory is host-visible and coherent, as Metal's shared storage
/// needs no flush. `Shared` memory prefers a host-cached type, then a device-local one: the host
/// reads uncached memory slowly (0.14 GB/s from RADV's device-local host-visible type on a Radeon
/// 780M, against 21 GB/s from its cached one, EV PERF-mem2; 0.01 GB/s from resizable BAR on an
/// RTX 3090, against 45 GB/s, EV PERF-types2), while an integrated GPU uses either at one speed.
/// A discrete GPU uses host-cached system memory across the bus (on the RTX 3090, 16-byte stores
/// at 1.7 GB/s and 4-byte stores 12 bytes apart at 0.08 GB/s, against 425 and 850 GB/s in
/// device-local memory), as a discrete Mac's GPU uses a shared resource; `Device::new_buffer`
/// announces it (`Degradation::SharedInSystemMemory`). So without unified memory
/// `WriteCombined` memory, which the host does not read, prefers a device-local type; with
/// unified memory it is `Shared` memory.
pub(super) fn candidates(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Vec<u32> {
    use vk::MemoryPropertyFlags as P;
    let usable = |index: usize| allowed & (1 << index) != 0;
    // The flags required; those preferred, the first outweighing the second; those avoided,
    // weighed as the first preference.
    let (required, preferred, avoided) = match class {
        MemoryClass::WriteCombined if !unified => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::DEVICE_LOCAL, P::empty()],
            P::empty(),
        ),
        MemoryClass::Shared | MemoryClass::WriteCombined => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::HOST_CACHED, P::DEVICE_LOCAL],
            P::empty(),
        ),
        MemoryClass::Private => (P::DEVICE_LOCAL, [P::empty(); 2], P::HOST_VISIBLE),
        MemoryClass::Memoryless => (
            P::DEVICE_LOCAL,
            [P::LAZILY_ALLOCATED, P::empty()],
            P::HOST_VISIBLE,
        ),
        MemoryClass::Managed => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::HOST_CACHED, P::empty()],
            P::empty(),
        ),
    };
    let lacks = |flags: P, wanted: P| u8::from(!wanted.is_empty() && !flags.contains(wanted));
    let mut ranked: Vec<((u8, u8), u32)> = types
        .iter()
        .enumerate()
        .filter(|&(index, ty)| {
            usable(index)
                && allocatable(ty.property_flags)
                && ty.property_flags.contains(required)
                && (class == MemoryClass::Memoryless
                    || !ty.property_flags.contains(P::LAZILY_ALLOCATED))
        })
        .map(|(index, ty)| {
            let flags = ty.property_flags;
            let first = lacks(flags, preferred[0])
                + u8::from(!avoided.is_empty() && flags.intersects(avoided));
            ((first, lacks(flags, preferred[1])), index as u32)
        })
        .collect();
    // Vulkan lists memory types best first within equal properties; the sort is stable.
    ranked.sort_by_key(|&(rank, _)| rank);
    ranked.into_iter().map(|(_, index)| index).collect()
}

/// The block bookkeeping of one memory type: pure, so the CPU tests cover it. `T` is what the
/// caller keeps per block (its `VkDeviceMemory` and mapping).
#[derive(Debug)]
pub(super) struct Pool<T> {
    blocks: Vec<Option<PoolBlock<T>>>,
    /// The largest alignment a pooled placement asked of it so far ([`Pool::has_room`]).
    align: u64,
}

#[derive(Debug)]
struct PoolBlock<T> {
    ranges: RangeAllocator,
    dedicated: bool,
    payload: T,
}

impl<T> Default for Pool<T> {
    fn default() -> Pool<T> {
        Pool {
            blocks: Vec::new(),
            align: 1,
        }
    }
}

impl<T> Pool<T> {
    /// A place for `size` bytes aligned to `align` in a pooled block that has room: (block,
    /// offset).
    pub(super) fn place(&mut self, size: u64, align: u64) -> Option<(usize, u64)> {
        self.align = self.align.max(align);
        self.blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, block)| {
                let block = block.as_mut().filter(|block| !block.dedicated)?;
                block
                    .ranges
                    .allocate(size, align)
                    .map(|offset| (index, offset))
            })
    }

    /// Add a block of `size` bytes holding `payload`, and place `used` bytes at its start when
    /// given: the block's index.
    pub(super) fn add(&mut self, size: u64, dedicated: bool, payload: T, used: u64) -> usize {
        let mut ranges = RangeAllocator::new(0, size);
        if used > 0 {
            let at = ranges.allocate(used, 1);
            debug_assert_eq!(at, Some(0));
        }
        let block = Some(PoolBlock {
            ranges,
            dedicated,
            payload,
        });
        match self.blocks.iter().position(Option::is_none) {
            Some(index) => {
                self.blocks[index] = block;
                index
            }
            None => {
                self.blocks.push(block);
                self.blocks.len() - 1
            }
        }
    }

    /// Whether a pooled block has room for `size` bytes at the largest alignment asked of the
    /// pool so far.
    pub(super) fn has_room(&self, size: u64) -> bool {
        (self.blocks.iter().flatten())
            .any(|block| !block.dedicated && block.ranges.longest(self.align) >= size)
    }

    /// The payload of block `index`.
    pub(super) fn payload(&self, index: usize) -> Option<&T> {
        self.blocks.get(index)?.as_ref().map(|block| &block.payload)
    }

    /// The payload of block `index`, to change.
    fn payload_mut(&mut self, index: usize) -> Option<&mut T> {
        self.blocks
            .get_mut(index)?
            .as_mut()
            .map(|block| &mut block.payload)
    }

    /// Return `[offset, offset + size)` of block `index`. The block's payload when the block
    /// must now be freed: a dedicated block, or a pooled block that emptied while another empty
    /// pooled block is kept.
    pub(super) fn release(&mut self, index: usize, offset: u64, size: u64) -> Option<T> {
        let block = self.blocks.get_mut(index)?.as_mut()?;
        block.ranges.free(offset, size);
        if !block.ranges.is_empty() {
            return None;
        }
        let keep = !block.dedicated
            && !self.blocks.iter().enumerate().any(|(other, block)| {
                other != index
                    && block
                        .as_ref()
                        .is_some_and(|block| !block.dedicated && block.ranges.is_empty())
            });
        if keep {
            return None;
        }
        self.blocks[index].take().map(|block| block.payload)
    }

    /// Every block's payload, emptying the pool.
    pub(super) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.blocks.drain(..).flatten().map(|block| block.payload)
    }

    /// The number of live blocks.
    #[cfg(test)]
    pub(super) fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }
}

/// A host pointer the runtime may send between threads: the mapping of a block it owns.
#[derive(Clone, Copy, Debug)]
pub(super) struct Mapped(NonNull<u8>);

// SAFETY: the pointer names persistently mapped device memory the runtime owns; who writes
// through it is the resource API's contract (Metal's `contents`).
unsafe impl Send for Mapped {}
// SAFETY: as above.
unsafe impl Sync for Mapped {}

impl Mapped {
    pub(super) fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
}

/// One `VkDeviceMemory` block.
#[derive(Debug)]
pub(super) struct Block {
    memory: vk::DeviceMemory,
    /// The bytes its placements take.
    size: u64,
    mapped: Option<Mapped>,
    /// Its buffer of addresses, from the first placement that needs an address (the module
    /// docs).
    addresses: Option<BlockAddresses>,
}

/// A buffer over the first `Block::size` bytes of a block, with every usage of an addressed
/// buffer (`buffer::usage`), whose device address gives each buffer placed in the block its own
/// (the module docs).
#[derive(Debug)]
struct BlockAddresses {
    raw: vk::Buffer,
    address: u64,
}

impl Block {
    /// Free the block's memory, its buffer of addresses first.
    ///
    /// # Safety
    ///
    /// Nothing is placed in the block any more, and no command that reads it is pending.
    unsafe fn release(self, device: &ash::Device) {
        if let Some(addresses) = self.addresses {
            // SAFETY: the caller's contract.
            unsafe { release::object(device, addresses.raw) };
        }
        // SAFETY: the caller's contract; freeing unmaps it.
        unsafe { release::object(device, self.memory) };
    }
}

/// A buffer of `size` bytes with every usage of an addressed buffer, shared as `sharing` says,
/// unbound, and the bytes of memory it requires.
fn address_buffer(
    device: &ash::Device,
    size: u64,
    sharing: Sharing<'_>,
) -> Result<(vk::Buffer, u64), Error> {
    let info = sharing.buffer(
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage(true)),
    );
    // SAFETY: the device is live and `info` valid.
    let raw = unsafe { device.create_buffer(&info, None) }.map_err(vulkan)?;
    // SAFETY: `raw` is a live buffer of this device.
    let required = unsafe { device.get_buffer_memory_requirements(raw) }.size;
    Ok((raw, required))
}

/// `raw` ([`address_buffer`]'s) bound at the start of `memory`, with its device address; released
/// when the binding fails.
///
/// # Safety
///
/// `memory` is the device's, allocated with the device-address flag, of a type `raw` takes and
/// at least as long as `raw` requires.
unsafe fn bind_addresses(
    device: &ash::Device,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
) -> Result<BlockAddresses, Error> {
    // SAFETY: the caller's contract.
    if let Err(error) = unsafe { device.bind_buffer_memory(raw, memory, 0) } {
        // SAFETY: the buffer is unused.
        unsafe { release::object(device, raw) };
        return Err(vulkan(error));
    }
    let info = vk::BufferDeviceAddressInfo::default().buffer(raw);
    // SAFETY: the buffer has the device-address usage and is bound whole to memory allocated
    // with the device-address flag.
    let address = unsafe { device.get_buffer_device_address(&info) };
    Ok(BlockAddresses { raw, address })
}

/// Where a resource's memory is.
#[derive(Debug)]
pub(super) struct Placement {
    type_index: u32,
    /// What its block holds.
    holds: Holds,
    /// The runtime's block the placement is in; `None` for memory the caller made and keeps
    /// (`Device::new_buffer_with_vk_buffer`), which the runtime never frees.
    block: Option<usize>,
    size: u64,
    pub(super) memory: vk::DeviceMemory,
    pub(super) offset: u64,
    /// The host address of `offset` when the memory is host-visible.
    pub(super) mapped: Option<Mapped>,
    /// The device address of `offset`, from the block's buffer of addresses, where the placement
    /// asked for one ([`allocate_addressed`]) on a device that enables
    /// `bufferDeviceAddress`.
    pub(super) address: Option<u64>,
    /// The imports of the caller's dma-bufs the placement lies in, with the source that made them
    /// (`dma_buf.rs`), which no block of the runtime's holds: kept until the placement goes, and
    /// freed, the source dropped, when their last holder goes.
    #[cfg(unix)]
    dma_buf: Option<Arc<DmaBufPages>>,
}

impl Placement {
    /// The placement of `size` bytes at `offset` in `memory`, of memory type `type_index`, that
    /// the caller made and keeps (`wrapped_buffer.rs`): no block of the runtime's, so freeing it
    /// returns nothing. `mapped` is the host address of `offset`, where the caller gives one.
    pub(super) fn caller(
        memory: vk::DeviceMemory,
        offset: u64,
        size: u64,
        type_index: u32,
        mapped: Option<NonNull<u8>>,
    ) -> Placement {
        Placement {
            type_index,
            holds: Holds::Buffers,
            block: None,
            size,
            memory,
            offset,
            mapped: mapped.map(Mapped),
            address: None,
            #[cfg(unix)]
            dma_buf: None,
        }
    }

    /// The placement of a no-copy buffer's `size` bytes of host pages in the imports of the
    /// caller's dma-bufs of them, `pages`: at offset 0 of `memory`, of type `type_index`, where
    /// they are one import, or no memory (`VK_NULL_HANDLE`) where they are several bound to a
    /// sparse buffer; their host address is `mapped`, the pages' pointer (the runtime never maps
    /// a dma-buf). No block of the runtime's, so freeing it returns nothing; it keeps the imports
    /// until it goes.
    #[cfg(unix)]
    pub(super) fn dma_buf(
        pages: Arc<DmaBufPages>,
        memory: vk::DeviceMemory,
        type_index: u32,
        size: u64,
        mapped: NonNull<u8>,
    ) -> Placement {
        Placement {
            type_index,
            holds: Holds::Buffers,
            block: None,
            size,
            memory,
            offset: 0,
            mapped: Some(Mapped(mapped)),
            address: None,
            dma_buf: Some(pages),
        }
    }

    /// The index of the memory type the placement's memory is of.
    pub(super) fn type_index(&self) -> u32 {
        self.type_index
    }

    /// The bytes placed.
    pub(super) fn size(&self) -> u64 {
        self.size
    }

    /// Whether the placement lies in imports of the caller's dma-bufs (`dma_buf.rs`), to which no
    /// image is bound (an image over them is bound to a dma-buf of its own pages).
    pub(super) fn over_dma_buf(&self) -> bool {
        #[cfg(unix)]
        {
            self.dma_buf.is_some()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// The imports of the caller's dma-bufs the placement lies in, with their source
    /// (`dma_buf.rs`); `None` for any other placement.
    #[cfg(unix)]
    pub(super) fn dma_buf_pages(&self) -> Option<&Arc<DmaBufPages>> {
        self.dma_buf.as_ref()
    }

    /// Wait until the sparse binding of the buffer over the placement's dma-bufs completed,
    /// where it may not have (`dma_buf.rs`'s `DmaBufPages::settle`): before that buffer is
    /// destroyed. Any other placement has nothing to wait for.
    pub(super) fn settle(&self) {
        #[cfg(unix)]
        if let Some(pages) = &self.dma_buf {
            pages.settle();
        }
    }

    /// The same placement, for another owner to free from now on: an image's first memory, which
    /// goes with its first `VkImage` once the image takes its storage form
    /// (`texture/storage_form.rs`). The image frees its storage form's memory instead, never
    /// this one.
    pub(super) fn handed_over(&self) -> Placement {
        Placement {
            type_index: self.type_index,
            holds: self.holds,
            block: self.block,
            size: self.size,
            memory: self.memory,
            offset: self.offset,
            mapped: self.mapped,
            address: self.address,
            #[cfg(unix)]
            dma_buf: self.dma_buf.clone(),
        }
    }
}

/// The device's memory: one [`Pool`] per memory type for buffers, and one for images.
pub(super) struct Memory {
    types: Vec<vk::MemoryType>,
    /// The buffers' pools by memory type, then the images' ([`Memory::pool`]).
    pools: Vec<Pool<Block>>,
    block_bytes: u64,
    /// Blocks are allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`: the device enabled
    /// `bufferDeviceAddress`, so buffers carry device addresses.
    device_address: bool,
    /// The memory types a buffer of addresses takes, as bits, once asked
    /// ([`allocate_addressed`]).
    address_types: Option<u32>,
    /// The pools the runtime's encodings and commits place their own resources in, once asked
    /// ([`Memory::kept`], [`keep_room`]).
    kept: Option<Vec<(Holds, u32)>>,
    /// The device has unified memory (`DeviceProperties::has_unified_memory`), which
    /// [`candidates`] ranks `WriteCombined` memory by.
    unified: bool,
    /// The device's queue families, which its buffers of addresses are shared among
    /// (`sharing.rs`).
    families: Vec<u32>,
    /// The bytes of the blocks allocated so far ([`Memory::allocated`]).
    allocated: u64,
}

impl Memory {
    /// The memory of a device with `properties`, whose largest allocation is
    /// `max_allocation` bytes, with unified memory or without (`unified`), whose queue families
    /// are `families`.
    pub(super) fn new(
        properties: &vk::PhysicalDeviceMemoryProperties,
        max_allocation: u64,
        device_address: bool,
        unified: bool,
        families: Vec<u32>,
    ) -> Memory {
        let count = properties.memory_type_count as usize;
        let types = properties.memory_types[..count.min(vk::MAX_MEMORY_TYPES)].to_vec();
        Memory {
            pools: (types.iter().chain(&types))
                .map(|_| Pool::default())
                .collect(),
            types,
            block_bytes: BLOCK_BYTES.min(max_allocation.max(1)),
            device_address,
            address_types: None,
            kept: None,
            unified,
            families,
            allocated: 0,
        }
    }

    /// The bytes of `VkDeviceMemory` allocated so far for blocks of the runtime's own memory,
    /// pooled or dedicated, whether freed since or not: host memory imported counts in none.
    pub(super) fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Whether one of the memory types `allowed` names is of `class`.
    pub(super) fn has_type(&self, allowed: u32, class: MemoryClass) -> bool {
        self.class_types(allowed, class) != 0
    }

    /// Whether an image of `class` with host-transfer usage may lie in memory the host caches
    /// ([`host_cached_placement`]), whatever types its requirements allow.
    pub(super) fn has_host_cached(&self, class: MemoryClass) -> bool {
        host_cached_placement(&self.types, u32::MAX, class, self.unified).is_some()
    }

    /// The memory class an image of `class` with host-transfer usage takes in memory the host
    /// caches, and its memory types among `allowed`, as bits ([`host_cached_placement`]).
    pub(super) fn host_cached_placement(
        &self,
        allowed: u32,
        class: MemoryClass,
    ) -> Option<(MemoryClass, u32)> {
        host_cached_placement(&self.types, allowed, class, self.unified)
    }

    /// The memory types the host maps without coherence, as bits ([`host_incoherent`]).
    pub(super) fn host_incoherent(&self) -> u32 {
        host_incoherent(&self.types)
    }

    /// The memory types of `class` among those `allowed` names, as bits.
    pub(super) fn class_types(&self, allowed: u32, class: MemoryClass) -> u32 {
        candidates(&self.types, allowed, class, self.unified)
            .into_iter()
            .fold(0, |bits, index| bits | 1 << index)
    }

    /// The index in `pools` of the pool of blocks of memory type `type_index` that hold `holds`.
    fn pool_at(&self, holds: Holds, type_index: u32) -> usize {
        type_index as usize + usize::from(holds == Holds::Images) * self.types.len()
    }

    /// The pool of blocks of memory type `type_index` that hold `holds`.
    fn pool(&mut self, holds: Holds, type_index: u32) -> &mut Pool<Block> {
        let at = self.pool_at(holds, type_index);
        &mut self.pools[at]
    }

    /// [`pool`](Self::pool), to read.
    fn pool_ref(&self, holds: Holds, type_index: u32) -> &Pool<Block> {
        &self.pools[self.pool_at(holds, type_index)]
    }

    /// The memory types a buffer of addresses takes, as bits, asked once: `None` without
    /// `bufferDeviceAddress`, where blocks carry no buffer of addresses.
    fn address_types(&mut self, device: &ash::Device) -> Result<Option<u32>, Error> {
        if !self.device_address {
            return Ok(None);
        }
        if self.address_types.is_none() {
            let sharing = Sharing::new(&self.families);
            self.address_types = Some(buffer_types(device, usage(true), sharing)?);
        }
        Ok(self.address_types)
    }

    /// `size` bytes aligned to `align` in a pooled block of `pool` that has room, in a block
    /// with a buffer of addresses where `addressed` gives the counts of such buffers made:
    /// `None` where no block has room.
    fn place_pooled(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        size: u64,
        align: u64,
        addressed: Option<&Counters>,
    ) -> Result<Option<Placement>, Error> {
        let Some((block, offset)) = self.pool(holds, type_index).place(size, align) else {
            return Ok(None);
        };
        if let Some(counts) = addressed {
            if let Err(error) = self.address_block(device, type_index, block, counts) {
                self.release_range(device, (holds, type_index), block, offset, size);
                return Err(error);
            }
        }
        let addressed = addressed.is_some();
        Ok(Some(self.placement(
            (holds, type_index),
            block,
            offset,
            size,
            addressed,
        )))
    }

    /// Block `block` of `size` bytes, just made, added to the pool of memory type `type_index`
    /// that holds `holds`, dedicated or pooled, with `used` bytes at its start placed: its index.
    fn add_block(
        &mut self,
        (holds, type_index): (Holds, u32),
        block: Block,
        size: u64,
        dedicated: bool,
        used: u64,
    ) -> usize {
        self.allocated += size;
        self.pool(holds, type_index)
            .add(size, dedicated, block, used)
    }

    /// What making a block takes, copied out so that the block is made with the memory's lock
    /// released ([`place`], [`keep_room`]).
    fn maker(&self) -> BlockMaker {
        let host_visible = (self.types.iter().enumerate())
            .filter(|(_, ty)| (ty.property_flags).contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
            .fold(0, |bits, (index, _)| bits | 1 << index);
        BlockMaker {
            block_bytes: self.block_bytes,
            host_visible,
            device_address: self.device_address,
            families: self.families.clone(),
        }
    }

    /// The pools the runtime's encodings and commits place their own resources in, found once:
    /// of buffers, the best memory type of `Private`, `WriteCombined` and `Shared` memory among
    /// those a buffer the runtime places takes (`buffer::usage(false)`, its address from a buffer
    /// of addresses, [`allocate_addressed`]); of images, the best `Private` type among those a
    /// color image of optimal tiling takes, which every such image takes (the Vulkan
    /// specification's `VkMemoryRequirements`).
    fn kept(&mut self, device: &ash::Device) -> Result<Vec<(Holds, u32)>, Error> {
        if let Some(kept) = &self.kept {
            return Ok(kept.clone());
        }
        let addresses = self.address_types(device)?.unwrap_or(u32::MAX);
        let sharing = Sharing::new(&self.families);
        let buffers = buffer_types(device, usage(false), sharing)? & addresses;
        let images = color_image_types(device, sharing)?;
        let best = |bits: u32, class: MemoryClass| {
            candidates(&self.types, bits, class, self.unified)
                .first()
                .copied()
        };
        let mut kept = Vec::new();
        let classes = [
            MemoryClass::Private,
            MemoryClass::WriteCombined,
            MemoryClass::Shared,
        ];
        let buffer_pools = (classes.into_iter())
            .filter_map(|class| best(buffers, class))
            .map(|type_index| (Holds::Buffers, type_index));
        let image_pool =
            best(images, MemoryClass::Private).map(|type_index| (Holds::Images, type_index));
        for pool in buffer_pools.chain(image_pool) {
            if !kept.contains(&pool) {
                kept.push(pool);
            }
        }
        self.kept = Some(kept.clone());
        Ok(kept)
    }

    /// The placement of `size` bytes at `offset` in block `block` of memory type `type_index`,
    /// with the block's device address of `offset` where `addressed`. A placement that asked for
    /// none has none, though its block has a buffer of addresses: its buffer has an address of
    /// its own where it needs one (a preprocess buffer's must be its own, of its usage).
    fn placement(
        &self,
        (holds, type_index): (Holds, u32),
        block: usize,
        offset: u64,
        size: u64,
        addressed: bool,
    ) -> Placement {
        let payload = (self.pool_ref(holds, type_index))
            .payload(block)
            .expect("a placed block is live");
        Placement {
            type_index,
            holds,
            block: Some(block),
            size,
            memory: payload.memory,
            offset,
            mapped: payload.mapped.map(|mapped| {
                // SAFETY: `offset + size` is inside the block, which is mapped whole.
                Mapped(unsafe { NonNull::new_unchecked(mapped.as_ptr().add(offset as usize)) })
            }),
            address: payload
                .addresses
                .as_ref()
                .filter(|_| addressed)
                .map(|addresses| addresses.address + offset),
            #[cfg(unix)]
            dma_buf: None,
        }
    }

    /// Give pooled block `block` of memory type `type_index` a buffer of addresses where it has
    /// none: over its whole size, which it takes since the block is of a type such a buffer
    /// takes ([`allocate_addressed`]); refused (an invariant) where the buffer would
    /// need more bytes than the block has.
    fn address_block(
        &mut self,
        device: &ash::Device,
        type_index: u32,
        block: usize,
        counts: &Counters,
    ) -> Result<(), Error> {
        let at = self.pool_at(Holds::Buffers, type_index);
        let sharing = Sharing::new(&self.families);
        let payload = self.pools[at]
            .payload_mut(block)
            .expect("a placed block is live");
        if payload.addresses.is_some() {
            return Ok(());
        }
        let (raw, required) = address_buffer(device, payload.size, sharing)?;
        if required > payload.size {
            // SAFETY: the buffer is unused.
            unsafe { release::object(device, raw) };
            return Err(internal(
                "a buffer over a whole memory block requires more bytes than the block has",
            ));
        }
        // SAFETY: the block's memory is the device's, allocated with the device-address flag
        // (`device_address`), of a type the buffer takes, and as long as the buffer requires.
        payload.addresses = Some(unsafe { bind_addresses(device, raw, payload.memory) }?);
        counts.buffer_made(usage(true));
        Ok(())
    }

    /// Return `[offset, offset + size)` of block `block` of memory type `type_index`, freeing the
    /// block when it empties ([`Pool::release`]).
    fn release_range(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        block: usize,
        offset: u64,
        size: u64,
    ) {
        if let Some(block) = self.pool(holds, type_index).release(block, offset, size) {
            // SAFETY: no resource is placed in the block any more.
            unsafe { block.release(device) };
        }
    }

    /// Import the caller's host memory, `size` bytes at `pointer` (both multiples of the device's
    /// import alignment), as a dedicated block of the best memory type of `class` among
    /// `allowed` (the types both the pointer and the resource take): its placement at offset 0,
    /// whose `mapped` is `pointer`. The block is never pooled and is freed with the placement,
    /// as a dedicated block is (`host_memory.rs`). Allocated for export of its
    /// `MTLBuffer` where `export_metal` (`VK_EXT_metal_objects`, `metal_texture.rs`). Refused
    /// when no type of `class` is among `allowed`, else with the failure of the last type tried.
    ///
    /// # Safety
    ///
    /// The `size` bytes at `pointer` are host memory of this process that stays mapped until the
    /// placement is freed.
    pub(super) unsafe fn import_host(
        &mut self,
        device: &ash::Device,
        pointer: NonNull<u8>,
        size: u64,
        allowed: u32,
        class: MemoryClass,
        export_metal: bool,
    ) -> Result<Placement, ImportFailure> {
        let mut failure = ImportFailure::NoMemoryType;
        for type_index in candidates(&self.types, allowed, class, self.unified) {
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(pointer.as_ptr().cast());
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut export = vk::ExportMetalObjectCreateInfoEXT::default()
                .export_object_type(vk::ExportMetalObjectTypeFlagsEXT::METAL_BUFFER);
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(type_index)
                .push_next(&mut import);
            if self.device_address {
                info = info.push_next(&mut flags);
            }
            if export_metal {
                info = info.push_next(&mut export);
            }
            // SAFETY: `info` and its chain live for the call; the type index is the device's and
            // allows the pointer; the caller's contract keeps the memory mapped.
            match unsafe { device.allocate_memory(&info, None) } {
                Ok(memory) => {
                    let block = Block {
                        memory,
                        size,
                        mapped: Some(Mapped(pointer)),
                        addresses: None,
                    };
                    let index = self
                        .pool(Holds::Buffers, type_index)
                        .add(size, true, block, size);
                    return Ok(self.placement((Holds::Buffers, type_index), index, 0, size, false));
                }
                Err(error) => failure = ImportFailure::Vulkan(error),
            }
        }
        Err(failure)
    }

    /// Import the dma-buf `fd`, `size` bytes, as memory of the best type of `class` among
    /// `allowed` (the types both the dma-buf and what is bound in it take): the memory and its
    /// type, the caller's to free (`dma_buf.rs`'s `DmaBufPages`), in no block and never mapped.
    /// For a buffer, allocated with the device-address flag where the device enables
    /// `bufferDeviceAddress`; for `image`, its dedicated allocation (`size` its requirements').
    /// Refused when no type of `class` is among `allowed`, else with the failure of the last type
    /// tried; the descriptor is Vulkan's once imported, and closed on failure.
    #[cfg(unix)]
    pub(super) fn import_dma_buf(
        &self,
        device: &ash::Device,
        fd: std::os::fd::OwnedFd,
        size: u64,
        (allowed, class): (u32, MemoryClass),
        image: Option<vk::Image>,
    ) -> Result<(vk::DeviceMemory, u32), ImportFailure> {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let mut failure = ImportFailure::NoMemoryType;
        for type_index in candidates(&self.types, allowed, class, self.unified) {
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(fd.as_raw_fd());
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default();
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(type_index)
                .push_next(&mut import);
            match image {
                Some(image) => {
                    dedicated = dedicated.image(image);
                    info = info.push_next(&mut dedicated);
                }
                None if self.device_address => info = info.push_next(&mut flags),
                None => {}
            }
            // SAFETY: `info` and its chain live for the call; the type index is the device's and
            // allows the dma-buf (and the image, unbound and made for dma-bufs, whose size this
            // is). A failed import leaves the descriptor open, the caller's.
            match unsafe { device.allocate_memory(&info, None) } {
                Ok(memory) => {
                    // A successful import owns the descriptor: Vulkan closes it with the memory.
                    let _ = fd.into_raw_fd();
                    return Ok((memory, type_index));
                }
                Err(error) => failure = ImportFailure::Vulkan(error),
            }
        }
        Err(failure)
    }

    /// The property flags of memory type `index`, where the device has it.
    pub(super) fn type_flags(&self, index: u32) -> Option<vk::MemoryPropertyFlags> {
        self.types
            .get(index as usize)
            .map(|memory_type| memory_type.property_flags)
    }

    /// Whether `placement` lies in device-local memory.
    pub(super) fn is_device_local(&self, placement: &Placement) -> bool {
        self.type_flags(placement.type_index)
            .is_some_and(|flags| flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
    }

    /// Return `placement`'s memory, freeing its block when it empties.
    pub(super) fn free(&mut self, device: &ash::Device, placement: &Placement) {
        // The caller's memory is the caller's to free.
        let Some(index) = placement.block else {
            return;
        };
        self.release_range(
            device,
            (placement.holds, placement.type_index),
            index,
            placement.offset,
            placement.size,
        );
    }

    /// Free every block: the device is being destroyed and no resource is left.
    pub(super) fn destroy(&mut self, device: &ash::Device) {
        for pool in &mut self.pools {
            for block in pool.drain() {
                // SAFETY: no resource is left, and the device is idle.
                unsafe { block.release(device) };
            }
        }
    }
}

/// What making a block of the runtime's memory takes ([`Memory::maker`]): a copy of the memory's
/// constants, so that the block is made with its lock released.
struct BlockMaker {
    /// A pooled block's size ([`BLOCK_BYTES`], or the device's largest allocation).
    block_bytes: u64,
    /// The host-visible memory types, as bits: a block of one is mapped whole.
    host_visible: u32,
    /// As [`Memory`]'s.
    device_address: bool,
    /// As [`Memory`]'s.
    families: Vec<u32>,
}

impl BlockMaker {
    /// A new block of `size` bytes of memory type `type_index`, with a buffer of addresses where
    /// `addressed` gives the counts of such buffers made (its memory then as long as that buffer
    /// requires, if longer).
    fn make(
        &self,
        device: &ash::Device,
        type_index: u32,
        size: u64,
        addressed: Option<&Counters>,
    ) -> Result<Block, Error> {
        let addresses = if addressed.is_some() && self.device_address {
            Some(address_buffer(device, size, Sharing::new(&self.families))?)
        } else {
            None
        };
        let release_addresses = || {
            if let Some((raw, _)) = addresses {
                // SAFETY: the buffer is unused.
                unsafe { release::object(device, raw) };
            }
        };
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(addresses.map_or(size, |(_, required)| size.max(required)))
            .memory_type_index(type_index);
        if self.device_address {
            info = info.push_next(&mut flags);
        }
        // SAFETY: `info` and its chain live for the call; the type index is the device's.
        let memory = match unsafe { device.allocate_memory(&info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                release_addresses();
                return Err(vulkan(error));
            }
        };
        let host_visible = self.host_visible & 1 << type_index != 0;
        let mapped = if host_visible {
            // SAFETY: the memory is host-visible, unmapped, and mapped whole once.
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(pointer) => NonNull::new(pointer.cast::<u8>()).map(Mapped),
                Err(error) => {
                    release_addresses();
                    // SAFETY: nothing uses the memory yet.
                    unsafe { release::object(device, memory) };
                    return Err(vulkan(error));
                }
            }
        } else {
            None
        };
        let addresses = match addresses {
            // SAFETY: the memory was just allocated with the device-address flag, of a type the
            // buffer takes (`allocate_addressed`), and as long as it requires.
            Some((raw, _)) => match unsafe { bind_addresses(device, raw, memory) } {
                Ok(addresses) => {
                    if let Some(counts) = addressed {
                        counts.buffer_made(usage(true));
                    }
                    Some(addresses)
                }
                Err(error) => {
                    // SAFETY: nothing uses the memory yet; freeing unmaps it.
                    unsafe { release::object(device, memory) };
                    return Err(error);
                }
            },
            None => None,
        };
        Ok(Block {
            memory,
            size,
            mapped,
            addresses,
        })
    }
}

/// The memory types a buffer of `usage` takes, as bits: those of one made for the query and
/// destroyed at once, with no memory bound. Every buffer of one usage takes the same types.
fn buffer_types(
    device: &ash::Device,
    usage: vk::BufferUsageFlags,
    sharing: Sharing<'_>,
) -> Result<u32, Error> {
    let info = sharing.buffer(vk::BufferCreateInfo::default().size(16).usage(usage));
    // SAFETY: the device is live and `info` valid.
    let raw = unsafe { device.create_buffer(&info, None) }.map_err(vulkan)?;
    // SAFETY: `raw` is a live buffer of this device, destroyed right after the query with no
    // memory bound.
    Ok(unsafe {
        let types = device.get_buffer_memory_requirements(raw).memory_type_bits;
        release::object(device, raw);
        types
    })
}

/// The memory types a color image of optimal tiling takes, as bits: those of a 1 x 1
/// `R8G8B8A8_UNORM` image made for the query and destroyed at once, with no memory bound.
fn color_image_types(device: &ash::Device, sharing: Sharing<'_>) -> Result<u32, Error> {
    let info = sharing.image(
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    );
    // SAFETY: the device is live and `info` valid: the format is sampled and a transfer
    // destination with optimal tiling on every device.
    let raw = unsafe { crate::creation::image(device, &info) }?;
    // SAFETY: `raw` is a live image of this device, destroyed right after the query with no
    // memory bound.
    Ok(unsafe {
        let types = device.get_image_memory_requirements(raw).memory_type_bits;
        release::object(device, raw);
        types
    })
}

/// Place memory meeting a buffer's `requirements` of `class` in `memory`: in a pooled block with
/// room, else in a new block (dedicated when larger than half a block), trying the class's
/// memory types best first ([`place`]).
pub(super) fn allocate(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Buffers,
        Fit::Pooled,
    )
}

/// [`allocate`] for an image: in a block that holds images alone ([`Holds`]).
pub(super) fn allocate_image(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Images,
        Fit::Pooled,
    )
}

/// [`allocate_image`] in a block of its own (`Fit::Apart`): for an image the runtime keeps for the
/// device's lifetime and binds while work that uses others of its kind may be queued, a neutral
/// image (`neutral.rs`). Freed, its block goes with it.
pub(super) fn allocate_image_apart(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Images,
        Fit::Apart,
    )
}

/// [`allocate`] for a buffer made without the device-address usage that needs a device address
/// (the module docs): in a block of a memory type a buffer of addresses takes, which gets one if
/// it has none (counted in `counts`), and with the placement's `address`. Without
/// `bufferDeviceAddress`, as `allocate`.
pub(super) fn allocate_addressed(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    counts: &Counters,
) -> Result<Placement, Error> {
    let Some(types) = lock(memory).address_types(device)? else {
        return allocate(memory, device, requirements, class);
    };
    let requirements = vk::MemoryRequirements {
        memory_type_bits: requirements.memory_type_bits & types,
        ..*requirements
    };
    place(
        memory,
        device,
        &requirements,
        class,
        Some(counts),
        Holds::Buffers,
        Fit::Pooled,
    )
}

/// [`allocate`], in a block with a buffer of addresses where `addressed` gives the counts of the
/// buffers of addresses made, of the blocks that hold `holds`, pooled or in a block of its own
/// as `fit` says. A new block is made with `memory`'s lock released: the device's other threads'
/// placements, which the lock orders, wait for no `vkAllocateMemory`, which on NVIDIA's driver
/// waited 2.37 s for other processes' GPU work.
fn place(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    addressed: Option<&Counters>,
    holds: Holds,
    fit: Fit,
) -> Result<Placement, Error> {
    let size = requirements.size.max(1);
    let align = requirements.alignment.max(1);
    let (candidates, maker) = {
        let memory = lock(memory);
        let bits = requirements.memory_type_bits;
        (
            candidates(&memory.types, bits, class, memory.unified),
            memory.maker(),
        )
    };
    if candidates.is_empty() {
        return Err(Error::OutOfMemory(MemoryKind::Device));
    }
    let dedicated = fit == Fit::Apart || size > maker.block_bytes / 2;
    let block_size = if dedicated { size } else { maker.block_bytes };
    let mut failure = Error::OutOfMemory(MemoryKind::Device);
    for type_index in candidates {
        let pool = (holds, type_index);
        if fit == Fit::Pooled {
            if let Some(placement) =
                lock(memory).place_pooled(device, pool, size, align, addressed)?
            {
                return Ok(placement);
            }
        }
        match maker.make(device, type_index, block_size, addressed) {
            Ok(block) => {
                let mut memory = lock(memory);
                let index = memory.add_block(pool, block, block_size, dedicated, size);
                return Ok(memory.placement(pool, index, 0, size, addressed.is_some()));
            }
            Err(error @ Error::OutOfMemory(_)) => failure = error,
            Err(error) => return Err(error),
        }
    }
    Err(failure)
}

/// Keep room for any pooled placement (half a block) in each pool the runtime's encodings and
/// commits place their own resources in ([`Memory::kept`]), making a block, with `memory`'s lock
/// released, for a pool that lacks it. The device's creation calls, which may wait, keep it, so
/// that an encoding or a commit, which must wait for no other process's GPU work, finds a block
/// with room rather than making one. A block of buffers gets its buffer of addresses (counted in
/// `counts`) where blocks carry one. A block that fails to be made, or a pool the query fails for,
/// is left to the placement that needs it.
pub(super) fn keep_room(memory: &Mutex<Memory>, device: &ash::Device, counts: &Counters) {
    let (lacking, maker) = {
        let mut memory = lock(memory);
        let Ok(kept) = memory.kept(device) else {
            return;
        };
        let room = memory.block_bytes / 2;
        let lacking: Vec<(Holds, u32)> = (kept.into_iter())
            .filter(|&(holds, type_index)| !memory.pool_ref(holds, type_index).has_room(room))
            .collect();
        (lacking, memory.maker())
    };
    for pool in lacking {
        let addressed = (pool.0 == Holds::Buffers).then_some(counts);
        let Ok(block) = maker.make(device, pool.1, maker.block_bytes, addressed) else {
            continue;
        };
        let mut memory = lock(memory);
        if memory
            .pool_ref(pool.0, pool.1)
            .has_room(maker.block_bytes / 2)
        {
            drop(memory);
            // SAFETY: nothing is placed in the block, which no command reads.
            unsafe { block.release(device) };
            continue;
        }
        memory.add_block(pool, block, maker.block_bytes, false, 0);
    }
}
