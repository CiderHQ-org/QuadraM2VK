//! A command buffer's upload arena: host-visible memory the runtime writes a command's own data
//! into while the command is encoded — argument, handle and root tables, constant blobs, neutral
//! buffers, and the command buffer's execution status words.
//!
//! - **Chunks.** The arena is a list of `Shared` buffers of the device in the default CPU cache
//!   mode, with Vulkan device addresses where the device has them, each filled from its start:
//!   host-coherent, host-cached where the device has such memory, else device-local where it maps
//!   such (`MemoryClass::Shared`), so system memory on a discrete GPU. The encoding thread writes
//!   them, and a write into device-local memory waits for a powered-down GPU to wake, where one
//!   into system memory does not (an RTX 5080 laptop GPU in NVIDIA's runtime D3 took 970 to 980 ms
//!   for its first write into device-local memory after 20 s idle, against 0.01 ms into system
//!   memory). The host writes host-cached memory faster too, and the GPU reads it as fast: on an RX
//!   7900 XTX (RADV), 3.7 us of thread time a draw that sets its bytes, against 4.3 in device-local
//!   memory, and a vertex-heavy draw's GPU time the same (28.19 ms against 28.18). A region is
//!   never reused while the command buffer lives: every command reads data no other command wrote,
//!   and the host writes it before the command's batch is submitted (a host step's, after the
//!   batches before it). The regions the GPU writes: where a pass's visibility results land before
//!   they are copied on into a no-copy buffer (`visibility.rs`), within one command of barriers,
//!   and an index bounds pass's bounds words and the command buffer's index pool
//!   (`index_bounds.rs`), written before the pass between barriers; and a flat dispatch's region
//!   record and `VkDispatchIndirectCommand` (`binding::Tables::region_record`), which its
//!   preprocess writes before the dispatch between barriers; and an execution's device replay
//!   scratch on the GPU alone (its slot tables, region records, sequences and counts, and the sink;
//!   `generated_execute.rs`), which the replay's preprocess writes before the executions between
//!   barriers; and an emulated tessellation draw's work buffer, whose first words an indirect
//!   draw's copy of its arguments and whose header and chunk slots its scan write before its draws
//!   between barriers (`render/tessellation_scan.rs`): every chunk carries indirect and storage
//!   usage and, where the device has them, a device address. The command buffer releases the arena
//!   when it completes.
//! - **Sizes.** A command buffer's first chunk holds [`FIRST_CHUNK_BYTES`], and each next one
//!   twice as many, up to [`CHUNK_BYTES`]; a region of more than half of that takes a chunk of its
//!   own. A new chunk is cleared whole (a new buffer reads zero), so a command buffer of a few
//!   commands clears a few KiB, not 64 (clearing 64 KiB took 11.5 us of an RTX 3090's host-visible
//!   memory and 2.6 us of a Radeon 780M's).
//! - **Alignment.** Every region starts at a multiple of the device's upload alignment (16 bytes,
//!   and the storage and uniform buffer offset alignments), so it can be bound at its offset and
//!   its device address is 16-aligned. A root table and the resident set it searches start at a
//!   device address that is a multiple of 32 too (`upload_aligned`). A region's bytes past the data
//!   are zero (a new buffer reads zero).

use super::buffer::Buffer;
use super::error::{Error, MemoryKind};
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use ash::vk;

/// The size of a command buffer's first ordinary chunk.
const FIRST_CHUNK_BYTES: u64 = 4 * 1024;

/// The size of the largest ordinary chunk; a region of more than half of it gets a chunk of its
/// own.
const CHUNK_BYTES: u64 = 64 * 1024;

/// The arena of one command buffer.
pub(super) struct Uploads {
    /// The chunk filled now is the last; dedicated chunks go before it.
    chunks: Vec<Chunk>,
    /// The ordinary chunks opened: the next is [`FIRST_CHUNK_BYTES`] times two to this power,
    /// at most [`CHUNK_BYTES`].
    opened: u32,
}

struct Chunk {
    buffer: Buffer,
    /// Bytes from the chunk's start that regions took.
    used: u64,
}

/// A region of the arena: `len` bytes of `buffer` from `offset`.
#[derive(Clone, Debug)]
pub(super) struct Upload {
    buffer: Buffer,
    offset: u64,
    len: u64,
}

impl Uploads {
    /// An empty arena.
    pub(super) fn new() -> Uploads {
        Uploads {
            chunks: Vec::new(),
            opened: 0,
        }
    }

    /// Release every chunk: no command reads the arena any more.
    pub(super) fn clear(&mut self) {
        self.chunks.clear();
        self.opened = 0;
    }

    /// A region of `len` bytes (at least `bytes.len()`) holding `bytes` and then zeros.
    pub(super) fn upload(
        &mut self,
        device: &Device,
        bytes: &[u8],
        len: u64,
    ) -> Result<Upload, Error> {
        self.upload_aligned(device, bytes, len, 1)
    }

    /// [`upload`](Self::upload) at a device address that is a multiple of `address_alignment`
    /// too, a power of two (a root table and the resident set it searches, 32 bytes): on a
    /// device without device addresses, at an offset that is.
    pub(super) fn upload_aligned(
        &mut self,
        device: &Device,
        bytes: &[u8],
        len: u64,
        address_alignment: u64,
    ) -> Result<Upload, Error> {
        let len = len.max(bytes.len() as u64);
        let alignment = device.shared().upload_alignment;
        // The first offset from `from`, a multiple of the upload alignment, whose device address
        // in `chunk` is a multiple of `address_alignment`: both are powers of two, so the walk
        // takes at most `address_alignment / alignment` steps.
        let aligned = |chunk: &Chunk, from: u64| {
            let base = chunk.buffer.device_address().unwrap_or(0);
            let mut offset = from.checked_next_multiple_of(alignment)?;
            while !base.wrapping_add(offset).is_multiple_of(address_alignment) {
                offset = offset.checked_add(alignment)?;
            }
            Some(offset)
        };
        let fits = |chunk: &Chunk| {
            aligned(chunk, chunk.used)
                .and_then(|offset| Some((offset, offset.checked_add(len)?)))
                .filter(|&(_, end)| end <= chunk.buffer.length())
        };
        let current = self
            .chunks
            .last()
            .and_then(fits)
            .map(|(offset, end)| (self.chunks.len() - 1, offset, end));
        let (index, offset, end) = match current {
            Some(found) => found,
            None => {
                // Room for the region past the first aligned address.
                let need = len
                    .checked_add(address_alignment - 1)
                    .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
                let dedicated = need > CHUNK_BYTES / 2;
                let size = if dedicated {
                    need.checked_next_multiple_of(CHUNK_BYTES)
                        .ok_or(Error::OutOfMemory(MemoryKind::Device))?
                } else {
                    // At most `CHUNK_BYTES`: `need` is at most half of it.
                    (FIRST_CHUNK_BYTES << self.opened.min(4)).max(need.next_power_of_two())
                };
                // The encoding thread writes the chunk: host-cached memory where the device has
                // such (the module docs).
                let buffer = device.own_buffer(
                    size,
                    ResourceOptions::new(
                        StorageMode::Shared,
                        CpuCacheMode::DefaultCache,
                        HazardTrackingMode::Untracked,
                    ),
                )?;
                if !device.properties().has_unified_memory
                    && buffer.own_pages_device_local() == Some(true)
                {
                    device.shared().counts.upload_chunk_in_device_memory();
                }
                // A large region takes a chunk of its own, leaving the current chunk current.
                let at = if dedicated && !self.chunks.is_empty() {
                    self.chunks.len() - 1
                } else {
                    self.chunks.len()
                };
                if !dedicated {
                    self.opened += 1;
                }
                let chunk = Chunk { buffer, used: 0 };
                let offset = aligned(&chunk, 0).ok_or(Error::OutOfMemory(MemoryKind::Device))?;
                self.chunks.insert(at, chunk);
                (at, offset, offset + len)
            }
        };
        let chunk = &mut self.chunks[index];
        let Some(mapped) = chunk.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        // SAFETY: `[offset, offset + bytes.len())` is inside the chunk's mapping (`fits`, or a
        // new chunk of at least `len` bytes), no region overlaps it, and no submitted command
        // reads it: regions are written while the command buffer is encoded, or by a host step
        // for a batch not submitted yet, past every region submitted before.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add(offset as usize),
                bytes.len(),
            );
        }
        chunk.used = end;
        Ok(Upload {
            buffer: chunk.buffer.clone(),
            offset,
            len,
        })
    }
}

impl Upload {
    /// The `VkBuffer` the region is in.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer.at(self.offset).0
    }

    /// The region's offset in its `VkBuffer`.
    pub(super) fn offset(&self) -> u64 {
        self.buffer.at(self.offset).1
    }

    /// The region's length.
    pub(super) fn len(&self) -> u64 {
        self.len
    }

    /// Write `bytes` at the region's start again, before any batch that reads the region is
    /// submitted: as the command buffer commits (a function-table snapshot). A write longer than
    /// the region is cut to it.
    pub(super) fn write(&self, bytes: &[u8]) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = bytes.len().min(self.len as usize);
        // SAFETY: `[offset, offset + len)` is inside the region, which no other region overlaps;
        // the caller writes it before the batches that read it are submitted, so no command
        // reads it meanwhile.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add(self.offset as usize),
                len,
            );
        }
    }

    /// [`write`](Self::write) at byte `at` of the region: a root table's resident-set fields as
    /// the command buffer commits, or as a host step uploads it. A write past the region is cut.
    pub(super) fn write_at(&self, at: u64, bytes: &[u8]) {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let len = (bytes.len() as u64).min(self.len.saturating_sub(at)) as usize;
        // SAFETY: `[offset + at, offset + at + len)` is inside the region, which no other region
        // overlaps; the caller writes it before the batches that read it are submitted.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                mapped.as_ptr().add((self.offset + at) as usize),
                len,
            );
        }
    }

    /// The region's Vulkan device address, where the device has device addresses.
    pub(super) fn device_address(&self) -> Option<u64> {
        self.buffer
            .device_address()
            .map(|address| address + self.offset)
    }

    /// The region's first `N` bytes as the device left them. The caller has waited for every
    /// command that writes them, and made their writes visible to the host.
    pub(super) fn read<const N: usize>(&self) -> [u8; N] {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        assert!(N as u64 <= self.len, "a read past the region");
        let mut bytes = [0; N];
        // SAFETY: the region holds at least `N` bytes of the mapping, and no command writes them
        // any more (the caller's contract); the memory is host-coherent.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().add(self.offset as usize),
                bytes.as_mut_ptr(),
                N,
            );
        }
        bytes
    }

    /// The region's bytes as the device left them, under [`read`](Self::read)'s contract.
    pub(super) fn read_bytes(&self) -> Vec<u8> {
        let Some(mapped) = self.buffer.contents() else {
            unreachable!("upload chunks are Shared buffers")
        };
        let mut bytes = vec![0; self.len as usize];
        // SAFETY: the region lies inside the mapping, and no command writes it any more (the
        // caller's contract); the memory is host-coherent.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr().add(self.offset as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        bytes
    }
}
