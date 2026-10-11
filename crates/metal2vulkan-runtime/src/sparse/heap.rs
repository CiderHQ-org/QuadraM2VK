//! A sparse heap's own state: Metal's count of its tiles, and no memory but the zeros its textures'
//! unmapped texels are cleared from. Each Vulkan unit a texture binds takes its
//! own placement from the sub-allocator as it is bound (`texture.rs`), so the heap only counts what
//! Metal counts: `usedSize`, and the room a mapping takes, in Metal's tiles whatever the device's
//! units.

use super::SparsePageSize;
use crate::address::KeptToken;
use crate::attributes::ResourceAttributes;
use crate::buffer::Buffer;
use crate::device::lock;
use crate::error::Error;
use crate::values::ResourceOptions;
use crate::Device;
use std::sync::Mutex;

/// What a sparse heap holds.
pub(crate) struct SparseHeap {
    pub(crate) device: Device,
    /// The descriptor's size rounded up to the page.
    pub(crate) size: u64,
    /// `Private` storage, the descriptor's CPU cache mode, and the tracking mode resolved (a
    /// sparse heap's default is untracked).
    pub(crate) options: ResourceOptions,
    pub(crate) page: SparsePageSize,
    pub(crate) attributes: ResourceAttributes,
    /// The tiles mapped, of the heap's `size / page`.
    tiles: Mutex<u64>,
    /// Zeros its textures' unmapped texels are copied from (`emulated.rs`), made at the first
    /// texture: two pages, more than any tile's bytes.
    zeros: Mutex<Option<Buffer>>,
    /// The caller's token for the heap's range (`Device::new_heap_with_range_token`), set once
    /// as the heap is made: dropped as the heap goes, its textures gone before it (each holds
    /// it), no lock held; left to the process's exit where the heap goes as the process exits.
    pub(crate) range_token: KeptToken,
}

impl Drop for SparseHeap {
    fn drop(&mut self) {
        // The heap holds no memory of the address space's: its zeros lie outside it
        // (`Device::own_buffer`), and its textures' units went with them.
        self.range_token.release();
    }
}

impl SparseHeap {
    /// A heap of `size` bytes (a multiple of `page`) on `device` with `options`.
    pub(crate) fn new(
        device: Device,
        size: u64,
        options: ResourceOptions,
        page: SparsePageSize,
    ) -> SparseHeap {
        SparseHeap {
            device,
            size,
            options,
            page,
            attributes: ResourceAttributes::new(),
            tiles: Mutex::new(0),
            zeros: Mutex::new(None),
            range_token: KeptToken::default(),
        }
    }

    /// The heap's zeros, made at the first call.
    pub(crate) fn zeros(&self) -> Result<Buffer, Error> {
        let mut zeros = lock(&self.zeros);
        if let Some(zeros) = &*zeros {
            return Ok(zeros.clone());
        }
        let made = self
            .device
            .own_buffer(2 * self.page.bytes(), ResourceOptions::private())?;
        *zeros = Some(made.clone());
        Ok(made)
    }

    /// `usedSize`: the mapped tiles' bytes.
    pub(crate) fn used_size(&self) -> u64 {
        *lock(&self.tiles) * self.page.bytes()
    }

    /// Take `tiles` more tiles, when the heap has room for all of them.
    pub(crate) fn take(&self, tiles: u64) -> bool {
        let mut used = lock(&self.tiles);
        let capacity = self.size / self.page.bytes();
        match used.checked_add(tiles) {
            Some(total) if total <= capacity => {
                *used = total;
                true
            }
            _ => false,
        }
    }

    /// Give `tiles` tiles back.
    pub(crate) fn give(&self, tiles: u64) {
        let mut used = lock(&self.tiles);
        *used = used.saturating_sub(tiles);
    }
}
