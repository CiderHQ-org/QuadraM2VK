//! The buffer constructors that take contents: `newBufferWithBytes:length:options:`,
//! `newBufferWithBytesNoCopy:length:options:deallocator:`, and [`BufferDescriptor`], the runtime
//! extension for Apple's private `newBufferWithDescriptor:` that takes either. Placement (a
//! pinned address) joins the descriptor in step 7.
//!
//! The descriptor is taken by value: a no-copy buffer's deallocator is `FnOnce`, so a descriptor
//! makes one buffer.

use super::address::{
    AddressPlacement, AddressRange, GpuAddress, Held, Place, PlacementConflict, RangeToken,
};
use super::address_pool::AddressPool;
#[cfg(unix)]
use super::buffer::Pages;
use super::buffer::{usage, Buffer, BufferParts, HostCell, Lying, Modes, Owner, ReadApart};
use super::device::lock;
use super::error::MemoryKind;
use super::error::{internal, invalid, nil, vulkan, Error, Problem, RuntimeGap};
use super::heap::{Heap, HeapProblem};
use super::host_memory::{
    based_size, binds_in, buffer_base, import_refused, import_window, not_importable,
    refuse_unwritable, refused, Deallocator, HostMemory, HostRelease, ImportWindow, Importer,
    KeepAlive, Source,
};
#[cfg(unix)]
use super::host_residency::Again;
use super::memory::{ImportFailure, MemoryClass, Placement};
use super::rebind::RebindProblem;
use super::texture::stand_in::TextureImages;
use super::texture::Remade;
use super::values::{ResourceOptions, StorageMode};
use super::Device;
use crate::release;
use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

/// What a new buffer holds.
#[non_exhaustive]
#[derive(Default)]
pub enum BufferContents<'a> {
    /// Zero bytes, as `newBufferWithLength:options:` makes them.
    #[default]
    Zero,
    /// A copy of these bytes (`newBufferWithBytes:length:options:`); their length is the
    /// descriptor's.
    Copy(&'a [u8]),
    /// The caller's host memory itself (`newBufferWithBytesNoCopy:length:options:deallocator:`),
    /// given back to `deallocator` when the buffer is released.
    NoCopy {
        /// The caller's memory: its first byte, of the descriptor's length.
        memory: HostMemory,
        /// `deallocator`: called when the buffer is released; `None` for none.
        deallocator: Option<Deallocator>,
        /// What keeps the memory mapped (a runtime extension: [`KeepAlive`]), dropped once the
        /// device's import of its pages is freed; `None` for none. With one, `deallocator` runs
        /// when the buffer is released, as Metal's does; with none, once the import is freed,
        /// which may be later (`host-deallocator-delayed`).
        keep_alive: Option<KeepAlive>,
    },
}

impl std::fmt::Debug for BufferContents<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferContents::Zero => f.write_str("Zero"),
            BufferContents::Copy(bytes) => write!(f, "Copy({} bytes)", bytes.len()),
            BufferContents::NoCopy {
                memory,
                deallocator,
                keep_alive,
            } => f
                .debug_struct("NoCopy")
                .field("pointer", &memory.pointer())
                .field("deallocator", &deallocator.is_some())
                .field("keep_alive", &keep_alive.is_some())
                .finish(),
        }
    }
}

/// `MTLBufferDescriptor` (Apple's private class, a runtime extension): a buffer's length, resource
/// options, contents, and address. Its default is Metal's: length 0, options 0, zero contents, an
/// address the runtime picks (`pinnedGPUAddress` 0).
#[non_exhaustive]
#[derive(Debug, Default)]
pub struct BufferDescriptor<'a> {
    /// `length`, in bytes.
    pub length: u64,
    /// `resourceOptions`: the storage mode, CPU cache mode and hazard tracking mode.
    pub resource_options: ResourceOptions,
    /// What the buffer holds when made: zeros, a copy, or the caller's memory itself.
    pub contents: BufferContents<'a>,
    /// `pinnedGPUAddress`: where the buffer goes in the Metal address space (`address.rs`).
    pub placement: AddressPlacement,
    /// A token the runtime drops once the buffer's range of the address space is free
    /// ([`RangeToken`]); `None` for none. A runtime extension: Metal has none.
    pub range_token: Option<RangeToken>,
    /// A live no-copy buffer whose range the buffer shares, at its address: the buffer is a
    /// no-copy one over the same first byte of host memory, no longer than it (`address.rs`);
    /// `None` for its own range. A runtime extension: Metal answers an IOSurface's address for
    /// every buffer over the surface (`newBufferWithIOSurface:`), which a caller standing a
    /// surface in with its host memory gives so.
    pub range_of: Option<&'a Buffer>,
    /// A pool and the offset in it of the slot the buffer goes in (`address.rs`): the buffer
    /// answers the pool's address plus the offset and holds its length rounded up to 256 bytes
    /// of the pool; `None` for a range of the space. A runtime extension: Metal places a buffer
    /// shorter than a page in a slot of a pool of its own, which its API does not name.
    pub pool_slot: Option<(&'a AddressPool, u64)>,
}

impl<'a> BufferDescriptor<'a> {
    /// A descriptor of `length` bytes with `resource_options` and `contents`.
    pub fn new(
        length: u64,
        resource_options: ResourceOptions,
        contents: BufferContents<'a>,
    ) -> BufferDescriptor<'a> {
        BufferDescriptor {
            length,
            resource_options,
            contents,
            placement: AddressPlacement::Any,
            range_token: None,
            range_of: None,
            pool_slot: None,
        }
    }

    /// This descriptor at `placement` in the address space.
    pub fn with_placement(self, placement: AddressPlacement) -> BufferDescriptor<'a> {
        BufferDescriptor { placement, ..self }
    }

    /// This descriptor with `token`, dropped once the buffer's range is free ([`RangeToken`]).
    pub fn with_range_token(self, token: RangeToken) -> BufferDescriptor<'a> {
        BufferDescriptor {
            range_token: Some(token),
            ..self
        }
    }

    /// This descriptor sharing `buffer`'s range, at its address (`range_of`).
    pub fn with_range_of(self, buffer: &'a Buffer) -> BufferDescriptor<'a> {
        BufferDescriptor {
            range_of: Some(buffer),
            ..self
        }
    }

    /// This descriptor in the slot at `offset` of `pool` (`pool_slot`).
    pub fn with_pool_slot(self, pool: &'a AddressPool, offset: u64) -> BufferDescriptor<'a> {
        BufferDescriptor {
            pool_slot: Some((pool, offset)),
            ..self
        }
    }
}

impl Device {
    /// `newBufferWithDescriptor:` (a runtime extension): a buffer of the descriptor's length and
    /// options holding its contents, as [`Device::new_buffer`], [`Device::new_buffer_with_bytes`]
    /// or [`Device::new_buffer_with_bytes_no_copy`] makes it.
    ///
    /// Refused as those are; `Copy` bytes of another length than the descriptor's are
    /// `Invalid` (`LengthMismatch`); a pinned address whose range meets another resource's bytes,
    /// or leaves the space, is `PlacementConflict`. A refused call never runs the deallocator: it
    /// is dropped. The descriptor's `range_token` is dropped once the buffer's range is free
    /// ([`RangeToken`]), or, for a refused call, before it returns, after any range it took is
    /// free again.
    ///
    /// With `range_of`, the buffer answers that buffer's address and holds its range with it, free
    /// once both are gone (each one's token drops with it). Refused: a buffer of another device
    /// (`Invalid`, `ForeignObject`); `PlacementConflict::Unshared` unless the contents are `NoCopy`
    /// over the first byte of host memory that buffer's bytes are now, of no greater length, that
    /// buffer is a no-copy buffer of its own range, and the placement is `Any` or its address.
    ///
    /// With `pool_slot`, the buffer answers the pool's address plus the offset and holds
    /// `round_up(length, 256)` bytes of the pool from there, free once it is gone (its token drops
    /// then). Refused: a pool of another device (`Invalid`, `ForeignObject`);
    /// `PlacementConflict::OffSlot` for an offset that is not a multiple of 256 or a placement
    /// other than `Any` or that address; `OutsidePool` for bytes past the pool's end; `Held` for
    /// bytes a live buffer in the pool holds; `Unshared` with `range_of`.
    pub fn new_buffer_with_descriptor(
        &self,
        descriptor: BufferDescriptor<'_>,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer_with_descriptor";
        self.keep_room();
        let BufferDescriptor {
            length,
            resource_options,
            contents,
            placement,
            range_token,
            range_of,
            pool_slot,
        } = descriptor;
        let place = match (range_of, pool_slot) {
            (Some(other), slot) => {
                let in_pool = slot.is_some();
                self.shared_place(CALL, other, &contents, length, placement, in_pool)?
            }
            (None, Some((pool, offset))) => {
                if !pool.device().same(self) {
                    return Err(invalid(CALL, Problem::ForeignObject));
                }
                Place::InPool {
                    pool: pool.gpu_address().0,
                    offset,
                    placement,
                }
            }
            (None, None) => Place::At(placement),
        };
        let resource_options = self.take_options(CALL, resource_options);
        let buffer = match contents {
            BufferContents::Zero => self.zero_buffer(CALL, length, resource_options, place),
            BufferContents::Copy(bytes) => {
                if bytes.len() as u64 != length {
                    return Err(invalid(CALL, Problem::LengthMismatch));
                }
                self.buffer_with_bytes(CALL, bytes, resource_options, place)
            }
            // SAFETY: the `HostMemory` constructor's caller vouched for `length` bytes at the
            // pointer.
            BufferContents::NoCopy {
                memory,
                deallocator,
                keep_alive,
            } => unsafe {
                self.buffer_over_host_memory(
                    CALL,
                    memory,
                    length,
                    resource_options,
                    HostRelease::new(deallocator, keep_alive),
                    place,
                    false,
                )
            },
        }?;
        if let Some(token) = range_token {
            buffer.keep_range_token(token)?;
        }
        Ok(self.visible(buffer))
    }

    /// `newBufferWithBytes:length:options:`: a buffer holding a copy of `bytes`.
    ///
    /// Refused as [`Device::new_buffer`] is (an empty `bytes` is a length of 0).
    pub fn new_buffer_with_bytes(
        &self,
        bytes: &[u8],
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer_with_bytes";
        self.keep_room();
        let options = self.take_options(CALL, options);
        let buffer = self.buffer_with_bytes(CALL, bytes, options, AddressPlacement::Any)?;
        Ok(self.visible(buffer))
    }

    /// `newBufferWithBytesNoCopy:length:options:deallocator:`, its pointer wrapped in `memory`
    /// ([`HostMemory`]): a buffer whose bytes are the caller's `length` bytes at
    /// `memory.pointer()` (module docs of `host_memory.rs`). `deallocator` runs once, with the
    /// pointer and `length`, when the buffer's last reference goes: where that is a command
    /// buffer's, in its release on its queue's handler thread, after its completed handlers, where
    /// a wait in it for that command buffer returns at once, as Metal's does. A refused call never
    /// runs it.
    ///
    /// The pages are imported here, as Metal wires them when it makes the buffer: the GPU uses
    /// the pages mapped at the pointer now, and memory the device cannot import is refused here,
    /// as Metal answers nil. The import costs work per page here (on macOS about 11 to 20 ms per
    /// GiB measured on an M4 and an M3 Max), where Metal makes the buffer at once and pays at the
    /// first command using it (24 to 88 ms per GiB). Where the device refuses the pointer and
    /// `memory` carries a dma-buf source ([`HostMemory::with_dma_buf`]), the dma-bufs it makes of
    /// the pages are imported instead.
    ///
    /// Refused: a length of 0 or past `max_buffer_length` (`Nil`, as plain Metal answers nil);
    /// `Private` or `Memoryless` storage, as Metal stops them, and an undefined
    /// option (`Invalid`); a device
    /// without `VK_EXT_external_memory_host` (`Unsupported`, naming it); memory this device
    /// cannot import exactly, by its pointer or by its dma-bufs (`Unsupported`, `HostMemory`),
    /// among it on Linux a page this process cannot write ([`HostMemory::new`]); dma-bufs that
    /// break the source's contract (`Invalid`, `Problem::DmaBuf`); pages past the
    /// device's allocation size (`OutOfMemory`). Pages not mapped, or mapped without access, which
    /// Metal makes no buffer over (nil), are refused here too, typed as the check or the import
    /// answers, though such memory breaks [`HostMemory::new`]'s contract.
    ///
    /// # Safety
    ///
    /// `memory` meets the contract of the [`HostMemory`] constructor that made it
    /// ([`HostMemory::new`], [`HostMemory::with_dma_buf`]) for `length` bytes.
    pub unsafe fn new_buffer_with_bytes_no_copy(
        &self,
        memory: HostMemory,
        length: u64,
        options: ResourceOptions,
        deallocator: Option<Deallocator>,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer_with_bytes_no_copy";
        let options = self.take_options(CALL, options);
        // SAFETY: the caller's contract.
        let buffer = unsafe {
            self.buffer_over_host_memory(
                CALL,
                memory,
                length,
                options,
                HostRelease::new(deallocator, None),
                AddressPlacement::Any.into(),
                false,
            )
        }?;
        Ok(self.visible(buffer))
    }

    /// Tell the device that the pages under the `length` host bytes at `bytes` were replaced (an
    /// unmap and a map at the same addresses, as `mmap` with `MAP_FIXED` does): a no-copy buffer
    /// (or heap) made after this over any of those bytes imports the pages there then, as each
    /// of Metal's `newBufferWithBytesNoCopy:` buffers wires the pages at its address when it is
    /// made. Without the call, a no-copy buffer made over exactly the bytes a live one was made
    /// over, of its memory class, lies in that one's import (`host_aliases.rs`), and so over the
    /// pages that were there when that one was made. Buffers made before the call keep their own
    /// imports, and stay ordered with each other as before. Their pages are those their imports
    /// read: an import of dma-bufs keeps the pages it was made over, as Metal's buffer does, and so
    /// does an import of the pointer on a driver that keeps them; on a driver whose pointer imports
    /// follow their address (`DriverFact::PointerImportsFollowTheirAddress`: ANV on i915, RADV,
    /// lavapipe), an import of the pointer reads the pages mapped there at each submission, the new
    /// ones from the remap on (`docs/FAST-PATHS.md`, known gaps). Not a Metal call: Metal needs
    /// none. A `length` of 0 tells nothing.
    ///
    /// The call may come at the unmap or after the new map, but before a buffer is made over the
    /// new pages: a buffer made over those bytes while it runs may lie in either pages. The bytes
    /// are an address range only, never read: they may be unmapped, and may cover a whole
    /// reservation of many buffers. The call is safe from any thread while other calls run on
    /// others, a no-copy buffer's deallocator among them (the runtime holds no lock this call
    /// takes while it runs a deallocator). It is cheap: with no no-copy buffer live it returns at
    /// once, and otherwise it finds the buffers over the range by address, whether the range
    /// covers none of them or many, never reading them all.
    pub fn host_range_remapped(&self, bytes: NonNull<u8>, length: u64) {
        let start = bytes.as_ptr() as u64;
        let shared = self.shared();
        // The unit an import's pages round out to: an import by pointer over a page the call
        // covers is taken anew at its next submission.
        let unit = (shared.host_import.as_ref())
            .map_or(1, |importer| importer.page.max(importer.alignment));
        let imports = (shared.aliases).remapped(start, start.saturating_add(length), unit);
        shared.counts.host_range_remapped(imports);
    }

    /// A buffer of `length` zero bytes at `placement`, for `call`: refused as
    /// [`Device::new_buffer`] is, and for a pin the address space cannot honour; announced as
    /// `new_buffer` announces it.
    fn zero_buffer(
        &self,
        call: &'static str,
        length: u64,
        options: ResourceOptions,
        placement: impl Into<Place>,
    ) -> Result<Buffer, Error> {
        let (buffer, class) = self.buffer_of_length(call, length, options, placement)?;
        self.degrade_system_memory(&buffer, class, || {
            format!("a `Shared` buffer of {length} bytes")
        });
        Ok(buffer)
    }

    /// A buffer holding a copy of `bytes` at `placement`, for `call`: written through its
    /// mapping, and from there to a `Managed` buffer's own bytes where it has a host copy
    /// (`Buffer::upload`), or for `Private` storage copied from a staging buffer by the device's
    /// next submission, before any of its commands, which the call does not wait for
    /// (`Buffer::fill_first_from`).
    fn buffer_with_bytes(
        &self,
        call: &'static str,
        bytes: &[u8],
        options: ResourceOptions,
        placement: impl Into<Place>,
    ) -> Result<Buffer, Error> {
        let length = bytes.len() as u64;
        let buffer = self.zero_buffer(call, length, options, placement)?;
        let fill = |target: &Buffer| {
            let contents = target
                .contents()
                .ok_or_else(|| internal("a host-visible buffer without contents"))?;
            // SAFETY: the buffer holds `length` bytes at `contents`, and no command uses it yet.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            Ok::<(), Error>(())
        };
        if buffer.storage_mode() != StorageMode::Private {
            fill(&buffer)?;
            buffer.upload(0..length)?;
            return Ok(buffer);
        }
        let staging = self.own_buffer(length, ResourceOptions::shared())?;
        fill(&staging)?;
        buffer.fill_first_from(staging, length)?;
        Ok(buffer)
    }

    /// A no-copy buffer over `length` bytes of `memory` at `placement`, for `call`, given back to
    /// `release` (`host_memory.rs`): a heap's own
    /// buffer when `of_heap` (`heap.rs`, `Device::new_heap_with_bytes_no_copy`), whose memory
    /// must start at a multiple of the import's unit, the larger of the host page and the
    /// device's `minImportedHostPointerAlignment`, so that the heap's offsets are its memory's
    /// (`Invalid`, `HeapProblem::HostMemoryAlignment`, elsewhere).
    ///
    /// # Safety
    ///
    /// `memory` meets [`HostMemory::new`]'s contract for `length` bytes.
    pub(super) unsafe fn buffer_over_host_memory(
        &self,
        call: &'static str,
        memory: HostMemory,
        length: u64,
        options: ResourceOptions,
        release: HostRelease,
        placement: Place,
        of_heap: bool,
    ) -> Result<Buffer, Error> {
        let modes = Modes::of(options)?;
        let class = match modes.storage {
            StorageMode::Shared => MemoryClass::Shared,
            StorageMode::Managed => MemoryClass::Managed,
            StorageMode::Private | StorageMode::Memoryless => {
                return Err(invalid(call, Problem::StorageMode))
            }
        };
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(call, Problem::BufferLength { length }));
        }
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let importer = shared
            .host_import
            .as_ref()
            .map_err(|unsupported| Error::Unsupported(unsupported.clone()))?;
        let (pointer, source) = memory.into_parts();
        let host = host_window(call, importer, pointer, length, source)?;
        let window = host.window;
        if of_heap && window.offset != 0 {
            let alignment = importer.page.max(importer.alignment);
            let problem = HeapProblem::HostMemoryAlignment { alignment };
            return Err(invalid(call, Problem::Heap(problem)));
        }
        // A buffer over the bytes a live no-copy buffer was made over, of its memory class, lies
        // in its import (`host_aliases.rs`), its dma-buf source dropped uncalled; a heap's own
        // buffer imports its own.
        let made_over = (pointer.as_ptr() as u64, length);
        let release = release.of(pointer, length);
        let shares = (!of_heap)
            .then(|| shared.aliases.import(made_over, class))
            .flatten();
        if let Some(import) = shares {
            let address = self.place_window(placement, &window, length)?;
            let buffer =
                self.buffer_in_import(Arc::clone(&import), length, modes, address, release);
            shared
                .aliases
                .join(buffer.identity().key, made_over, class, &import);
            shared.aliases.place(buffer.identity().key, made_over);
            return Ok(buffer);
        }
        // Only a buffer that imports its pages pins them, and so has their writability asked: one
        // that lies in a live import pins none, that import's making asked for the same window,
        // and the pages stay writable while imported (`HostMemory::new`).
        refuse_unwritable(&window, &shared.counts)?;
        // SAFETY: the window lies in the pages of the caller's bytes, which stay mapped
        // (`HostMemory::new`), and the source's pieces hold them (`HostMemory::with_dma_buf`). No
        // lock of the runtime's is held.
        // A no-copy buffer's own import of dma-bufs keeps them, so the device may release it
        // while idle and import them again (`host_residency.rs`), as it may its own import of
        // the pointer where the driver's pointer imports follow their address; a heap's import
        // stays.
        let (buffer, memory, again) =
            match unsafe { self.import_by_pointer(importer, pointer, length, host, class) }? {
                Ok((buffer, memory)) => {
                    #[cfg(unix)]
                    let again = (!of_heap)
                        .then(|| Again::pointer(self, pointer, (length, window, class)))
                        .flatten();
                    #[cfg(not(unix))]
                    let again = None;
                    (buffer, memory, again)
                }
                // SAFETY: as above.
                Err(source) => unsafe {
                    let keep = !of_heap;
                    let window = &window;
                    self.import_from_source(
                        call, importer, pointer, length, window, source, class, keep,
                    )
                }?,
            };
        let (address, reserved) = match self.place_window(placement, &window, length) {
            Ok(held) => held,
            Err(error) => {
                self.discard_import(buffer, memory);
                return Err(error);
            }
        };
        let buffer = self.buffer_of(BufferParts {
            raw: buffer.raw,
            memory,
            offset: window.offset,
            base: buffer.base,
            length,
            reach: buffer.size - buffer.base,
            modes,
            address,
            reserved,
            owner: Owner::Caller,
            release,
            of_heap,
            mirror: None,
            #[cfg(unix)]
            again,
        });
        // A buffer over the bytes another no-copy buffer covers joins it (`host_aliases.rs`), and
        // a later one lies in its import.
        // A buffer just made lies in its pages (`HostCell::current`).
        let current = buffer.host_cell().and_then(|cell| cell.current());
        if let (false, Some(current)) = (of_heap, current) {
            shared
                .aliases
                .join(buffer.identity().key, made_over, class, current.import());
        }
        // Its uses order against every other key's over bytes they meet, a heap's own buffer's
        // too (`host_aliases.rs`).
        shared.aliases.place(buffer.identity().key, made_over);
        Ok(buffer)
    }

    /// Give `buffer`, a no-copy buffer whose hold on its pages is `cell` (`heap`'s own buffer
    /// where given), the host memory `memory` of its length from now on, for `call`
    /// (`rebind.rs`): imported as [`Device::buffer_over_host_memory`] imports it, the heap's
    /// images bound to its bytes made again over it, its deallocator set once the pages are the
    /// buffer's. Once the rebind is made, the images swap in, and the old images and the texel
    /// views over the old pages go with those pages, once nothing holds them, their deallocator
    /// with them. Refused, changing nothing and running no deallocator, as the module says.
    ///
    /// # Safety
    ///
    /// `memory` meets [`HostMemory::new`]'s contract for the buffer's length; no command that
    /// uses the buffer is encoded or committed meanwhile.
    pub(super) unsafe fn rebind_host_pages(
        &self,
        call: &'static str,
        buffer: &Buffer,
        cell: &HostCell,
        memory: HostMemory,
        release: HostRelease,
        heap: Option<&Heap>,
    ) -> Result<(), Error> {
        let class = match buffer.storage_mode() {
            StorageMode::Shared => MemoryClass::Shared,
            StorageMode::Managed => MemoryClass::Managed,
            StorageMode::Private | StorageMode::Memoryless => {
                return Err(internal("a no-copy buffer of private storage"))
            }
        };
        let length = buffer.length();
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let key = buffer.identity().key;
        let encoded = || invalid(call, Problem::Rebind(RebindProblem::EncodedUses));
        if self.rebind_pending(key) {
            return Err(encoded());
        }
        let shared = self.shared();
        let importer = shared
            .host_import
            .as_ref()
            .map_err(|unsupported| Error::Unsupported(unsupported.clone()))?;
        let (pointer, source) = memory.into_parts();
        let host = host_window(call, importer, pointer, length, source)?;
        let window = host.window;
        if heap.is_some() && window.offset != 0 {
            let alignment = importer.page.max(importer.alignment);
            let problem = HeapProblem::HostMemoryAlignment { alignment };
            return Err(invalid(call, Problem::Heap(problem)));
        }
        // The texel views and copies of the textures over the bytes are planned at their offset
        // in the `VkBuffer`, the bytes' offset in their page.
        let texels = cell.texels();
        if !texels.is_empty() && window.offset != cell.offset() {
            return Err(Error::NotImplemented(RuntimeGap::RebindTextures));
        }
        // Detached bytes are given pages with no texture being made over them, and none is
        // made until the textures that presented stand-ins present their own images again.
        let detached_textures = || Error::NotImplemented(RuntimeGap::DetachTextures);
        let mut settled = match cell.detached() {
            true => Some(cell.settled().ok_or_else(detached_textures)?),
            false => None,
        };
        // Bytes a live no-copy buffer was made over, of its memory class, are given the pages of
        // its import, as a buffer made over them lies in it (`host_aliases.rs`), the dma-buf source
        // dropped uncalled: they pin no page, that import's making asked about the same window,
        // and the pages stay writable while imported (`HostMemory::new`). A heap's own buffer
        // imports its own, whose images bound to its bytes are made again over the new import's
        // memory or its dma-bufs.
        let made_over = (pointer.as_ptr() as u64, length);
        let shares = heap
            .is_none()
            .then(|| shared.aliases.import(made_over, class))
            .flatten();
        let (pages, remade) = match shares {
            Some(import) => (self.pages_in_import(import), Remade::default()),
            None => {
                // Only pages imported here are pinned, and so have their writability asked.
                refuse_unwritable(&window, &shared.counts)?;
                // Whether an import alive now holds every page of them: counted once the import
                // is made (`WorkCounts::host_rebind_imports_covered`).
                let unit = importer.page.max(importer.alignment);
                let start = pointer.as_ptr() as u64;
                let covered = (shared.aliases).covered(start, start.saturating_add(length), unit);
                // SAFETY: the window lies in the pages of the caller's bytes, which stay mapped
                // (`HostMemory::new`, `HostMemory::with_dma_buf`).
                let (host, placement, remade) = match unsafe {
                    self.import_by_pointer(importer, pointer, length, host, class)
                }? {
                    Ok((host, placement)) => (host, placement, Remade::default()),
                    // The source is called with no lock of the runtime's held (`dma_buf.rs`):
                    // the textures over detached bytes are let go of meanwhile, and must be
                    // settled again after it.
                    Err(source) => {
                        let was_settled = settled.take().is_some();
                        // SAFETY: as above, and the source's pieces hold the pages
                        // (`HostMemory::with_dma_buf`).
                        let (host, placement, _) = unsafe {
                            let window = &window;
                            self.import_from_source(
                                call, importer, pointer, length, window, source, class, false,
                            )
                        }?;
                        // The heap's images bound to its bytes, made over dma-bufs of their own
                        // pages, each of which the source makes: still with no lock held.
                        let again = match heap {
                            Some(heap) => self.dma_buf_images_again(heap, &placement, pointer),
                            None => Ok(Remade::default()),
                        };
                        let remade = match again {
                            Ok(remade) => remade,
                            Err(error) => {
                                self.discard_import(host, placement);
                                return Err(error);
                            }
                        };
                        if was_settled {
                            match cell.settled() {
                                Some(again) => settled = Some(again),
                                None => {
                                    drop(remade);
                                    self.discard_import(host, placement);
                                    return Err(detached_textures());
                                }
                            }
                        }
                        (host, placement, remade)
                    }
                };
                let pages = self.host_pages(
                    host.raw,
                    placement,
                    window.offset,
                    host.base,
                    host.size - host.base,
                );
                shared.counts.host_rebind_import_made(covered);
                (pages, remade)
            }
        };
        // The heap's images bound to its bytes, made again over the new pages. A refusal drops
        // them and the new pages unreleased: the deallocator goes unrun.
        let mut remade = remade;
        let swaps = match heap {
            Some(heap) => self.rebound_heap_images(heap, &pages, &mut remade)?,
            None => Vec::new(),
        };
        let (old, again) = shared
            .rebinds
            .replace(key, cell, Arc::clone(&pages))
            .map_err(|_| encoded())?;
        // The buffer, or the heap's buffers, are reached unnamed again only once a command buffer
        // uses it after the rebind, over the new pages.
        shared.live_buffers.rebound(buffer, heap);
        // Later buffers over the memory it was made over no longer join it.
        shared.aliases.leave(key);
        // Its uses from now on order against the keys over the new bytes.
        shared.aliases.place(key, (pointer.as_ptr() as u64, length));
        if let Some(release) = release.of(pointer, length) {
            pages.release_to(release);
        }
        // What lay in the old pages goes with them: the images swapped out, and the texel views.
        // A detached buffer lay in none: what took its stand-in goes with the new pages.
        let keeper = match &old {
            Lying::In(old) => Arc::clone(old),
            Lying::Detached(_) => Arc::clone(&pages),
        };
        let rebound = !swaps.is_empty() || !texels.is_empty();
        // The textures that presented stand-ins while the bytes were detached present their own
        // images again (`texture/stand_in.rs`): those bound to the pages the ones made over the
        // new pages, swapped in here; those in memory of their own the images they kept. What
        // the stand-ins lent them waits for the command buffers that may read it through the
        // heap sets (`heap_sets::retire`), not for the new pages.
        let mut stood_in = Vec::new();
        for swap in swaps {
            let objects = swap.apply();
            match objects.of_stand_in() {
                true => stood_in.push(objects),
                false => keeper.park(objects),
            }
        }
        if let Some(settled) = settled {
            let images = TextureImages::of(settled.images());
            stood_in.extend(images.present_own());
            debug_assert!(
                !images.standing_in(),
                "every image over the bytes presents its own again"
            );
            drop(settled);
            self.forget_stale(cell);
        }
        for views in &texels {
            keeper.park(views.park());
        }
        drop(keeper);
        // The device's heap sets view the remade images and views of the textures over the
        // bytes, and no other texture's elements change (`heap_sets::renew_over`).
        if rebound || !stood_in.is_empty() {
            super::heap_sets::renew_over(self, key);
        }
        if !stood_in.is_empty() {
            super::heap_sets::retire(self, Box::new(stood_in));
        }
        // The old pages go now where no command buffer holds them, their deallocator with them.
        drop(old);
        // The command buffers committed before the rebind and not yet first submitted whose
        // encoders named the buffer are encoded again over the new pages, in this call, before
        // their first submission (`command_buffer/reencode.rs`), as Metal resolves them then.
        for buffer in &again {
            buffer.reencode();
        }
        drop(again);
        Ok(())
    }

    /// The `VkBuffer` of a no-copy buffer of `length` bytes at `window.offset` in its window,
    /// made for host allocations, whose bytes start [`buffer_base`] bytes into it. The memory
    /// alignment is a buffer's whatever its size, so the buffer is made first for bytes at 0,
    /// and again, at its own size, where its alignment misses the bytes.
    ///
    /// It holds the rest of the window from its bytes, which a copy out of the buffer reads past
    /// its length as plain Metal does (`Buffer::copy_reach`); where the device's memory
    /// requirements of that buffer run past the window, it holds its padded length ([`based_size`])
    /// instead.
    fn host_buffer(&self, window: &ImportWindow, length: u64) -> Result<HostBuffer, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let make_sized = |base: u64, size: u64| {
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let info = shared.sharing().buffer(
                vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage(shared.device_address))
                    .push_next(&mut external),
            );
            // SAFETY: the device is live and `info` and its chain valid.
            let raw = unsafe { raw_device.create_buffer(&info, None) }.map_err(vulkan)?;
            shared.counts.buffer_made(info.usage);
            // SAFETY: `raw` is a live buffer of this device.
            let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
            Ok::<_, Error>(HostBuffer {
                raw,
                base,
                size,
                requirements,
            })
        };
        let make = |base: u64| {
            let padded = based_size(base, length).ok_or_else(not_importable)?;
            // The window from `base` bytes below the buffer's bytes to its end: where it binds
            // (`binds_in`), the room it has.
            let whole = window.size - window.offset + base;
            if whole > padded {
                let buffer = make_sized(base, whole)?;
                if buffer.requirements.size <= whole {
                    return Ok(buffer);
                }
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, buffer.raw) };
            }
            make_sized(base, padded)
        };
        let first = make(0)?;
        let base = buffer_base(window.offset, first.requirements.alignment);
        if base == 0 {
            return Ok(first);
        }
        // SAFETY: the buffer is unused.
        unsafe { release::object(raw_device, first.raw) };
        let buffer = make(base)?;
        if buffer_base(window.offset, buffer.requirements.alignment) != base {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, buffer.raw) };
            return Err(not_importable());
        }
        Ok(buffer)
    }

    /// The pointer's import of `host`'s window ([`Device::import_pointer`]), where the device's
    /// import alignment covers it, exactly as ever; and where the device refuses it ([`refused`]:
    /// no window, a buffer that does not fit it, no memory type, a driver refusal) and the caller
    /// gave a dma-buf source the device can use, the source, uncalled (`Err`), for the caller to
    /// call with no lock of the runtime's held (`Device::import_from_source`, `dma_buf.rs`). A
    /// refusal without such a source is the pointer's.
    ///
    /// # Safety
    ///
    /// The window's pages are host memory of this process that stays mapped until the
    /// placement is freed (`HostMemory::new`).
    unsafe fn import_by_pointer(
        &self,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        host: HostWindow,
        class: MemoryClass,
    ) -> Result<Result<(HostBuffer, Placement), Source>, Error> {
        let HostWindow {
            window,
            by_pointer,
            source,
        } = host;
        let refusal = match by_pointer {
            // SAFETY: the caller's.
            true => match unsafe { self.import_pointer(importer, pointer, length, &window, class) }
            {
                Ok(imported) => return Ok(Ok(imported)),
                Err(error) if refused(&error) => error,
                Err(error) => return Err(error),
            },
            false => not_importable(),
        };
        source.map(Err).ok_or(refusal)
    }

    /// The `VkBuffer` of a no-copy buffer of `length` bytes at `pointer`, made for host
    /// allocations and bound in the import of the window's pages at their pointer
    /// (`VK_EXT_external_memory_host`), `buffer.base` bytes below the window's offset, as memory
    /// of `class`: the buffer and the placement. Released again on failure.
    ///
    /// # Safety
    ///
    /// The window's `size` bytes are host memory of this process that stays mapped until the
    /// placement is freed.
    unsafe fn import_pointer(
        &self,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        window: &ImportWindow,
        class: MemoryClass,
    ) -> Result<(HostBuffer, Placement), Error> {
        let buffer = self.host_buffer(window, length)?;
        // The window starts `window.offset` bytes before the pointer, in the pages it touches.
        let start = pointer.as_ptr().wrapping_sub(window.offset as usize);
        // SAFETY: the caller's, and the buffer is a new one of this device.
        match unsafe { self.import_backing(importer, &buffer, start, window, class) } {
            Ok(placement) => Ok((buffer, placement)),
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(self.shared().vk(), buffer.raw) };
                Err(error)
            }
        }
    }

    /// The pages of a released import of the window's pages at `pointer`, made again there as
    /// `again` made it (`host_residency.rs`), on a driver whose pointer imports follow their
    /// address (`DriverFact::PointerImportsFollowTheirAddress`): a new `VkBuffer` made and bound
    /// as the first was, over an import of the pages mapped at the pointer now, which the kept
    /// import would read too. Refused as the first import may be, typed as the import answers
    /// (out of memory among them), or with the device lost.
    #[cfg(unix)]
    pub(super) fn import_pointer_again(
        &self,
        pointer: NonNull<u8>,
        again: &Again,
    ) -> Result<Pages, Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let importer = shared
            .host_import
            .as_ref()
            .map_err(|_| internal("a pointer import made again on a device that imports none"))?;
        let window = &again.window;
        // SAFETY: the caller's pages stay mapped at the pointer while the buffer lives
        // (`HostMemory::new`), and the buffer outlives the placement: the import holds it, and a
        // command buffer that reads it holds the buffer too.
        let (buffer, placement) =
            unsafe { self.import_pointer(importer, pointer, again.length, window, again.class) }?;
        Ok(self.pages_of(
            buffer.raw,
            placement,
            window.offset,
            buffer.base,
            buffer.size - buffer.base,
        ))
    }

    /// No dma-buf is imported on this system: there is no source to call.
    ///
    /// # Safety
    ///
    /// None: it does nothing.
    #[cfg(not(unix))]
    unsafe fn import_from_source(
        &self,
        _call: &'static str,
        _importer: &Importer,
        _pointer: NonNull<u8>,
        _length: u64,
        _window: &ImportWindow,
        source: Source,
        _class: MemoryClass,
        _keep: bool,
    ) -> Result<(HostBuffer, Placement, Option<std::convert::Infallible>), Error> {
        match source {}
    }

    /// Release `buffer` and its `placement`, an import made for a buffer or rebind that is not
    /// made after all: the `VkBuffer` first, once its sparse binding completed.
    fn discard_import(&self, buffer: HostBuffer, placement: Placement) {
        let shared = self.shared();
        placement.settle();
        // SAFETY: the buffer is unused, and its binding completed.
        unsafe { release::object(shared.vk(), buffer.raw) };
        lock(&shared.memory).free(shared.vk(), &placement);
    }

    /// The Metal addresses of a no-copy buffer of `length` bytes `window.offset` bytes into
    /// `window`, placed as `address` asks: the window's pages where the program names no address,
    /// so that the buffer's address keeps its pointer's offset in its page.
    fn place_window(
        &self,
        address: Place,
        window: &ImportWindow,
        length: u64,
    ) -> Result<(GpuAddress, Held), Error> {
        let mut addresses = lock(&self.shared().addresses);
        match address {
            Place::At(AddressPlacement::Any | AddressPlacement::Outside) => addresses
                .place(address, window.size)
                .map(|(range, held)| (GpuAddress(range.0 + window.offset), held)),
            _ => addresses.place(address, length),
        }
    }

    /// Where a buffer of `length` bytes holding `contents`, at `placement`, goes to share
    /// `other`'s range (`BufferDescriptor::range_of`), for `call`: refused as
    /// [`Device::new_buffer_with_descriptor`] says, and where it is to go `in_pool`.
    fn shared_place(
        &self,
        call: &'static str,
        other: &Buffer,
        contents: &BufferContents<'_>,
        length: u64,
        placement: AddressPlacement,
        in_pool: bool,
    ) -> Result<Place, Error> {
        if !other.device().same(self) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let unshared = || {
            let address = other.exposed_address(ReadApart::SecondBuffer);
            let held = AddressRange::of(address.0, other.length());
            Error::PlacementConflict(PlacementConflict::Unshared { held })
        };
        let (address, held, first) = other.sharable_range().ok_or_else(unshared)?;
        let over = match contents {
            BufferContents::NoCopy { memory, .. } => memory.pointer() == first,
            BufferContents::Zero | BufferContents::Copy(_) => false,
        };
        let placed = matches!(placement, AddressPlacement::Any)
            || placement == AddressPlacement::At(address);
        if !over || length > other.length() || !placed || in_pool {
            return Err(unshared());
        }
        Ok(Place::Shared { address, held })
    }

    /// Import the window's pages at `start` and bind `buffer` in them, `buffer.base` bytes below
    /// the window's offset, as memory of `class`: the placement. Released again on failure but
    /// the buffer.
    ///
    /// # Safety
    ///
    /// The window's `size` bytes at `start` are host memory of this process that stays mapped
    /// until the placement is freed; `buffer` is an unbound buffer of this device made for host
    /// allocations.
    unsafe fn import_backing(
        &self,
        importer: &Importer,
        buffer: &HostBuffer,
        start: *mut u8,
        window: &ImportWindow,
        class: MemoryClass,
    ) -> Result<Placement, Error> {
        let raw_device = self.shared().vk();
        let requirements = &buffer.requirements;
        let at = binds_in(window, buffer.base, buffer.size, requirements)
            .map_err(|_| not_importable())?;
        let start = NonNull::new(start).ok_or_else(not_importable)?;
        // SAFETY: `start` is host memory aligned to the import alignment (the window's).
        let types = unsafe { importer.memory_types(raw_device, start) }.map_err(import_refused)?;
        let allowed = types & requirements.memory_type_bits;
        // An allocation past the device's, refused before the driver sees it.
        let largest = self
            .compile_device()
            .allocation()
            .max_memory_allocation_size;
        if window.size > largest {
            return Err(Error::OutOfMemory(MemoryKind::Device));
        }
        // SAFETY: the caller's contract keeps the window mapped until the placement is freed,
        // and the buffer is unbound.
        unsafe { self.import_now(buffer.raw, start, window.size, allowed, class, at) }
    }

    /// Import `size` bytes of host pages at `start` as memory of `class` among the types
    /// `allowed`, and bind `raw` in them `at` bytes in: the placement. Released again on failure.
    ///
    /// # Safety
    ///
    /// The `size` bytes at `start` are host memory of this process that stays mapped until the
    /// placement is freed; `raw` is an unbound buffer of this device made for host allocations,
    /// whose requirements the pages meet from `at`.
    unsafe fn import_now(
        &self,
        raw: vk::Buffer,
        start: NonNull<u8>,
        size: u64,
        allowed: u32,
        class: MemoryClass,
        at: u64,
    ) -> Result<Placement, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        // SAFETY: the caller's contract keeps the pages mapped until the placement is freed.
        let placement =
            unsafe { lock(&shared.memory).import_host(raw_device, start, size, allowed, class) }
                .map_err(|failure| match failure {
                    ImportFailure::NoMemoryType => not_importable(),
                    ImportFailure::Vulkan(error) => import_refused(error),
                })?;
        // SAFETY: the placement is the pages' memory, the buffer's size and requirements of it
        // available from `at` (the caller's), and the buffer is unbound.
        let bound =
            unsafe { raw_device.bind_buffer_memory(raw, placement.memory, placement.offset + at) };
        match bound {
            Ok(()) => {
                shared.counts.host_pointer_import_made(size);
                Ok(placement)
            }
            Err(error) => {
                lock(&shared.memory).free(raw_device, &placement);
                Err(vulkan(error))
            }
        }
    }
}

/// The host pages a no-copy buffer (or a heap's own buffer) of some length imports
/// ([`host_window`]).
struct HostWindow {
    /// The pages: the range rounded out to the host page.
    window: ImportWindow,
    /// Whether the device's import alignment covers them, so that their pointer is imported
    /// first (`VK_EXT_external_memory_host`).
    by_pointer: bool,
    /// The caller's dma-buf source, where it gave one and the device imports dma-bufs.
    source: Option<Source>,
}

/// The host pages a no-copy object of `length` bytes at `pointer` imports on the device of
/// `importer`, for `call`: the pages the range touches ([`import_window`]), imported by their
/// pointer where the device's import alignment covers them, else only as a dma-buf where the
/// caller gave a source the device can use. A source the device cannot use is dropped uncalled.
/// Refused: memory past the address space's end (`Nil`, as Metal makes no buffer over it); a
/// range the import alignment cannot cover, with no usable source (`Unsupported`, `HostMemory`).
fn host_window(
    call: &'static str,
    importer: &Importer,
    pointer: NonNull<u8>,
    length: u64,
    source: Option<Source>,
) -> Result<HostWindow, Error> {
    let source = source.filter(|_| importer.imports_dma_bufs());
    let at = pointer.as_ptr() as u64;
    let window = import_window(at, length, importer.page, 1)
        .map_err(|_| nil(call, Problem::BufferLength { length }))?;
    let by_pointer = import_window(at, length, importer.page, importer.alignment).is_ok();
    if !by_pointer && source.is_none() {
        return Err(not_importable());
    }
    Ok(HostWindow {
        window,
        by_pointer,
        source,
    })
}

/// A no-copy buffer's unbound `VkBuffer` (`Device::host_buffer`): `size` bytes, its bytes from
/// `base`, with its memory requirements.
pub(super) struct HostBuffer {
    pub(super) raw: vk::Buffer,
    pub(super) base: u64,
    pub(super) size: u64,
    pub(super) requirements: vk::MemoryRequirements,
}
