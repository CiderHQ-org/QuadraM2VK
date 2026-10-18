//! Device tests of pinned GPU addresses (`setPinnedGPUAddress:`, a runtime extension) on the device
//! the cfg selects, under the validation layer: buffers (zero, copied and no-copy) and heaps made
//! at a caller's address answer it, aligned or not; a kernel walks a pointer chain from a pinned
//! address held in its argument buffer, as from one the runtime picked; and every pin Metal
//! answered nil for (bytes another resource holds, its guard pages, outside the space) is refused
//! typed, while a released pin's range takes pins again. The runtime's own buffers, and a caller's
//! helper buffer placed `AddressPlacement::Outside`, lie outside the space: a program's buffer made
//! while they live has the address it would have without them.

use super::host_memory::{gpu_read, not_importable, on_device, run, Mapping, GUEST_PAGE};
use super::sparse_textures::lacks_sparse;
use super::{bound_buffers, encode_case, pipeline_state, public_case, read_buffer, CaseResources};
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan_runtime::{
    AddressPlacement, BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBuffer,
    CommandQueue, Device, Error, GpuAddress, HeapDescriptor, HeapType, HostMemory, IdPlacement,
    Origin, PlacementConflict, RangeToken, ResourceOptions, SharedEvent, Size, Texture,
    TextureDescriptor,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Metal's page, the granule of the address space.
const PAGE: u64 = 16 << 10;

/// An address high in the space, `[2^32, 2^47)`, far from the ranges the runtime picks first.
const HIGH: u64 = 0x7000_0000_0000;

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0.
const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

fn at(address: u64) -> AddressPlacement {
    AddressPlacement::At(GpuAddress(address))
}

/// A buffer of `length` zero bytes pinned at `address`.
fn pinned(device: &Device, address: u64, length: u64) -> Result<Buffer, Error> {
    device.new_buffer_with_descriptor(
        BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero)
            .with_placement(at(address)),
    )
}

fn held(error: Error) -> bool {
    matches!(
        error,
        Error::PlacementConflict(PlacementConflict::Held { .. })
    )
}

fn outside(error: Error) -> bool {
    matches!(
        error,
        Error::PlacementConflict(PlacementConflict::OutsideSpace { .. })
    )
}

fn off_granule(error: Error) -> bool {
    matches!(
        error,
        Error::PlacementConflict(PlacementConflict::OffGranule { .. })
    )
}

/// Buffers and heaps made at a caller's address answer it, as Metal's do on an M3 Max and an M4:
/// zero, copied and no-copy buffers, aligned and not, and a placement heap whose buffer at an
/// offset has the heap's address plus the offset.
#[test]
fn pinned_buffers_and_heaps_answer_their_addresses() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        for (address, length) in [(HIGH, PAGE), (HIGH + 4 * PAGE + 0x100, 100)] {
            let buffer = pinned(device, address, length).unwrap();
            assert_eq!(buffer.gpu_address(), GpuAddress(address));
            assert_eq!(buffer.length(), length);
        }
        let bytes: Vec<u8> = (0..=255).collect();
        let copied = device
            .new_buffer_with_descriptor(
                BufferDescriptor::new(
                    256,
                    ResourceOptions::private(),
                    BufferContents::Copy(&bytes),
                )
                .with_placement(at(HIGH + 16 * PAGE)),
            )
            .unwrap();
        assert_eq!(copied.gpu_address(), GpuAddress(HIGH + 16 * PAGE));
        assert_eq!(gpu_read(&queue, &copied, 0..256), bytes);
        let heap = device
            .new_heap(
                &HeapDescriptor::new(HeapType::Placement, 4 * PAGE, ResourceOptions::shared())
                    .with_placement(at(HIGH + 32 * PAGE)),
            )
            .unwrap();
        assert_eq!(heap.gpu_address(), GpuAddress(HIGH + 32 * PAGE));
        let placed = heap
            .new_buffer_at_offset(PAGE, ResourceOptions::shared(), PAGE)
            .unwrap();
        assert_eq!(placed.gpu_address(), GpuAddress(HIGH + 33 * PAGE));
        // A pin inside the heap's range is refused, as Metal answered nil.
        assert!(held(pinned(device, HIGH + 34 * PAGE, PAGE).unwrap_err()));
        let memory = Mapping::anonymous(GUEST_PAGE);
        memory.fill(0x5a);
        // SAFETY: the mapping outlives the buffer, dropped first below.
        let no_copy = device.new_buffer_with_descriptor(
            BufferDescriptor::new(
                GUEST_PAGE as u64,
                ResourceOptions::shared(),
                BufferContents::NoCopy {
                    memory: unsafe { HostMemory::new(memory.pointer()) },
                    deallocator: None,
                    keep_alive: None,
                },
            )
            .with_placement(at(HIGH + 64 * PAGE)),
        );
        match no_copy {
            Ok(buffer) => {
                assert_eq!(buffer.gpu_address(), GpuAddress(HIGH + 64 * PAGE));
                assert_eq!(buffer.contents(), Some(memory.pointer()));
                eprintln!("runtime pinned no-copy buffer: made");
            }
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                eprintln!("runtime pinned no-copy buffer: refused (not importable)");
            }
        }
    });
}

/// Every pin Metal answered nil for is refused, typed, and changes nothing: the address of a live
/// buffer, a range overlapping one, the span of a pinned range (its guard page and the page that
/// rounds it to 32 KiB), a page off a 32-KiB granule, and addresses outside `[2^32, 2^47)`. A
/// released pin's span takes pins again, a 3-page pin leaves the next granule free (the consumer's
/// ask (c)), and a range the runtime picks never lands in a pinned one.
#[test]
fn pins_metal_refuses_are_refused_typed() {
    on_device(|device| {
        let any = device
            .new_buffer(4 * PAGE, ResourceOptions::shared())
            .unwrap();
        let start = any.gpu_address().0;
        assert!(held(pinned(device, start, PAGE).unwrap_err()));
        let inside = pinned(device, start + PAGE, PAGE).unwrap_err();
        assert!(
            if (start + PAGE).is_multiple_of(2 * PAGE) {
                held(inside)
            } else {
                off_granule(inside)
            },
            "inside a live buffer"
        );
        let one = pinned(device, HIGH, PAGE).unwrap();
        assert!(
            held(pinned(device, HIGH, PAGE).unwrap_err()),
            "the same address"
        );
        assert!(
            off_granule(pinned(device, HIGH + PAGE, PAGE).unwrap_err()),
            "its guard page, off a granule"
        );
        let after = pinned(device, HIGH + 2 * PAGE, PAGE).unwrap();
        let long = pinned(device, HIGH + 8 * PAGE, 2 * PAGE).unwrap();
        assert!(
            held(pinned(device, HIGH + 10 * PAGE, PAGE).unwrap_err()),
            "its guard page"
        );
        assert!(
            off_granule(pinned(device, HIGH + 11 * PAGE, PAGE).unwrap_err()),
            "the page rounding its span"
        );
        let _beyond = pinned(device, HIGH + 12 * PAGE, PAGE).unwrap();
        for address in [0x4000, 0xffff_c000, 1 << 47, (1 << 47) - PAGE, 1 << 56] {
            assert!(
                outside(pinned(device, address, PAGE).unwrap_err()),
                "{address:#x}"
            );
        }
        for _ in 0..16 {
            let picked = device
                .new_buffer(PAGE, ResourceOptions::shared())
                .unwrap()
                .gpu_address()
                .0;
            assert!(!(HIGH..HIGH + 16 * PAGE).contains(&picked), "{picked:#x}");
        }
        drop((one, after, long));
        // Three pages hold four: the next granule takes a pin (two guard pages refused it).
        let again = pinned(device, HIGH, 3 * PAGE).unwrap();
        assert_eq!(again.gpu_address(), GpuAddress(HIGH));
        let next = pinned(device, HIGH + 4 * PAGE, PAGE).unwrap();
        drop(next);
        drop(any);
        let granule = start.next_multiple_of(2 * PAGE);
        assert!(pinned(device, granule, PAGE).is_ok(), "a released range");
    });
}

/// The pointer walker's argument-buffer buffer made at a pinned address, aligned and a page plus
/// 256 bytes: the kernel loads that address from its argument buffer and walks the chain through
/// the resident table, its output Metal's.
#[test]
fn a_kernel_walks_a_chain_from_a_pinned_address() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = walker.compute.output
        else {
            panic!("{POINTER_WALKER}: a buffer output");
        };
        let walk = pipeline_state(device, &walker).unwrap();
        for address in [HIGH + 128 * PAGE, HIGH + 256 * PAGE + 0x100] {
            let mut resources = CaseResources::new(device, &walker).unwrap();
            let [(&field, member)] = resources.argument_buffers.iter().collect::<Vec<_>>()[..]
            else {
                panic!("{POINTER_WALKER}: one argument-buffer buffer");
            };
            let bytes = read_buffer(member, 0, member.length()).unwrap();
            let pinned_member = device
                .new_buffer_with_descriptor(
                    BufferDescriptor::new(
                        bytes.len() as u64,
                        ResourceOptions::shared(),
                        BufferContents::Copy(&bytes),
                    )
                    .with_placement(at(address)),
                )
                .unwrap();
            assert_eq!(pinned_member.gpu_address(), GpuAddress(address));
            resources
                .argument_buffers
                .insert(field, pinned_member.clone());
            let buffers = bound_buffers(device, &walker.compute, &resources).unwrap();
            assert_eq!(
                read_buffer(&buffers[&field.0], u64::from(field.1), 8).unwrap(),
                address.to_le_bytes(),
                "{address:#x}: the argument buffer holds the pinned address"
            );
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(pinned_member.as_resource(), ResourceUsage::READ)
                .unwrap();
            encode_case(&mut encoder, &walk, &walker.compute, &buffers, &resources).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            let output = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert_eq!(output, walker.metal_output, "{address:#x}");
        }
    });
}

/// A buffer the program makes while the runtime's own buffers live (an overlapping copy's
/// scratch, held until its command buffer completes) has the address it has without them, as
/// Metal's own allocations never move a program's; so does one made beside a helper buffer
/// placed `AddressPlacement::Outside`, whose address lies in `[2^47, 2^48)`.
#[test]
fn the_runtime_s_own_buffers_take_no_metal_address() {
    on_device(|device| {
        let shared = ResourceOptions::shared();
        let queue = device.new_command_queue().unwrap();
        let moved = device.new_buffer(4 * PAGE, shared).unwrap();
        // Where the program's next buffer goes with nothing else made.
        let probe = device.new_buffer(PAGE, shared).unwrap();
        let next = probe.gpu_address();
        drop(probe);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&moved, 0, &moved, 16, 1024)
            .unwrap();
        blit.end_encoding().unwrap();
        let during = device.new_buffer(PAGE, shared).unwrap();
        assert_eq!(during.gpu_address(), next, "beside an in-flight scratch");
        run(&command_buffer);
        drop(during);
        let helper = device
            .new_buffer_with_descriptor(
                BufferDescriptor::new(PAGE, shared, BufferContents::Zero)
                    .with_placement(AddressPlacement::Outside),
            )
            .unwrap();
        let address = helper.gpu_address().0;
        assert!((1 << 47..1 << 48).contains(&address), "{address:#x}");
        let beside = device.new_buffer(PAGE, shared).unwrap();
        assert_eq!(beside.gpu_address(), next, "beside a helper buffer");
        drop((helper, beside, moved));
    });
}

/// What a range token saw: whether it was dropped, and whether a pin at its range made from its
/// drop was made.
#[derive(Default)]
struct Seen {
    dropped: AtomicBool,
    remade: Mutex<Option<bool>>,
}

impl Seen {
    fn dropped(&self) -> bool {
        self.dropped.load(Ordering::SeqCst)
    }

    fn remade(&self) -> Option<bool> {
        *self.remade.lock().unwrap()
    }
}

/// A range token that pins `length` bytes at `address` as it drops, noting it in `seen`.
struct Token {
    device: Device,
    address: u64,
    length: u64,
    seen: Arc<Seen>,
}

impl Drop for Token {
    fn drop(&mut self) {
        let made = pinned(&self.device, self.address, self.length).is_ok();
        *self.seen.remade.lock().unwrap() = Some(made);
        self.seen.dropped.store(true, Ordering::SeqCst);
    }
}

/// A token for `length` bytes at `address` on `device`, and what it sees.
fn token(device: &Device, address: u64, length: u64) -> (RangeToken, Arc<Seen>) {
    let seen = Arc::new(Seen::default());
    let token = Token {
        device: device.clone(),
        address,
        length,
        seen: Arc::clone(&seen),
    };
    (Box::new(token), seen)
}

/// A buffer of `length` zero bytes pinned at `address`, keeping `token`.
fn pinned_with(
    device: &Device,
    address: u64,
    length: u64,
    token: RangeToken,
) -> Result<Buffer, Error> {
    device.new_buffer_with_descriptor(
        BufferDescriptor::new(length, ResourceOptions::shared(), BufferContents::Zero)
            .with_placement(at(address))
            .with_range_token(token),
    )
}

/// What a completed handler saw: whether the token had dropped, and whether a pin at its range
/// was made.
type InHandler = Arc<Mutex<Option<(bool, bool)>>>;

/// A command buffer filling `buffer` once `event` reaches 1, committed, with a completed handler
/// that notes whether `seen`'s token had dropped and whether a pin of `length` bytes at
/// `address` (the pinned range: the buffer's, or its heap's) was made there.
fn held_back(
    queue: &CommandQueue,
    event: &SharedEvent,
    buffer: &Buffer,
    (address, length): (u64, u64),
    seen: &Arc<Seen>,
) -> (CommandBuffer, InHandler) {
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(event.into(), 1)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..buffer.length(), 0x5a).unwrap();
    blit.end_encoding().unwrap();
    let in_handler = Arc::new(Mutex::new(None));
    let (noted, seen, device) = (
        Arc::clone(&in_handler),
        Arc::clone(seen),
        queue.device().clone(),
    );
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            let made = pinned(&device, address, length).is_ok();
            *noted.lock().unwrap() = Some((seen.dropped(), made));
        }))
        .unwrap();
    command_buffer.commit().unwrap();
    (command_buffer, in_handler)
}

/// A pinned buffer's and heap's range token (`RangeToken`) drops once the range is free, as Metal
/// frees a released resource's range (the consumer's rows, an Apple M3 Max): dropped by the
/// program, at once, and a pin at the range from the token's drop is made; released while a
/// committed command buffer uses it (1, 2 and 3 pages, and a heap whose placed buffer it fills),
/// still held in the command buffer's completed handler and dropped before its wait returns; held
/// by an uncommitted command buffer, dropped with it; a refused creation's, before the call
/// returns; a heap's, once the heap and its placed buffer are gone.
#[test]
fn a_range_token_drops_once_its_range_is_free() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let shared = ResourceOptions::shared();
        let (first, seen) = token(device, HIGH, PAGE);
        let buffer = pinned_with(device, HIGH, PAGE, first).unwrap();
        assert!(!seen.dropped());
        drop(buffer);
        assert!(seen.dropped(), "dropped with the buffer");
        assert_eq!(seen.remade(), Some(true), "its range free in its drop");
        for pages in [1, 2, 3] {
            let (address, length) = (HIGH + pages * 64 * PAGE, pages * PAGE);
            let (kept, seen) = token(device, address, length);
            let buffer = pinned_with(device, address, length, kept).unwrap();
            let event = device.new_shared_event().unwrap();
            let (command_buffer, in_handler) =
                held_back(&queue, &event, &buffer, (address, length), &seen);
            drop(buffer);
            assert!(!seen.dropped(), "{pages} pages: in flight");
            assert!(held(pinned(device, address, length).unwrap_err()));
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                *in_handler.lock().unwrap(),
                Some((false, false)),
                "{pages} pages: held in the completed handler"
            );
            assert!(
                seen.dropped(),
                "{pages} pages: dropped before the wait returns"
            );
            assert_eq!(seen.remade(), Some(true));
            pinned(device, address, length).unwrap();
        }
        // A heap whose placed buffer a command buffer fills.
        let heap_at = HIGH + 1024 * PAGE;
        let (kept, seen) = token(device, heap_at, 4 * PAGE);
        let heap = device
            .new_heap_with_range_token(
                &HeapDescriptor::new(HeapType::Placement, 4 * PAGE, shared)
                    .with_placement(at(heap_at)),
                kept,
            )
            .unwrap();
        let placed = heap.new_buffer_at_offset(PAGE, shared, PAGE).unwrap();
        let event = device.new_shared_event().unwrap();
        let (command_buffer, in_handler) =
            held_back(&queue, &event, &placed, (heap_at, 4 * PAGE), &seen);
        drop((heap, placed));
        assert!(!seen.dropped(), "a heap in flight");
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(*in_handler.lock().unwrap(), Some((false, false)), "a heap");
        assert!(seen.dropped() && seen.remade() == Some(true), "a heap");
        // An uncommitted command buffer's hold.
        let address = HIGH + 2048 * PAGE;
        let (kept, seen) = token(device, address, PAGE);
        let buffer = pinned_with(device, address, PAGE, kept).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..PAGE, 1).unwrap();
        blit.end_encoding().unwrap();
        drop(buffer);
        assert!(!seen.dropped(), "an uncommitted command buffer holds it");
        drop(command_buffer);
        assert!(seen.dropped() && seen.remade() == Some(true), "uncommitted");
        // A refused creation's token, before the call returns; the live buffer keeps the range.
        let live = pinned(device, address, PAGE).unwrap();
        let (refused, seen) = token(device, address, PAGE);
        assert!(held(
            pinned_with(device, address, PAGE, refused).unwrap_err()
        ));
        assert!(seen.dropped(), "a refused creation's");
        assert_eq!(seen.remade(), Some(false));
        // A heap's token, once the heap and its placed buffer are gone.
        let (kept, seen) = token(device, heap_at, 4 * PAGE);
        let heap = device
            .new_heap_with_range_token(
                &HeapDescriptor::new(HeapType::Placement, 4 * PAGE, shared)
                    .with_placement(at(heap_at)),
                kept,
            )
            .unwrap();
        let placed = heap.new_buffer_at_offset(PAGE, shared, 0).unwrap();
        drop(heap);
        assert!(!seen.dropped(), "its placed buffer holds the heap");
        drop(placed);
        assert!(seen.dropped() && seen.remade() == Some(true), "a heap's");
        drop(live);
    });
}

/// A range token for a texture or a sparse heap, which holds no range of the runtime's: it notes
/// its drop in `seen`, and whether a buffer made from its drop was made (no runtime lock held).
struct Freed {
    device: Device,
    seen: Arc<Seen>,
}

impl Drop for Freed {
    fn drop(&mut self) {
        let made = self
            .device
            .new_buffer(PAGE, ResourceOptions::shared())
            .is_ok();
        *self.seen.remade.lock().unwrap() = Some(made);
        self.seen.dropped.store(true, Ordering::SeqCst);
    }
}

/// A [`Freed`] token on `device`, and what it sees.
fn freed(device: &Device) -> (RangeToken, Arc<Seen>) {
    let seen = Arc::new(Seen::default());
    let token = Freed {
        device: device.clone(),
        seen: Arc::clone(&seen),
    };
    (Box::new(token), seen)
}

/// A 64 x 64 `Private` `R32Float` texture's descriptor: one sparse tile.
fn texture_descriptor() -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::R32Float, 64, 64, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    descriptor
}

/// A command buffer that copies 4 x 4 texels of `texture` into a new buffer once `event`
/// reaches 1, with a completed handler that notes whether `seen`'s token had dropped; not
/// committed.
fn reading(
    queue: &CommandQueue,
    event: &SharedEvent,
    texture: &Texture,
    seen: &Arc<Seen>,
) -> (CommandBuffer, Arc<Mutex<Option<bool>>>) {
    let into = queue
        .device()
        .new_buffer(64, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(event.into(), 1)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(4, 4, 1),
        &into,
        0,
        16,
        64,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    let in_handler = Arc::new(Mutex::new(None));
    let (noted, seen) = (Arc::clone(&in_handler), Arc::clone(seen));
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            *noted.lock().unwrap() = Some(seen.dropped());
        }))
        .unwrap();
    (command_buffer, in_handler)
}

/// Run `command_buffer` (from [`reading`]) once a buffer was made meanwhile, its texture
/// released: `seen`'s token is held while it runs and in its completed handler, and dropped
/// before its wait returns, a buffer made from its drop made.
fn runs_then_frees(
    what: &str,
    (command_buffer, in_handler): (CommandBuffer, Arc<Mutex<Option<bool>>>),
    event: &SharedEvent,
    seen: &Seen,
) {
    let device = command_buffer.device().clone();
    let meanwhile = device.new_buffer(PAGE, ResourceOptions::shared()).unwrap();
    assert!(!seen.dropped(), "{what}: in flight");
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        *in_handler.lock().unwrap(),
        Some(false),
        "{what}: held in the completed handler"
    );
    assert!(seen.dropped(), "{what}: dropped before the wait returns");
    assert_eq!(
        seen.remade(),
        Some(true),
        "{what}: a buffer made in its drop"
    );
    drop(meanwhile);
}

/// The range token of a texture with an image of its own (`newTextureWithDescriptor:`,
/// `newSharedTextureWithDescriptor:`) and of a sparse heap drops as the runtime frees the
/// resource, as Metal frees its range: dropped by the program, at once; released right after the
/// commit of a command buffer that reads it, or after encoding and before the commit, a buffer
/// made meanwhile, still held in the completed handler and dropped before the wait returns; held
/// by an uncommitted command buffer, a view, and a texture opened from a shared texture's handle
/// until they go; a texture buffer's with its texels; a refused creation's before the call
/// returns; a sparse heap's once the heap and its texture are gone.
#[test]
fn a_texture_or_sparse_heap_token_drops_once_it_is_freed() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let descriptor = texture_descriptor();
        let (kept, seen) = freed(device);
        let texture = device
            .new_texture_with_range_token(&descriptor, kept)
            .unwrap();
        assert!(!seen.dropped());
        drop(texture);
        assert!(seen.dropped(), "dropped with the texture");
        assert_eq!(seen.remade(), Some(true), "a buffer made in its drop");
        for shareable in [false, true] {
            for released in ["after the commit", "before the commit"] {
                let what = format!("shareable {shareable}, released {released}");
                let (kept, seen) = freed(device);
                let texture = match shareable {
                    false => device.new_texture_with_range_token(&descriptor, kept),
                    true => device.new_shared_texture_with_range_token(&descriptor, kept),
                }
                .unwrap();
                let event = device.new_shared_event().unwrap();
                let (command_buffer, in_handler) = reading(&queue, &event, &texture, &seen);
                if released == "before the commit" {
                    drop(texture);
                    command_buffer.commit().unwrap();
                } else {
                    command_buffer.commit().unwrap();
                    drop(texture);
                }
                runs_then_frees(&what, (command_buffer, in_handler), &event, &seen);
            }
        }
        // An uncommitted command buffer's hold.
        let (kept, seen) = freed(device);
        let texture = device
            .new_texture_with_range_token(&descriptor, kept)
            .unwrap();
        let event = device.new_shared_event().unwrap();
        let (command_buffer, _) = reading(&queue, &event, &texture, &seen);
        drop(texture);
        assert!(!seen.dropped(), "an uncommitted command buffer holds it");
        drop(command_buffer);
        assert!(seen.dropped() && seen.remade() == Some(true), "uncommitted");
        // A view's, and a texture's opened from a shared texture's handle.
        let (kept, seen) = freed(device);
        let texture = device
            .new_texture_with_range_token(&descriptor, kept)
            .unwrap();
        let view = texture.new_texture_view(PixelFormat::R32Float).unwrap();
        drop(texture);
        assert!(!seen.dropped(), "a view holds its texture's image");
        drop(view);
        assert!(seen.dropped(), "a view's");
        let (kept, seen) = freed(device);
        let texture = device
            .new_shared_texture_with_range_token(&descriptor, kept)
            .unwrap();
        let handle = texture.new_shared_texture_handle().unwrap();
        let opened = device
            .new_shared_texture_with_handle(&handle, IdPlacement::Any)
            .unwrap();
        drop((texture, handle));
        assert!(!seen.dropped(), "an opened texture holds the image");
        drop(opened);
        assert!(seen.dropped(), "an opened texture's");
        // A texture buffer's, with its texels.
        let (kept, seen) = freed(device);
        let texels = TextureDescriptor::texture_buffer(
            PixelFormat::R32Float,
            1024,
            ResourceOptions::shared(),
            TextureUsage::SHADER_READ,
        );
        let texture = device.new_texture_with_range_token(&texels, kept).unwrap();
        assert!(!seen.dropped());
        drop(texture);
        assert!(seen.dropped(), "a texture buffer's");
        // A refused creation's, before the call returns: a width of 0, and a shareable texture
        // of `Shared` storage (nil).
        let (refused, seen) = freed(device);
        let mut empty = texture_descriptor();
        empty.width = 0;
        device
            .new_texture_with_range_token(&empty, refused)
            .unwrap_err();
        assert!(seen.dropped(), "a refused texture's");
        let (refused, seen) = freed(device);
        let mut in_shared = texture_descriptor();
        in_shared.resource_options = ResourceOptions::shared();
        device
            .new_shared_texture_with_range_token(&in_shared, refused)
            .unwrap_err();
        assert!(seen.dropped(), "a refused shareable texture's");
        // A sparse heap's: kept while the heap lives, and while a texture of it lives after it.
        let sparse = HeapDescriptor::new(HeapType::Sparse, 1 << 20, ResourceOptions::private());
        let (kept, seen) = freed(device);
        let heap = match device.new_heap_with_range_token(&sparse, kept) {
            Ok(heap) => heap,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {}: refused {error:?}", device.name());
                assert!(seen.dropped(), "a refused sparse heap's");
                return;
            }
            Err(error) => panic!("a sparse heap on {}: {error:?}", device.name()),
        };
        assert!(!seen.dropped(), "a sparse heap keeps its token");
        drop(heap);
        assert!(
            seen.dropped() && seen.remade() == Some(true),
            "a sparse heap's"
        );
        let (kept, seen) = freed(device);
        let heap = device.new_heap_with_range_token(&sparse, kept).unwrap();
        let texture = match heap.new_texture(&descriptor) {
            Ok(texture) => texture,
            Err(error) if lacks_sparse(&error) => {
                println!("SPARSE {}: texture refused {error:?}", device.name());
                return;
            }
            Err(error) => panic!("a sparse texture on {}: {error:?}", device.name()),
        };
        drop(heap);
        assert!(!seen.dropped(), "its texture holds the sparse heap");
        let event = device.new_shared_event().unwrap();
        let reads = reading(&queue, &event, &texture, &seen);
        reads.0.commit().unwrap();
        drop(texture);
        runs_then_frees("a sparse heap", reads, &event, &seen);
    });
}

/// A no-copy buffer of `length` bytes over `memory`'s first byte.
fn no_copy_over(memory: &Mapping, length: u64) -> BufferDescriptor<'static> {
    // SAFETY: each test keeps the mapping past the buffers made over it, of at most its length.
    let memory = unsafe { HostMemory::new(memory.pointer()) };
    BufferDescriptor::new(
        length,
        ResourceOptions::shared(),
        BufferContents::NoCopy {
            memory,
            deallocator: None,
            keep_alive: None,
        },
    )
}

fn unshared(error: &Error) -> bool {
    matches!(
        error,
        Error::PlacementConflict(PlacementConflict::Unshared { .. })
    )
}

/// A no-copy buffer made over the same first byte of host memory as a live no-copy buffer, no
/// longer than it (`BufferDescriptor::range_of`), answers its address and holds its range with
/// it, as Metal answers one IOSurface's address for every buffer over it: a pin there is refused
/// while either lives; each one's range token drops with it, the first one's while the other
/// holds the range, the last one's once the range is free. Released right after the commit of a
/// command buffer that fills the second, a buffer made meanwhile, the range stays held until it
/// completed. Refused `Unshared`: other memory, a longer buffer, zero contents, a buffer of no
/// no-copy range, a placement at another address.
#[test]
fn a_buffer_over_the_same_memory_shares_a_live_buffer_s_range() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let memory = Mapping::anonymous(3 * GUEST_PAGE);
        memory.fill(0x11);
        let length = 2 * GUEST_PAGE as u64;
        let at_x = HIGH + 4096 * PAGE;
        let span = 4 * PAGE;
        let (kept, first_seen) = token(device, at_x, span);
        let first = match device.new_buffer_with_descriptor(
            no_copy_over(&memory, length)
                .with_placement(at(at_x))
                .with_range_token(kept),
        ) {
            Ok(first) => first,
            Err(error) => {
                assert!(not_importable(&error), "{error:?}");
                println!("SHARED {}: no-copy refused {error:?}", device.name());
                return;
            }
        };
        let (kept, second_seen) = token(device, at_x, span);
        let second = device
            .new_buffer_with_descriptor(
                no_copy_over(&memory, length)
                    .with_range_of(&first)
                    .with_range_token(kept),
            )
            .unwrap();
        assert_eq!(second.gpu_address(), GpuAddress(at_x));
        assert_eq!(second.contents(), Some(memory.pointer()));
        // A shorter one, sharing through the second, at the address named.
        let shorter = device
            .new_buffer_with_descriptor(
                no_copy_over(&memory, GUEST_PAGE as u64)
                    .with_range_of(&second)
                    .with_placement(at(at_x)),
            )
            .unwrap();
        assert_eq!(shorter.gpu_address(), GpuAddress(at_x));
        assert!(held(pinned(device, at_x, PAGE).unwrap_err()));
        // Refused: other memory, longer, zero contents, a buffer of no no-copy range, another
        // address.
        let elsewhere = Mapping::anonymous(GUEST_PAGE);
        let zero = device.new_buffer(PAGE, ResourceOptions::shared()).unwrap();
        let refusals = [
            (
                "other memory",
                no_copy_over(&elsewhere, PAGE).with_range_of(&first),
            ),
            (
                "longer",
                no_copy_over(&memory, 3 * GUEST_PAGE as u64).with_range_of(&first),
            ),
            (
                "zero contents",
                BufferDescriptor::new(PAGE, ResourceOptions::shared(), BufferContents::Zero)
                    .with_range_of(&first),
            ),
            (
                "of a zero buffer",
                no_copy_over(&memory, PAGE).with_range_of(&zero),
            ),
            (
                "another address",
                no_copy_over(&memory, length)
                    .with_range_of(&first)
                    .with_placement(at(at_x + 64 * PAGE)),
            ),
        ];
        for (what, descriptor) in refusals {
            let error = device.new_buffer_with_descriptor(descriptor).unwrap_err();
            assert!(unshared(&error), "{what}: {error:?}");
        }
        drop((zero, elsewhere));
        // The first goes: its token drops while the others hold the range.
        drop(first);
        assert!(first_seen.dropped(), "the first one's, with it");
        assert_eq!(first_seen.remade(), Some(false), "the range still held");
        drop(shorter);
        assert!(held(pinned(device, at_x, PAGE).unwrap_err()));
        // The second, released right after the commit of a command buffer that fills it.
        let event = device.new_shared_event().unwrap();
        let (command_buffer, in_handler) =
            held_back(&queue, &event, &second, (at_x, span), &second_seen);
        drop(second);
        let meanwhile = device.new_buffer(PAGE, ResourceOptions::shared()).unwrap();
        assert_ne!(meanwhile.gpu_address(), GpuAddress(at_x), "made elsewhere");
        assert!(!second_seen.dropped(), "in flight");
        assert!(held(pinned(device, at_x, PAGE).unwrap_err()));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(*in_handler.lock().unwrap(), Some((false, false)));
        assert!(
            second_seen.dropped(),
            "the last one's, before the wait returns"
        );
        assert_eq!(second_seen.remade(), Some(true), "the range free");
        let bytes = memory.read(0..3 * GUEST_PAGE);
        assert!(
            bytes[..2 * GUEST_PAGE].iter().all(|&byte| byte == 0x5a),
            "filled"
        );
        assert!(
            bytes[2 * GUEST_PAGE..].iter().all(|&byte| byte == 0x11),
            "past it"
        );
        drop(meanwhile);
    });
}
