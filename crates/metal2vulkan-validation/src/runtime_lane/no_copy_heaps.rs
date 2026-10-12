//! Heaps over the caller's host memory (`Device::new_heap_with_bytes_no_copy`, the heap twin of
//! a no-copy buffer) on the device the cfg selects, under the validation layer:
//!
//! - **Buffers.** A placement heap's buffers are the caller's bytes at their offsets: the CPU's
//!   writes through the caller's pointer are what the GPU reads through each buffer, the GPU's
//!   writes through a buffer show at the pointer, and two buffers over overlapping offsets alias.
//!   `contents` is the pointer plus the offset, `gpu_address` the heap's plus the offset. The
//!   deallocator runs once, with the pointer and the heap's size, when the heap and its resources
//!   are gone.
//! - **Textures.** A placement heap's texture and a buffer over the same bytes alias them as in
//!   any heap, written from the CPU and the GPU both ways: texels the GPU writes into the texture
//!   change the caller's bytes and the buffer's; bytes the CPU writes at the pointer, and bytes
//!   the GPU writes through the buffer, change every texel the texture reads; `replaceRegion`
//!   and `getBytes` round-trip and reach the caller's bytes. Where the device binds no image of
//!   the texture to imported host memory, the texture is a shadow of the heap's bytes, announced
//!   once (`shadowed-texture`), whose texels lie in the bytes level by level, rows packed, or,
//!   for a texture Metal keeps uncompressed, in Metal's tiles (`heap_layouts`): there the bytes are
//!   asserted exactly. `NOCOPYHEAP` lines print each texture's route.
//! - **Refusals.** Storage other than `Shared`, a sparse heap and a size of 0 are `Invalid`, as
//!   is memory whose first byte is off the import's unit
//!   (`HeapProblem::HostMemoryAlignment`); each refused call drops its deallocator unrun. A
//!   device that imports no host memory refuses the heap typed (`Unsupported`, `HostMemory`):
//!   printed and skipped.

use super::heap_layouts::tiled_offset;
use super::host_memory::{gpu_fill, gpu_read, host_page, not_importable, run, Mapping};
use super::{read_buffer, validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandQueue, CpuCacheMode, Deallocator, Degradation,
    Device, Error, HazardTrackingMode, Heap, HeapDescriptor, HeapProblem, HeapType, HostMemory,
    Invalid, Nil, Origin, Problem, Region, ResourceOptions, RuntimeGap, Size, StorageMode, Texture,
    TextureDescriptor,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The heaps' size: a few pages of any host.
pub(super) const HEAP: usize = 1 << 20;

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// `Shared`, tracked: the heap's options.
pub(super) fn shared() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    )
}

/// What a heap's deallocator saw: its runs and their arguments, and whether it was dropped
/// unrun.
#[derive(Default)]
struct Seen {
    runs: AtomicUsize,
    dropped: AtomicUsize,
    args: Mutex<Option<(usize, u64)>>,
}

/// Counts its drop into its `Seen` (a deallocator dropped unrun).
struct DropCount(Arc<Seen>);

impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A deallocator that records into `seen` and then releases `mapping`.
fn recording(seen: &Arc<Seen>, mapping: Mapping) -> Deallocator {
    let seen = Arc::clone(seen);
    let count = DropCount(Arc::clone(&seen));
    Box::new(move |pointer: NonNull<u8>, length: u64| {
        *seen.args.lock().unwrap() = Some((pointer.as_ptr() as usize, length));
        seen.runs.fetch_add(1, Ordering::SeqCst);
        drop(mapping);
        std::mem::forget(count);
    })
}

/// A heap of `heap_type` and `options` over `HEAP` bytes at `pointer`; `None`, printed, where
/// the device imports no host memory.
///
/// # Safety
///
/// `HostMemory::new`'s contract for `HEAP` bytes at `pointer`.
pub(super) unsafe fn no_copy_heap(
    device: &Device,
    pointer: NonNull<u8>,
    heap_type: HeapType,
    deallocator: Option<Deallocator>,
) -> Option<Heap> {
    let descriptor = HeapDescriptor::new(heap_type, HEAP as u64, shared());
    // SAFETY: the caller's contract.
    match unsafe {
        device.new_heap_with_bytes_no_copy(HostMemory::new(pointer), &descriptor, deallocator, None)
    } {
        Ok(heap) => Some(heap),
        Err(error) if not_importable(&error) => {
            eprintln!("NOCOPYHEAP: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_heap_with_bytes_no_copy: {error:?}"),
    }
}

/// Bytes `(i * 7 + seed) % 251`.
fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 7 + seed) % 251) as u8).collect()
}

/// `len` bytes of the 64 values from `base`, in an order that varies along the bytes: bytes of
/// different bases never match, so a texel written from one base and read after a write from
/// another changed.
pub(super) fn band(len: usize, base: u8) -> Vec<u8> {
    (0..len)
        .map(|i| base + ((i * 5 + i / 64) % 64) as u8)
        .collect()
}

/// A Shared buffer holding `bytes`.
pub(super) fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// Run `test` on a validated device, then check that the layer reported nothing; gives the
/// device's degrade notices of `kind`.
fn on_device(kind: Degradation, test: impl FnOnce(&Device)) -> usize {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    let notices = messages
        .take_notices()
        .iter()
        .filter(|message| {
            message
                .degradation
                .as_ref()
                .is_some_and(|notice| notice.degradation == kind)
        })
        .count();
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    notices
}

/// A placement heap over the caller's pages: buffers at offsets are its bytes, read and written
/// by the CPU through the pointer and by the GPU through the buffers, two buffers over
/// overlapping offsets alias, `contents` and `gpu_address` follow the offsets, an automatic heap
/// places its first buffer at the pointer, and the deallocator runs once, with the pointer and
/// the size, after the heap and its buffers are gone.
#[test]
fn a_no_copy_heap_s_buffers_are_the_caller_s_bytes() {
    let seen = Arc::new(Seen::default());
    on_device(Degradation::ShadowedTexture, |device| {
        let queue = device.new_command_queue().unwrap();
        let mapping = Mapping::anonymous(HEAP);
        let pointer = mapping.pointer();
        let at = pointer.as_ptr() as usize;
        mapping.write(0, &pattern(HEAP, 3));
        let deallocator = recording(&seen, mapping);
        // SAFETY: the mapping holds `HEAP` writable bytes until the deallocator releases it.
        let Some(heap) =
            (unsafe { no_copy_heap(device, pointer, HeapType::Placement, Some(deallocator)) })
        else {
            return;
        };
        assert_eq!(heap.size(), HEAP as u64);
        assert_eq!(heap.storage_mode(), StorageMode::Shared);
        // `a` over the heap's first 64 KiB, `b` over the 64 KiB from 32 KiB: they share 32 KiB.
        let first = 64u64 << 10;
        let a = heap.new_buffer_at_offset(first, shared(), 0).unwrap();
        let b = heap
            .new_buffer_at_offset(first, shared(), 32 << 10)
            .unwrap();
        assert_eq!(a.contents().unwrap().as_ptr() as usize, at);
        assert_eq!(b.contents().unwrap().as_ptr() as usize, at + 32 * 1024);
        assert_eq!(b.gpu_address().0, heap.gpu_address().0 + 32 * 1024);
        // The CPU's bytes, read by the GPU through each buffer.
        assert_eq!(
            gpu_read(&queue, &a, 0..first),
            pattern(HEAP, 3)[..first as usize]
        );
        // SAFETY: the bytes are the mapping's, which the deallocator has not released; no
        // command uses them now.
        let view = unsafe { std::slice::from_raw_parts_mut(pointer.as_ptr(), HEAP) };
        view[40 << 10..48 << 10].fill(0x3C);
        assert_eq!(gpu_read(&queue, &b, 8 << 10..16 << 10), vec![0x3C; 8 << 10]);
        // The GPU's writes through one buffer, at the pointer and through the other.
        gpu_fill(&queue, &a, 48 << 10..64 << 10, 0xA5);
        assert!(view[48 << 10..64 << 10].iter().all(|&byte| byte == 0xA5));
        assert_eq!(
            gpu_read(&queue, &b, 16 << 10..32 << 10),
            vec![0xA5; 16 << 10]
        );
        assert_eq!(
            read_buffer(&b, 16 << 10, 16 << 10).unwrap(),
            vec![0xA5; 16 << 10]
        );
        assert_eq!(
            seen.runs.load(Ordering::SeqCst),
            0,
            "ran while the heap lives"
        );
        drop((a, b, heap));
        assert_eq!(seen.runs.load(Ordering::SeqCst), 1);
        assert_eq!(*seen.args.lock().unwrap(), Some((at, HEAP as u64)));
        eprintln!("NOCOPYHEAP buffers: {HEAP} bytes at {at:#x}");

        // An automatic heap places its first buffer at the pointer.
        let mapping = Mapping::anonymous(HEAP);
        let pointer = mapping.pointer();
        // SAFETY: the mapping outlives the heap, dropped below first.
        let heap = unsafe { no_copy_heap(device, pointer, HeapType::Automatic, None) }.unwrap();
        let buffer = heap.new_buffer(4096, shared()).unwrap();
        assert_eq!(buffer.contents(), Some(pointer));
        gpu_fill(&queue, &buffer, 0..4096, 0x77);
        assert_eq!(mapping.read(0..4096), vec![0x77; 4096]);
        drop((buffer, heap));
        drop(mapping);
    });
    assert_eq!(seen.dropped.load(Ordering::SeqCst), 0);
}

/// Encode a blit of `texture`'s first level, `size` texels of `bytes` bytes, into `out`, rows
/// packed.
pub(super) fn encode_texels_out(
    command_buffer: &CommandBuffer,
    texture: &Texture,
    (width, height): (u64, u64),
    bytes: u64,
    out: &Buffer,
) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        out,
        0,
        width * bytes,
        width * height * bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

/// Encode a blit of `from`'s bytes, rows packed, into `texture`'s first level, `size` texels of
/// `bytes` bytes.
pub(super) fn encode_texels_in(
    command_buffer: &CommandBuffer,
    from: &Buffer,
    (width, height): (u64, u64),
    bytes: u64,
    texture: &Texture,
) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        from,
        0,
        width * bytes,
        width * height * bytes,
        Size::new(width, height, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
}

/// The texels of `texture`'s first level (`size` texels, `bytes` a texel), read back through a
/// blit, rows packed.
pub(super) fn blit_texels(
    queue: &CommandQueue,
    texture: &Texture,
    size: (u64, u64),
    bytes: u64,
) -> Vec<u8> {
    let len = size.0 * size.1 * bytes;
    let copy = (queue.device())
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode_texels_out(&command_buffer, texture, size, bytes, &copy);
    run(&command_buffer);
    read_buffer(&copy, 0, len).unwrap()
}

/// Blit `texels` (`size` texels, `bytes` a texel, rows packed) into `texture`'s first level.
pub(super) fn blit_into(
    queue: &CommandQueue,
    texture: &Texture,
    size: (u64, u64),
    bytes: u64,
    texels: &[u8],
) {
    let source = staged(queue.device(), texels);
    let command_buffer = queue.command_buffer().unwrap();
    encode_texels_in(&command_buffer, &source, size, bytes, texture);
    run(&command_buffer);
}

/// How many `bytes`-byte texels of `after` equal `before`'s.
fn same_texels(after: &[u8], before: &[u8], bytes: u64) -> usize {
    let n = bytes as usize;
    after
        .chunks_exact(n)
        .zip(before.chunks_exact(n))
        .filter(|(a, b)| a == b)
        .count()
}

/// The aliasing of a `side`-square texture of `pf` (`bytes` a texel) placed at `offset` of a
/// no-copy placement heap and a buffer over the same bytes, both ways from the CPU and the GPU
/// (the module docs). Gives whether the device announced the texture's shadow; `None`, printed,
/// where the device imports no host memory or refuses the texture as its lack.
fn texture_round_trip(pf: PixelFormat, side: u64, bytes: u64, offset: u64) -> Option<bool> {
    let what = format!("{pf:?} {side}x{side} at {offset}");
    // Whether the heap made the texture, and whether its texels lay in the caller's bytes rows
    // packed from its offset, or in Metal's tiles there where the runtime counts it tiled.
    let (mut made, mut laid) = (false, false);
    let shadows = on_device(Degradation::ShadowedTexture, |device| {
        let queue = device.new_command_queue().unwrap();
        let mapping = Mapping::anonymous(HEAP);
        let pointer = mapping.pointer();
        // SAFETY: the mapping outlives the heap, dropped at the end of the closure after it.
        let Some(heap) = (unsafe { no_copy_heap(device, pointer, HeapType::Placement, None) })
        else {
            return;
        };
        let mut descriptor = TextureDescriptor::texture_2d(pf, side, side, false);
        descriptor.resource_options = shared();
        descriptor.usage = if is_depth_stencil(pf) {
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET
        } else {
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE
        };
        let tiled_before = device.work_counts().heap_textures_tiled;
        let texture = match heap.new_texture_at_offset(&descriptor, offset) {
            Ok(texture) => texture,
            Err(error @ Error::NotImplemented(RuntimeGap::HeapTextureMemory)) => {
                eprintln!("NOCOPYHEAP {what}: refused {error:?}");
                return;
            }
            Err(error) => panic!("{what}: {error:?}"),
        };
        made = true;
        let tiled = device.work_counts().heap_textures_tiled - tiled_before == 1;
        assert_eq!(texture.heap_offset(), offset);
        let size = device
            .heap_texture_size_and_align(&descriptor)
            .unwrap()
            .size;
        let buffer = heap.new_buffer_at_offset(size, shared(), offset).unwrap();
        let len = (side * side * bytes) as usize;
        let range = offset as usize..(offset + size) as usize;
        let start = offset as usize;

        // The GPU writes the texture: the caller's bytes and the buffer's change.
        mapping.fill(0);
        let first = band(len, 0x00);
        blit_into(&queue, &texture, (side, side), bytes, &first);
        assert_eq!(
            blit_texels(&queue, &texture, (side, side), bytes),
            first,
            "{what}"
        );
        let host = mapping.read(range.clone());
        let changed = host.iter().filter(|&&byte| byte != 0).count();
        assert!(
            changed > 0,
            "{what}: the texture's write left the caller's bytes zero"
        );
        assert_eq!(gpu_read(&queue, &buffer, 0..size), host, "{what}");
        let mut texels = vec![0u8; len];
        let row = side * bytes;
        let region = Region::new_2d(0, 0, side, side);
        texture
            .get_bytes(&mut texels, row, 0, region, 0, 0)
            .unwrap();
        assert_eq!(texels, first, "{what}: getBytes");

        // The CPU writes the caller's bytes: every texel changes.
        let second = band(size as usize, 0x80);
        mapping.write(start, &second);
        let read = blit_texels(&queue, &texture, (side, side), bytes);
        assert_eq!(
            same_texels(&read, &first, bytes),
            0,
            "{what}: the CPU's write"
        );

        // The GPU writes through the buffer: every texel changes again.
        gpu_fill(&queue, &buffer, 0..size, 0x5A);
        let after = blit_texels(&queue, &texture, (side, side), bytes);
        assert_eq!(
            same_texels(&after, &read, bytes),
            0,
            "{what}: the buffer's write"
        );
        assert_eq!(mapping.read(range.clone()), vec![0x5A; size as usize]);

        // `replaceRegion` reaches the caller's bytes; `getBytes` reads them back.
        let third = band(len, 0xC0);
        texture
            .replace_region(region, 0, 0, &third, row, 0)
            .unwrap();
        let host = mapping.read(range.clone());
        assert!(
            host.iter().any(|&byte| byte != 0x5A),
            "{what}: replaceRegion left the caller's bytes"
        );
        texture
            .get_bytes(&mut texels, row, 0, region, 0, 0)
            .unwrap();
        assert_eq!(texels, third, "{what}: getBytes after replaceRegion");
        assert_eq!(
            blit_texels(&queue, &texture, (side, side), bytes),
            third,
            "{what}"
        );
        laid = if tiled {
            let n = bytes as usize;
            (0..side).all(|y| {
                (0..side).all(|x| {
                    let at = start + tiled_offset(bytes, [side, side], x, y) as usize;
                    let from = ((y * side + x) * bytes) as usize;
                    mapping.read(at..at + n) == third[from..from + n]
                })
            })
        } else {
            mapping.read(start..start + len) == third
        };
        eprintln!(
            "NOCOPYHEAP {what}: size {size}, {changed} bytes changed by the texture's write, \
             texels laid at the offset (tiled {tiled}): {laid}"
        );
        drop((texture, buffer, heap));
        drop(mapping);
    });
    eprintln!("NOCOPYHEAP {what}: {shadows} shadowed-texture notices");
    // A shadow's texels are the heap's bytes, rows packed from the texture's offset or in Metal's
    // tiles there.
    assert!(
        shadows == 0 || laid,
        "{what}: a shadow's texels lie elsewhere"
    );
    made.then_some(shadows == 1)
}

/// A no-copy heap's colour texture and a buffer over its bytes alias them both ways from the
/// CPU and the GPU (the module docs), at the heap's start and at a later multiple of the
/// texture's alignment. Where the texture is a shadow of the heap's bytes, the device announced
/// it once and its texels lie in the bytes rows packed from its offset, or in Metal's tiles there
/// (a texture of a plain format Metal keeps uncompressed, as this one).
#[test]
fn a_no_copy_heap_s_texture_and_buffer_alias_both_ways() {
    for offset in [0, 256 << 10] {
        texture_round_trip(PixelFormat::RGBA8Unorm, 64, 4, offset);
    }
}

/// A no-copy heap's depth texture aliases as its colour textures do: where the device binds no
/// depth image to imported host memory (every device measured), a shadow of the heap's bytes,
/// announced once.
#[test]
fn a_no_copy_heap_s_depth_texture_aliases_both_ways() {
    if let Some(shadow) = texture_round_trip(PixelFormat::Depth16Unorm, 32, 2, 0) {
        eprintln!("NOCOPYHEAP Depth16Unorm: shadow {shadow}");
    }
}

/// A no-copy heap is refused as Metal's heaps and the no-copy import's rules say: `Private`
/// storage, a sparse heap, a size of 0 (`Invalid`), memory whose first byte is off the import's
/// unit (`Invalid`, `HeapProblem::HostMemoryAlignment`), each dropping its deallocator unrun.
#[test]
fn a_no_copy_heap_is_refused_as_its_rules_say() {
    on_device(Degradation::ShadowedTexture, |device| {
        const CALL: &str = "new_heap_with_bytes_no_copy";
        let page = host_page();
        let mapping = Mapping::anonymous(HEAP + 4 * page);
        let pointer = mapping.pointer();
        let refuse = |descriptor: HeapDescriptor, at: NonNull<u8>| {
            let seen = Arc::new(Seen::default());
            let count = DropCount(Arc::clone(&seen));
            let deallocator: Deallocator = Box::new(move |_, _| drop(count));
            // SAFETY: the mapping holds every byte asked for and outlives any heap.
            let made = unsafe {
                device.new_heap_with_bytes_no_copy(
                    HostMemory::new(at),
                    &descriptor,
                    Some(deallocator),
                    None,
                )
            };
            assert_eq!(seen.runs.load(Ordering::SeqCst), 0);
            assert_eq!(seen.dropped.load(Ordering::SeqCst), 1, "{descriptor:?}");
            made.err()
        };
        // SAFETY: the mapping holds every byte the heap would take.
        let Some(heap) = (unsafe { no_copy_heap(device, pointer, HeapType::Placement, None) })
        else {
            return;
        };
        drop(heap);
        let private = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Tracked,
        );
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP as u64, private);
        assert_eq!(
            refuse(descriptor, pointer),
            Some(invalid(CALL, Problem::StorageMode))
        );
        let descriptor = HeapDescriptor::new(HeapType::Sparse, HEAP as u64, shared());
        assert_eq!(
            refuse(descriptor, pointer),
            Some(invalid(CALL, Problem::StorageMode))
        );
        let descriptor = HeapDescriptor::new(HeapType::Placement, 0, shared());
        assert_eq!(
            refuse(descriptor, pointer),
            Some(Error::Nil(Nil::new(CALL, Problem::Heap(HeapProblem::Size))))
        );
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP as u64, shared());
        // SAFETY: 256 bytes into the mapping, which holds the heap's bytes past it.
        let off = unsafe { NonNull::new_unchecked(pointer.as_ptr().add(256)) };
        let refused = refuse(descriptor, off);
        let Some(Error::Invalid(invalid)) = &refused else {
            panic!("an unaligned heap: {refused:?}");
        };
        let Problem::Heap(HeapProblem::HostMemoryAlignment { alignment }) = invalid.problem else {
            panic!("an unaligned heap: {refused:?}");
        };
        assert!(alignment >= page as u64 && alignment.is_power_of_two());
        eprintln!("NOCOPYHEAP refusals: import unit {alignment} bytes");
        drop(mapping);
    });
}
