//! Textures in heaps over host pages laid out as Metal lays them out, on the device the cfg
//! selects, under the validation layer:
//!
//! - **Tiles.** A texture of a plain format that Metal keeps uncompressed lies in the heap's
//!   bytes as Metal's tiles hold it: each element (a texel) in the tile of the largest 16 KiB
//!   whose side is the level's own where shorter, the tiles in rows, the elements of a tile in
//!   Morton order (`x0 y0 x1 y1 ...`, the longer side's extra bits on top). The bytes it is made
//!   over, its `replaceRegion` and `getBytes`, the GPU's blits, and the CPU's and a buffer's
//!   writes over the bytes all meet the texels at those bytes, and no byte between them changes.
//!   The runtime counts the texture (`WorkCounts::heap_textures_tiled`), announces no layout
//!   notice, and writes no zeros making it. Each lane-side offset
//!   is restated here from the measured rule, apart from the runtime's
//!   (`texture/heap_tiled.rs`).
//! - **Two ways in one command.** A dispatch that writes the tiled texture's texel (0, 0) and
//!   four of its bytes through a buffer over the heap keeps both writes: the next element's bytes
//!   (`k.two_ways_disjoint`'s shape) or bytes between the elements (`k.two_ways_gap`'s), every
//!   other byte the heap's, through the merge (`shadow-merge` once).
//! - **Blocks**. A BC, ETC2, EAC or ASTC texture lies as its device's Metal level lays it out, at
//!   each level: each row's span, every block numbered as the probes numbered it, hashes to the
//!   M3 Max's and the M4's at 27.0.1 through `replaceRegion` and blits, and at 26.7 to the M3
//!   Max's CPU's through `replaceRegion` and its GPU's through blits; `getBytes` reads the CPU's
//!   writes of the CPU's bytes.
//! - **Layouts the runtime does not follow** ([`HEAP_LAYOUTS`]). A texture Metal keeps
//!   losslessly compressed (`heap-texture-lossless-layout`), one of a block-compressed format
//!   whose placement no level measures (PVRTC, `heap-texture-block-compressed-layout`) and one
//!   of a format whose layout is not measured
//!   (`heap-texture-untiled-layout`, a two-aspect depth and stencil format) are each counted
//!   (`Device::degraded_objects`, and its kind's `WorkCounts` field) and usable: the notice is
//!   not fatal. Each kind is announced once per device, in the call that made its first texture,
//!   with that texture's ID, and at each texture only where the device's options ask.
//! - **Multisample textures**. One Metal keeps losslessly compressed lies in memory of its own:
//!   counted and announced `heap-texture-lossless-layout`, usable (a pass clears and resolves it),
//!   the heap's bytes keeping every byte they held; the heap's detach is refused while it lives (no
//!   stand-in zeroes samples). One Metal keeps uncompressed is refused
//!   `NotImplemented(HostHeapUncompressedMultisample)`. The size query answers at least Metal's
//!   for both.

use super::degrade_notices::{counts, on_device, on_device_with};
use super::host_memory::{not_importable, run, Mapping};
use super::no_copy_heaps::{shared, HEAP};
use super::{buffer_with, read_buffer, Messages};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandQueue, Degradation, Device, DeviceOptions, Error, Heap,
    HeapDescriptor, HeapType, HostMemory, LoadAction, MetalLevel, ObjectId, Origin,
    PipelineOptions, Region, RenderPassDescriptor, ResourceOptions, RuntimeGap, Size, StoreAction,
    Texture, TextureDescriptor, WorkCounts,
};
use std::path::Path;

/// The kinds a texture in a heap over host pages laid out otherwise than Metal's tiles takes, each
/// counted per texture and announced once per device with the first's ID.
pub(super) const HEAP_LAYOUTS: [Degradation; 3] = [
    Degradation::HeapTextureLosslessLayout,
    Degradation::HeapTextureBlockCompressedLayout,
    Degradation::HeapTextureUntiledLayout,
];

/// Where each texture lies in its heap: a multiple of every alignment Metal answers.
const OFFSET: u64 = 64 << 10;

/// Metal's largest tile of `unit`-byte elements, 16 KiB.
fn largest_tile(unit: u64) -> [u64; 2] {
    match unit {
        1 => [128, 128],
        2 => [128, 64],
        4 => [64, 64],
        8 => [64, 32],
        16 => [32, 32],
        _ => panic!("no tile of {unit}-byte elements"),
    }
}

/// The Morton index of `(x, y)` in a tile of `tw x th` elements: x's bit lowest, then y's, the
/// longer side's extra bits on top.
fn morton(x: u64, y: u64, [tw, th]: [u64; 2]) -> u64 {
    let (bx, by) = (tw.trailing_zeros(), th.trailing_zeros());
    let (mut index, mut bit) = (0, 0);
    for i in 0..bx.max(by) {
        if i < bx {
            index |= ((x >> i) & 1) << bit;
            bit += 1;
        }
        if i < by {
            index |= ((y >> i) & 1) << bit;
            bit += 1;
        }
    }
    index
}

/// The byte, from the texture's first, of element `(x, y)` of a one-level 2D texture of
/// `[w, h]` `unit`-byte elements in Metal's tiles.
pub(super) fn tiled_offset(unit: u64, [w, h]: [u64; 2], x: u64, y: u64) -> u64 {
    let [tw, th] = largest_tile(unit);
    let tile = [tw.min(w.next_power_of_two()), th.min(h.next_power_of_two())];
    let across = w.div_ceil(tile[0]);
    let index = (y / tile[1]) * across + x / tile[0];
    (index * tile[0] * tile[1] + morton(x % tile[0], y % tile[1], tile)) * unit
}

/// A plain texture's shape: its format, element bytes and extent.
#[derive(Clone, Copy, Debug)]
struct Plain {
    format: PixelFormat,
    unit: u64,
    extent: [u64; 2],
}

impl Plain {
    /// Its descriptor in a `Shared` heap: read and written by shaders, so Metal keeps it
    /// uncompressed.
    fn descriptor(&self) -> TextureDescriptor {
        let [w, h] = self.extent;
        let mut descriptor = TextureDescriptor::texture_2d(self.format, w, h, false);
        descriptor.resource_options = shared();
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor
    }

    fn texels(&self) -> usize {
        (self.extent[0] * self.extent[1]) as usize
    }

    fn row(&self) -> u64 {
        self.extent[0] * self.unit
    }

    fn region(&self) -> Region {
        Region::new_2d(0, 0, self.extent[0], self.extent[1])
    }

    /// The heap's bytes from its first: `outside` everywhere but at the texture's elements, which
    /// hold `texels` (packed rows) where Metal's tiles put them.
    fn laid(&self, texels: &[u8], outside: &[u8]) -> Vec<u8> {
        let mut bytes = outside.to_vec();
        let n = self.unit as usize;
        let [w, h] = self.extent;
        for y in 0..h {
            for x in 0..w {
                let at = (OFFSET + tiled_offset(self.unit, self.extent, x, y)) as usize;
                let from = ((y * w + x) * self.unit) as usize;
                bytes[at..at + n].copy_from_slice(&texels[from..from + n]);
            }
        }
        bytes
    }

    /// The bytes one past the texture's last element, from its first.
    fn span(&self) -> u64 {
        let [w, h] = self.extent;
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| tiled_offset(self.unit, self.extent, x, y) + self.unit)
            .max()
            .unwrap()
    }
}

/// The shapes the lane lays out: each element size, levels narrower and shorter than a tile,
/// tiles across and down, and a level of more than one tile across.
fn plains() -> Vec<Plain> {
    let plain = |format, unit, extent| Plain {
        format,
        unit,
        extent,
    };
    vec![
        plain(PixelFormat::R8Unorm, 1, [100, 37]),
        plain(PixelFormat::RG8Unorm, 2, [300, 20]),
        plain(PixelFormat::RGBA8Unorm, 4, [64, 64]),
        plain(PixelFormat::RGBA8Unorm, 4, [48, 48]),
        plain(PixelFormat::RGBA16Float, 8, [70, 10]),
        plain(PixelFormat::RGBA32Float, 16, [33, 17]),
    ]
}

/// Bytes that name their place, from `seed`: no two shapes' and no two seeds' alike.
fn named(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 13 + i / 251 + seed) % 251) as u8 + 1)
        .collect()
}

/// A placement heap over `mapping`'s `HEAP` bytes; `None`, printed, where the device imports no
/// host memory.
fn heap_over(device: &Device, mapping: &Mapping) -> Option<Heap> {
    let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP as u64, shared());
    // SAFETY: the mapping holds `HEAP` writable bytes and outlives the heap (each test drops the
    // heap first).
    match unsafe {
        device.new_heap_with_bytes_no_copy(
            HostMemory::new(mapping.pointer()),
            &descriptor,
            None,
            None,
        )
    } {
        Ok(heap) => Some(heap),
        Err(error) if not_importable(&error) => {
            eprintln!("HEAPLAYOUT: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_heap_with_bytes_no_copy: {error:?}"),
    }
}

/// The texels of `plain`'s texture read back through a blit.
fn blit_out(queue: &CommandQueue, texture: &Texture, plain: &Plain) -> Vec<u8> {
    let len = plain.texels() as u64 * plain.unit;
    let copy = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [w, h] = plain.extent;
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &copy,
        0,
        plain.row(),
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&copy, 0, len).unwrap()
}

/// Blit `texels` into `plain`'s texture.
fn blit_in(queue: &CommandQueue, texture: &Texture, plain: &Plain, texels: &[u8]) {
    let source = buffer_with(queue.device(), texels).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [w, h] = plain.extent;
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        plain.row(),
        texels.len() as u64,
        Size::new(w, h, 1),
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// The degrade notices delivered so far of the [`HEAP_LAYOUTS`] kinds: each kind and the ID it
/// names.
fn layout_notices(messages: &Messages) -> Vec<(Degradation, Option<ObjectId>)> {
    let notices = messages.take_notices();
    // Every notice decodes by its kind (`counts`).
    counts(&notices);
    notices
        .into_iter()
        .filter_map(|message| message.degradation)
        .filter(|notice| HEAP_LAYOUTS.contains(&notice.degradation))
        .map(|notice| (notice.degradation, notice.object_id))
        .collect()
}

/// How many textures each layout counter of `after` counted past `before`: tiled, lossless,
/// block-compressed, untiled.
fn layouts_counted(before: &WorkCounts, after: &WorkCounts) -> [u64; 4] {
    [
        after.heap_textures_tiled - before.heap_textures_tiled,
        after.heap_texture_lossless_layouts - before.heap_texture_lossless_layouts,
        after.heap_texture_block_compressed_layouts - before.heap_texture_block_compressed_layouts,
        after.heap_texture_untiled_layouts - before.heap_texture_untiled_layouts,
    ]
}

/// A plain texture in a heap over host pages lies in Metal's tiles (the module docs): its
/// answered size holds them; the bytes it is made over, `replaceRegion`, a blit into it, the CPU's
/// writes and a buffer's over the bytes meet its texels there, and `getBytes` and a blit out of it
/// read them; no byte between them changes. Counted tiled, with no layout notice; made with no
/// zeros (`WorkCounts::images_started_on_the_host`, `staging_submissions`).
#[test]
fn a_host_heap_s_plain_texture_lies_in_metal_s_tiles() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        for plain in plains() {
            let what = format!("{:?} {:?}", plain.format, plain.extent);
            let mapping = Mapping::anonymous(HEAP);
            let Some(heap) = heap_over(device, &mapping) else {
                return;
            };
            let descriptor = plain.descriptor();
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            assert!(
                answer.size >= plain.span(),
                "{what}: the answer {answer:?} holds Metal's tiles ({} bytes)",
                plain.span()
            );
            let len = plain.texels() * plain.unit as usize;
            let zero = vec![0u8; HEAP];

            // What the heap's bytes held before the texture was made is its texels: making it
            // writes nothing (its image starts as its memory lies, each move in writing every
            // element first: no copy on the host, no staging submission), and a blit reads them.
            let held = named(len, 5);
            mapping.write(0, &plain.laid(&held, &zero));
            let before = device.work_counts();
            let texture = heap.new_texture_at_offset(&descriptor, OFFSET).unwrap();
            let made = device.work_counts();
            assert_eq!(
                layouts_counted(&before, &made),
                [1, 0, 0, 0],
                "{what}: counted tiled"
            );
            assert_eq!(
                (made.images_started_on_the_host, made.staging_submissions),
                (
                    before.images_started_on_the_host,
                    before.staging_submissions
                ),
                "{what}: made with no zeros"
            );
            assert_eq!(layout_notices(messages), [], "{what}");
            assert_eq!(
                blit_out(&queue, &texture, &plain),
                held,
                "{what}: the bytes it was made over"
            );
            mapping.fill(0);

            // `replaceRegion` writes the texels into Metal's tiles; `getBytes` and a blit read
            // them back.
            let first = named(len, 1);
            texture
                .replace_region(plain.region(), 0, 0, &first, plain.row(), 0)
                .unwrap();
            assert!(
                mapping.read(0..HEAP) == plain.laid(&first, &zero),
                "{what}: replaceRegion's bytes"
            );
            let mut read = vec![0u8; len];
            texture
                .get_bytes(&mut read, plain.row(), 0, plain.region(), 0, 0)
                .unwrap();
            assert_eq!(read, first, "{what}: getBytes");
            assert_eq!(blit_out(&queue, &texture, &plain), first, "{what}: a blit");

            // A blit into the texture lands in the tiles, no other byte.
            let second = named(len, 2);
            blit_in(&queue, &texture, &plain, &second);
            assert!(
                mapping.read(0..HEAP) == plain.laid(&second, &zero),
                "{what}: a blit's bytes"
            );

            // The CPU's writes at the tiles are the texels the GPU and `getBytes` read.
            let third = named(len, 3);
            mapping.write(0, &plain.laid(&third, &zero));
            assert_eq!(
                blit_out(&queue, &texture, &plain),
                third,
                "{what}: the CPU's"
            );
            texture
                .get_bytes(&mut read, plain.row(), 0, plain.region(), 0, 0)
                .unwrap();
            assert_eq!(read, third, "{what}: getBytes of the CPU's");

            // A buffer over the texture's bytes: its write reaches every texel, and the texture's
            // next write its elements alone.
            let span = answer.size;
            let buffer = heap.new_buffer_at_offset(span, shared(), OFFSET).unwrap();
            let fill = |value| {
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..span, value).unwrap();
                blit.end_encoding().unwrap();
                run(&command_buffer);
            };
            fill(0x5a);
            assert_eq!(
                blit_out(&queue, &texture, &plain),
                vec![0x5a; len],
                "{what}: the buffer's write"
            );
            let fourth = named(len, 4);
            blit_in(&queue, &texture, &plain, &fourth);
            let mut around = zero.clone();
            around[OFFSET as usize..(OFFSET + span) as usize].fill(0x5a);
            assert!(
                mapping.read(0..HEAP) == plain.laid(&fourth, &around),
                "{what}: a blit after the buffer's write"
            );
            drop((buffer, texture, heap));
            drop(mapping);
            eprintln!("HEAPLAYOUT {what}: in Metal's tiles, size {}", answer.size);
        }
        let counted = counts(&messages.take_notices());
        assert!(
            HEAP_LAYOUTS.iter().all(|kind| !counted.contains_key(kind)),
            "{counted:?}"
        );
    });
}

/// A dispatch writes a tiled texture's texel (0, 0) and four bytes of a buffer over the heap,
/// those of element (1, 0) or ones between the elements, and both writes stay: every other byte
/// is the heap's (the module docs), merged once (`shadow-merge`).
#[test]
fn a_command_writing_a_tiled_texture_two_ways_keeps_both_writes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let ll =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = library.new_function_with_name("tob_two_ways").unwrap();
        let kernel = device
            .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
            .unwrap();
        let rgba = |side| Plain {
            format: PixelFormat::RGBA8Unorm,
            unit: 4,
            extent: [side, side],
        };
        // Element (1, 0), and the bytes of (48, 0), past a 48-square level's elements.
        let cases = [
            (rgba(64), tiled_offset(4, [64, 64], 1, 0)),
            (rgba(48), tiled_offset(4, [64, 64], 48, 0)),
        ];
        let mut made = false;
        for (plain, off) in cases {
            let what = format!("{:?} writing bytes {off}", plain.extent);
            let mapping = Mapping::anonymous(HEAP);
            let Some(heap) = heap_over(device, &mapping) else {
                return;
            };
            let texture = heap
                .new_texture_at_offset(&plain.descriptor(), OFFSET)
                .unwrap();
            let span = device
                .heap_texture_size_and_align(&plain.descriptor())
                .unwrap()
                .size;
            let buffer = heap.new_buffer_at_offset(span, shared(), OFFSET).unwrap();
            let before = named(HEAP, 7);
            mapping.write(0, &before);
            let at = buffer_with(device, &(off as u32).to_le_bytes()).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut e = command_buffer.compute_command_encoder().unwrap();
            e.set_compute_pipeline_state(&kernel).unwrap();
            e.set_texture(Some(&texture), 0).unwrap();
            e.set_buffer(Some(&buffer), 0, 0).unwrap();
            e.set_buffer(Some(&at), 0, 1).unwrap();
            e.dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
            e.end_encoding().unwrap();
            run(&command_buffer);
            let mut want = before.clone();
            let first = OFFSET as usize;
            want[first..first + 4].copy_from_slice(&[0xff, 0, 0, 0xff]);
            let off = first + off as usize;
            want[off..off + 4].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
            let after = mapping.read(0..HEAP);
            let differ: Vec<usize> = (0..HEAP).filter(|&i| after[i] != want[i]).collect();
            assert!(
                differ.is_empty(),
                "{what}: {} bytes differ, the first at {:?}",
                differ.len(),
                differ.first()
            );
            made = true;
            drop((buffer, texture, heap));
            drop(mapping);
            eprintln!("HEAPLAYOUT two ways {what}: both writes kept");
        }
        let counted = counts(&messages.take_notices());
        if made {
            assert_eq!(
                counted.get(&Degradation::ShadowMerge),
                Some(&1),
                "merged, announced once: {counted:?}"
            );
        }
    });
}

/// A texture Metal keeps losslessly compressed, one of a block-compressed format whose placement
/// no level measures (PVRTC) and one of a two-aspect depth and stencil format, each in a heap
/// over host pages, are counted in their kind's field and `Device::degraded_objects`, and the
/// first of each kind is announced in the call that made it, with its ID; each is usable after.
/// Two compressed textures, one notice, a count of two.
#[test]
fn heap_textures_laid_out_otherwise_than_metal_s_tiles_are_counted_and_announced_once() {
    on_device(|device, messages| {
        let mapping = Mapping::anonymous(HEAP);
        let Some(heap) = heap_over(device, &mapping) else {
            return;
        };
        messages.take_notices();
        let read = TextureUsage::SHADER_READ;
        let mut compressed = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        compressed.resource_options = shared();
        compressed.usage = read;
        let mut blocks = TextureDescriptor::texture_2d(PixelFormat::PVRTCRGBA4BPP, 64, 64, false);
        blocks.resource_options = shared();
        blocks.usage = read;
        let mut aspects =
            TextureDescriptor::texture_2d(PixelFormat::Depth32FloatStencil8, 32, 32, false);
        aspects.resource_options = shared();
        aspects.usage = read | TextureUsage::RENDER_TARGET;
        // Metal's answer for the compressed shape (`heap_textures`): its
        // metadata included.
        let answer = device.heap_texture_size_and_align(&compressed).unwrap();
        assert!(answer.size >= 16512, "{answer:?}");
        let cases = [
            (compressed, Degradation::HeapTextureLosslessLayout, 1),
            (compressed, Degradation::HeapTextureLosslessLayout, 1),
            (blocks, Degradation::HeapTextureBlockCompressedLayout, 2),
            (aspects, Degradation::HeapTextureUntiledLayout, 3),
        ];
        let mut offset = 0u64;
        let mut kept = Vec::new();
        let mut announced = Vec::new();
        for (descriptor, kind, counter) in cases {
            let what = format!("{:?} {kind:?}", descriptor.pixel_format);
            let size = device.heap_texture_size_and_align(&descriptor).unwrap();
            offset = offset.next_multiple_of(size.align.max(OFFSET));
            let before = device.work_counts();
            let objects = device
                .degraded_objects(kind)
                .expect("a kind counted per texture");
            let texture = match heap.new_texture_at_offset(&descriptor, offset) {
                Ok(texture) => texture,
                Err(
                    error @ (Error::NotImplemented(RuntimeGap::HeapTextureMemory)
                    | Error::Unsupported(_)),
                ) => {
                    eprintln!("HEAPLAYOUT {what}: refused {error:?}");
                    continue;
                }
                Err(error) => panic!("{what}: {error:?}"),
            };
            offset += size.size;
            // The first of its kind announced in the call, with the texture's ID; each counted.
            let want: &[_] = match announced.contains(&kind) {
                true => &[],
                false => &[(kind, Some(texture.object_id()))],
            };
            assert_eq!(layout_notices(messages), want, "{what}");
            announced.push(kind);
            let mut want = [0; 4];
            want[counter] = 1;
            assert_eq!(
                layouts_counted(&before, &device.work_counts()),
                want,
                "{what}: counted"
            );
            assert_eq!(
                device.degraded_objects(kind),
                Some(objects + 1),
                "{what}: counted"
            );
            // Usable: its texels round-trip.
            let (row, rows) = match descriptor.pixel_format {
                PixelFormat::PVRTCRGBA4BPP => (16 * 8, 16),
                PixelFormat::RGBA8Unorm => (64 * 4, 64),
                _ => (0, 0),
            };
            if row > 0 {
                let texels = named((row * rows) as usize, 5);
                let region = Region::new_2d(0, 0, descriptor.width, descriptor.height);
                texture
                    .replace_region(region, 0, 0, &texels, row, 0)
                    .unwrap();
                let mut back = vec![0u8; texels.len()];
                texture.get_bytes(&mut back, row, 0, region, 0, 0).unwrap();
                assert_eq!(back, texels, "{what}: round trip");
            }
            eprintln!("HEAPLAYOUT {what}: counted, announced where first");
            kept.push(texture);
        }
        drop((kept, heap));
        drop(mapping);
    });
}

/// The textures [`many_compressed`] makes.
const MANY: usize = 12;

/// Make [`MANY`] textures Metal keeps losslessly compressed in one heap over host pages, and
/// answer the `heap-texture-lossless-layout` notices their calls announced (each with its object
/// ID), the textures' IDs, and how many textures `Device::degraded_objects` counted; `None`,
/// printed, where the device imports no host memory or refuses the textures.
fn many_compressed(
    device: &Device,
    messages: &Messages,
) -> Option<(Vec<Option<ObjectId>>, Vec<ObjectId>, u64)> {
    let kind = Degradation::HeapTextureLosslessLayout;
    let mapping = Mapping::anonymous(HEAP);
    let heap = heap_over(device, &mapping)?;
    messages.take_notices();
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
    descriptor.resource_options = shared();
    descriptor.usage = TextureUsage::SHADER_READ;
    let size = device.heap_texture_size_and_align(&descriptor).unwrap();
    let before = device
        .degraded_objects(kind)
        .expect("a kind counted per texture");
    let (mut offset, mut textures) = (0u64, Vec::new());
    for _ in 0..MANY {
        offset = offset.next_multiple_of(size.align.max(OFFSET));
        assert!(
            offset + size.size <= HEAP as u64,
            "{MANY} textures fit the heap"
        );
        match heap.new_texture_at_offset(&descriptor, offset) {
            Ok(texture) => textures.push(texture),
            Err(
                error @ (Error::NotImplemented(RuntimeGap::HeapTextureMemory)
                | Error::Unsupported(_)),
            ) => {
                eprintln!("HEAPLAYOUT many: refused {error:?}");
                return None;
            }
            Err(error) => panic!("many compressed textures: {error:?}"),
        }
        offset += size.size;
    }
    let notices = layout_notices(messages);
    assert!(notices.iter().all(|(k, _)| *k == kind), "{notices:?}");
    let counted = device
        .degraded_objects(kind)
        .expect("a kind counted per texture")
        - before;
    let ids = textures.iter().map(Texture::object_id).collect();
    drop((textures, heap));
    drop(mapping);
    Some((
        notices.into_iter().map(|(_, id)| id).collect(),
        ids,
        counted,
    ))
}

/// Making many textures of one layout kind in heaps over host pages announces the kind once,
/// naming the first texture, and counts each (`Device::degraded_objects`): no line per texture.
/// A device whose options ask for the kind at each occurrence (`DeviceOptions::announce_each`)
/// announces each texture, with its ID, and counts them alike.
#[test]
fn making_many_textures_of_a_layout_kind_announces_it_once_and_counts_each() {
    on_device(|device, messages| {
        let Some((notices, ids, counted)) = many_compressed(device, messages) else {
            return;
        };
        assert_eq!(notices, [Some(ids[0])], "announced once, naming the first");
        assert_eq!(counted, MANY as u64, "each counted");
        eprintln!("HEAPLAYOUT many: {MANY} textures, one notice, counted {counted}");
    });
    let each = |options: &mut DeviceOptions| {
        options.announce_each = [Degradation::HeapTextureLosslessLayout].into();
    };
    on_device_with(each, |device, messages| {
        let Some((notices, ids, counted)) = many_compressed(device, messages) else {
            return;
        };
        let want: Vec<_> = ids.into_iter().map(Some).collect();
        assert_eq!(notices, want, "asked for each: announced at each");
        assert_eq!(counted, MANY as u64, "each counted");
        eprintln!("HEAPLAYOUT many: asked for each, {MANY} notices");
    });
}

/// A block-compressed texture's shape, one of the rows of `runtime::tests::heap_tiled_blocks`
/// (and the same probes on an Apple M3 Max): its format, block bytes and texels, type, level-0
/// extent, levels and array length, Metal's size, and the FNV-1a hashes of its span with every
/// block numbered in turn, as each level's Metal lays it out.
struct Blocks {
    format: PixelFormat,
    unit: u64,
    block: [u64; 2],
    texture_type: TextureType,
    extent: [u64; 2],
    levels: u64,
    array: u64,
    size: u64,
    /// An Apple M4's under macOS 27.0.1, by `replaceRegion` and by a blit alike.
    m4: u64,
    /// An Apple M3 Max's under macOS 26.7: by `replaceRegion`, and by a blit.
    m3: (u64, u64),
}

/// The rows the lane lays out: BC, ETC2, EAC and ASTC of square and other blocks, levels narrower
/// than a tile and wider than one, a cube and a 2D array.
fn block_rows() -> Vec<Blocks> {
    use PixelFormat as P;
    use TextureType as T;
    let row =
        |format, (unit, block), texture_type, extent, (levels, array), (size, m4), m3| Blocks {
            format,
            unit,
            block,
            texture_type,
            extent,
            levels,
            array,
            size,
            m4,
            m3,
        };
    let (bc1, bc7, astc4) = ((8, [4, 4]), (16, [4, 4]), (16, [4, 4]));
    vec![
        row(
            P::BC1RGBA,
            bc1,
            T::Type2D,
            [100, 33],
            (7, 1),
            (5888, 0x4558c9fea80831cb),
            (0x35839f04fcebe60b, 0x35839f04fcebe60b),
        ),
        row(
            P::BC1RGBA,
            bc1,
            T::Type2D,
            [513, 300],
            (10, 1),
            (251648, 0x22ea3ab679043212),
            (0x22ea3ab679043212, 0x8a8f3ecd2a8d3212),
        ),
        row(
            P::BC3RGBA,
            bc7,
            T::Type2D,
            [129, 300],
            (9, 1),
            (142336, 0x9b6045c40ad46074),
            (0x56e93915d9393274, 0x56e93915d9393274),
        ),
        row(
            P::BC1RGBA,
            bc1,
            T::Cube,
            [100, 100],
            (7, 1),
            (67584, 0x8af37d6151423431),
            (0x8af37d6151423431, 0x8af37d6151423431),
        ),
        row(
            P::BC7RGBAUnorm,
            bc7,
            T::Type2DArray,
            [260, 70],
            (9, 2),
            (196608, 0x4d4b187c72ec750a),
            (0x15f023d6e521510a, 0x15f023d6e521510a),
        ),
        row(
            P::ETC2RGB8,
            bc1,
            T::Type2D,
            [67, 300],
            (9, 1),
            (44160, 0xc670a09509ba2c26),
            (0x9524fa25b7c2a8e6, 0x9524fa25b7c2a8e6),
        ),
        row(
            P::EACRG11Unorm,
            bc7,
            T::Type2D,
            [260, 70],
            (9, 1),
            (87808, 0xcd5ceb35bcad1b11),
            (0x1225e98160f8ff11, 0x1225e98160f8ff11),
        ),
        row(
            P::ASTC4x4LDR,
            astc4,
            T::Type2D,
            [100, 33],
            (7, 1),
            (11264, 0xdbaa664cd6a1e329),
            (0xdbaa664cd6a1e329, 0xdbaa664cd6a1e329),
        ),
        row(
            P::ASTC4x4LDR,
            astc4,
            T::Type2D,
            [257, 300],
            (9, 1),
            (251520, 0x72d9c6368399c396),
            (0x72d9c6368399c396, 0x2169ff70ee34c396),
        ),
        row(
            P::ASTC5x4LDR,
            (16, [5, 4]),
            T::Type2D,
            [260, 70],
            (9, 1),
            (44160, 0xbdaaba6680137944),
            (0xbdaaba6680137944, 0xbdaaba6680137944),
        ),
    ]
}

impl Blocks {
    /// Its span's hashes as `level`'s Metal lays it out: by `replaceRegion`, and by a blit.
    fn spans(&self, level: MetalLevel) -> (u64, u64) {
        match level {
            MetalLevel::MacOS26_7 => self.m3,
            MetalLevel::MacOS27_0_1 => (self.m4, self.m4),
            other => panic!("no spans measured at {other}"),
        }
    }

    /// Its descriptor in a `Shared` heap, read by shaders.
    fn descriptor(&self) -> TextureDescriptor {
        let [w, h] = self.extent;
        let mut descriptor = TextureDescriptor::texture_2d(self.format, w, h, false);
        descriptor.texture_type = self.texture_type;
        descriptor.mipmap_level_count = self.levels;
        descriptor.array_length = self.array;
        descriptor.resource_options = shared();
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor
    }

    /// Its slices (six a cube's).
    fn slices(&self) -> u64 {
        match self.texture_type {
            TextureType::Cube => 6 * self.array,
            _ => self.array,
        }
    }

    /// Level `level`'s texels and blocks, across and down.
    fn level(&self, level: u64) -> ([u64; 2], [u64; 2]) {
        let texels = self.extent.map(|side| (side >> level).max(1));
        let blocks = [0, 1].map(|axis| texels[axis].div_ceil(self.block[axis]));
        (texels, blocks)
    }

    /// Each slice's levels' blocks in turn, numbered as the M4's probe wrote them (unit `j` of
    /// block `e`, 4 bytes, holds `e · (B / 4) + j + 1`): slice, level and its packed rows.
    fn numbered(&self) -> Vec<(u64, u64, Vec<u8>)> {
        let per = self.unit / 4;
        let mut element = 0u64;
        let mut out = Vec::new();
        for slice in 0..self.slices() {
            for level in 0..self.levels {
                let (_, [bw, bh]) = self.level(level);
                let mut bytes = Vec::new();
                for _ in 0..bw * bh {
                    for j in 0..per {
                        bytes.extend_from_slice(&((element * per + j + 1) as u32).to_le_bytes());
                    }
                    element += 1;
                }
                out.push((slice, level, bytes));
            }
        }
        out
    }
}

/// FNV-1a over `bytes`.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}

/// A block-compressed texture in a heap over host pages lies as its device's Metal level lays it
/// out (the M4's under 27.0.1, `MetalLevel::MacOS27_0_1`; the M3 Max's under 26.7,
/// `MetalLevel::MacOS26_7`; each level a device naming it, `DeviceOptions::metal_level`): its
/// answer holds Metal's size; `replaceRegion` of every block leaves the span the level's CPU's
/// bytes, and a blit of each level the level's GPU's bytes (their hashes; the M3 Max's GPU counts
/// the 513 x 300 `BC1_RGBA` and 257 x 300 `ASTC_4x4` level 1's tiles a row otherwise than its
/// CPU); `getBytes` reads each level back from the CPU's writes of the CPU's bytes. Counted tiled,
/// with no layout notice. A device whose image of the format holds other bytes than its blocks (a
/// format it decodes) announces `heap-texture-untiled-layout` instead (each row's notice: the
/// device asks for each, `DeviceOptions::announce_each`).
#[test]
fn a_host_heap_s_block_compressed_texture_lies_as_its_level_s_metal_lays_it_out() {
    for metal_level in MetalLevel::ALL {
        let each = |options: &mut DeviceOptions| {
            options.announce_each = HEAP_LAYOUTS.into();
            options.metal_level = Some(metal_level);
        };
        on_device_with(each, |device, messages| {
            block_rows_lie_as_their_level_lays_them_out(device, messages, metal_level);
        });
    }
}

/// `a_host_heap_s_block_compressed_texture_lies_as_its_level_s_metal_lays_it_out` on `device`,
/// whose Metal level is `metal_level`.
fn block_rows_lie_as_their_level_lays_them_out(
    device: &Device,
    messages: &Messages,
    metal_level: MetalLevel,
) {
    assert_eq!(device.properties().metal_level, metal_level);
    let queue = device.new_command_queue().unwrap();
    let mut tiled = 0;
    for row in block_rows() {
        let what = format!(
            "{metal_level} {:?} {:?} {:?}",
            row.format, row.texture_type, row.extent
        );
        let (cpu_span, gpu_span) = row.spans(metal_level);
        let mapping = Mapping::anonymous(HEAP);
        let Some(heap) = heap_over(device, &mapping) else {
            return;
        };
        let descriptor = row.descriptor();
        // A format the device has no image of is refused before any layout, and an ASTC texture
        // on a device without ASTC (`RuntimeGap::DecodedHeapTexture`); an ETC2 or EAC texture on a
        // device without them lies in a carrier of the same blocks (`etc.rs`).
        let answer = match device.heap_texture_size_and_align(&descriptor) {
            Ok(answer) => answer,
            Err(
                error @ (Error::Unsupported(_)
                | Error::NotImplemented(RuntimeGap::DecodedHeapTexture)),
            ) => {
                eprintln!("HEAPLAYOUT {what}: refused {error:?}");
                continue;
            }
            Err(error) => panic!("{what}: {error:?}"),
        };
        assert!(
            answer.size >= row.size,
            "{what}: {answer:?} holds {}",
            row.size
        );
        messages.take_notices();
        let before = device.work_counts();
        let texture = match heap.new_texture_at_offset(&descriptor, OFFSET) {
            Ok(texture) => texture,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("HEAPLAYOUT {what}: refused {error:?}");
                continue;
            }
            Err(error) => panic!("{what}: {error:?}"),
        };
        let counted = layouts_counted(&before, &device.work_counts());
        let notices = layout_notices(messages);
        match counted {
            [1, 0, 0, 0] => assert_eq!(notices, [], "{what}"),
            [0, 0, 0, 1] => {
                let kind = Degradation::HeapTextureUntiledLayout;
                assert_eq!(notices, [(kind, Some(texture.object_id()))], "{what}");
                eprintln!("HEAPLAYOUT {what}: the device's image holds other bytes");
                continue;
            }
            other => panic!("{what}: counted {other:?}"),
        }
        let span = OFFSET as usize..(OFFSET + row.size) as usize;
        let numbered = row.numbered();

        // `replaceRegion` of every level of every slice: the level's CPU's bytes.
        mapping.fill(0);
        for (slice, level, bytes) in &numbered {
            let ([w, h], [bw, _]) = row.level(*level);
            let region = Region::new_2d(0, 0, w, h);
            texture
                .replace_region(region, *level, *slice, bytes, bw * row.unit, 0)
                .unwrap();
        }
        let laid = mapping.read(span.clone());
        assert_eq!(fnv(&laid), cpu_span, "{what}: replaceRegion's bytes");

        // A blit of each level writes them as the level's GPU lays them out, over bytes of zeros.
        mapping.fill(0);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let mut sources = Vec::new();
        for (slice, level, bytes) in &numbered {
            let ([w, h], [bw, _]) = row.level(*level);
            let source = buffer_with(device, bytes).unwrap();
            let (size, at) = (Size::new(w, h, 1), Origin::new(0, 0, 0));
            let (rows, image) = (bw * row.unit, bytes.len() as u64);
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                rows,
                image,
                size,
                &texture,
                *slice,
                *level,
                at,
                BlitOption::None,
            )
            .unwrap();
            sources.push(source);
        }
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(
            fnv(&mapping.read(span.clone())),
            gpu_span,
            "{what}: a blit's bytes"
        );

        // The CPU's writes of its bytes are the blocks `getBytes` reads.
        mapping.fill(0);
        mapping.write(OFFSET as usize, &laid);
        for (slice, level, bytes) in &numbered {
            let ([w, h], [bw, _]) = row.level(*level);
            let region = Region::new_2d(0, 0, w, h);
            let mut back = vec![0u8; bytes.len()];
            texture
                .get_bytes(&mut back, bw * row.unit, 0, region, *level, *slice)
                .unwrap();
            assert!(
                back == *bytes,
                "{what}: getBytes of slice {slice} level {level}"
            );
        }
        tiled += 1;
        drop((sources, texture, heap));
        drop(mapping);
        let apart = if cpu_span == gpu_span {
            ""
        } else {
            ", its blit's apart"
        };
        eprintln!(
            "HEAPLAYOUT {what}: as its level lays it out{apart}, size {}",
            answer.size
        );
    }
    assert!(
        tiled > 0,
        "{metal_level}: no block-compressed texture tiled"
    );
}

/// A 4-sample 64 x 64 `RGBA8Unorm` render target in a heap over host pages: with
/// `allowGPUOptimizedContents`, which Metal keeps losslessly compressed, made in memory of its own,
/// counted and announced (`heap-texture-lossless-layout`, the first of its kind, with its ID); the
/// heap's detach refused while it lives (`NotImplemented(DetachTextures)`); a pass clears it and
/// resolves it into a texture that reads the clear colour in every texel, and the heap's bytes
/// keep every byte they held. Without it, which Metal keeps uncompressed, refused
/// `NotImplemented(HostHeapUncompressedMultisample)`. The size query answers at least Metal's for
/// each on every device (66048 and 65536 bytes at 128; and measured for the
/// uncompressed one where the device cannot hold its samples in the heap's bytes, an NVIDIA GPU's:
/// no optimal image in host-visible memory, no multisample linear image).
#[test]
fn a_host_heap_s_compressed_multisample_texture_lies_apart_from_its_bytes() {
    on_device(|device, messages| {
        if device.properties().supports_texture_sample_count(4) != Ok(true) {
            eprintln!("HEAPLAYOUT multisample: the device takes no 4-sample texture");
            return;
        }
        let mapping = Mapping::anonymous(HEAP);
        let Some(heap) = heap_over(device, &mapping) else {
            return;
        };
        messages.take_notices();
        let mut compressed = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        compressed.texture_type = TextureType::Type2DMultisample;
        compressed.sample_count = 4;
        compressed.usage = TextureUsage::RENDER_TARGET;
        compressed.resource_options = shared();
        let mut plain = compressed;
        plain.allow_gpu_optimized_contents = false;
        for (descriptor, size) in [(&compressed, 66048), (&plain, 65536)] {
            match device.heap_texture_size_and_align(descriptor) {
                Ok(answer) => assert!(answer.size >= size && answer.align >= 128, "{answer:?}"),
                Err(error) => panic!("the size query: {error:?}"),
            }
        }
        match heap.new_texture_at_offset(&plain, OFFSET) {
            Err(Error::NotImplemented(RuntimeGap::HostHeapUncompressedMultisample)) => {}
            other => panic!("an uncompressed multisample texture: {other:?}"),
        }
        let before = named(HEAP, 11);
        mapping.write(0, &before);
        let kind = Degradation::HeapTextureLosslessLayout;
        let objects = device
            .degraded_objects(kind)
            .expect("a kind counted per texture");
        let texture = match heap.new_texture_at_offset(&compressed, OFFSET) {
            Ok(texture) => texture,
            Err(error @ Error::Unsupported(_)) => {
                eprintln!("HEAPLAYOUT multisample: refused {error:?}");
                return;
            }
            Err(error) => panic!("a compressed multisample texture: {error:?}"),
        };
        assert_eq!(
            layout_notices(messages),
            [(kind, Some(texture.object_id()))]
        );
        assert_eq!(device.degraded_objects(kind), Some(objects + 1));
        match heap.detach_backing() {
            Err(Error::NotImplemented(RuntimeGap::DetachTextures)) => {}
            other => panic!("a detach with a multisample texture: {other:?}"),
        }
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 64, 64, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = shared();
        let resolved = device.new_texture(&target).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(texture.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
        color.store_action = StoreAction::MultisampleResolve;
        color.resolve_texture = Some(resolved.clone());
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let mut texels = vec![0u8; 64 * 64 * 4];
        let region = Region::new_2d(0, 0, 64, 64);
        resolved
            .get_bytes(&mut texels, 64 * 4, 0, region, 0, 0)
            .unwrap();
        assert!(
            texels.chunks(4).all(|texel| texel == [51, 102, 153, 255]),
            "resolved: {:?}",
            &texels[..16]
        );
        let after = mapping.read(0..HEAP);
        let differ = (0..HEAP).filter(|&i| after[i] != before[i]).count();
        assert_eq!(differ, 0, "the heap's bytes changed");
        drop((texture, resolved, heap));
        drop(mapping);
        eprintln!("HEAPLAYOUT multisample: compressed made apart, uncompressed refused");
    });
}
