//! Tiled heap textures whose elements move through their images' unsigned-integer views
//! (`texture/heap_tiled_words.rs`), on the device the cfg selects, under the validation layer:
//! the heap's bytes after each move are the bytes the texture's CPU copies, which lay each element
//! where Metal's tiles put it (`TiledLayout::element`, apart from every GPU move), leave for the
//! same writes.
//!
//! - **Every class with a view** ([`classes`]): the 8-bit four-channel formats (`RGBA8Unorm`,
//!   `BGRA8Unorm`, `RGBA8Uint`), `RGBA16Float`, `RG32Float`, `RGBA32Float` and `R32Float`, as a
//!   one-level 2D texture, a mipmapped one, a 2D array and a cube. The CPU writes every element of
//!   every level and slice (`replaceRegion`) over bytes that name their place; a blit reads each
//!   back (the copy in); blits into parts of two levels and slices (kept and merged) and into
//!   every texel of a one-level texture (copied back) leave the heap's bytes, every byte of them,
//!   as the CPU's writes of the same texels over the same bytes leave them. Each texture counts
//!   whether its moves went through its views (`WorkCounts::heap_tiled_view_moves`, printed
//!   `TILEDWORDS`); a class whose image takes no view on the device moves through staging and
//!   leaves the same bytes.
//! - **Depth and stencil** (`Depth32Float`, `Depth16Unorm`, `Stencil8`, sides under 16 so Metal
//!   keeps them uncompressed): no view, their moves stay staged (no view move counted), and their
//!   blits leave the bytes the CPU model of Metal's tiles names.
//! - **The CPU's write while the command buffer runs**: a blit writes part of each class's
//!   texture, and the CPU writes an element no command touches while the command buffer runs
//!   (copies between two large buffers hold its run open); the heap's bytes hold both writes and
//!   no other change, as Metal keeps them. The GPU times show whether the write landed inside the
//!   run (printed).
//! - **One scratch a queue**: staged textures (`RG8Unorm` and `R8Unorm` arrays, which no image
//!   takes a view of, and a `Depth16Unorm` texture, whose keep moves its image's elements back)
//!   at offsets of one heap, a part of each written by one blit encoder in each of two command
//!   buffers: the queue's one scratch serves both encoders (`WorkCounts::tiled_scratches_made`
//!   and `tiled_scratches_reused`, `texture/heap_tiled_scratch.rs`), and the heap's bytes are
//!   those the CPU's writes leave.

use super::degrade_notices::on_device;
use super::heap_layouts::{named, tiled_offset};
use super::host_memory::{not_importable, run, Mapping};
use super::no_copy_heaps::shared;
use super::{buffer_with, read_buffer, WAIT};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    host_time, BlitOption, CommandQueue, Device, Heap, HeapDescriptor, HeapType, HostMemory,
    Origin, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::time::{Duration, Instant};

/// The heap's bytes.
const HEAP: usize = 4 << 20;
/// Where each texture lies in its heap: a multiple of every alignment Metal answers.
const OFFSET: u64 = 64 << 10;

/// A texture's shape: its format, element bytes, type, level 0's sides, levels and slices.
#[derive(Clone, Copy, Debug)]
struct Shape {
    format: PixelFormat,
    unit: u64,
    texture_type: TextureType,
    extent: [u64; 2],
    levels: u64,
    slices: u64,
}

impl Shape {
    /// Its descriptor in a `Shared` heap: read and written by shaders, so Metal keeps it
    /// uncompressed (a depth or stencil texture keeps sides under 16 instead).
    fn descriptor(&self) -> TextureDescriptor {
        let [w, h] = self.extent;
        let mut descriptor = match self.texture_type {
            TextureType::Cube => TextureDescriptor::texture_cube(self.format, w, self.levels > 1),
            _ => TextureDescriptor::texture_2d(self.format, w, h, self.levels > 1),
        };
        if self.texture_type == TextureType::Type2DArray {
            descriptor.texture_type = TextureType::Type2DArray;
            descriptor.array_length = self.slices;
        }
        descriptor.mipmap_level_count = self.levels;
        descriptor.resource_options = shared();
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        descriptor
    }

    /// Level `level`'s sides.
    fn sides(&self, level: u64) -> [u64; 2] {
        self.extent.map(|side| (side >> level).max(1))
    }

    /// Level `level`'s packed rows: its bytes a row and a slice.
    fn rows(&self, level: u64) -> (u64, u64) {
        let [w, h] = self.sides(level);
        (w * self.unit, w * h * self.unit)
    }

    /// Every level and slice.
    fn subresources(&self) -> Vec<(u64, u64)> {
        (0..self.levels)
            .flat_map(|level| (0..self.slices).map(move |slice| (level, slice)))
            .collect()
    }
}

/// The shapes of `format` (`unit` bytes an element): a one-level 2D texture across tiles, a
/// mipmapped one, a 2D array of three slices, and a cube.
fn shapes_of(format: PixelFormat, unit: u64) -> Vec<Shape> {
    let shape = |texture_type, extent, levels, slices| Shape {
        format,
        unit,
        texture_type,
        extent,
        levels,
        slices,
    };
    vec![
        shape(TextureType::Type2D, [70, 45], 1, 1),
        shape(TextureType::Type2D, [100, 37], 7, 1),
        shape(TextureType::Type2DArray, [33, 17], 1, 3),
        shape(TextureType::Cube, [24, 24], 2, 6),
    ]
}

/// The formats of every class with a view, and their element bytes.
fn classes() -> Vec<(PixelFormat, u64)> {
    vec![
        (PixelFormat::RGBA8Unorm, 4),
        (PixelFormat::BGRA8Unorm, 4),
        (PixelFormat::RGBA8Uint, 4),
        (PixelFormat::R32Float, 4),
        (PixelFormat::RGBA16Float, 8),
        (PixelFormat::RG32Float, 8),
        (PixelFormat::RGBA32Float, 16),
    ]
}

/// Bytes of `len` whose float components, read through any of the classes' formats, are finite
/// (no NaN a conversion could change, though no move converts): each byte below 0x40.
fn finite(len: usize, seed: usize) -> Vec<u8> {
    named(len, seed).into_iter().map(|b| b & 0x3f).collect()
}

/// A placement heap over `mapping`'s [`HEAP`] bytes; `None`, printed, where the device imports no
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
            eprintln!("TILEDWORDS: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_heap_with_bytes_no_copy: {error:?}"),
    }
}

/// Level `level` of slice `slice` of `texture` read back through a blit.
fn blit_out(
    queue: &CommandQueue,
    texture: &Texture,
    shape: &Shape,
    (level, slice): (u64, u64),
) -> Vec<u8> {
    let (row, image) = shape.rows(level);
    let copy = queue
        .device()
        .new_buffer(image, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let [w, h] = shape.sides(level);
    blit.copy_from_texture_to_buffer(
        texture,
        slice,
        level,
        Origin::new(0, 0, 0),
        Size::new(w, h, 1),
        &copy,
        0,
        row,
        image,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&copy, 0, image).unwrap()
}

/// One write of packed `texels` into `region` of level `level` of slice `slice`.
struct Write {
    level: u64,
    slice: u64,
    region: Region,
    texels: Vec<u8>,
}

impl Write {
    /// Its texels' bytes a row.
    fn row(&self, unit: u64) -> u64 {
        self.region.size.width * unit
    }
}

/// Record `writes` as blits into `texture` in `blit`, from buffers kept in `sources`.
fn blit_writes(
    device: &Device,
    blit: &mut metal2vulkan_runtime::BlitCommandEncoder,
    texture: &Texture,
    unit: u64,
    writes: &[Write],
    sources: &mut Vec<metal2vulkan_runtime::Buffer>,
) {
    for write in writes {
        let source = buffer_with(device, &write.texels).unwrap();
        let size = write.region.size;
        let origin = write.region.origin;
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            write.row(unit),
            write.texels.len() as u64,
            Size::new(size.width, size.height, 1),
            texture,
            write.slice,
            write.level,
            Origin::new(origin.x, origin.y, 0),
            BlitOption::None,
        )
        .unwrap();
        sources.push(source);
    }
}

/// The heap's bytes after the CPU writes `writes` into `texture` (`replaceRegion`) over `held`,
/// the heap's bytes before them; the mapping is left as it was found.
fn cpu_writes(
    mapping: &Mapping,
    texture: &Texture,
    unit: u64,
    held: &[u8],
    writes: &[Write],
) -> Vec<u8> {
    let now = mapping.read(0..HEAP);
    mapping.write(0, held);
    for write in writes {
        texture
            .replace_region(
                write.region,
                write.level,
                write.slice,
                &write.texels,
                write.row(unit),
                0,
            )
            .unwrap();
    }
    let laid = mapping.read(0..HEAP);
    mapping.write(0, &now);
    laid
}

/// That `got` and `want`, the heap's bytes, are alike, naming the first byte that differs.
fn assert_bytes(got: &[u8], want: &[u8], what: &str) {
    let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        differ.is_empty(),
        "{what}: {} bytes differ, the first at {:?} ({:?} where {:?})",
        differ.len(),
        differ.first(),
        differ.first().map(|&i| got[i]),
        differ.first().map(|&i| want[i]),
    );
}

/// Each class's shapes (the module docs): the CPU's writes read back by blits, then blits into
/// parts of two subresources and into every texel of a one-level texture, each leaving the bytes
/// the CPU's writes of the same texels leave.
#[test]
fn every_class_moves_its_elements_where_its_cpu_copies_lay_them() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        for (format, unit) in classes() {
            for shape in shapes_of(format, unit) {
                let what = format!(
                    "{format:?} {:?} {:?} x{} levels x{} slices",
                    shape.texture_type, shape.extent, shape.levels, shape.slices
                );
                let mapping = Mapping::anonymous(HEAP);
                let Some(heap) = heap_over(device, &mapping) else {
                    return;
                };
                mapping.write(0, &finite(HEAP, 11));
                let before = device.work_counts();
                let texture = heap
                    .new_texture_at_offset(&shape.descriptor(), OFFSET)
                    .unwrap();
                // The CPU's elements of every level and slice, read back by blits.
                for (k, &(level, slice)) in shape.subresources().iter().enumerate() {
                    let [w, h] = shape.sides(level);
                    let (row, image) = shape.rows(level);
                    let texels = finite(image as usize, 20 + k);
                    let region = Region::new_2d(0, 0, w, h);
                    texture
                        .replace_region(region, level, slice, &texels, row, 0)
                        .unwrap();
                    assert_eq!(
                        blit_out(&queue, &texture, &shape, (level, slice)),
                        texels,
                        "{what}: level {level} slice {slice} read back"
                    );
                }
                let held = mapping.read(0..HEAP);
                // Parts of level 0 of slice 0 and of the last level of the last slice.
                let last = (shape.levels - 1, shape.slices - 1);
                let part = |(level, slice): (u64, u64), seed| {
                    let [w, h] = shape.sides(level);
                    let (x, y) = ((w / 3).min(w - 1), (h / 2).min(h - 1));
                    let (pw, ph) = ((w - x).min(3), (h - y).min(2));
                    Write {
                        level,
                        slice,
                        region: Region::new_2d(x, y, pw, ph),
                        texels: finite((pw * ph * unit) as usize, seed),
                    }
                };
                let writes = vec![part((0, 0), 40), part(last, 41)];
                let want = cpu_writes(&mapping, &texture, unit, &held, &writes);
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                let mut sources = Vec::new();
                blit_writes(device, &mut blit, &texture, unit, &writes, &mut sources);
                blit.end_encoding().unwrap();
                run(&command_buffer);
                assert_bytes(&mapping.read(0..HEAP), &want, &format!("{what}: parts"));
                // Every texel of a one-level, one-slice texture: copied back, not merged.
                if shape.levels == 1 && shape.slices == 1 {
                    let held = mapping.read(0..HEAP);
                    let [w, h] = shape.extent;
                    let whole = vec![Write {
                        level: 0,
                        slice: 0,
                        region: Region::new_2d(0, 0, w, h),
                        texels: finite((w * h * unit) as usize, 50),
                    }];
                    let want = cpu_writes(&mapping, &texture, unit, &held, &whole);
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit_writes(device, &mut blit, &texture, unit, &whole, &mut sources);
                    blit.end_encoding().unwrap();
                    run(&command_buffer);
                    assert_bytes(
                        &mapping.read(0..HEAP),
                        &want,
                        &format!("{what}: every texel"),
                    );
                }
                let moves =
                    device.work_counts().heap_tiled_view_moves - before.heap_tiled_view_moves;
                println!(
                    "TILEDWORDS on {}: {what}: {}",
                    device.name(),
                    match moves {
                        0 => "staged".to_owned(),
                        n => format!("{n} moves through its views"),
                    }
                );
                drop((texture, heap));
                drop(mapping);
            }
        }
    });
}

/// Depth and stencil textures in Metal's tiles take no view: their moves stay staged, and a
/// blit's texels land at the elements Metal's tiles name, no other byte changing.
#[test]
fn depth_and_stencil_textures_keep_their_staged_moves() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let formats = [
            (PixelFormat::Depth32Float, 4),
            (PixelFormat::Depth16Unorm, 2),
            (PixelFormat::Stencil8, 1),
        ];
        for (format, unit) in formats {
            let extent = [40u64, 12];
            let what = format!("{format:?} {extent:?}");
            let mapping = Mapping::anonymous(HEAP);
            let Some(heap) = heap_over(device, &mapping) else {
                return;
            };
            let held = finite(HEAP, 13);
            mapping.write(0, &held);
            let [w, h] = extent;
            let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
            descriptor.resource_options = shared();
            descriptor.usage = TextureUsage::SHADER_READ;
            let before = device.work_counts();
            let Ok(texture) = heap.new_texture_at_offset(&descriptor, OFFSET) else {
                println!(
                    "TILEDWORDS on {}: {what}: not made in the heap",
                    device.name()
                );
                continue;
            };
            // Depth texels within [0, 1] (a float's bytes below 0x3f80_0000), stencil any.
            let texels: Vec<u8> = finite((w * h * unit) as usize, 30)
                .chunks(4)
                .flat_map(|c| {
                    let mut c = c.to_vec();
                    if let (PixelFormat::Depth32Float, Some(top)) = (format, c.get_mut(3)) {
                        *top &= 0x3e;
                    }
                    c
                })
                .collect();
            let source = buffer_with(device, &texels).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                w * unit,
                texels.len() as u64,
                Size::new(w, h, 1),
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            run(&command_buffer);
            let mut want = held.clone();
            for y in 0..h {
                for x in 0..w {
                    let at = (OFFSET + tiled_offset(unit, extent, x, y)) as usize;
                    let from = ((y * w + x) * unit) as usize;
                    want[at..at + unit as usize]
                        .copy_from_slice(&texels[from..from + unit as usize]);
                }
            }
            assert_bytes(&mapping.read(0..HEAP), &want, &what);
            let moves = device.work_counts().heap_tiled_view_moves - before.heap_tiled_view_moves;
            assert_eq!(moves, 0, "{what}: no view");
            println!(
                "TILEDWORDS on {}: {what}: staged, in Metal's tiles",
                device.name()
            );
            drop((source, texture, heap));
            drop(mapping);
        }
    });
}

/// The bytes the copies that hold the command buffer's run open move each time, and the most
/// copies one command buffer takes.
const HOLD_BYTES: u64 = 16 << 20;
const MOST_HOLDS: u64 = 256;
/// The GPU time the command buffer is grown toward, and the attempts at a write inside it.
const TARGET: Duration = Duration::from_millis(60);
const ATTEMPTS: usize = 4;

/// Each class's one-level texture: a blit writes a part of it, then copies between two large
/// buffers hold the command buffer's run open while the CPU writes an element no command touches;
/// the heap's bytes hold both writes and no other change (the module docs).
#[test]
fn a_cpu_write_beside_a_blit_s_part_while_the_command_buffer_runs_stays() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let hold = [
            device
                .new_buffer(HOLD_BYTES, ResourceOptions::private())
                .unwrap(),
            device
                .new_buffer(HOLD_BYTES, ResourceOptions::private())
                .unwrap(),
        ];
        let mut holds = 4;
        let mut ran = Duration::from_millis(5);
        for (format, unit) in classes() {
            let shape = shapes_of(format, unit)[0];
            let [w, h] = shape.extent;
            let what = format!("{format:?} {:?}", shape.extent);
            let cpu_element = (w - 2, h - 3);
            let cpu_bytes = finite(unit as usize, 61);
            let mut inside = false;
            for attempt in 1..=ATTEMPTS {
                let mapping = Mapping::anonymous(HEAP);
                let Some(heap) = heap_over(device, &mapping) else {
                    return;
                };
                mapping.write(0, &finite(HEAP, 12));
                let texture = heap
                    .new_texture_at_offset(&shape.descriptor(), OFFSET)
                    .unwrap();
                let held = mapping.read(0..HEAP);
                let writes = vec![Write {
                    level: 0,
                    slice: 0,
                    region: Region::new_2d(1, 1, 3, 2),
                    texels: finite((6 * unit) as usize, 60),
                }];
                let mut want = cpu_writes(&mapping, &texture, unit, &held, &writes);
                let at = (OFFSET + tiled_offset(unit, shape.extent, cpu_element.0, cpu_element.1))
                    as usize;
                want[at..at + unit as usize].copy_from_slice(&cpu_bytes);
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                let mut sources = Vec::new();
                blit_writes(device, &mut blit, &texture, unit, &writes, &mut sources);
                for k in 0..holds {
                    let (from, to) = (&hold[k % 2], &hold[1 - k % 2]);
                    blit.copy_from_buffer_to_buffer(from, 0, to, 0, HOLD_BYTES)
                        .unwrap();
                }
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                let deadline = Instant::now() + WAIT;
                while command_buffer.kernel_end_time() == 0.0 {
                    assert!(Instant::now() < deadline, "the batch was never submitted");
                    std::thread::yield_now();
                }
                std::thread::sleep(ran / 3);
                let wrote = host_time();
                mapping.write(at, &cpu_bytes);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None, "{what}");
                assert_bytes(
                    &mapping.read(0..HEAP),
                    &want,
                    &format!("{what}: attempt {attempt}"),
                );
                let (Ok(start), Ok(end)) = (
                    command_buffer.gpu_start_time(),
                    command_buffer.gpu_end_time(),
                ) else {
                    println!(
                        "TILEDWORDS on {}: {what}: the device times no command buffer; both \
                         writes stayed",
                        device.name()
                    );
                    break;
                };
                inside = start < wrote && wrote < end;
                ran = Duration::from_secs_f64((end - start).max(1e-4));
                println!(
                    "TILEDWORDS on {}: {what}: attempt {attempt}, {holds} copies, GPU run {:.2} \
                     ms, the CPU's write at {:+.2} ms from its start (inside {inside}); both \
                     writes stayed",
                    device.name(),
                    (end - start) * 1e3,
                    (wrote - start) * 1e3,
                );
                drop((sources, texture, heap));
                drop(mapping);
                if inside {
                    break;
                }
                if ran < TARGET {
                    let grow = (TARGET.as_secs_f64() / ran.as_secs_f64()).ceil() as u64;
                    holds = (holds * grow.max(2) as usize).min(MOST_HOLDS as usize);
                }
            }
            if !inside {
                println!(
                    "TILEDWORDS on {}: {what}: no attempt's write landed inside the GPU's run",
                    device.name()
                );
            }
        }
    });
}

/// Where the scratch lane's textures lie in its heap: a multiple of [`OFFSET`] apart, each past
/// the one before.
fn scratch_offset(k: usize) -> u64 {
    OFFSET * (1 + 2 * k as u64)
}

/// Textures of formats no image takes a view of (`RG8Unorm`, `R8Unorm`), whose moves are staged
/// on every device, and a `Depth16Unorm` one, whose keep moves its image's elements back, at
/// offsets of one heap: in each of two command buffers one blit encoder writes a part of each,
/// every staged move, keep and merge going through the queue's one scratch, made in the first
/// and taken as it was in the second (`WorkCounts::tiled_scratches_made` and
/// `tiled_scratches_reused`), and the heap's bytes are those the CPU's writes of the same texels
/// leave (the depth texture's at the elements Metal's tiles name).
#[test]
fn staged_textures_move_through_their_queue_s_one_scratch() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let mapping = Mapping::anonymous(HEAP);
        let Some(heap) = heap_over(device, &mapping) else {
            return;
        };
        mapping.write(0, &finite(HEAP, 14));
        let array = |format, unit| Shape {
            format,
            unit,
            texture_type: TextureType::Type2DArray,
            extent: [33, 17],
            levels: 1,
            slices: 3,
        };
        let shapes = [
            array(PixelFormat::RG8Unorm, 2),
            array(PixelFormat::RG8Unorm, 2),
            array(PixelFormat::R8Unorm, 1),
            array(PixelFormat::R8Unorm, 1),
        ];
        let colours: Vec<(Shape, Texture)> = (shapes.iter().enumerate())
            .map(|(k, shape)| {
                let texture = heap
                    .new_texture_at_offset(&shape.descriptor(), scratch_offset(k))
                    .unwrap();
                (*shape, texture)
            })
            .collect();
        let (depth_extent, depth_unit) = ([40u64, 12], 2u64);
        let depth_at = scratch_offset(shapes.len());
        let mut descriptor = TextureDescriptor::texture_2d(
            PixelFormat::Depth16Unorm,
            depth_extent[0],
            depth_extent[1],
            false,
        );
        descriptor.resource_options = shared();
        descriptor.usage = TextureUsage::SHADER_READ;
        let depth = heap.new_texture_at_offset(&descriptor, depth_at).ok();
        if depth.is_none() {
            println!(
                "TILEDWORDS on {}: Depth16Unorm {depth_extent:?}: not made in the heap",
                device.name()
            );
        }
        for round in 0..2 {
            let what = format!("round {round}");
            let mut want = mapping.read(0..HEAP);
            let writes: Vec<Vec<Write>> = (colours.iter().enumerate())
                .map(|(k, (shape, texture))| {
                    let part = |slice: u64, seed| {
                        let [w, h] = shape.extent;
                        let (x, y) = (w / 3, h / 2);
                        let (pw, ph) = (5.min(w - x), 3.min(h - y));
                        Write {
                            level: 0,
                            slice,
                            region: Region::new_2d(x, y, pw, ph),
                            texels: finite((pw * ph * shape.unit) as usize, seed),
                        }
                    };
                    let seed = 70 + 10 * round + 2 * k;
                    let writes = vec![part(0, seed), part(shape.slices - 1, seed + 1)];
                    want = cpu_writes(&mapping, texture, shape.unit, &want, &writes);
                    writes
                })
                .collect();
            let before = device.work_counts();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let mut sources = Vec::new();
            for ((shape, texture), writes) in colours.iter().zip(&writes) {
                blit_writes(device, &mut blit, texture, shape.unit, writes, &mut sources);
            }
            if let Some(depth) = &depth {
                let (x0, y0, w, h) = (5u64, 3u64, 9u64, 6u64);
                let texels = finite((w * h * depth_unit) as usize, 90 + round);
                let source = buffer_with(device, &texels).unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    w * depth_unit,
                    texels.len() as u64,
                    Size::new(w, h, 1),
                    depth,
                    0,
                    0,
                    Origin::new(x0, y0, 0),
                    BlitOption::None,
                )
                .unwrap();
                sources.push(source);
                for y in 0..h {
                    for x in 0..w {
                        let element = tiled_offset(depth_unit, depth_extent, x0 + x, y0 + y);
                        let at = (depth_at + element) as usize;
                        let from = ((y * w + x) * depth_unit) as usize;
                        want[at..at + depth_unit as usize]
                            .copy_from_slice(&texels[from..from + depth_unit as usize]);
                    }
                }
            }
            blit.end_encoding().unwrap();
            run(&command_buffer);
            let after = device.work_counts();
            assert_bytes(&mapping.read(0..HEAP), &want, &what);
            let made = after.tiled_scratches_made - before.tiled_scratches_made;
            let reused = after.tiled_scratches_reused - before.tiled_scratches_reused;
            // The queue's first staged move makes its scratch, which the next encoder takes.
            assert_eq!(
                (made, reused),
                (u64::from(round == 0), u64::from(round > 0)),
                "{what}: the queue's one scratch"
            );
            println!(
                "TILEDWORDS on {}: {what}: {} staged textures, the queue's one scratch ({made} \
                 made, {reused} reused); every byte as the CPU's writes leave it",
                device.name(),
                colours.len() + usize::from(depth.is_some()),
            );
            drop((sources, command_buffer));
        }
        drop((colours, depth, heap));
        drop(mapping);
    });
}
