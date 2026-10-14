//! Device tests of textures over no-copy buffers at a guest browser's padded row pitches, on the
//! device the cfg selects, under the validation layer. Each texture is made as the consumer makes
//! a surface's (`new_texture_with_iosurface_plane`, render target and shader read usage) over a
//! no-copy buffer of private anonymous pages, at one of [`LAYOUTS`]: the browser's 1307x512
//! `BGRA8Unorm` tile with rows of 5248 bytes (its texels 5228), and the same tile and others with
//! rows padded to 16, 256 and 512 bytes, a small tile, a full-width strip, and an `RGBA16Float`
//! tile, so the rows' padding takes each power of two from 16 to 512 bytes.
//!
//! Whatever form the device gives the texture (its own linear image, a `DRM_FORMAT_MOD_LINEAR`
//! image laid out at Metal's row pitch, or a shadow), every step reads and writes Metal's bytes at
//! Metal's places, and no write reaches the bytes between the rows:
//!
//! - **reads**: a blit out, `get_bytes`, a fragment function's `read`, and its `sample` through a
//!   nearest sampler give the caller's texels; a linear sampler gives what it gives over a private
//!   texture of the same texels, byte for byte; the caller's later writes are what the next read
//!   reads;
//! - **writes**: a blit of a buffer into the texture, `replace_region` of a box at an odd origin,
//!   and a blit of a private texture's box into it land in the pages;
//! - **renders**: a pass that loads, blends a rectangle and stores, and one that clears, blends and
//!   stores, leave in the pages what they leave in a private texture of the same texels;
//! - **blits out**: a box at an odd origin copied into a private texture holds the pages' texels.
//!
//! Printed per layout (`PITCHED-SURFACE`): the form, the images the making counted
//! (`WorkCounts::pitched_host_images`, `shadowed_textures`), and a shadow's reason. On ANV, whose
//! own linear images pack rows tight, every layout is an image at Metal's row pitch, never a
//! shadow.

use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::no_copy_textures::{blending, floats, staged};
use super::{validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::DriverId;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandQueue, Degradation, Device, IoSurfacePlane, LoadAction,
    Origin, PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};

/// A surface's plane: its name, pixel format and texel bytes, width, height and bytes per row.
#[derive(Clone, Copy, Debug)]
struct Layout {
    name: &'static str,
    pf: PixelFormat,
    texel: u64,
    w: u64,
    h: u64,
    bpr: u64,
}

impl Layout {
    const fn bgra8(name: &'static str, w: u64, h: u64, bpr: u64) -> Layout {
        Layout {
            name,
            pf: PixelFormat::BGRA8Unorm,
            texel: 4,
            w,
            h,
            bpr,
        }
    }

    /// The bytes of a row's texels.
    fn row(&self) -> u64 {
        self.w * self.texel
    }

    /// The bytes the texels span, the last row's padding with them.
    fn len(&self) -> usize {
        (self.bpr * self.h) as usize
    }
}

/// The layouts, each its own test.
const LAYOUTS: [Layout; 8] = [
    // The browser's tile: rows padded to 64 bytes (5248 = 41 x 128).
    Layout::bgra8("1307x512/5248", 1307, 512, 5248),
    // Rows padded to 16 bytes (5232 = 327 x 16).
    Layout::bgra8("1307x512/5232", 1307, 512, 5232),
    // Rows padded to 256 bytes (5376 = 21 x 256).
    Layout::bgra8("1307x512/5376", 1307, 512, 5376),
    // Rows padded past 256, to 512 bytes (5632 = 11 x 512).
    Layout::bgra8("1307x512/5632", 1307, 512, 5632),
    // A small tile, rows padded to 64 bytes (1216 = 19 x 64).
    Layout::bgra8("300x200/1216", 300, 200, 1216),
    // A full-width strip, rows padded to 256 bytes (7680 = 15 x 512; 7652 of texels).
    Layout::bgra8("1913x96/7680", 1913, 96, 7680),
    // A narrow tile, rows padded to 32 bytes (1056 = 33 x 32).
    Layout::bgra8("257x64/1056", 257, 64, 1056),
    // An `RGBA16Float` tile, rows padded to 64 bytes (5312 = 83 x 64).
    Layout {
        name: "657x128/5312 RGBA16Float",
        pf: PixelFormat::RGBA16Float,
        texel: 8,
        w: 657,
        h: 128,
        bpr: 5312,
    },
];

/// Texels of `layout` for `n` bytes, varied by `k`, each a value every read, sample and blend
/// keeps: `RGBA16Float` halves in [2^-7, 1), other formats any bits.
fn texels_of(layout: &Layout, n: usize, k: usize) -> Vec<u8> {
    if layout.texel == 8 {
        (0..n / 2)
            .flat_map(|i| (0x2000u16 + ((i * 97 + k * 13) % 0x1c00) as u16).to_le_bytes())
            .collect()
    } else {
        counting(n, k)
    }
}

/// The texels of `bytes`, rows `layout.bpr` apart: what a texture over them reads and writes, row
/// after row.
fn rows(layout: &Layout, bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(layout.bpr as usize)
        .take(layout.h as usize)
        .flat_map(|r| &r[..layout.row() as usize])
        .copied()
        .collect()
}

/// `texels` (rows of `layout.row()` bytes) laid into `bytes` at rows `layout.bpr` apart, the bytes
/// between the rows left.
fn lay(layout: &Layout, bytes: &mut [u8], texels: &[u8]) {
    let row = layout.row() as usize;
    for (y, texels) in texels.chunks(row).enumerate() {
        let at = y * layout.bpr as usize;
        bytes[at..at + row].copy_from_slice(texels);
    }
}

/// The texels (`texel` bytes each) of `a` and `b` that differ, and the first.
fn differing(a: &[u8], b: &[u8], texel: u64) -> (usize, Option<usize>) {
    let mut count = 0;
    let mut first = None;
    for (i, (x, y)) in a
        .chunks_exact(texel as usize)
        .zip(b.chunks_exact(texel as usize))
        .enumerate()
    {
        if x != y {
            count += 1;
            first.get_or_insert(i);
        }
    }
    (count, first)
}

/// The bytes of a `Shared` buffer, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// Commit `command_buffer` and wait for it, which must complete without an error.
fn run(command_buffer: &metal2vulkan_runtime::CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// `texture`'s texels, blitted out into a new `Shared` buffer at tight rows.
fn blit_out(queue: &CommandQueue, layout: &Layout, texture: &Texture) -> Vec<u8> {
    let (row, len) = (layout.row(), layout.row() * layout.h);
    let out = queue
        .device()
        .new_buffer(len, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(layout.w, layout.h, 1),
        &out,
        0,
        row,
        len,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    bytes_of(&out)
}

/// `texels` (tight rows) blitted into `texture` from a `Shared` buffer.
fn blit_in(queue: &CommandQueue, layout: &Layout, texture: &Texture, texels: &[u8]) {
    let source = staged(queue.device(), texels);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        layout.row(),
        texels.len() as u64,
        Size::new(layout.w, layout.h, 1),
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

/// A blit of `size` texels from `source` at `from` into `destination` at `to`.
fn blit_box(
    queue: &CommandQueue,
    (source, from): (&Texture, Origin),
    (destination, to): (&Texture, Origin),
    size: Size,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_texture(source, 0, 0, from, size, destination, 0, 0, to)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// A private texture of `layout`'s format and size holding `texels` (tight rows).
fn private_with(queue: &CommandQueue, layout: &Layout, texels: &[u8]) -> Texture {
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let texture = queue.device().new_texture(&d).unwrap();
    blit_in(queue, layout, &texture, texels);
    texture
}

/// The owned fixture's `tob_rect` over the whole target with `fragment` (`tob_read` or
/// `tob_sample`), into one attachment of `pf`.
fn reading(device: &Device, pf: PixelFormat, fragment: &str) -> RenderPipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble(&engine_dir().join("tob_renders.ll")).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut d = RenderPipelineDescriptor::new(function("tob_rect"));
    d.fragment = Some(function(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// What a draw of `pipeline` over a whole target of `layout`'s format and size reads from
/// `texture` (through `sampler`, where given): the target's texels, tight rows.
fn drawn(
    queue: &CommandQueue,
    layout: &Layout,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let device = queue.device();
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::RENDER_TARGET;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let corners = staged(device, &floats([-1.0, -1.0, 1.0, 1.0]));
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    blit_out(queue, layout, &target)
}

/// A pass over `texture` that loads it (clears it to (0.1, 0.2, 0.3, 0.4) where `clear`), blends
/// `color` over the rectangle `corners` and stores.
fn blend_pass(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    clear: bool,
    (corners, color): ([f32; 4], [f32; 4]),
) {
    let device = queue.device();
    let corners = staged(device, &floats(corners));
    let color = staged(device, &floats(color));
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = if clear {
        LoadAction::Clear
    } else {
        LoadAction::Load
    };
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.1, 0.2, 0.3, 0.4);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// A sampler of `filter`, clamped to the edge.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// Whether the device is ANV, whose own linear images pack rows tight: a texture over host pages
/// with padded rows is there an image at Metal's row pitch.
fn anv(device: &Device) -> bool {
    device.properties().description.driver_id == Some(DriverId::INTEL_OPEN_SOURCE_MESA)
}

/// The module docs' steps over a texture of `layout` over a no-copy buffer: each read gives the
/// pages' texels (or the private texture's answer), each write lands in the pages, and the bytes
/// between the rows stay the caller's.
fn reads_writes_and_renders_as_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let (n, texel) = (layout.len(), layout.texel);
        let len = n.next_multiple_of(GUEST_PAGE);
        let mapping = Mapping::anonymous(len);
        // The pages: texels a blend keeps finite, and the bytes between the rows a pattern of
        // their own, which no write through the texture reaches.
        let mut pages = counting(len, 7);
        lay(
            &layout,
            &mut pages,
            &texels_of(&layout, (layout.row() * layout.h) as usize, 1),
        );
        mapping.write(0, &pages);
        // SAFETY: the mapping outlives the buffer and the texture (dropped first, below).
        let made = unsafe {
            no_copy(
                &device,
                mapping.pointer(),
                len,
                ResourceOptions::default(),
                None,
            )
        };
        let buffer = match made {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                eprintln!("PITCHED-SURFACE: the device imports no host memory: {error:?}");
                return;
            }
            Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
        };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let before = device.work_counts();
        let texture = buffer
            .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, layout.bpr, false))
            .unwrap();
        let counted = device.work_counts();
        let pitched = counted.pitched_host_images - before.pitched_host_images;
        let shadowed = counted.shadowed_textures - before.shadowed_textures;
        // SAFETY: the handle is only compared, while the texture lives.
        let aliased = unsafe { texture.vk_image() }.is_some();
        let what = format!("{} ({})", layout.name, layout.pf.name());
        assert_eq!(
            u64::from(!aliased),
            shadowed,
            "{what}: aliased {aliased}, but {shadowed} shadows counted"
        );
        if anv(&device) {
            assert!(
                aliased && pitched == 1,
                "{what}: on ANV not an image at Metal's row pitch (aliased {aliased}, pitched \
                 images {pitched})"
            );
        }
        // The pages as Metal leaves them, after each step.
        let mut expected = pages.clone();
        let check = |step: &str, expected: &[u8]| {
            let now = mapping.read(0..len);
            let (count, first) = differing(&rows(&layout, &now), &rows(&layout, expected), texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels in the pages differ from Metal's, the first at \
                 texel {first:?}"
            );
            assert!(
                now == expected,
                "{what}, {step}: the bytes between the rows (or past the last) changed"
            );
        };
        let read = |step: &str, ours: &[u8], theirs: &[u8]| {
            let (count, first) = differing(ours, theirs, texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels differ, the first at texel {first:?}"
            );
        };

        // Reads of the caller's bytes as they were before the texture was made.
        let texels = rows(&layout, &expected);
        read("a blit out", &blit_out(&queue, &layout, &texture), &texels);
        let mut got = vec![0u8; texels.len()];
        let whole = Region::new_2d(0, 0, layout.w, layout.h);
        texture
            .get_bytes(&mut got, layout.row(), 0, whole, 0, 0)
            .unwrap();
        read("get_bytes", &got, &texels);
        let reads = reading(&device, layout.pf, "tob_read");
        read(
            "a fragment function's read",
            &drawn(&queue, &layout, &reads, &texture, None),
            &texels,
        );
        let samples = reading(&device, layout.pf, "tob_sample");
        let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
        let linear = sampler(&device, SamplerMinMagFilter::Linear);
        read(
            "a nearest sample",
            &drawn(&queue, &layout, &samples, &texture, Some(&nearest)),
            &texels,
        );
        let private = private_with(&queue, &layout, &texels);
        read(
            "a linear sample, against a private texture's",
            &drawn(&queue, &layout, &samples, &texture, Some(&linear)),
            &drawn(&queue, &layout, &samples, &private, Some(&linear)),
        );
        check("the reads", &expected);

        // The caller's later bytes, read by a sample and a blit.
        let later = texels_of(&layout, texels.len(), 2);
        lay(&layout, &mut expected, &later);
        mapping.write(0, &expected);
        read(
            "a nearest sample of the caller's later bytes",
            &drawn(&queue, &layout, &samples, &texture, Some(&nearest)),
            &later,
        );
        read(
            "a blit out of the caller's later bytes",
            &blit_out(&queue, &layout, &texture),
            &later,
        );

        // Writes: a blit of a buffer, `replace_region` of a box, a blit of a private texture's box.
        let written = texels_of(&layout, later.len(), 3);
        blit_in(&queue, &layout, &texture, &written);
        lay(&layout, &mut expected, &written);
        check("a blit's write", &expected);
        let (bx, by, bw, bh) = (3, 5, layout.w / 2 + 1, layout.h / 3 + 1);
        let boxed = texels_of(&layout, (bw * bh * texel) as usize, 4);
        texture
            .replace_region(Region::new_2d(bx, by, bw, bh), 0, 0, &boxed, bw * texel, 0)
            .unwrap();
        for (y, row) in boxed.chunks((bw * texel) as usize).enumerate() {
            let at = ((by + y as u64) * layout.bpr + bx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("replace_region", &expected);
        let source = private_with(&queue, &layout, &texels_of(&layout, later.len(), 5));
        let (fx, fy, tx, ty) = (7, 1, layout.w / 3, 2);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        blit_box(
            &queue,
            (&source, Origin::new(fx, fy, 0)),
            (&texture, Origin::new(tx, ty, 0)),
            size,
        );
        let from = blit_out(&queue, &layout, &source);
        for y in 0..size.height {
            let start = ((fy + y) * layout.row() + fx * texel) as usize;
            let row = &from[start..start + (size.width * texel) as usize];
            let at = ((ty + y) * layout.bpr + tx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("a blit of a private texture's box", &expected);

        // Renders that store, against a private texture of the same texels.
        let blend = blending(&device, layout.pf);
        let reference = private_with(&queue, &layout, &rows(&layout, &expected));
        let left = ([-1.0, -1.0, 0.0, 1.0], [0.2, 0.4, 0.6, 0.5]);
        let right = ([0.0, -1.0, 1.0, 1.0], [0.9, 0.6, 0.1, 0.25]);
        for (step, clear, rect) in [
            ("a pass that loads, blends and stores", false, left),
            ("a pass that clears, blends and stores", true, right),
        ] {
            blend_pass(&queue, &blend, &reference, clear, rect);
            blend_pass(&queue, &blend, &texture, clear, rect);
            lay(
                &layout,
                &mut expected,
                &blit_out(&queue, &layout, &reference),
            );
            check(step, &expected);
        }

        // A blit of the texture's box out into a private texture.
        let (fx, fy) = (layout.w / 4 + 1, 3);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        let out = private_with(&queue, &layout, &vec![0; later.len()]);
        blit_box(
            &queue,
            (&texture, Origin::new(fx, fy, 0)),
            (&out, Origin::new(0, 0, 0)),
            size,
        );
        let copied = blit_out(&queue, &layout, &out);
        let pages = rows(&layout, &expected);
        for y in 0..size.height {
            let at = (y * layout.row()) as usize;
            let from = ((fy + y) * layout.row() + fx * texel) as usize;
            let width = (size.width * texel) as usize;
            assert!(
                copied[at..at + width] == pages[from..from + width],
                "{what}: a blit of the texture's box out: row {y} differs from the pages'"
            );
        }
        check("the last blit", &expected);
        // A shadow's reason, as the device announced it.
        let reason = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .find(|notice| notice.degradation == Degradation::ShadowedTexture)
            .map(|notice| format!(": {}", notice.reason))
            .unwrap_or_default();
        eprintln!(
            "PITCHED-SURFACE on {}: {what}: {} (pitched host images {pitched}, shadows \
             {shadowed}){reason}",
            device.name(),
            if aliased { "aliased" } else { "shadowed" },
        );
        drop((texture, buffer));
        drop(mapping);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// One test a layout, each its own process: `$name` runs [`LAYOUTS`]`[$index]`.
macro_rules! per_layout {
    ($($name:ident => $index:expr;)*) => {
        $(
            #[test]
            fn $name() {
                reads_writes_and_renders_as_metal(LAYOUTS[$index]);
            }
        )*
    };
}

per_layout! {
    a_browser_tile_at_5248_byte_rows_reads_writes_and_renders_as_metal => 0;
    a_tile_at_16_byte_padded_rows_reads_writes_and_renders_as_metal => 1;
    a_tile_at_256_byte_padded_rows_reads_writes_and_renders_as_metal => 2;
    a_tile_at_512_byte_padded_rows_reads_writes_and_renders_as_metal => 3;
    a_small_tile_reads_writes_and_renders_as_metal => 4;
    a_full_width_strip_reads_writes_and_renders_as_metal => 5;
    a_narrow_tile_reads_writes_and_renders_as_metal => 6;
    a_half_float_tile_reads_writes_and_renders_as_metal => 7;
}
