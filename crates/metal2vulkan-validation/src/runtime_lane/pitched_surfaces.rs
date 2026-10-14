//! Device tests of textures over no-copy buffers at a guest browser's padded row pitches, on the
//! device the cfg selects, under the validation layer. Each texture is made as the consumer makes
//! a surface's (`new_texture_with_iosurface_plane`, render target and shader read usage) over a
//! no-copy buffer of private anonymous pages, at one of [`LAYOUTS`]: the browser's 1307x512
//! `BGRA8Unorm` tile with rows of 5248 bytes (its texels 5228), and the same tile and others with
//! rows padded to 16, 256 and 512 bytes, a small tile, a full-width strip, and an `RGBA16Float`
//! tile, so the rows' padding takes each power of two from 16 to 512 bytes; and tiles whose rows
//! lie one after another, of a height that is no multiple of 4, with the pages' later bytes past
//! their last row (a surface's next plane).
//!
//! Whatever form the device gives the texture (its own linear image, a `DRM_FORMAT_MOD_LINEAR`
//! image laid out at Metal's row pitch, or a shadow), every step reads and writes Metal's bytes at
//! Metal's places, and no write reaches the bytes between the rows or past the last. A texture
//! whose rows lie with bytes between them is never a linear image over the pages (Vulkan lets an
//! image's layout transitions, and a render pass's load and store operations, write every byte of
//! its subresource), but on a driver that runs on Metal, whose image is a Metal texture over the
//! buffer:
//!
//! - **reads**: a blit out, `get_bytes`, a fragment function's `read`, and its `sample` through a
//!   nearest sampler give the caller's texels; a linear sampler gives what it gives over a private
//!   texture of the same texels, byte for byte; the caller's later writes are what the next read
//!   reads;
//! - **writes**: a blit of a buffer into the texture, `replace_region` of a box at an odd origin,
//!   and a blit of a private texture's box into it land in the pages;
//! - **renders**: a pass that loads, blends a rectangle and stores, and one that clears, blends and
//!   stores, leave in the pages what they leave in a private texture of the same texels; one that
//!   loads, blends and stores `DontCare` leaves the pages as they were, as Apple's GPU leaves the
//!   attachment's earlier texels;
//! - **blits out**: a box at an odd origin copied into a private texture holds the pages' texels.
//!
//! [`a_fragment_function_s_writes_leave_the_bytes_between_rows`] makes the textures with shader
//! read and write usage instead, and writes every texel through a fragment function's `write`: the
//! pages' texels are a private texture's after the same writes, and no other byte changes.
//!
//! [`a_padded_texture_s_shadow_names_what_may_write_its_padding`] checks the shadow's reason over
//! an ordinary buffer's padded rows: a texture that only reads names its layout transitions, a
//! render target a render pass's load and store operations too.
//!
//! Printed per layout (`PITCHED-SURFACE`): the form, the images the making counted
//! (`WorkCounts::pitched_host_images`, `shadowed_textures`), and a shadow's reason.

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
    SamplerState, ScissorRect, Size, StoreAction, Texture, TextureDescriptor,
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

    /// Whether the rows lie with bytes between them that are not texels.
    fn padded(&self) -> bool {
        self.bpr != self.row()
    }
}

/// The layouts, each its own test.
const LAYOUTS: [Layout; 11] = [
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
    // Rows one after another (a surface's rows are multiples of 16 bytes), 510 of them: a
    // driver's image of 512 reaches past the texels.
    Layout::bgra8("1308x510/5232", 1308, 510, 5232),
    // Rows one after another, of a whole number of cache lines (1200 = 18.75 x 64).
    Layout::bgra8("300x200/1200", 300, 200, 1200),
    // An `RGBA16Float` tile, rows one after another, 126 of them.
    Layout {
        name: "658x126/5264 RGBA16Float",
        pf: PixelFormat::RGBA16Float,
        texel: 8,
        w: 658,
        h: 126,
        bpr: 5264,
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

/// Where `now` differs from `expected` outside `layout`'s texels, in words: how many bytes, the
/// rows they lie in, and the first, with its place in its row and the bytes it holds.
fn padding_changes(layout: &Layout, now: &[u8], expected: &[u8]) -> String {
    let (bpr, row) = (layout.bpr as usize, layout.row() as usize);
    let changed: Vec<usize> = (0..now.len().min(expected.len()))
        .filter(|&i| now[i] != expected[i] && (i % bpr >= row || i / bpr >= layout.h as usize))
        .collect();
    let Some(&first) = changed.first() else {
        return "none outside the texels".to_owned();
    };
    let rows: std::collections::BTreeSet<usize> = changed.iter().map(|i| i / bpr).collect();
    let end = (first + 8).min(now.len());
    format!(
        "{} bytes outside the texels in {} rows (rows {:?}..={:?}), the first at row {} byte {} \
         (a row's texels end at byte {row}), now {:02x?}, was {:02x?}",
        changed.len(),
        rows.len(),
        rows.first(),
        rows.last(),
        first / bpr,
        first % bpr,
        &now[first..end],
        &expected[first..end]
    )
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
/// `color` over the rectangle `corners` and stores (by `store`).
fn blend_pass(
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    (texture, store): (&Texture, StoreAction),
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
    attachment.store_action = store;
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

/// Whether the device runs on Metal (MoltenVK), where a texture over host pages is an image of a
/// Metal texture over the buffer, which keeps the bytes between its rows as Metal does.
fn on_metal(device: &Device) -> bool {
    device.properties().description.driver_id == Some(DriverId::MOLTENVK)
}

/// A texture of `layout` over a no-copy buffer of new anonymous pages, made as a surface's with
/// `usage`: the pages hold a pattern of their own, the texels `texels_of(.., 1)` laid at the rows;
/// with the buffer, the mapping and the pages as written. `None` where the device imports no host
/// memory. Its form is checked against the making's counts: a shadow counted where it is not
/// aliased, and never a linear image over padded rows off Metal.
fn surface(device: &Device, layout: &Layout, usage: TextureUsage) -> Option<Surface> {
    let len = layout.len().next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    // The pages: texels a blend keeps finite, and the bytes between the rows a pattern of
    // their own, which no write through the texture reaches.
    let mut pages = counting(len, 7);
    lay(
        layout,
        &mut pages,
        &texels_of(layout, (layout.row() * layout.h) as usize, 1),
    );
    mapping.write(0, &pages);
    // SAFETY: the mapping outlives the buffer and the texture (dropped first, by `Surface`).
    let made = unsafe {
        no_copy(
            device,
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
            return None;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = usage;
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
    let what = format!("{} ({}, {usage:?})", layout.name, layout.pf.name());
    assert_eq!(
        u64::from(!aliased),
        shadowed,
        "{what}: aliased {aliased}, but {shadowed} shadows counted"
    );
    assert!(
        !(aliased && layout.padded()) || on_metal(device),
        "{what}: rows with bytes between them aliased by a linear image (pitched images \
         {pitched})"
    );
    Some(Surface {
        texture,
        buffer,
        mapping,
        pages,
        len,
        what,
        form: (aliased, pitched, shadowed),
    })
}

/// A texture over a no-copy buffer's pages ([`surface`]).
struct Surface {
    texture: Texture,
    buffer: Buffer,
    mapping: Mapping,
    /// The pages as written before the texture was made.
    pages: Vec<u8>,
    /// The pages' bytes.
    len: usize,
    /// The layout, format and usage, in words.
    what: String,
    /// Whether it is aliased, and the pitched images and shadows its making counted.
    form: (bool, u64, u64),
}

impl Surface {
    /// Check that the pages hold `expected`: its texels (by `layout`), then every byte.
    fn check(&self, layout: &Layout, step: &str, expected: &[u8]) {
        let what = &self.what;
        let now = self.mapping.read(0..self.len);
        let (count, first) = differing(&rows(layout, &now), &rows(layout, expected), layout.texel);
        assert!(
            count == 0,
            "{what}, {step}: {count} texels in the pages differ from Metal's, the first at texel \
             {first:?}"
        );
        assert!(
            now == expected,
            "{what}, {step}: the bytes between the rows (or past the last) changed: {}",
            padding_changes(layout, &now, expected)
        );
    }

    /// Print the form and a shadow's reason (`PITCHED-SURFACE`), and drop the texture, then the
    /// buffer, then the pages.
    fn report(self, device: &Device, messages: &Messages) {
        let (aliased, pitched, shadowed) = self.form;
        let reason = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .find(|notice| notice.degradation == Degradation::ShadowedTexture)
            .map(|notice| format!(": {}", notice.reason))
            .unwrap_or_default();
        eprintln!(
            "PITCHED-SURFACE on {}: {}: {} (pitched host images {pitched}, shadows \
             {shadowed}){reason}",
            device.name(),
            self.what,
            if aliased { "aliased" } else { "shadowed" },
        );
        let Surface {
            texture,
            buffer,
            mapping,
            ..
        } = self;
        drop((texture, buffer));
        drop(mapping);
    }
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
        let texel = layout.texel;
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(surface) = surface(&device, &layout, usage) else {
            return;
        };
        let (texture, what) = (&surface.texture, &surface.what);
        // The pages as Metal leaves them, after each step.
        let mut expected = surface.pages.clone();
        let check = |step: &str, expected: &[u8]| surface.check(&layout, step, expected);
        let read = |step: &str, ours: &[u8], theirs: &[u8]| {
            let (count, first) = differing(ours, theirs, texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels differ, the first at texel {first:?}"
            );
        };

        // Reads of the caller's bytes as they were before the texture was made.
        let texels = rows(&layout, &expected);
        read("a blit out", &blit_out(&queue, &layout, texture), &texels);
        let mut got = vec![0u8; texels.len()];
        let whole = Region::new_2d(0, 0, layout.w, layout.h);
        texture
            .get_bytes(&mut got, layout.row(), 0, whole, 0, 0)
            .unwrap();
        read("get_bytes", &got, &texels);
        let reads = reading(&device, layout.pf, "tob_read");
        read(
            "a fragment function's read",
            &drawn(&queue, &layout, &reads, texture, None),
            &texels,
        );
        let samples = reading(&device, layout.pf, "tob_sample");
        let nearest = sampler(&device, SamplerMinMagFilter::Nearest);
        let linear = sampler(&device, SamplerMinMagFilter::Linear);
        read(
            "a nearest sample",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &texels,
        );
        let private = private_with(&queue, &layout, &texels);
        read(
            "a linear sample, against a private texture's",
            &drawn(&queue, &layout, &samples, texture, Some(&linear)),
            &drawn(&queue, &layout, &samples, &private, Some(&linear)),
        );
        check("the reads", &expected);

        // The caller's later bytes, read by a sample and a blit.
        let later = texels_of(&layout, texels.len(), 2);
        lay(&layout, &mut expected, &later);
        surface.mapping.write(0, &expected);
        read(
            "a nearest sample of the caller's later bytes",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &later,
        );
        read(
            "a blit out of the caller's later bytes",
            &blit_out(&queue, &layout, texture),
            &later,
        );

        // Writes: a blit of a buffer, `replace_region` of a box, a blit of a private texture's box.
        let written = texels_of(&layout, later.len(), 3);
        blit_in(&queue, &layout, texture, &written);
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
            (texture, Origin::new(tx, ty, 0)),
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
            blend_pass(
                &queue,
                &blend,
                (&reference, StoreAction::Store),
                clear,
                rect,
            );
            blend_pass(&queue, &blend, (texture, StoreAction::Store), clear, rect);
            lay(
                &layout,
                &mut expected,
                &blit_out(&queue, &layout, &reference),
            );
            check(step, &expected);
        }
        // A pass that does not store leaves the pages as they were.
        blend_pass(
            &queue,
            &blend,
            (texture, StoreAction::DontCare),
            false,
            left,
        );
        check("a pass that loads, blends and stores DontCare", &expected);

        // A blit of the texture's box out into a private texture.
        let (fx, fy) = (layout.w / 4 + 1, 3);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        let out = private_with(&queue, &layout, &vec![0; later.len()]);
        blit_box(
            &queue,
            (texture, Origin::new(fx, fy, 0)),
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
        surface.report(&device, &messages);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A texture of `layout` over a no-copy buffer whose first use reads it through its image (a
/// fragment function's `read`), before any command reads or writes the pages another way: the
/// image's first move to `GENERAL` (from `UNDEFINED`, as Vulkan has an image of host allocations
/// made, `VUID-VkImageCreateInfo-pNext-01443`) keeps the bytes, which Vulkan does not promise
/// ("doing so will cause the contents of the image's memory to be undefined", docs/FAST-PATHS.md's
/// known gaps). A driver that discards them fails here: the read differs from the texels, or the
/// pages change.
fn first_use_reads_the_bytes(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(surface) = surface(&device, &layout, usage) else {
            return;
        };
        let texels = rows(&layout, &surface.pages);
        let reads = reading(&device, layout.pf, "tob_read");
        let read = drawn(&queue, &layout, &reads, &surface.texture, None);
        let (count, first) = differing(&read, &texels, layout.texel);
        assert!(
            count == 0,
            "{}, its first use: {count} texels read differ from the pages', the first at texel \
             {first:?}",
            surface.what
        );
        surface.check(&layout, "its first use", &surface.pages);
        surface.report(&device, &messages);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// [`first_use_reads_the_bytes`] over each of [`LAYOUTS`]: an aliased texture's image moves at its
/// first use (a shadow's copy in reads the pages through a buffer).
#[test]
fn a_texture_s_first_use_reads_the_bytes_it_was_made_over() {
    for layout in LAYOUTS {
        first_use_reads_the_bytes(layout);
    }
}

/// A texture over a buffer's padded rows (the browser's tile, [`LAYOUTS`]' first) is a shadow on
/// every device, its buffer an ordinary one's, which no image of Metal's aliases either; and its
/// `shadowed-texture` notice names what Vulkan lets write the padding: for a texture that only
/// reads, its image's layout transitions (the first move from the layout it is made in), never a
/// render pass, which no image of its takes; for a render target, a render pass's load and store
/// operations too. Each usage on a device of its own, which announces the kind once.
#[test]
fn a_padded_texture_s_shadow_names_what_may_write_its_padding() {
    let layout = LAYOUTS[0];
    assert!(layout.padded(), "{}", layout.name);
    for (usage, renders) in [
        (TextureUsage::SHADER_READ, false),
        (
            TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            true,
        ),
    ] {
        let cache = ScratchCache::new().unwrap();
        let messages = Messages::default();
        {
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            // What the device announced as it was made.
            messages.take_notices();
            let buffer = device
                .new_buffer(layout.bpr * layout.h, ResourceOptions::default())
                .unwrap();
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            let aliased = unsafe { texture.vk_image() }.is_some();
            assert!(!aliased, "{} ({usage:?}): aliased", layout.name);
            let reason = messages
                .take_notices()
                .into_iter()
                .filter_map(|message| message.degradation)
                .find(|notice| notice.degradation == Degradation::ShadowedTexture)
                .map(|notice| notice.reason)
                .unwrap_or_else(|| {
                    panic!("{} ({usage:?}): no shadowed-texture notice", layout.name)
                });
            eprintln!(
                "PITCHED-SURFACE on {}: {} ({usage:?}), an ordinary buffer's: shadowed: {reason}",
                device.name(),
                layout.name
            );
            let rows = format!(
                "its rows lie {} bytes apart, {} bytes of texels each",
                layout.bpr,
                layout.row()
            );
            assert!(reason.contains(&rows), "{usage:?}: {reason}");
            assert!(
                reason.contains("its layout transitions read and write"),
                "{usage:?}: {reason}"
            );
            assert_eq!(
                reason.contains("a render pass's load and store operations write back"),
                renders,
                "{usage:?}: {reason}"
            );
            drop((texture, buffer));
        }
        let messages = messages.take();
        assert!(
            messages.is_empty(),
            "the validation layer reported: {messages:#?}"
        );
    }
}

/// A pass over a shadowed texture (the browser's padded tile over an ordinary buffer, a shadow on
/// every device) that loads it and draws nothing leaves its texels as they were, as Metal's leaves
/// the tiles no draw reaches: it takes no copy in and no copy back (`WorkCounts::shadow_copies`),
/// whether it stores the attachment or stores it `DontCare`, and the buffer keeps every byte. A
/// pass that draws a rectangle copies in before it and back after it.
#[test]
fn a_pass_that_draws_nothing_copies_no_shadow() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let copies = || device.work_counts().shadow_copies;
        for store in [StoreAction::Store, StoreAction::DontCare] {
            let before = copies();
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(texture.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = store;
            let command_buffer = queue.command_buffer().unwrap();
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
            assert_eq!(copies() - before, 0, "{store:?}: a pass drawing nothing");
            assert!(
                bytes_of(&buffer) == bytes,
                "{store:?}: the buffer's bytes changed"
            );
        }
        let before = copies();
        let blend = blending(&device, layout.pf);
        let rect = ([-1.0, -1.0, 0.0, 1.0], [0.25, 0.5, 0.75, 0.5]);
        blend_pass(&queue, &blend, (&texture, StoreAction::Store), false, rect);
        assert_eq!(
            copies() - before,
            2,
            "a pass drawing a rectangle: in and back"
        );
        drop((texture, buffer));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A pass that loads a shadowed texture (the browser's padded tile over an ordinary buffer, a
/// shadow on every device) and draws under a scissor copies back only the texels inside it
/// (`WorkCounts::shadow_texels_copied_back`): a fragment outside the scissor has no coverage, and
/// the pass's load and store write back what they read. The buffer holds the blend inside the
/// scissor, as a private texture of the same texels blended alike does, and every other byte as
/// it was. Without a scissor, the copy back moves every texel.
#[test]
fn a_pass_copies_back_only_the_texels_its_scissors_reach() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 5);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = buffer.resource_options();
        let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        let blend = blending(&device, layout.pf);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let scissor = ScissorRect::new(100, 40, 300, 200);
        let draw = |target: &Texture, scissor: Option<ScissorRect>| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            if let Some(scissor) = scissor {
                encoder.set_scissor_rect(scissor).unwrap();
            }
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        let moved = || device.work_counts().shadow_texels_copied_back;
        let before = moved();
        draw(&texture, Some(scissor));
        assert_eq!(moved() - before, 300 * 200, "a scissored pass");
        draw(&private, Some(scissor));
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, first) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{count} texels differ from the private texture's, the first at texel {first:?}"
        );
        assert!(now == expected, "the bytes between the rows changed");
        let before = moved();
        draw(&texture, None);
        assert_eq!(moved() - before, layout.w * layout.h, "an unscissored pass");
        drop((texture, buffer, private));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A texture of `layout` over a no-copy buffer, with shader read and write usage, written at every
/// texel by a fragment function's `write` (`tob_write`): the pages' texels are what a private
/// texture of the same texels holds after the same writes, and no other byte of the pages changes.
fn writes_as_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let Some(surface) = surface(&device, &layout, usage) else {
            return;
        };
        let mut expected = surface.pages.clone();
        let texels = rows(&layout, &expected);
        let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
        d.usage = usage;
        d.resource_options = ResourceOptions::private();
        let reference = device.new_texture(&d).unwrap();
        blit_in(&queue, &layout, &reference, &texels);
        let writes = reading(&device, layout.pf, "tob_write");
        drawn(&queue, &layout, &writes, &reference, None);
        drawn(&queue, &layout, &writes, &surface.texture, None);
        lay(
            &layout,
            &mut expected,
            &blit_out(&queue, &layout, &reference),
        );
        surface.check(&layout, "a fragment function's writes", &expected);
        surface.report(&device, &messages);
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
    a_tile_of_510_packed_rows_reads_writes_and_renders_as_metal => 8;
    a_small_tile_of_packed_rows_reads_writes_and_renders_as_metal => 9;
    a_half_float_tile_of_126_packed_rows_reads_writes_and_renders_as_metal => 10;
}

/// A fragment function's writes over [`LAYOUTS`]: padded rows (a browser's tile, a narrow tile, a
/// half-float tile) and packed ones of a height that is no multiple of 4, and of one that is.
#[test]
fn a_fragment_function_s_writes_leave_the_bytes_between_rows() {
    for index in [0, 6, 7, 8, 9, 10] {
        writes_as_metal(LAYOUTS[index]);
    }
}
