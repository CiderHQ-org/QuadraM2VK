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
//! that renders or writes, whose rows lie with bytes between them, is never a linear image over
//! the pages (Vulkan lets a render pass's load and store operations write every byte of an
//! attachment's subresource), but on a driver that runs on Metal, whose image is a Metal texture
//! over the buffer; nor is one that only reads under the lane's policy (`exact_answers`,
//! `PaddedReadsPolicy::Shadow`). The steps:
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
//! Under the policy's default (`PaddedReadsPolicy::Alias`), a texture that only reads (`ShaderRead`
//! usage) is an image over its padded rows where the device lays one at their pitch.
//! [`a_texture_that_only_reads_keeps_every_byte_of_its_pages`] makes one over each layout and
//! reads it first through its image, before anything else touches the pages: its image's first
//! move keeps its texels (the runtime copies them out and back around it) and the bytes between
//! its rows as far as the device's move does, and a lapse fails here, naming the bytes. Then it
//! samples, blits out, reads the
//! caller's later bytes (the bytes between the rows rewritten too), blits a buffer's box and a
//! private texture's box in and replaces a box: after each, every byte of the pages is checked.
//! [`a_padded_texture_that_only_reads_copies_no_shadow_by_default`] counts its copies
//! (`WorkCounts::shadow_copies`: none across its uses, and a shadow's under
//! `PaddedReadsPolicy::Shadow`), and
//! [`a_padded_texture_that_renders_or_writes_is_a_shadow_under_either_policy`] keeps a render
//! target's, a written texture's and an `Unknown` usage's shadow.
//!
//! [`a_texture_s_first_move_keeps_the_bytes_it_lies_over`] makes a texture over each layout, and
//! over packed 1080p and tile rows, with no submission reaching it before its first use: that
//! batch makes an aliasing image's move from `UNDEFINED`, the host keeping the bytes the image is
//! bound to (`WorkCounts::first_move_keeps` counts one per aliased texture, none per shadow, none
//! at a second use; `first_move_kept_bytes` those bytes, from the first texel to at most the last
//! row's pitch), the read gives the pages' texels, and every byte of the pages stays
//! (`FIRST-MOVE-KEEP`). [`a_cpu_write_during_a_texture_s_first_move_stays_outside_its_bytes`]
//! writes bytes between the rows and past the image's from the CPU while that first use runs, and
//! finds the texels and each write past the image's bytes in the pages after it; the writes
//! between rows the keep undid are printed (`FIRST-MOVE-WINDOW`).
//!
//! [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] draws under a scissor
//! beside a depth attachment the pass clears and stores, of the color's size and larger, whose
//! clear then reaches past the render area: the color's copies stay inside the scissor, and its
//! bytes are a private texture's.
//!
//! [`a_shadowed_color_beside_a_memoryless_scratch_blends_over_part_of_it`] blends over a shadowed
//! texture beside a memoryless scratch loaded and stored `DontCare`, over a render area smaller
//! than both: the bytes are a private texture's inside the render area and unchanged past it.
//!
//! [`a_pass_s_copies_in_share_their_barriers`] reads several shadowed textures in one pass that
//! loads another: their copies in share one barrier before and one after, and the pass records
//! as many pipeline barriers whatever the number of textures it reads.
//!
//! (`SHADOW-COPY-BACK`, [`a_pass_copies_in_and_back_only_the_texels_its_scissors_reach`]): whether
//! a shadow's copy back ran as a dispatch (`WorkCounts::shadow_copies_back_dispatched`).

use super::draw_work::pushes;
use super::host_memory::{counting, no_copy, not_importable, Mapping, GUEST_PAGE};
use super::memoryless::memoryless_options;
use super::no_copy_textures::{blending, blending_beside_scratch, floats, rects_beside, staged};
use super::{validated_device, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BarrierScope, PixelFormat, PrimitiveType, RenderStages, SamplerMinMagFilter, TextureUsage,
};
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
pub(super) fn reading(device: &Device, pf: PixelFormat, fragment: &str) -> RenderPipelineState {
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
    let (command_buffer, target) = draw_reading(queue, layout, pipeline, texture, sampler);
    run(&command_buffer);
    blit_out(queue, layout, &target)
}

/// [`drawn`]'s command buffer, encoded and not committed, and its target.
fn draw_reading(
    queue: &CommandQueue,
    layout: &Layout,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> (metal2vulkan_runtime::CommandBuffer, Texture) {
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
    (command_buffer, target)
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
pub(super) fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
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
/// aliased, and never a linear image over padded rows off Metal but where `padded_alias` (a
/// texture that only reads, under `PaddedReadsPolicy::Alias`).
fn surface(
    device: &Device,
    layout: &Layout,
    usage: TextureUsage,
    padded_alias: bool,
) -> Option<Surface> {
    surface_with(device, layout, usage, padded_alias, true)
}

/// [`surface`], its image's `VkImage` asked (`Texture::vk_image`, which makes its first move on a
/// submission of its own) where `handed_out`; else whether it is aliased is told by the shadows
/// its making counted alone, and no submission has reached it yet.
fn surface_with(
    device: &Device,
    layout: &Layout,
    usage: TextureUsage,
    padded_alias: bool,
    handed_out: bool,
) -> Option<Surface> {
    surface_past(device, layout, usage, (padded_alias, handed_out), 0)
}

/// [`surface_with`], the pages reaching at least `past` bytes past the last row's pitch.
fn surface_past(
    device: &Device,
    layout: &Layout,
    usage: TextureUsage,
    (padded_alias, handed_out): (bool, bool),
    past: usize,
) -> Option<Surface> {
    let len = (layout.len() + past).next_multiple_of(GUEST_PAGE);
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
    let aliased = match handed_out {
        // SAFETY: the handle is only compared, while the texture lives.
        true => unsafe { texture.vk_image() }.is_some(),
        false => shadowed == 0,
    };
    let what = format!("{} ({}, {usage:?})", layout.name, layout.pf.name());
    assert_eq!(
        u64::from(!aliased),
        shadowed,
        "{what}: aliased {aliased}, but {shadowed} shadows counted"
    );
    assert!(
        !(aliased && layout.padded()) || on_metal(device) || padded_alias,
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
        let Some(surface) = surface(&device, &layout, usage, false) else {
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
/// made, `VUID-VkImageCreateInfo-pNext-01443`, which "will cause the contents of the image's
/// memory to be undefined") keeps the texels, which the runtime copies out and back around it,
/// and the bytes past them as far as the device's move does. A lapse fails here: the read differs
/// from the texels, or the pages change.
fn first_use_reads_the_bytes(device: &Device, messages: &Messages, layout: Layout) {
    {
        let queue = device.new_command_queue().unwrap();
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let Some(surface) = surface(device, &layout, usage, false) else {
            return;
        };
        let texels = rows(&layout, &surface.pages);
        let reads = reading(device, layout.pf, "tob_read");
        let read = drawn(&queue, &layout, &reads, &surface.texture, None);
        let (count, first) = differing(&read, &texels, layout.texel);
        assert!(
            count == 0,
            "{}, its first use: {count} texels read differ from the pages', the first at texel \
             {first:?}",
            surface.what
        );
        surface.check(&layout, "its first use", &surface.pages);
        surface.report(device, messages);
    }
    assert_eq!(
        messages.take(),
        Vec::new(),
        "{}: the validation layer reported",
        layout.name
    );
}

/// [`first_use_reads_the_bytes`] over each of [`LAYOUTS`], on one device: an aliased texture's
/// image moves at its first use (a shadow's copy in reads the pages through a buffer).
#[test]
fn a_texture_s_first_use_reads_the_bytes_it_was_made_over() {
    on_shared_device(None, |device, messages| {
        for layout in LAYOUTS {
            first_use_reads_the_bytes(device, messages, layout);
        }
    });
}

/// A texture of `layout` over a no-copy buffer, of `usage`, whose first use (a fragment
/// function's `read`, in a command buffer of its own) is the first submission to reach it: where
/// an image aliases the pages, that batch makes the image's move from `UNDEFINED` (an image of host
/// allocations, `VUID-VkImageCreateInfo-pNext-01443`, or on Metal a Metal texture over the
/// buffer), whose bytes the host keeps (`WorkCounts::first_move_keeps`: one;
/// `first_move_kept_bytes`: from the first texel to at most the last row's pitch); a shadow keeps
/// none. The read
/// gives the pages' texels and every byte of the pages stays; a second use keeps nothing more.
/// Whether it was aliased; `None` where the device imports no host memory.
fn first_move_keeps_the_bytes(
    device: &Device,
    messages: &Messages,
    layout: Layout,
    usage: TextureUsage,
) -> Option<bool> {
    let aliased = {
        let queue = device.new_command_queue().unwrap();
        let reads = reading(device, layout.pf, "tob_read");
        let before = device.work_counts();
        let padded_alias = !usage.contains(TextureUsage::RENDER_TARGET);
        let surface = surface_with(device, &layout, usage, padded_alias, false)?;
        let (aliased, _, _) = surface.form;
        let texels = rows(&layout, &surface.pages);
        for use_ in ["its first use", "its second use"] {
            let read = drawn(&queue, &layout, &reads, &surface.texture, None);
            let (count, first) = differing(&read, &texels, layout.texel);
            assert!(
                count == 0,
                "{}, {use_}: {count} texels read differ from the pages', the first at texel \
                 {first:?}",
                surface.what
            );
            surface.check(&layout, use_, &surface.pages);
            let after = device.work_counts();
            let kept = after.first_move_keeps - before.first_move_keeps;
            assert_eq!(
                kept,
                u64::from(aliased),
                "{}, {use_}: {kept} first moves kept their bytes (aliased {aliased})",
                surface.what
            );
            let bytes = after.first_move_kept_bytes - before.first_move_kept_bytes;
            let span = u64::from(aliased) * ((layout.h - 1) * layout.bpr + layout.row());
            let rows = u64::from(aliased) * layout.h * layout.bpr;
            assert!(
                (span..=rows).contains(&bytes),
                "{}: the first move kept {bytes} bytes, not the image's, from the first texel to \
                 at most the last row's pitch ({span} to {rows})",
                surface.what
            );
            let restored = after.first_move_restored_bytes - before.first_move_restored_bytes;
            eprintln!(
                "FIRST-MOVE-KEEP on {}: {}, {use_}: {} ({kept} kept, {bytes} bytes, {restored} \
                 restored)",
                device.name(),
                surface.what,
                if aliased { "aliased" } else { "shadowed" },
            );
        }
        surface.report(device, messages);
        aliased
    };
    assert_eq!(
        messages.take(),
        Vec::new(),
        "{}: the validation layer reported",
        layout.name
    );
    Some(aliased)
}

/// Packed rows whose pitch every measured driver lays a linear image at (multiples of 256
/// bytes): a presented 1080p surface's, and a tile's.
const PACKED: [Layout; 2] = [
    Layout::bgra8("1920x1080/7680", 1920, 1080, 7680),
    Layout::bgra8("1024x512/4096", 1024, 512, 4096),
];

/// [`first_move_keeps_the_bytes`] over each of [`LAYOUTS`] and [`PACKED`], for a texture that only
/// reads (an image over padded rows too, under the policy's default) and for a render target that
/// reads (an image over packed rows alone, but on Metal), on one device. Where the device imports
/// host memory, at least one texture is aliased, or no keep was checked.
#[test]
fn a_texture_s_first_move_keeps_the_bytes_it_lies_over() {
    on_shared_device(Some(PaddedReadsPolicy::Alias), |device, messages| {
        let reads = TextureUsage::SHADER_READ;
        let renders = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let before = device.work_counts().first_move_keeps;
        let mut aliased = 0;
        for usage in [reads, renders] {
            for layout in PACKED.into_iter().chain(LAYOUTS) {
                match first_move_keeps_the_bytes(device, messages, layout, usage) {
                    Some(alias) => aliased += u64::from(alias),
                    None => return,
                }
            }
        }
        let kept = device.work_counts().first_move_keeps - before;
        eprintln!(
            "FIRST-MOVE-KEEP on {}: {kept} keeps in all, {aliased} textures aliased",
            device.name()
        );
        assert_eq!(kept, aliased);
        assert!(
            aliased > 0,
            "no texture was aliased, so no keep was checked"
        );
    });
}

/// A texture that only reads, of `layout`, over a no-copy buffer whose pages the process maps
/// read-only by its first use: made read-only after the buffer was made (Linux refuses read-only
/// pages at the import, `a_read_only_page_is_refused_on_linux_and_read_elsewhere`, but not later),
/// or, where `from_the_start`, before it (a guest's pages the host maps read-only, which MoltenVK
/// imports in place). Its first use makes the image's first move, across which the host keeps the
/// bytes the image is bound to and stores back only those the move changed
/// (`texture/first_move.rs`'s `KeptBytes::write_back`): a store into the read-only pages would end
/// the process (`SIGBUS` on macOS, `SIGSEGV` on Linux), as every byte's store did. The read gives
/// the pages' texels and every byte of the pages stays, unless the move changed some, which the
/// device then announces (`unrestored-first-move-bytes`). Printed (`FIRST-MOVE-READ-ONLY`): the
/// form and the kept and restored bytes. Whether it was aliased; `None` where the device makes no
/// buffer over the pages.
fn first_move_over_read_only_pages(
    device: &Device,
    messages: &Messages,
    layout: Layout,
    from_the_start: bool,
) -> Option<bool> {
    let queue = device.new_command_queue().unwrap();
    let reads = reading(device, layout.pf, "tob_read");
    let len = layout.len().next_multiple_of(GUEST_PAGE);
    let mapping = Mapping::anonymous(len);
    let mut pages = counting(len, 7);
    lay(
        &layout,
        &mut pages,
        &texels_of(&layout, (layout.row() * layout.h) as usize, 1),
    );
    mapping.write(0, &pages);
    if from_the_start {
        mapping.protect_read_only();
    }
    // SAFETY: the mapping outlives the buffer and the texture (dropped first), and nothing writes
    // the buffer.
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
            eprintln!("FIRST-MOVE-READ-ONLY: no buffer over the pages: {error:?}");
            return None;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    if !from_the_start {
        mapping.protect_read_only();
    }
    let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    let before = device.work_counts();
    let texture = buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, layout.bpr, false))
        .unwrap();
    let aliased = device.work_counts().shadowed_textures == before.shadowed_textures;
    let what = format!(
        "{} ({}), read-only {}",
        layout.name,
        layout.pf.name(),
        if from_the_start {
            "from the start"
        } else {
            "after the buffer"
        }
    );
    let read = drawn(&queue, &layout, &reads, &texture, None);
    let (count, first) = differing(&read, &rows(&layout, &pages), layout.texel);
    let after = device.work_counts();
    let kept = after.first_move_kept_bytes - before.first_move_kept_bytes;
    let restored = after.first_move_restored_bytes - before.first_move_restored_bytes;
    let unrestored = messages
        .take_notices()
        .into_iter()
        .filter_map(|message| message.degradation)
        .any(|notice| notice.degradation == Degradation::UnrestoredFirstMoveBytes);
    eprintln!(
        "FIRST-MOVE-READ-ONLY on {}: {what}: {} ({kept} bytes kept, {restored} restored, \
         unrestored announced {unrestored})",
        device.name(),
        if aliased { "aliased" } else { "shadowed" },
    );
    let keeps = after.first_move_keeps - before.first_move_keeps;
    assert_eq!(
        keeps,
        u64::from(aliased),
        "{what}: {keeps} first moves kept their bytes (aliased {aliased})"
    );
    if !unrestored {
        assert!(
            count == 0,
            "{what}: {count} texels read differ from the pages', the first at texel {first:?}"
        );
        let now = mapping.read(0..len);
        assert!(
            now == pages,
            "{what}: the read-only pages changed: {}",
            padding_changes(&layout, &now, &pages)
        );
    }
    drop((texture, buffer));
    drop(mapping);
    Some(aliased)
}

/// [`first_move_over_read_only_pages`] over [`PACKED`], with the pages made read-only after the
/// buffer, and off Linux also before it, on one device of the policy's default. Where the device
/// imports host memory, at least one texture is aliased, or no keep over read-only pages was
/// checked. On Linux only lavapipe and NVIDIA's driver keep an import running once its pages lose
/// write permission (`HostMemory::new`'s Linux contract: RADV failed the command buffers submitted
/// after it), so only they take the case.
#[test]
fn a_texture_over_read_only_pages_takes_its_first_move() {
    on_shared_device(Some(PaddedReadsPolicy::Alias), |device, messages| {
        let driver = device.properties().description.driver_id;
        let keeps_running = matches!(
            driver,
            Some(DriverId::MESA_LLVMPIPE | DriverId::NVIDIA_PROPRIETARY)
        );
        if cfg!(target_os = "linux") && !keeps_running {
            eprintln!(
                "FIRST-MOVE-READ-ONLY on {}: {driver:?} is held to the writable pages of \
                 HostMemory::new's Linux contract",
                device.name()
            );
            return;
        }
        let mut aliased = 0;
        for layout in PACKED {
            for from_the_start in [false, true] {
                if from_the_start && cfg!(target_os = "linux") {
                    continue;
                }
                match first_move_over_read_only_pages(device, messages, layout, from_the_start) {
                    Some(alias) => aliased += u64::from(alias),
                    None => return,
                }
            }
        }
        assert!(
            aliased > 0,
            "no texture over read-only pages was aliased, so no keep over them was checked"
        );
    });
}

/// How long the CPU waits after the commit before its writes beside a texture
/// ([`a_cpu_write_during_the_first_move_stays_outside`]), one fresh texture each: the first move
/// runs when the device reaches the submission, which takes a time of its own on each device, so
/// the writes start at several times and one of them overlaps the move's keep.
const PADDING_WRITE_DELAYS: [std::time::Duration; 6] = [
    std::time::Duration::from_micros(0),
    std::time::Duration::from_micros(500),
    std::time::Duration::from_micros(1000),
    std::time::Duration::from_micros(2000),
    std::time::Duration::from_micros(3000),
    std::time::Duration::from_micros(5000),
];

/// Spin for `wait`.
fn spin(wait: std::time::Duration) {
    let spun = std::time::Instant::now();
    while spun.elapsed() < wait {
        std::hint::spin_loop();
    }
}

/// The bytes past a texture's last row that [`a_cpu_write_during_the_first_move_stays_outside`]
/// writes: a cache line's.
const PAST: u64 = 64;

/// The CPU writes bytes beside a texture that only reads, over a no-copy buffer's padded rows,
/// while its first use's command buffer runs, which makes its image's first move: each row's first
/// byte past its texels and its last before the next row, inside the bytes the image is bound to,
/// and the first and last of the [`PAST`] bytes past the last row's pitch, outside them; one row
/// after another, a few microseconds apart, from a time after the commit
/// ([`PADDING_WRITE_DELAYS`], a fresh texture each). Metal never writes those bytes, so each write
/// stays. The host keeps the bytes the image is bound to across its move (`texture/first_move.rs`'s
/// `Keep`), so a write past them stays, and so do the texels; a write between rows made between
/// the keep's read and its write back is undone (`docs/FAST-PATHS.md`'s known gap): printed, as
/// `FIRST-MOVE-WINDOW`, not failed. The lapses outside the image's bytes or in the texels, one line
/// each, and whether the texture was aliased.
fn a_cpu_write_during_the_first_move_stays_outside(
    device: &Device,
    messages: &Messages,
    layout: Layout,
) -> Option<(bool, Vec<String>)> {
    let mut lapses = Vec::new();
    let mut aliased = false;
    let queue = device.new_command_queue().unwrap();
    let reads = reading(device, layout.pf, "tob_read");
    let bound = layout.bpr * layout.h;
    for delay in PADDING_WRITE_DELAYS {
        let usage = TextureUsage::SHADER_READ;
        let surface = surface_past(device, &layout, usage, (true, false), PAST as usize)?;
        aliased = surface.form.0;
        let (command_buffer, _target) =
            draw_reading(&queue, &layout, &reads, &surface.texture, None);
        let mut expected = surface.pages.clone();
        let write = |at: u64, expected: &mut Vec<u8>| {
            let at = at as usize;
            let byte = surface.pages[at] ^ 0xA5;
            surface.mapping.write(at, &[byte]);
            expected[at] = byte;
        };
        command_buffer.commit().unwrap();
        spin(delay);
        for row in 0..layout.h {
            write(row * layout.bpr + layout.row(), &mut expected);
            write((row + 1) * layout.bpr - 1, &mut expected);
            if row == layout.h / 2 {
                write(bound, &mut expected);
            }
            spin(std::time::Duration::from_micros(8));
        }
        write(bound + PAST - 1, &mut expected);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let now = surface.mapping.read(0..surface.len);
        let (count, first) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        if count != 0 {
            lapses.push(format!(
                "{}, writes from {delay:?} after the commit: {count} texels differ from Metal's, \
                 the first at texel {first:?}",
                surface.what
            ));
        }
        let outside: Vec<usize> = (bound as usize..surface.len)
            .filter(|&at| now[at] != expected[at])
            .collect();
        if let Some(&at) = outside.first() {
            lapses.push(format!(
                "{}, writes from {delay:?} after the commit: {} bytes past the image's changed, \
                 the first at byte {at} (its bytes end at {bound}), now {:02x}, was {:02x}",
                surface.what,
                outside.len(),
                now[at],
                expected[at]
            ));
        }
        let inside = (0..bound as usize)
            .filter(|&at| now[at] != expected[at])
            .count();
        eprintln!(
            "FIRST-MOVE-WINDOW on {}: {}: {}, writes from {delay:?} after the commit: {inside} of \
             {} bytes between rows undone, {} of 2 past its bytes",
            device.name(),
            surface.what,
            if aliased { "aliased" } else { "shadowed" },
            2 * layout.h,
            outside.len(),
        );
        surface.report(device, messages);
    }
    assert_eq!(
        messages.take(),
        Vec::new(),
        "{}: the validation layer reported",
        layout.name
    );
    Some((aliased, lapses))
}

/// [`a_cpu_write_during_the_first_move_stays_outside`] over each padded layout of [`LAYOUTS`], on
/// one device of the policy's default. Off Metal, where the device imports host memory, at least
/// one is aliased, or no first move over padded rows was checked.
#[test]
fn a_cpu_write_during_a_texture_s_first_move_stays_outside_its_bytes() {
    on_shared_device(Some(PaddedReadsPolicy::Alias), |device, messages| {
        let (mut aliased, mut lapses) = (0, Vec::new());
        for layout in LAYOUTS.into_iter().filter(Layout::padded) {
            match a_cpu_write_during_the_first_move_stays_outside(device, messages, layout) {
                Some((alias, lapsed)) => {
                    aliased += u64::from(alias);
                    lapses.extend(lapsed);
                }
                None => return,
            }
        }
        eprintln!(
            "FIRST-MOVE-PADDING on {}: {aliased} padded textures aliased",
            device.name()
        );
        assert!(
            lapses.is_empty(),
            "a first use changed the texels, or a CPU write past the image's bytes:\n{}",
            lapses.join("\n")
        );
        assert!(
            aliased > 0 || on_metal(device),
            "no padded texture was aliased, so no first move over padded rows was checked"
        );
    });
}

/// Textures of one image over the same bytes of one import share one `VkImage`
/// (`texture/shared_alias.rs`), held by the import, as a consumer makes a texture over the
/// presented surface at each present: a new no-copy buffer over its pages, which joins the import
/// of a buffer that lives across presents (here the surface's own, with its texture of render
/// target usage), and a texture of its plane over it, the one before released first. Over packed
/// 1080p rows, where the textures are aliased: a texture released unused makes the image, whose
/// move the next submission leaves waiting, no texture over it alive
/// (`WorkCounts::shared_alias_moves_waited`); each present's texture takes that image
/// (`shared_alias_images`), though no texture over it lived meanwhile, the first one's submission
/// keeping its texels once (`first_move_keeps`), and reads the pages' texels; a texture of the
/// surface texture's usage takes that texture's image, and keeps nothing. The pages stay as
/// written. Printed: `SHARED-ALIAS`.
#[test]
fn textures_of_one_image_over_the_same_bytes_share_it() {
    const PRESENTS: u64 = 5;
    on_shared_device(Some(PaddedReadsPolicy::Alias), |device, messages| {
        let layout = PACKED[0];
        let reads = TextureUsage::SHADER_READ;
        let renders = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let queue = device.new_command_queue().unwrap();
        let pipeline = reading(device, layout.pf, "tob_read");
        let Some(surface) = surface_with(device, &layout, renders, false, false) else {
            return;
        };
        let (aliased, _, _) = surface.form;
        let texels = rows(&layout, &surface.pages);
        let over = |usage: TextureUsage| {
            // SAFETY: the mapping outlives the buffer: both are dropped before the surface.
            let made = unsafe {
                no_copy(
                    device,
                    surface.mapping.pointer(),
                    surface.len,
                    ResourceOptions::default(),
                    None,
                )
            };
            let buffer = made.unwrap();
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            let plane = IoSurfacePlane::new(0, layout.bpr, false);
            let texture = buffer.new_texture_with_iosurface_plane(&d, plane).unwrap();
            (texture, buffer)
        };
        let reads_texels = |texture: &Texture, step: &str| {
            let read = drawn(&queue, &layout, &pipeline, texture, None);
            let (count, first) = differing(&read, &texels, layout.texel);
            assert!(
                count == 0,
                "{}, {step}: {count} texels read differ from the pages', the first at texel \
                 {first:?}",
                surface.what
            );
            surface.check(&layout, step, &surface.pages);
        };
        reads_texels(&surface.texture, "the surface's texture");
        let before = device.work_counts();
        drop(over(reads));
        reads_texels(
            &surface.texture,
            "the surface's texture, the image's move waiting",
        );
        let waiting = device.work_counts();
        let waited = waiting.shared_alias_moves_waited - before.shared_alias_moves_waited;
        let kept_waiting = waiting.first_move_keeps - before.first_move_keeps;
        for present in 0..PRESENTS {
            let made = over(reads);
            reads_texels(&made.0, &format!("present {present}"));
            drop(made);
        }
        let counted = device.work_counts();
        let shared = counted.shared_alias_images - before.shared_alias_images;
        let kept = counted.first_move_keeps - before.first_move_keeps;
        let presents_aliased = counted.shadowed_textures == before.shadowed_textures;
        let same = over(renders);
        reads_texels(&same.0, "a texture of the surface texture's usage");
        drop(same);
        let after = device.work_counts();
        let shared_after = after.shared_alias_images - counted.shared_alias_images;
        let kept_after = after.first_move_keeps - counted.first_move_keeps;
        eprintln!(
            "SHARED-ALIAS on {}: {}: aliased {aliased}; {PRESENTS} presents (aliased \
             {presents_aliased}), each texture released before the next, took {shared} shared \
             images and kept {kept} times, the image's move left waiting {waited} times, keeping \
             {kept_waiting}; the surface texture's usage: {shared_after} shared, {kept_after} \
             keeps",
            device.name(),
            surface.what,
        );
        let (presents, presents_kept) = match presents_aliased {
            true => (PRESENTS, 1),
            false => (0, 0),
        };
        assert_eq!(
            (shared, kept),
            (presents, presents_kept),
            "each present's texture takes the image a texture released unused made, kept once"
        );
        assert!(
            !presents_aliased || (waited > 0 && kept_waiting == 0),
            "a submission with no texture over the shared image alive leaves its move waiting \
             (waited {waited}, kept {kept_waiting})"
        );
        assert_eq!(
            (shared_after, kept_after),
            (u64::from(aliased), 0),
            "a texture of the surface texture's usage takes its image"
        );
        assert!(
            aliased || presents_aliased || !on_metal(device),
            "no texture was aliased on Metal"
        );
        surface.report(device, messages);
    });
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
/// shadow on every device) and draws under a scissor copies in and back only the texels inside it
/// (`WorkCounts::shadow_texels_copied_in`, `shadow_texels_copied_back`): a fragment outside the
/// scissor has no coverage, and the pass's load and store write back what they read. The buffer
/// holds the blend inside the scissor, as a private texture of the same texels blended alike
/// does, and every other byte as it was; a later draw reading every texel of the texture reads
/// the buffer's, none the image kept from before. Without a scissor, both copies move every
/// texel. The CPU may write the `Shared` buffer's bytes, so its texture's copy back is a merge,
/// which moves no texel back as a copy and dispatches no copy back; its keep moves the texels
/// inside the scissor alone (`WorkCounts::shadow_texels_kept`). On a device that takes the copy
/// Neither merge zeroes its scratch (`WorkCounts::shadow_merge_scratches_zeroed`). Where the
/// device pushes set 0, the keep's and the merge's dispatches push their descriptors and make no
/// descriptor pool; elsewhere each makes one (`WorkCounts::owned_descriptor_pools`).
#[test]
fn a_pass_copies_in_and_back_only_the_texels_its_scissors_reach() {
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
        // `WorkCounts::shadow_merges_fetched`); both copy everywhere else.
        let moved = || {
            let counts = device.work_counts();
            (
                counts.shadow_texels_copied_in,
                counts.shadow_texels_copied_back,
                counts.shadow_copies_back_dispatched,
                counts.shadow_merge_copies_dispatched,
                counts.shadow_texels_kept,
                counts.shadow_merges_fetched,
                counts.owned_descriptor_pools,
                counts.shadow_copies_in_dispatched,
            )
        };
            let now = moved();
            (
                now.0 - before.0,
                now.1 - before.1,
                now.2 - before.2,
                now.3 - before.3,
                now.4 - before.4,
                now.5 - before.5,
                now.6 - before.6,
                now.7 - before.7,
            )
        };
        // The merge's kernel, and the keep where it is a dispatch, each make a descriptor pool
        // where the device does not push set 0.
        let pools = if pushes(&device) { 0 } else { 1 + merged };
        // The keep's dispatch takes the copy in of the same texels with it, where the device takes
        let zeroed = device.work_counts().shadow_merge_scratches_zeroed;
        let before = moved();
        draw(&texture, Some(scissor));
        let scissored = since(before);
        println!(
            "SHADOW-COPY-BACK on {}: a scissored pass dispatched {} copies back and {} of its \
            device.name(),
            scissored.2,
            scissored.3,
            scissored.4,
            scissored.5,
            scissored.7,
            if dispatches { "takes" } else { "does not take" },
        );
        assert_eq!(
            scissored,
            (
                300 * 200,
                back(300 * 200),
                dispatched,
                merged,
                300 * 200,
                fetched,
                pools,
            ),
            "a scissored pass"
        );
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
        // Every texel, read in a later encoder: the buffer's, though the scissored pass left the
        // image's texels outside its scissor as they were before.
        let read = drawn(
            &queue,
            &layout,
            &reading(&device, layout.pf, "tob_read"),
            &texture,
            None,
        );
        let (count, first) = differing(&read, &rows(&layout, &expected), layout.texel);
        assert!(
            count == 0,
            "{count} texels read differ from the buffer's, the first at texel {first:?}"
        );
        let before = moved();
        draw(&texture, None);
        let every = layout.w * layout.h;
        assert_eq!(
            since(before),
            (
                every,
                back(every),
                dispatched,
                merged,
                every,
                fetched,
                pools,
            ),
            "an unscissored pass"
        );
        // Both merges go in place over whole words of each row, so their copies write every byte
        // their kernels read, and neither zeroes its scratch.
        assert_eq!(
            device.work_counts().shadow_merge_scratches_zeroed,
            zeroed,
            "merge scratches zeroed"
        );
        drop((texture, buffer, private));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A pass that loads a shadowed texture (the browser's padded tile over an ordinary buffer) over
/// a render area smaller than it, of whole 32-pixel tiles on neither axis, beside a memoryless
/// `RGBA16Float` scratch of its size loaded and stored `DontCare` (the consumer's guest's shadowed
/// passes' shape): the buffer holds the blend inside the render area, as a private texture of the
/// same texels drawn alike does, and every other byte as it was.
#[test]
fn a_shadowed_color_beside_a_memoryless_scratch_blends_over_part_of_it() {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 9);
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
        let mut d =
            TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, layout.w, layout.h, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = memoryless_options();
        let scratch = device.new_texture(&d).unwrap();
        let blend = blending_beside_scratch(&device, layout.pf, PixelFormat::RGBA16Float);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let area = [layout.w - 37, layout.h - 21];
        assert!(area.iter().all(|side| side % 32 != 0), "tiles cut");
        let draw = |target: &Texture| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let attachment = &mut pass.color_attachments[1];
            attachment.texture = Some(scratch.clone());
            attachment.load_action = LoadAction::DontCare;
            attachment.store_action = StoreAction::DontCare;
            [pass.render_target_width, pass.render_target_height] = area;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        draw(&texture);
        draw(&private);
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
        drop((texture, buffer, private, scratch));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A pass's copies in share their barriers (`encoder.rs`'s `ShadowRun`): a pass that loads a
/// shadowed texture over a `Shared` buffer and draws `n` rectangles, each under its own scissor
/// reading its own shadowed texture over the same buffer, copies the `n` textures in, and the
/// loaded texture in and kept (it merges back), before it. The copies in of different textures
/// share one barrier before them and one after; the keep, which reads the shadow its texture's
/// copy in wrote, takes a barrier of its own. So the pass records as many pipeline barriers
/// (`WorkCounts::commands_replayed`) whatever `n`, where a barrier before and one after each copy
/// grew them by two a texture (counted after a first pass of the shape). The buffer holds each
/// texture's texels inside its rectangle, as the draws read them, and every other byte as it
/// was. Printed: `SHADOW-RUN`.
#[test]
fn a_pass_s_copies_in_share_their_barriers() {
    let layout = LAYOUTS[4];
    assert!(layout.padded(), "{}", layout.name);
    let most = 4;
    let stride = (layout.len() as u64).next_multiple_of(4096);
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = stride * (most + 1);
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        // The textures each rectangle reads at the buffer's start, the loaded one after them.
        let texture_at = |at: u64| {
            let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
            d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            d.resource_options = buffer.resource_options();
            let texture = buffer.new_texture(&d, at, layout.bpr).unwrap();
            // SAFETY: the handle is only compared, while the texture lives.
            assert!(unsafe { texture.vk_image() }.is_none(), "a shadow");
            texture
        };
        let read: Vec<Texture> = (0..most).map(|k| texture_at(k * stride)).collect();
        let loaded = texture_at(most * stride);
        let reads = reading(&device, layout.pf, "tob_read");
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        // Rectangle `k` of `n`: a band of the texture's rows.
        let band = |k: u64, n: u64| {
            let rows = layout.h / n;
            ScissorRect::new(0, k * rows, layout.w, rows)
        };
        let pass = |n: u64| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(loaded.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&reads).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            for (k, texture) in read.iter().take(n as usize).enumerate() {
                encoder.set_scissor_rect(band(k as u64, n)).unwrap();
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
            let before = device.work_counts().commands_replayed.cmd_pipeline_barrier;
            run(&command_buffer);
            device.work_counts().commands_replayed.cmd_pipeline_barrier - before
        };
        // A first pass makes what every pass of the shape needs (the textures' first moves, the
        // pipelines' sets), then the counted passes.
        let mut expected = bytes.clone();
        let mut barriers = Vec::new();
        for n in [most, 1, most] {
            barriers.push(pass(n));
            // Each band of the loaded texture's rows holds its texture's texels there.
            for k in 0..n {
                let rect = band(k, n);
                for y in rect.y..rect.y + rect.height {
                    let row = (y * layout.bpr) as usize;
                    let at = (most * stride) as usize + row;
                    let from = (k * stride) as usize + row;
                    let width = layout.row() as usize;
                    let texels = bytes[from..from + width].to_vec();
                    expected[at..at + width].copy_from_slice(&texels);
                }
            }
            let now = bytes_of(&buffer);
            let (count, first) = differing(&now, &expected, 1);
            assert!(
                count == 0,
                "after a pass of {n} rectangles {count} bytes differ, the first at {first:?}"
            );
        }
        println!(
            "SHADOW-RUN on {}: a pass reading 1 shadowed texture recorded {} pipeline barriers, \
             one reading {most} recorded {}",
            device.name(),
            barriers[1],
            barriers[2],
        );
        assert_eq!(
            barriers[1], barriers[2],
            "barriers of a pass of 1 and of {most} copies in"
        );
        drop((read, loaded, buffer));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A pass that loads a shadowed texture and draws under a scissor beside a depth attachment it
/// clears and stores, which reaches every tile (`clear_reach::PassReach::Every`), `past` pixels
/// wider and taller than the color: where that is some, the depth's clear reaches past the render
/// area to whole tiles and is made before the pass (`clear_reach::ReachClears`,
/// `WorkCounts::attachment_clears_recorded`). Neither clear writes the color, so the draws alone
/// write it, inside the scissor, and its copy in and keep move the texels inside the scissor alone
/// (`WorkCounts::shadow_texels_copied_in`, `shadow_texels_kept`,
/// `shadow_attachment_loads_within`). The buffer holds the blend inside the scissor, as a private
/// texture of the same texels drawn alike does, and every other byte as it was.
fn a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(past: u64) {
    let layout = LAYOUTS[0];
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let len = layout.bpr * layout.h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 7);
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
        // The render area is the color's, 1307 pixels wide, which the depth's clear reaches past
        // to 1312, a whole number of 32-pixel tiles, where the depth is wider.
        let mut z = TextureDescriptor::texture_2d(
            PixelFormat::Depth32Float,
            layout.w + past,
            layout.h + past,
            false,
        );
        z.usage = TextureUsage::RENDER_TARGET;
        z.resource_options = ResourceOptions::private();
        let depth = device.new_texture(&z).unwrap();
        let blend = rects_beside(&device, layout.pf, true, Some(PixelFormat::Depth32Float));
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let color = staged(&device, &floats([0.25, 0.5, 0.75, 0.5]));
        let scissor = ScissorRect::new(100, 40, 300, 200);
        let draw = |target: &Texture| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::Store;
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_scissor_rect(scissor).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            encoder.set_fragment_buffer(Some(&color), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            run(&command_buffer);
        };
        let moved = || {
            let counts = device.work_counts();
            [
                counts.attachment_clears_recorded,
                counts.shadow_texels_copied_in,
                counts.shadow_texels_kept,
                counts.shadow_attachment_loads_within,
                counts.shadow_attachment_loads_whole,
            ]
        };
        let before = moved();
        draw(&texture);
        let after = moved();
        let [clears, copied_in, kept, within, whole] =
            [0, 1, 2, 3, 4].map(|i| after[i] - before[i]);
        println!(
            "CLEARED-DEPTH-BOUND on {}: a scissored pass beside a cleared depth {past} pixels \
             past the color recorded {clears} attachment clears, copied in {copied_in} texels \
             and kept {kept} ({within} colors loaded within the scissor, {whole} whole)",
            device.name(),
        );
        assert_eq!(
            clears > 0,
            past > 0,
            "the depth's clear past the render area"
        );
        assert_eq!(
            (copied_in, kept, within, whole),
            (300 * 200, 300 * 200, 1, 0),
            "the color's copies inside the scissor"
        );
        draw(&private);
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
        drop((texture, buffer, private, depth));
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] beside a depth of the
/// color's size.
#[test]
fn a_loaded_color_beside_a_cleared_depth_of_its_size_copies_inside_its_scissor() {
    a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(0);
}

/// [`a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor`] beside a larger depth,
/// whose clear reaches past the render area.
#[test]
fn a_clear_past_the_render_area_leaves_a_loaded_color_s_copies_inside_its_scissor() {
    a_loaded_color_beside_a_cleared_depth_copies_inside_its_scissor(64);
}

/// One draw of [`merged_writes_match_metal`]'s: its scissor (none: the whole target), its
/// viewport (none: the whole target), its color, and a barrier over textures before it, which
/// splits the pass there.
#[derive(Clone, Copy)]
struct Stroke {
    scissor: Option<ScissorRect>,
    viewport: Option<[f64; 4]>,
    color: usize,
    split: bool,
}

/// [`Stroke`] under `scissor` alone.
fn stroke(scissor: (u64, u64, u64, u64), color: usize) -> Stroke {
    let (x, y, w, h) = scissor;
    Stroke {
        scissor: Some(ScissorRect::new(x, y, w, h)),
        viewport: None,
        color,
        split: false,
    }
}

/// Passes that load a shadowed texture over a buffer's bytes and draw under scissors that change
/// from draw to draw, in instances a barrier splits, across encoders and command buffers, with a
/// viewport and no scissor, under scissors at the right and bottom edges, and through a second
/// texture over the same bytes between them, leave the bytes as a private texture of the same
/// texels drawn alike holds them: each merge's keep and merge are cut to the rectangle the
/// encoder's commands write in (`shadow_merge.rs`), and no write outside it is lost. Over a
/// `Shared` buffer and over a no-copy buffer's host pages, of `layout`.
fn merged_writes_match_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let blend = blending(&device, layout.pf);
        let corners = staged(&device, &floats([-1.0, -1.0, 1.0, 1.0]));
        let colors = [
            staged(&device, &floats([0.25, 0.5, 0.75, 0.5])),
            staged(&device, &floats([0.9, 0.1, 0.3, 0.75])),
        ];
        let (w, h) = (layout.w, layout.h);
        // Each command buffer's passes, each pass's strokes, the second texture's pass marked.
        // Every scissor lies inside the target, as Metal's validation requires, at fractions of
        // its size, so each stroke draws over every layout.
        let passes: Vec<Vec<(bool, Vec<Stroke>)>> = vec![
            vec![
                (
                    false,
                    vec![
                        stroke((w / 13, h / 12, w / 4, h / 3), 0),
                        stroke((w / 2, h / 2, w / 4, h / 4), 1),
                    ],
                ),
                (
                    false,
                    vec![
                        stroke((w / 26, h / 6, w / 5, h / 5), 1),
                        Stroke {
                            split: true,
                            ..stroke((w * 5 / 8, h * 5 / 8, w / 4, h / 5), 0)
                        },
                        Stroke {
                            split: true,
                            ..stroke((5, h * 5 / 8, w / 9, h / 3), 1)
                        },
                    ],
                ),
            ],
            vec![
                (true, vec![stroke((w / 3, h / 10, w / 4, h / 2), 0)]),
                (
                    false,
                    vec![Stroke {
                        scissor: None,
                        viewport: Some([
                            w as f64 / 4.0,
                            h as f64 / 5.0,
                            w as f64 / 3.0,
                            h as f64 / 4.0,
                        ]),
                        color: 1,
                        split: false,
                    }],
                ),
                (
                    false,
                    vec![
                        stroke((w - 13, h - 7, 13, 7), 0),
                        stroke((w - 1, 0, 1, h), 1),
                    ],
                ),
                (false, vec![stroke((0, 0, w, h), 0)]),
            ],
        ];
        let encode = |command_buffer: &metal2vulkan_runtime::CommandBuffer,
                      target: &Texture,
                      strokes: &[Stroke]| {
            let mut pass = RenderPassDescriptor::new();
            let attachment = &mut pass.color_attachments[0];
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Load;
            attachment.store_action = StoreAction::Store;
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&blend).unwrap();
            encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
            for s in strokes {
                if s.split {
                    encoder.memory_barrier_with_scope(
                        BarrierScope::TEXTURES,
                        RenderStages::FRAGMENT,
                        RenderStages::FRAGMENT,
                    );
                }
                let scissor = s.scissor.unwrap_or(ScissorRect::new(0, 0, w, h));
                assert!(
                    scissor.width > 0
                        && scissor.height > 0
                        && scissor.x + scissor.width <= w
                        && scissor.y + scissor.height <= h,
                    "{}: the scissor {scissor:?} lies outside the {w}x{h} target",
                    layout.name
                );
                encoder.set_scissor_rect(scissor).unwrap();
                let [x, y, vw, vh] = s.viewport.unwrap_or([0.0, 0.0, w as f64, h as f64]);
                encoder
                    .set_viewport(Viewport::new(x, y, vw, vh, 0.0, 1.0))
                    .unwrap();
                encoder
                    .set_fragment_buffer(Some(&colors[s.color]), 0, 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
        };
        // The passes through `first` (and `second`, a texture over the same bytes, where marked),
        // command buffer by command buffer, waited for at the end.
        let draw_all = |first: &Texture, second: &Texture| {
            let mut committed = Vec::new();
            for buffer_passes in &passes {
                let command_buffer = queue.command_buffer().unwrap();
                for (twin, strokes) in buffer_passes {
                    encode(&command_buffer, if *twin { second } else { first }, strokes);
                }
                command_buffer.commit().unwrap();
                committed.push(command_buffer);
            }
            for command_buffer in &committed {
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.error(), None);
            }
        };
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let texture_over = |buffer: &Buffer| {
            let mut d = TextureDescriptor::texture_2d(layout.pf, w, h, false);
            d.usage = usage;
            d.resource_options = buffer.resource_options();
            buffer.new_texture(&d, 0, layout.bpr).unwrap()
        };
        // Over a `Shared` buffer.
        let len = layout.bpr * h;
        let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
        let bytes = texels_of(&layout, len as usize, 3);
        let contents = buffer.contents().unwrap();
        // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        let (first, second) = (texture_over(&buffer), texture_over(&buffer));
        // SAFETY: the handle is only compared, while the texture lives.
        assert!(unsafe { first.vk_image() }.is_none(), "a shadow");
        let private = private_with(&queue, &layout, &rows(&layout, &bytes));
        draw_all(&first, &second);
        draw_all(&private, &private);
        let mut expected = bytes.clone();
        lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
        let now = bytes_of(&buffer);
        let (count, at) = differing(
            &rows(&layout, &now),
            &rows(&layout, &expected),
            layout.texel,
        );
        assert!(
            count == 0,
            "{}, a Shared buffer: {count} texels differ from the private texture's, the first at \
             texel {at:?}",
            layout.name
        );
        assert!(
            now == expected,
            "{}, a Shared buffer: the bytes between the rows changed",
            layout.name
        );
        drop((first, second, buffer, private));
        // Over a no-copy buffer's host pages.
        if let Some(surface) = surface(&device, &layout, usage) {
            let second = surface
                .buffer
                .new_texture(
                    &{
                        let mut d = TextureDescriptor::texture_2d(layout.pf, w, h, false);
                        d.usage = usage;
                        d.resource_options = surface.buffer.resource_options();
                        d
                    },
                    0,
                    layout.bpr,
                )
                .unwrap();
            let private = private_with(&queue, &layout, &rows(&layout, &surface.pages));
            draw_all(&surface.texture, &second);
            draw_all(&private, &private);
            let mut expected = surface.pages.clone();
            lay(&layout, &mut expected, &blit_out(&queue, &layout, &private));
            surface.check(&layout, "passes cut to their scissors", &expected);
            drop((second, private));
            surface.report(&device, &messages);
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

#[test]
fn merged_writes_match_metal_over_the_browser_tile() {
    merged_writes_match_metal(LAYOUTS[0]);
}

#[test]
fn merged_writes_match_metal_over_a_half_float_tile() {
    merged_writes_match_metal(LAYOUTS[7]);
}

#[test]
fn merged_writes_match_metal_over_a_narrow_tile() {
    merged_writes_match_metal(LAYOUTS[6]);
}

/// A texture of `layout` over a no-copy buffer, with shader read and write usage, written at every
/// texel by a fragment function's `write` (`tob_write`): the pages' texels are what a private
/// texture of the same texels holds after the same writes, and no other byte of the pages changes.
fn writes_as_metal(layout: Layout) {
    let cache = ScratchCache::new().unwrap();
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

/// Run `test` on one lane device, of `padded_reads` where given (else the lane's own policy,
/// `PaddedReadsPolicy::Shadow` with `exact_answers`),
/// which announces `shadowed-texture` for each texture so that each case prints its own reason;
/// then fail on any validation message, leaks included. Each case takes its own validation
/// messages first, to fail by name.
fn on_shared_device(
    padded_reads: Option<PaddedReadsPolicy>,
    test: impl FnOnce(&Device, &Messages),
) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            if let Some(policy) = padded_reads {
                options.policy.padded_reads = policy;
            }
            options.announce_each.insert(Degradation::ShadowedTexture);
        })
        .unwrap();
        test(&device, &messages);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// `boxed` (rows of `bw` texels) laid into `expected` at texel (`bx`, `by`) of `layout`'s rows.
fn lay_box(layout: &Layout, expected: &mut [u8], (bx, by, bw): (u64, u64, u64), boxed: &[u8]) {
    for (y, row) in boxed.chunks((bw * layout.texel) as usize).enumerate() {
        let at = ((by + y as u64) * layout.bpr + bx * layout.texel) as usize;
        expected[at..at + row.len()].copy_from_slice(row);
    }
}

/// A texture of `layout` that only reads (`ShaderRead` usage), over a no-copy buffer, on a device
/// of the policy's default (`PaddedReadsPolicy::Alias`): its first use reads it through its image
/// (a fragment function's `read`) before any command reads or writes the pages another way, and
/// must read the pages' texels and leave every byte of the pages, the bytes between the rows
/// among them, as it was: the host keeps them across the image's first move
/// (docs/FAST-PATHS.md's known gaps). Then its reads (a nearest and a linear sample, `get_bytes`,
/// a blit out), the caller's later bytes (texels and the bytes between the rows alike) read by a
/// sample and a blit, and its writes that Metal makes of any texture (a buffer's box blitted in
/// at an odd origin, `replace_region` of a box, a private texture's box blitted in): each reads
/// Metal's texels, and after each every byte of the pages is the program's. Whether it aliased,
/// and whether the device runs on Metal; `None` where it imports no host memory.
fn reads_and_blits_as_metal(
    device: &Device,
    messages: &Messages,
    layout: Layout,
) -> Option<(bool, bool)> {
    let made = {
        let queue = device.new_command_queue().unwrap();
        let texel = layout.texel;
        let surface = surface(device, &layout, TextureUsage::SHADER_READ, true)?;
        let (texture, what) = (&surface.texture, &surface.what);
        let mut expected = surface.pages.clone();
        let check = |step: &str, expected: &[u8]| surface.check(&layout, step, expected);
        let read = |step: &str, ours: &[u8], theirs: &[u8]| {
            let (count, first) = differing(ours, theirs, texel);
            assert!(
                count == 0,
                "{what}, {step}: {count} texels differ, the first at texel {first:?}"
            );
        };

        // The first use, through the image, before anything else touches the pages.
        let texels = rows(&layout, &expected);
        let reads = reading(device, layout.pf, "tob_read");
        read(
            "its first use, a fragment function's read",
            &drawn(&queue, &layout, &reads, texture, None),
            &texels,
        );
        check("its first use", &expected);

        // Its other reads.
        let samples = reading(device, layout.pf, "tob_sample");
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
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
        let mut got = vec![0u8; texels.len()];
        let whole = Region::new_2d(0, 0, layout.w, layout.h);
        texture
            .get_bytes(&mut got, layout.row(), 0, whole, 0, 0)
            .unwrap();
        read("get_bytes", &got, &texels);
        read("a blit out", &blit_out(&queue, &layout, texture), &texels);
        check("the reads", &expected);

        // The caller's later bytes: new texels, and new bytes between the rows, which no later
        // command may move.
        let mut later_pages = counting(surface.len, 9);
        let later = texels_of(&layout, texels.len(), 2);
        lay(&layout, &mut later_pages, &later);
        expected.copy_from_slice(&later_pages);
        surface.mapping.write(0, &expected);
        read(
            "a nearest sample of the caller's later bytes",
            &drawn(&queue, &layout, &samples, texture, Some(&nearest)),
            &later,
        );
        read(
            "a fragment function's read of the caller's later bytes",
            &drawn(&queue, &layout, &reads, texture, None),
            &later,
        );
        check("the reads of the caller's later bytes", &expected);

        // Writes Metal makes of any texture: a buffer's box blitted in at an odd origin.
        let (bx, by, bw, bh) = (3, 5, layout.w / 2 + 1, layout.h / 3 + 1);
        let boxed = texels_of(&layout, (bw * bh * texel) as usize, 3);
        let source = staged(device, &boxed);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            bw * texel,
            boxed.len() as u64,
            Size::new(bw, bh, 1),
            texture,
            0,
            0,
            Origin::new(bx, by, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        lay_box(&layout, &mut expected, (bx, by, bw), &boxed);
        check("a buffer's box blitted in", &expected);
        // `replace_region` of a box.
        let (rx, ry, rw, rh) = (layout.w / 3, 1, layout.w / 4 + 1, layout.h / 2);
        let replaced = texels_of(&layout, (rw * rh * texel) as usize, 4);
        texture
            .replace_region(
                Region::new_2d(rx, ry, rw, rh),
                0,
                0,
                &replaced,
                rw * texel,
                0,
            )
            .unwrap();
        lay_box(&layout, &mut expected, (rx, ry, rw), &replaced);
        check("replace_region", &expected);
        // A private texture's box blitted in.
        let other = private_with(&queue, &layout, &texels_of(&layout, texels.len(), 5));
        let (fx, fy, tx, ty) = (7, 1, layout.w / 3, 2);
        let size = Size::new(layout.w / 2, layout.h / 2, 1);
        blit_box(
            &queue,
            (&other, Origin::new(fx, fy, 0)),
            (texture, Origin::new(tx, ty, 0)),
            size,
        );
        let from = blit_out(&queue, &layout, &other);
        for y in 0..size.height {
            let start = ((fy + y) * layout.row() + fx * texel) as usize;
            let row = &from[start..start + (size.width * texel) as usize];
            let at = ((ty + y) * layout.bpr + tx * texel) as usize;
            expected[at..at + row.len()].copy_from_slice(row);
        }
        check("a private texture's box blitted in", &expected);
        // And the writes read back through the image.
        read(
            "a fragment function's read after the writes",
            &drawn(&queue, &layout, &reads, texture, None),
            &rows(&layout, &expected),
        );
        check("the last read", &expected);
        let made = (surface.form.0, on_metal(device));
        surface.report(device, messages);
        made
    };
    assert_eq!(
        messages.take(),
        Vec::new(),
        "{}: the validation layer reported",
        layout.name
    );
    Some(made)
}

/// [`reads_and_blits_as_metal`] over each of [`LAYOUTS`]: a texture that only reads, aliased
/// where the device lays an image at the rows' pitch, keeps every byte of its pages through its
/// first move, its reads, the caller's writes and Metal's blits into it. Printed: how many of the
/// padded layouts aliased (`PITCHED-READS`); off Metal at least one must, or no first move over
/// padded rows was checked. Every layout on one device of the policy's default.
#[test]
fn a_texture_that_only_reads_keeps_every_byte_of_its_pages() {
    let (mut aliased, mut padded, mut metal) = (0, 0, false);
    let mut imports = true;
    on_shared_device(Some(PaddedReadsPolicy::Alias), |device, messages| {
        for layout in LAYOUTS {
            let Some((alias, on_metal)) = reads_and_blits_as_metal(device, messages, layout) else {
                imports = false;
                return;
            };
            metal = on_metal;
            if layout.padded() {
                aliased += u64::from(alias);
                padded += 1;
            }
        }
    });
    if !imports {
        return;
    }
    eprintln!("PITCHED-READS: {aliased} of {padded} padded layouts that only read aliased");
    assert!(
        aliased > 0 || metal,
        "no padded layout that only reads was aliased, so no first move over padded rows was \
         checked"
    );
}

/// A texture that only reads over padded rows (each padded layout of [`LAYOUTS`] over a no-copy
/// buffer) copies no shadow under the policy's default where it is aliased:
/// `WorkCounts::shadow_copies` stays the same across three sampling passes and a blit out. Under
/// `PaddedReadsPolicy::Shadow` the same texture is a shadow (but on Metal, whose image of it is a
/// Metal texture over the buffer either way), copied in for its uses. Off Metal at least one layout
/// is aliased by default. Every layout of a policy on one device of that policy.
#[test]
fn a_padded_texture_that_only_reads_copies_no_shadow_by_default() {
    let mut aliased_layouts = 0;
    let mut metal = false;
    let mut imports = true;
    for policy in [PaddedReadsPolicy::Alias, PaddedReadsPolicy::Shadow] {
        on_shared_device(Some(policy), |device, messages| {
            for layout in LAYOUTS.into_iter().filter(Layout::padded) {
                metal = on_metal(device);
                let queue = device.new_command_queue().unwrap();
                let alias = policy == PaddedReadsPolicy::Alias;
                let usage = TextureUsage::SHADER_READ;
                let Some(surface) = surface(device, &layout, usage, alias) else {
                    imports = false;
                    return;
                };
                let (aliased, _, _) = surface.form;
                let what = format!("{} under {policy:?}", surface.what);
                if !alias {
                    assert!(!aliased || metal, "{what}: aliased");
                }
                let texels = rows(&layout, &surface.pages);
                let samples = reading(device, layout.pf, "tob_sample");
                let nearest = sampler(device, SamplerMinMagFilter::Nearest);
                let copies = || device.work_counts().shadow_copies;
                let before = copies();
                for _ in 0..3 {
                    let got = drawn(&queue, &layout, &samples, &surface.texture, Some(&nearest));
                    assert!(
                        got == texels,
                        "{what}: a sample differs from the pages' texels"
                    );
                }
                assert!(
                    blit_out(&queue, &layout, &surface.texture) == texels,
                    "{what}"
                );
                let copied = copies() - before;
                eprintln!(
                    "PITCHED-READS on {}: {what}: {}, {copied} shadow copies over three samples \
                     and a blit out",
                    device.name(),
                    if aliased { "aliased" } else { "shadowed" }
                );
                if aliased {
                    assert_eq!(copied, 0, "{what}: an aliased texture copied a shadow");
                    aliased_layouts += u64::from(alias);
                } else {
                    assert!(copied > 0, "{what}: a shadow its uses copied nothing into");
                }
                surface.check(&layout, "its uses", &surface.pages);
                surface.report(device, messages);
                assert_eq!(
                    messages.take(),
                    Vec::new(),
                    "{what}: the validation layer reported"
                );
            }
        });
        if !imports {
            return;
        }
    }
    assert!(
        aliased_layouts > 0 || metal,
        "no padded layout that only reads was aliased by default"
    );
}

/// A padded texture that renders (`RenderTarget`), that a shader writes (`ShaderWrite`), or of
/// `Unknown` usage (which Metal reports as both) is a shadow under either policy, off Metal: over
/// a no-copy buffer's pages and over an ordinary buffer alike (the browser's tile, [`LAYOUTS`]'
/// first). Every usage of a policy on one device of that policy.
#[test]
fn a_padded_texture_that_renders_or_writes_is_a_shadow_under_either_policy() {
    let layout = LAYOUTS[0];
    assert!(layout.padded(), "{}", layout.name);
    for policy in [PaddedReadsPolicy::Alias, PaddedReadsPolicy::Shadow] {
        on_shared_device(Some(policy), |device, messages| {
            for usage in [
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                TextureUsage::UNKNOWN,
            ] {
                // Over a no-copy buffer's pages (`surface` checks the form against the counts).
                if let Some(surface) = surface(device, &layout, usage, false) {
                    surface.report(device, messages);
                }
                // Over an ordinary buffer.
                let buffer = device
                    .new_buffer(layout.bpr * layout.h, ResourceOptions::default())
                    .unwrap();
                let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
                d.usage = usage;
                d.resource_options = buffer.resource_options();
                let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
                // SAFETY: the handle is only compared, while the texture lives.
                let aliased = unsafe { texture.vk_image() }.is_some();
                assert!(
                    !aliased,
                    "{} ({usage:?}) under {policy:?}, an ordinary buffer's: aliased",
                    layout.name
                );
                messages.take_notices();
                drop((texture, buffer));
                assert_eq!(
                    messages.take(),
                    Vec::new(),
                    "{} ({usage:?}) under {policy:?}: the validation layer reported",
                    layout.name
                );
            }
        });
    }
}

/// A texture that only reads over an ordinary `Shared` buffer's padded rows (each padded layout
/// of [`LAYOUTS`]): under the policy's default it is the device's own linear image over the
/// buffer's memory, made `PREINITIALIZED`, where the device lays one at the rows' pitch, and its
/// first use, a fragment function's read through it, reads the buffer's texels and leaves every
/// byte of the buffer, the bytes between the rows among them, as it was; an aliased one copies
/// no shadow over that read and a nearest sample. Under `PaddedReadsPolicy::Shadow` it is a
/// shadow on every device. Printed (`PITCHED-READS`): each layout's form, and a shadow's reason.
/// Every layout of a policy on one device of that policy.
#[test]
fn a_padded_texture_that_only_reads_over_an_ordinary_buffer_keeps_its_bytes() {
    for policy in [PaddedReadsPolicy::Alias, PaddedReadsPolicy::Shadow] {
        on_shared_device(Some(policy), |device, messages| {
            for layout in LAYOUTS.into_iter().filter(Layout::padded) {
                let queue = device.new_command_queue().unwrap();
                let len = layout.bpr * layout.h;
                let buffer = device.new_buffer(len, ResourceOptions::default()).unwrap();
                let mut bytes = counting(len as usize, 11);
                lay(
                    &layout,
                    &mut bytes,
                    &texels_of(&layout, (layout.row() * layout.h) as usize, 6),
                );
                let contents = buffer.contents().unwrap();
                // SAFETY: a new `Shared` buffer of `len` bytes, which no command uses yet.
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
                };
                let mut d = TextureDescriptor::texture_2d(layout.pf, layout.w, layout.h, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = buffer.resource_options();
                let texture = buffer.new_texture(&d, 0, layout.bpr).unwrap();
                // SAFETY: the handle is only compared, while the texture lives.
                let aliased = unsafe { texture.vk_image() }.is_some();
                let what = format!(
                    "{} (ShaderRead) under {policy:?}, an ordinary buffer's",
                    layout.name
                );
                if policy == PaddedReadsPolicy::Shadow {
                    assert!(!aliased, "{what}: aliased");
                }
                let texels = rows(&layout, &bytes);
                let copies = || device.work_counts().shadow_copies;
                let before = copies();
                let reads = reading(device, layout.pf, "tob_read");
                let read = drawn(&queue, &layout, &reads, &texture, None);
                let (count, first) = differing(&read, &texels, layout.texel);
                assert!(
                    count == 0,
                    "{what}, its first use: {count} texels read differ from the buffer's, the \
                     first at texel {first:?}"
                );
                let now = bytes_of(&buffer);
                assert!(
                    now == bytes,
                    "{what}, its first use: the bytes between the rows (or past the last) \
                     changed: {}",
                    padding_changes(&layout, &now, &bytes)
                );
                let samples = reading(device, layout.pf, "tob_sample");
                let nearest = sampler(device, SamplerMinMagFilter::Nearest);
                let got = drawn(&queue, &layout, &samples, &texture, Some(&nearest));
                assert!(
                    got == texels,
                    "{what}: a sample differs from the buffer's texels"
                );
                let copied = copies() - before;
                if aliased {
                    assert_eq!(copied, 0, "{what}: an aliased texture copied a shadow");
                }
                assert!(
                    bytes_of(&buffer) == bytes,
                    "{what}: the buffer's bytes changed"
                );
                let reason = messages
                    .take_notices()
                    .into_iter()
                    .filter_map(|message| message.degradation)
                    .find(|notice| notice.degradation == Degradation::ShadowedTexture)
                    .map(|notice| format!(": {}", notice.reason))
                    .unwrap_or_default();
                eprintln!(
                    "PITCHED-READS on {}: {what}: {}, {copied} shadow copies over a read and a \
                     sample{reason}",
                    device.name(),
                    if aliased { "aliased" } else { "shadowed" }
                );
                drop((texture, buffer));
                assert_eq!(
                    messages.take(),
                    Vec::new(),
                    "{what}: the validation layer reported"
                );
            }
        });
    }
}
