//! Render passes that load an `R10Unorm_X6` or `RG10Unorm_X12` texture, or one of their `_sRGB`
//! twins, over a buffer and draw on the device the cfg selects, under the validation layer, against
//! Metal's bytes.
//!
//! Each case replays one of Metal probes `x6i`, `x6j` and `x6k` (an Apple M4 under macOS 27.0.1 and
//! an Apple M3 Max under macOS 26.7, byte-identical): a 200 x 100 texture over a buffer holding the
//! probe's words (texel `i`'s red word `(i * 2654435761) >> 7` and green word `i * 40503`, each to
//! 16 bits), loaded and stored, and the owned fixture `rt6_samples`'s full-screen triangle filling
//! `float4` values `((37 i) mod 1024 / 1023, (101 i) mod 1024 / 1023, 0.5, 1)` at texel `i`
//! (`rt6_fill_rgba32float`), under a write mask: drawn whole, under a scissor rectangle (37, 13,
//! 54, 57), in a pass of a smaller render target size, as a draw of no vertices, as an indirect
//! command buffer's execution of the draw, or not drawn. And the probe `x6l` (byte-identical): the
//! triangle drawn into a view of the texture as a 2D array of its one slice (in a pass of a render
//! target array length of 0 and of 1), and into a texture buffer of one row over the buffer's
//! first, whose pass rewrites that row alone. And the probe `x6is` (byte-identical): the same words
//! and draw over `R10Unorm_X6_sRGB` and `RG10Unorm_X12_sRGB`, drawn whole, as a draw that
//! rasterizes nothing (which the lane draws as one of no vertices: any draw rewrites alike), or not
//! drawn. Metal's answer: the SHA-256 of the texture's words after the pass, row after row
//! ([`METAL`]); the bytes between the rows stay as they were.
//!
//! Metal's pass that loads such a texture and draws writes every word of its render area rounded
//! up to 32 texels back, the words no fragment writes and the channels the mask leaves out as
//! the write of their code's read (`ExtendedRange::load_code`: their code with its low six bits
//! zeroed, an `_sRGB` twin's through Metal's table). The runtime rewrites them so before the
//! pass (`render/unused_bits_load.rs`), so every case is Metal's, and none announces
//! `extended-range-bits-kept` (once `padded-bits-kept`). (The probes' passes also cleared and
//! stored a second color attachment, under which Metal's pass rewrites every tile of the render
//! area, the tiles no draw reaches too. The lane's draw of no vertices and its scissored draw have
//! it too, since without it the first reaches no tile and the pass keeps every word, and the second
//! only the tiles its scissor rectangle overlaps; the lane's other passes have none, and their
//! draws reach every tile alike.) The device's work counts show the path ran: one load pass per
//! case that draws, none for the case that does not, over the region's texels
//! (`WorkCounts::unused_bits_load_passes` and `unused_bits_load_texels`).

use super::degrade_notices::counts;
use super::extended_range_targets::digest;
use super::resolves::{lane, quiet, Lane};
use super::{buffer_with, read_buffer};
use crate::candidate::runtime_samples::{library, shape};
use metal2vulkan::metal::{
    ColorWriteMask, IndirectCommandType, PixelFormat, PrimitiveType, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Buffer, Degradation, IndirectCommandBufferDescriptor, LoadAction, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, ScissorRect, StoreAction, TextureDescriptor,
};

/// The probes' texture size.
const WIDTH: u64 = 200;
const HEIGHT: u64 = 100;

/// The byte the buffer holds between rows.
const PADDING: u8 = 0xa5;

/// How a case's pass draws.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Pass {
    /// The full-screen triangle.
    Whole,
    /// The triangle under the scissor rectangle (37, 13, 54, 57).
    Scissored,
    /// The triangle in a pass whose render target size is the one given.
    Sized(u64, u64),
    /// A draw of no vertices.
    NoVertices,
    /// An indirect command buffer's execution of the triangle's draw.
    Executed,
    /// No draw.
    Undrawn,
    // The triangle into another attachment over the same bytes:
    /// a view of the texture as a 2D array of its one slice;
    ArrayView,
    /// that view, in a pass of a render target array length of 1;
    ArrayViewOfOne,
    /// a texture buffer of one row over the buffer's first.
    TextureBuffer,
}

/// Metal's words after each case (the module docs): the format, the write mask's raw bits (red
/// 8, green 4), the pass, and the SHA-256 of the words.
const METAL: [(PixelFormat, u64, Pass, &str); 57] = {
    use Pass::*;
    use PixelFormat::{
        R10UnormX6 as R, R10UnormX6SRGB as RS, RG10UnormX12 as RG, RG10UnormX12SRGB as RGS,
    };
    [
        (
            R,
            0,
            Whole,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            Scissored,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            Sized(120, 60),
            "b8e7532fe68533dc27f198c7d2d86bb0124bf35f73b0c51a0dd9dc710ac106e4",
        ),
        (
            R,
            0,
            NoVertices,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            Executed,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            Undrawn,
            "df24dccce9d2b31886a249eeccd6f4ac331fc09c009bd555b65f7950d4ae865a",
        ),
        (
            R,
            8,
            Whole,
            "b25968c68482feadf323c22d2e6a6a028d2a740121f6862170f0d241d5714844",
        ),
        (
            R,
            8,
            Scissored,
            "eb05566ff4ba4cebe6e6f28da6119349498fce9051d4f5814dc8e51f9b7f275e",
        ),
        (
            R,
            15,
            Whole,
            "b25968c68482feadf323c22d2e6a6a028d2a740121f6862170f0d241d5714844",
        ),
        (
            R,
            15,
            Scissored,
            "eb05566ff4ba4cebe6e6f28da6119349498fce9051d4f5814dc8e51f9b7f275e",
        ),
        (
            R,
            15,
            Sized(120, 60),
            "b402eb14be70b3e76664ca98cde398f7b7d4b255b2810987a09dc876c280315e",
        ),
        (
            R,
            15,
            Sized(70, 20),
            "c4588c294038bdaad3ecc1d6394672e29c4909a2db9577b9a4cc2658cc76ef02",
        ),
        (
            R,
            15,
            Sized(33, 17),
            "745939fa5736a5a032ad8c8e7cee48dc418b3928c34f8ddf94301cbec347c95d",
        ),
        (
            R,
            15,
            NoVertices,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            15,
            Executed,
            "b25968c68482feadf323c22d2e6a6a028d2a740121f6862170f0d241d5714844",
        ),
        (
            RG,
            0,
            Whole,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            Scissored,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            Sized(120, 60),
            "5aa6fc1f9cb8deff727ec76f01fd6c1895a8e5954088e7a29252bc2c26aba9c8",
        ),
        (
            RG,
            0,
            NoVertices,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            Executed,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            Undrawn,
            "e0bdea941ab83e05377a3a285907e590f4fdc114a03a82870a70ce4bceaa84e9",
        ),
        (
            RG,
            4,
            Whole,
            "a483777b700e2706f2679ba0c7299feb4c789501bf06b1bc98c475e42cc3caab",
        ),
        (
            RG,
            4,
            Scissored,
            "bcc256dca258ac1562dccfa6629d3330dc004b433d64e5bf4aa7aa47526218b1",
        ),
        (
            RG,
            4,
            Sized(120, 60),
            "056bf32144be66899c9ff7adc013b4ffbb0d263cd3b41307ee8abdd00620a63e",
        ),
        (
            RG,
            4,
            NoVertices,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            4,
            Executed,
            "a483777b700e2706f2679ba0c7299feb4c789501bf06b1bc98c475e42cc3caab",
        ),
        (
            RG,
            8,
            Whole,
            "bbac224c5e31a1367f83e93571932fb81b3d7893e308dbd8b14ec96f487730e7",
        ),
        (
            RG,
            8,
            Scissored,
            "d52fd056a1204f847c209959717e5c7e2db28235537a676d55bb01e935f6013a",
        ),
        (
            RG,
            12,
            Whole,
            "cc74eaff3268792676e5c68e6bd2dc9ab4bc1f866b67b35a21779f4fc6b21e92",
        ),
        (
            RG,
            12,
            Scissored,
            "8c3d9ffe4f828be5620ec3a5ec7826e096e40c7e9c88613da036e5d8282499aa",
        ),
        (
            RG,
            12,
            Sized(120, 60),
            "5659771c1df3aae18c9aec0f44241371d792f8ed4b02575bb6283e0a3f7018a4",
        ),
        (
            RG,
            12,
            Sized(70, 20),
            "c9c2bd0b69cd7feaab207d9eb763ac5ebac1760f82fb36f90815c17625e11dd2",
        ),
        (
            RG,
            12,
            Sized(33, 17),
            "fb4bfcf264d80fe866de8cf4de7537fbd86a5f2d09da41519db1a887582b5e59",
        ),
        (
            RG,
            12,
            NoVertices,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            12,
            Executed,
            "cc74eaff3268792676e5c68e6bd2dc9ab4bc1f866b67b35a21779f4fc6b21e92",
        ),
        (
            R,
            0,
            ArrayView,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            ArrayViewOfOne,
            "dcb62d4893135cc2cb97545a6b8495d948fda7f31edf32e70b29c087b7b3f0b8",
        ),
        (
            R,
            0,
            TextureBuffer,
            "e3c3a746fb6c19851156aa4e9ba6423ab4c49334897d7ffc2a938831f7505d05",
        ),
        (
            R,
            8,
            ArrayView,
            "b25968c68482feadf323c22d2e6a6a028d2a740121f6862170f0d241d5714844",
        ),
        (
            R,
            8,
            ArrayViewOfOne,
            "b25968c68482feadf323c22d2e6a6a028d2a740121f6862170f0d241d5714844",
        ),
        (
            R,
            8,
            TextureBuffer,
            "de9930571e85e405d5229db52fd517334e43d64048f39d31a169a67594af4939",
        ),
        (
            RG,
            0,
            ArrayView,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            ArrayViewOfOne,
            "3361813774da80ef348bad3f34ee5b3b7e43b24db72c4ce17a84141735aa1b68",
        ),
        (
            RG,
            0,
            TextureBuffer,
            "8909d3c662abf331dc4f8236e2d81a077732bde4be813bc2a451f0cc7a203b27",
        ),
        (
            RG,
            12,
            ArrayView,
            "cc74eaff3268792676e5c68e6bd2dc9ab4bc1f866b67b35a21779f4fc6b21e92",
        ),
        (
            RG,
            12,
            ArrayViewOfOne,
            "cc74eaff3268792676e5c68e6bd2dc9ab4bc1f866b67b35a21779f4fc6b21e92",
        ),
        (
            RG,
            12,
            TextureBuffer,
            "9cca74d32e9edc5d5193cd09adbe525badfc352237879b336cecbeacf3a28c8a",
        ),
        (
            RS,
            0,
            Whole,
            "f8de684e5752b7a7050b8b5f1c33c6aeec56f94603112d2c5e77bb7d95ddccb7",
        ),
        (
            RS,
            0,
            NoVertices,
            "f8de684e5752b7a7050b8b5f1c33c6aeec56f94603112d2c5e77bb7d95ddccb7",
        ),
        (
            RS,
            0,
            Undrawn,
            "df24dccce9d2b31886a249eeccd6f4ac331fc09c009bd555b65f7950d4ae865a",
        ),
        (
            RS,
            8,
            Whole,
            "5c44072611e6628614a48d4113596596488d0ba04384b87e173887264d98a34f",
        ),
        (
            RS,
            8,
            NoVertices,
            "f8de684e5752b7a7050b8b5f1c33c6aeec56f94603112d2c5e77bb7d95ddccb7",
        ),
        (
            RGS,
            0,
            Whole,
            "5e52dc71d9a88d3906d5b19643a9174787b8478cb61d9359c575516e6040ac7b",
        ),
        (
            RGS,
            0,
            NoVertices,
            "5e52dc71d9a88d3906d5b19643a9174787b8478cb61d9359c575516e6040ac7b",
        ),
        (
            RGS,
            0,
            Undrawn,
            "e0bdea941ab83e05377a3a285907e590f4fdc114a03a82870a70ce4bceaa84e9",
        ),
        (
            RGS,
            12,
            Whole,
            "e84c60cf0fd47c1abdf0d212a7362a5a82055aa8e03b751e518e85270abd428e",
        ),
        (
            RGS,
            12,
            NoVertices,
            "5e52dc71d9a88d3906d5b19643a9174787b8478cb61d9359c575516e6040ac7b",
        ),
    ]
};

/// The write mask of Metal's raw bits `bits`.
fn mask(bits: u64) -> ColorWriteMask {
    [
        (8, ColorWriteMask::RED),
        (4, ColorWriteMask::GREEN),
        (2, ColorWriteMask::BLUE),
        (1, ColorWriteMask::ALPHA),
    ]
    .into_iter()
    .filter(|(bit, _)| bits & bit != 0)
    .fold(ColorWriteMask::NONE, |m, (_, c)| m.union(c))
}

/// The probe's words, row after row: each texel's red word, and its green word for `RG`.
fn words(channels: u64) -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .flat_map(|i| {
            let red = (i.wrapping_mul(2_654_435_761) >> 7) as u16;
            let green = i.wrapping_mul(40_503) as u16;
            let both = [red, green];
            both.into_iter().take(channels as usize).collect::<Vec<_>>()
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// The probe's `float4` values, one a texel.
fn values() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .flat_map(|i| {
            [
                ((37 * i) % 1024) as f32 / 1023.0,
                ((101 * i) % 1024) as f32 / 1023.0,
                0.5,
                1.0,
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// Run case `(pf, bits, pass)` on `lane`: the texture's words after it, row after row, and
/// whether every byte between the rows kept [`PADDING`].
fn run(lane: &Lane, (pf, bits, pass): (PixelFormat, u64, Pass)) -> (Vec<u8>, bool) {
    let channels = match pf {
        PixelFormat::R10UnormX6 | PixelFormat::R10UnormX6SRGB => 1,
        _ => 2,
    };
    let row_bytes = 2 * channels * WIDTH;
    let bpr = row_bytes.next_multiple_of(256);
    let mut bytes = vec![PADDING; (bpr * HEIGHT) as usize];
    for (row, words) in words(channels).chunks(row_bytes as usize).enumerate() {
        let at = row * bpr as usize;
        bytes[at..at + words.len()].copy_from_slice(words);
    }
    let buffer = buffer_with(&lane.device, &bytes).unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(pf, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = buffer
        .new_texture(&descriptor, 0, bpr)
        .unwrap_or_else(|e| panic!("{pf:?} over a buffer: {e}"));
    let library = library(&lane.device).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut state = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
    state.fragment = Some(function("rt6_fill_rgba32float"));
    let mut color = ColorAttachmentDescriptor::new(pf);
    color.write_mask = mask(bits);
    state.color_attachments[0] = Some(color);
    // The draw of no vertices and the scissored draw beside the probes' second color attachment,
    // cleared and stored, under which their pass reaches every tile (without it, the draw of no
    // vertices reaches none, and the scissored one the tiles its scissor rectangle overlaps: the
    // module docs).
    let cleared = matches!(pass, Pass::NoVertices | Pass::Scissored);
    if cleared {
        state.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    }
    state.support_indirect_command_buffers = pass == Pass::Executed;
    let state = lane
        .device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap();
    let values = buffer_with(&lane.device, &values()).unwrap();
    let shape: Buffer = shape(&lane.device, WIDTH as u32, 1).unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    // The attachment: the texture, a view of it, or a texture buffer over its first row.
    let target = match pass {
        Pass::ArrayView | Pass::ArrayViewOfOne => texture
            .new_texture_view_with_levels_slices(pf, TextureType::Type2DArray, 0..1, 0..1)
            .unwrap_or_else(|e| panic!("{pf:?}'s 2D array view: {e}")),
        Pass::TextureBuffer => {
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let row =
                TextureDescriptor::texture_buffer(pf, WIDTH, ResourceOptions::shared(), usage);
            buffer
                .new_texture(&row, 0, bpr)
                .unwrap_or_else(|e| panic!("{pf:?}'s texture buffer: {e}"))
        }
        _ => texture.clone(),
    };
    attachment.texture = Some(target);
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    if let Pass::Sized(w, h) = pass {
        descriptor.render_target_width = w;
        descriptor.render_target_height = h;
    }
    if pass == Pass::ArrayViewOfOne {
        descriptor.render_target_array_length = 1;
    }
    if cleared {
        let mut second =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
        second.usage = TextureUsage::RENDER_TARGET;
        second.resource_options = ResourceOptions::private();
        let attachment = &mut descriptor.color_attachments[1];
        attachment.texture = Some(lane.device.new_texture(&second).unwrap());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
    }
    let icb = (pass == Pass::Executed).then(|| {
        let mut icb = IndirectCommandBufferDescriptor::new();
        icb.command_types = IndirectCommandType::DRAW;
        icb.inherit_pipeline_state = true;
        icb.inherit_buffers = true;
        let icb = lane
            .device
            .new_indirect_command_buffer(&icb, 1, ResourceOptions::shared())
            .unwrap();
        icb.indirect_render_command_at(0)
            .unwrap()
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        icb
    });
    lane.run(&descriptor, |_, e, _| {
        e.set_render_pipeline_state(&state).unwrap();
        e.set_fragment_buffer(Some(&values), 0, 0).unwrap();
        e.set_fragment_buffer(Some(&shape), 0, 1).unwrap();
        let triangle = |e: &mut metal2vulkan_runtime::RenderCommandEncoder| {
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap()
        };
        match pass {
            Pass::Whole
            | Pass::Sized(..)
            | Pass::ArrayView
            | Pass::ArrayViewOfOne
            | Pass::TextureBuffer => triangle(e),
            Pass::Scissored => {
                e.set_scissor_rect(ScissorRect {
                    x: 37,
                    y: 13,
                    width: 54,
                    height: 57,
                })
                .unwrap();
                triangle(e);
            }
            Pass::NoVertices => e
                .draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
                .unwrap(),
            Pass::Executed => e
                .execute_commands_in_buffer(icb.as_ref().unwrap(), 0..1)
                .unwrap(),
            Pass::Undrawn => {}
        }
    })
    .unwrap_or_else(|e| panic!("{pf:?} {bits} {pass:?}: the pass: {e}"));
    let after = read_buffer(&buffer, 0, bpr * HEIGHT).unwrap();
    let mut words = Vec::with_capacity((row_bytes * HEIGHT) as usize);
    let mut padded = true;
    for row in after.chunks(bpr as usize) {
        words.extend_from_slice(&row[..row_bytes as usize]);
        padded &= row[row_bytes as usize..].iter().all(|&b| b == PADDING);
    }
    (words, padded)
}

/// The load passes case `pass` records and the texels they zero: none where it does not draw,
/// else one over its render area (the attachment's, or the smaller render target size) rounded
/// up to 32 texels a side, within the attachment.
fn loads(pass: Pass) -> (u64, u64) {
    // The render area and the attachment's extent: a texture buffer's is its one row.
    let ((w, h), within) = match pass {
        Pass::Undrawn => return (0, 0),
        Pass::TextureBuffer => ((WIDTH, 1), (WIDTH, 1)),
        Pass::Sized(w, h) => ((w, h), (WIDTH, HEIGHT)),
        _ => ((WIDTH, HEIGHT), (WIDTH, HEIGHT)),
    };
    let region = |side: u64, within: u64| side.next_multiple_of(32).min(within);
    (1, region(w, within.0) * region(h, within.1))
}

/// Every case's words are Metal's, the bytes between the rows stay, no case announces
/// `extended-range-bits-kept`, and each case that draws recorded one load pass over its region's texels
/// (the module docs).
#[test]
fn a_loading_pass_that_draws_leaves_metals_padded_words() {
    let (messages, _cache, lane) = lane();
    let mut differ = Vec::new();
    let (mut passes, mut texels) = (0, 0);
    for case in METAL {
        let (pf, bits, pass, metal) = case;
        let before = lane.device.work_counts();
        let (words, padded) = run(&lane, (pf, bits, pass));
        let after = lane.device.work_counts();
        let recorded = (
            after.unused_bits_load_passes - before.unused_bits_load_passes,
            after.unused_bits_load_texels - before.unused_bits_load_texels,
        );
        assert_eq!(
            recorded,
            loads(pass),
            "{pf:?} {bits} {pass:?}: the load passes"
        );
        passes += recorded.0;
        texels += recorded.1;
        assert!(
            padded,
            "{pf:?} {bits} {pass:?}: the bytes between the rows changed"
        );
        let got = digest(&words);
        if got != metal {
            eprintln!("PADDED-LOADS-DIFFERS {pf:?} mask {bits} {pass:?}: {got}");
            differ.push(case);
        }
        let announced = counts(&messages.take_notices());
        assert!(
            !announced.contains_key(&Degradation::ExtendedRangeBitsKept),
            "{pf:?} {bits} {pass:?}: {announced:?}"
        );
    }
    eprintln!(
        "PADDED-LOADS {} cases, {} differ: {passes} load passes recorded, {texels} texels zeroed",
        METAL.len(),
        differ.len()
    );
    assert!(passes > 0 && texels > 0, "no load pass ran");
    assert!(
        differ.is_empty(),
        "{} of {} cases differ: {differ:?}",
        differ.len(),
        METAL.len()
    );
    quiet(lane, &messages);
}
