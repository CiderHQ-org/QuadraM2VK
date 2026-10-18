//! Which tiles of a render pass Metal's tile store writes back: the tiles its draws' primitives
//! reach.
//!
//! An Apple GPU bins a pass's primitives into tiles and loads and stores only the tiles a
//! primitive reaches, which decides what a `DontCare` load clears (`clear_reach`), which tiles a
//! rotated texture's frame turns, and which tiles an extended-range or padded 10-bit attachment's
//! store makes canonical ([`unused_bits_module`](super::unused_bits_module)). Measured on an Apple
//! M4 under macOS 27.0.1 (111 cases, each in a process of its own; alike in every case on an Apple
//! M3 Max under macOS 26.7, the kit `share-reach`; 168 cases on the small-primitive cull; and our
//! probes on a point's square):
//!
//! - **Tiles** are 32 by 32 pixels from the render area's origin at 1 and 2 samples, 32 by 16 at
//!   4, whatever the attachments' formats and count (up to eight `RGBA32Float`); each is
//!   half-open, `[32i, 32i + 32)` along x.
//! - **Vertices** snap to 1/256 of a pixel, to the nearest: a right edge at `32 - 1/512` reaches
//!   the next tile, one at `32 - 1/256` does not. Coordinates of 1e5 pixels snap exactly.
//! - **A triangle** reaches every tile its closed area meets, clipped by the viewport, the
//!   scissor and the render area, each a half-open rectangle (a viewport over `[32, 64)` reaches
//!   tile 1 alone): its area, not its bounding box, and an edge on `x = 32` reaches the tile
//!   right of it. It reaches none where it is culled: by the cull mode and winding, at zero area
//!   (through pixel centres too), wholly past the near or far plane (but under depth clamp), or by
//!   the small-primitive cull: where its closed box, in window coordinates snapped to 1/256 (the
//!   viewport's, before the scissor and the render area clip it), holds at most 12 pixel centres
//!   (any shape, 1 by 12 to 12 by 1) and its closed area none of them (a sample on an edge
//!   counts, though the fill rule that shades is top-left). A box of 13 or more is kept, holding
//!   a sample or not. At 2 samples a box of up to 12 of their positions is tested too, at 4 one
//!   of up to 24 (Metal's default positions, Vulkan's standard ones), in every slope measured. A
//!   fragment that discards or fails a test still reaches. A triangle across the near plane
//!   reaches by the clipped polygon.
//! - **A point** reaches what a closed square about its snapped centre meets (a size of 0 or 0.25
//!   too, a centre past the viewport too; no sample cull): from half its size back, rounded half
//!   up to 1/256 of a pixel (its left and top edges reach a tile only from more than half a unit
//!   past it), over its size rounded to the nearest 1/16 of a pixel (its right and bottom edges
//!   follow from the left and top ones, up to four units past the exact square); a size past 511
//!   draws as 511.
//! - **A line** reaches what a parallelogram meets: half a pixel each way across its major axis,
//!   extended half a pixel past each end along it, a 45-degree line taken as y-major; no sample
//!   cull.
//! - **A draw** reaches the same directly or indirectly; one of no instances, one by a pipeline
//!   that does not rasterize, and a pass without a draw call reach nothing. A layered pass's
//!   draw reaches the tiles of the slice it renders alone.
//! - **A clear.** In a pass with a draw call (of no instances too), an attachment loaded `Clear`
//!   and stored, color, depth or stencil, makes every tile of the render area reached; one whose
//!   store is `DontCare` does not.
//!
//! The rule is checked here as a plain function the tests hold, against every measured case.
//! [`tile_reach_module`] bins a pass's captured primitives on the device by it; its consumers in
//! the runtime follow (the later steps).
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`TileReachModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of 4-byte
//! words in set 0:
//!
//! - [`PRIMITIVES_BINDING`](TileReachModule::PRIMITIVES_BINDING): the pass's primitives, each
//!   [`PRIMITIVE_WORDS`](TileReachModule::PRIMITIVE_WORDS) words
//!   ([`primitive`](TileReachModule::primitive)): its kind, its state, its layer, a point's size,
//!   and three clip-space vertices (a point's first, a line's first two);
//! - [`PASS_BINDING`](TileReachModule::PASS_BINDING): the header
//!   ([`header`](TileReachModule::header)), [`HEADER_WORDS`](TileReachModule::HEADER_WORDS)
//!   words: the render area, the samples, the layers, the primitives' count, and the two counts the
//!   kernel adds to, then the states, each [`STATE_WORDS`](TileReachModule::STATE_WORDS) words
//!   ([`state`](TileReachModule::state)): the viewport, the scissor, the cull mode, the winding and
//!   the depth clip mode a primitive was drawn under;
//! - [`TILES_BINDING`](TileReachModule::TILES_BINDING): the tile mask, one bit a tile of the
//!   pass (32 by 32 pixels, 32 by 16 at 4 samples; bit `t % 32` of word `t / 32`), tile `t =
//!   (layer * tile_rows + y) * tiles_per_row + x`, which the kernel ORs bits into.
//!
//! Invocation `i` below the count bins primitive `i` into its slice's tiles (a layer past the
//! pass's into the first), and adds one to the header's word
//! [`NEAR_FAR_WORD`](TileReachModule::NEAR_FAR_WORD) where the primitive crossed the near or far
//! plane under depth clipping (its reach is its clipped polygon's, whose new vertices Apple's
//! clipper may place otherwise) and to [`UNMEASURED_WORD`](TileReachModule::UNMEASURED_WORD) where
//! its reach is past what was measured (a vertex not finite, with a subnormal component, at or
//! behind the eye or snapped past 2^21 pixels; a viewport edge past that; a line of no length; a
//! point past the depth range, or of a size negative, not a number or subnormal; a state past the
//! header's or a kind not one of [`POINT`](TileReachModule::POINT),
//! [`LINE`](TileReachModule::LINE) and [`TRIANGLE`](TileReachModule::TRIANGLE)). The kernel's
//! arithmetic is exact on every device (`kernel.rs`): the reach of each primitive is the rule's.
//!
//! **Synchronization** (consumer): the writes of the primitives, the pass and the cleared mask
//! (`TRANSFER_WRITE` or `SHADER_WRITE`) → `SHADER_STORAGE_READ` and `SHADER_STORAGE_WRITE` in
//! `COMPUTE_SHADER` before the dispatch; its writes → the mask's and the counts' readers.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::{Requirements, VkFeature};
use crate::device::Device;
use crate::error::{Capability, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::Symbol;

mod kernel;

/// The word positions within the kernel's records ([`TileReachModule`]); the capture assembly
/// writes primitives and the capture settle reads the header by them (`capture_assembly.rs`).
pub(super) mod layout {
    /// The header's: the render area's width and height, the samples, the layers, the count.
    pub(in crate::bind) const AREA_W: u32 = 0;
    pub(in crate::bind) const AREA_H: u32 = 1;
    pub(in crate::bind) const SAMPLES: u32 = 2;
    pub(in crate::bind) const LAYERS: u32 = 3;
    pub(super) const COUNT: u32 = 4;
    /// A state's, from its first word: the viewport (four `f32`s), the scissor (four words), the
    /// cull mode, the winding and the depth clip mode.
    pub(super) const VIEWPORT: u32 = 0;
    pub(super) const SCISSOR: u32 = 4;
    pub(super) const CULL: u32 = 8;
    pub(super) const WINDING: u32 = 9;
    pub(super) const DEPTH_CLIP: u32 = 10;
    /// A primitive's: its kind, state, layer, size (an `f32`) and vertices (twelve `f32`s).
    pub(in crate::bind) const KIND: u32 = 0;
    pub(in crate::bind) const STATE: u32 = 1;
    pub(in crate::bind) const LAYER: u32 = 2;
    pub(in crate::bind) const SIZE: u32 = 3;
    pub(in crate::bind) const VERTICES: u32 = 4;
}

/// The crate-owned tile reach module of one device (the module docs), validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileReachModule {
    /// SPIR-V 1.5 words, entry point `main`, its bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: within every Vulkan 1.2 device with
    /// `shaderInt64`.
    pub requirements: Requirements,
}

impl TileReachModule {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The primitives, a storage buffer the kernel reads.
    pub const PRIMITIVES_BINDING: u32 = 0;
    /// The header and the states, a storage buffer the kernel reads and counts into.
    pub const PASS_BINDING: u32 = 1;
    /// The tile mask, a storage buffer the kernel ORs bits into.
    pub const TILES_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one primitive each.
    pub const LOCAL_SIZE: u32 = 64;
    /// A primitive's words ([`primitive`](Self::primitive)).
    pub const PRIMITIVE_WORDS: u32 = 16;
    /// The header's words ([`header`](Self::header)), before the states.
    pub const HEADER_WORDS: u32 = 8;
    /// A state's words ([`state`](Self::state)).
    pub const STATE_WORDS: u32 = 12;
    /// The header's word the kernel adds one to for each primitive that crossed the near or far
    /// plane under depth clipping.
    pub const NEAR_FAR_WORD: u32 = 5;
    /// The header's word the kernel adds one to for each primitive whose reach is past what was
    /// measured (the module docs).
    pub const UNMEASURED_WORD: u32 = 6;
    /// The kind of a point (`MTLPrimitiveTypePoint`'s primitives).
    pub const POINT: u32 = 0;
    /// The kind of a line (`MTLPrimitiveTypeLine` and `LineStrip`'s).
    pub const LINE: u32 = 1;
    /// The kind of a triangle (`MTLPrimitiveTypeTriangle` and `TriangleStrip`'s).
    pub const TRIANGLE: u32 = 2;

    /// The header's bytes: the render area's width and height in pixels (at most 16384 each, a
    /// larger side binned as 16384), its samples (1, 2 or 4; another count bins as 1), its layers
    /// (1 for a pass that is not layered), and the count of primitives; the two counts zero.
    pub fn header(area: [u32; 2], samples: u32, layers: u32, count: u32) -> [u8; 32] {
        let mut words = [0; Self::HEADER_WORDS as usize];
        words[layout::AREA_W as usize] = area[0];
        words[layout::AREA_H as usize] = area[1];
        words[layout::SAMPLES as usize] = samples;
        words[layout::LAYERS as usize] = layers;
        words[layout::COUNT as usize] = count;
        bytes_of(words)
    }

    /// A state's bytes: the viewport (`MTLViewport`'s `originX`, `originY`, `width` and `height`
    /// as the rasterizer takes them, in `f32`), the scissor rectangle (`MTLScissorRect`'s `x`,
    /// `y`, `width` and `height`), and the raw values of the `MTLCullMode`, the `MTLWinding` and
    /// the `MTLDepthClipMode` a primitive was drawn under.
    pub fn state(
        viewport: [f32; 4],
        scissor: [u32; 4],
        cull: u32,
        winding: u32,
        depth_clip: u32,
    ) -> [u8; 48] {
        let mut words = [0; Self::STATE_WORDS as usize];
        for i in 0..4 {
            words[layout::VIEWPORT as usize + i] = viewport[i].to_bits();
            words[layout::SCISSOR as usize + i] = scissor[i];
        }
        words[layout::CULL as usize] = cull;
        words[layout::WINDING as usize] = winding;
        words[layout::DEPTH_CLIP as usize] = depth_clip;
        bytes_of(words)
    }

    /// A primitive's bytes: its `kind` ([`POINT`](Self::POINT), [`LINE`](Self::LINE) or
    /// [`TRIANGLE`](Self::TRIANGLE)), the index of the `state` it was drawn under among the
    /// pass's, the `layer` it renders (`[[render_target_array_index]]`; 0 where none), a point's
    /// `size` in pixels (`[[point_size]]`), and its clip-space `vertices` (`[[position]]`): a
    /// point's first, a line's first two, a triangle's three, in the order the draw assembled
    /// them.
    pub fn primitive(
        kind: u32,
        state: u32,
        layer: u32,
        size: f32,
        vertices: [[f32; 4]; 3],
    ) -> [u8; 64] {
        let mut words = [0; Self::PRIMITIVE_WORDS as usize];
        words[layout::KIND as usize] = kind;
        words[layout::STATE as usize] = state;
        words[layout::LAYER as usize] = layer;
        words[layout::SIZE as usize] = size.to_bits();
        for (v, vertex) in vertices.iter().enumerate() {
            for (c, value) in vertex.iter().enumerate() {
                words[layout::VERTICES as usize + 4 * v + c] = value.to_bits();
            }
        }
        bytes_of(words)
    }

    /// The tile mask's words of a pass of render area `area`, `samples` and `layers` (the
    /// header's): one bit a tile of each slice.
    pub fn mask_words(area: [u32; 2], samples: u32, layers: u32) -> u32 {
        let tile_rows = match samples {
            4 => 16,
            _ => 32,
        };
        let [w, h] = area.map(|a| a.min(AREA_LIMIT));
        let tiles = w.div_ceil(32) * h.div_ceil(tile_rows) * layers;
        tiles.div_ceil(32)
    }
}

/// The largest render area side the kernel bins, in pixels (Metal's): a larger side is binned as
/// this one, which keeps every product of the kernel's integer tests within an `i64`.
pub(super) const AREA_LIMIT: u32 = 16384;

/// The little-endian bytes of `N` words.
fn bytes_of<const N: usize, const B: usize>(words: [u32; N]) -> [u8; B] {
    let mut bytes = [0; B];
    for (at, w) in words.iter().enumerate() {
        bytes[4 * at..4 * at + 4].copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(TileReachModule::SET),
        Symbol::Binding(s) if *s == kernel::PRIMITIVES => Some(TileReachModule::PRIMITIVES_BINDING),
        Symbol::Binding(s) if *s == kernel::PASS => Some(TileReachModule::PASS_BINDING),
        Symbol::Binding(s) if *s == kernel::TILES => Some(TileReachModule::TILES_BINDING),
        _ => None,
    }
}

/// The tile reach module of `device`: built with the typed builder, checked by the owned verifier
/// against the device's profile (requirements included), serialized, and numbered with the fixed
/// bindings of [`TileReachModule`]. A device without `shaderInt64`, which the kernel's exact
/// arithmetic needs, is [`Feature::TileReach`] needing it.
pub fn tile_reach_module(device: &Device) -> Result<TileReachModule, Error> {
    let profile = device.profile();
    if !profile
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderInt64)
    {
        let int64 = Capability::Feature(VkFeature::ShaderInt64);
        return Err(Unsupported::needs_any_of(Feature::TileReach, vec![vec![int64]]).into());
    }
    let m = kernel::typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the tile reach kernel needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(TileReachModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
