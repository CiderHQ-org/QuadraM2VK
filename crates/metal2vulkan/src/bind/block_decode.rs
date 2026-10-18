//! A BC1 to BC5 texture's texels as Metal decodes them, from its blocks (a fitted decode).
//!
//! **Metal's decode** (an Apple M4 under macOS 27.0.1: every texel of random
//! blocks of every BC format read, which MoltenVK's equal; checked again by
//! `runtime_lane::compressed_decode` on the venues). Each 565 endpoint is widened to 8 bits by
//! replicating its bits; an index's weight `w` of 64 is BC7's: 21 and 43 for a four-color
//! block's thirds, 32 for a BC1 three-color block's half, 9, 18, 27, 37, 46 and 55 for an
//! eight-value channel block's sevenths, 13, 26, 38 and 51 for a six-value one's fifths; the sum
//! is `(64 - w) a + w b`.
//!
//! - **BC1, BC2 and BC3's color** ([`BlockDecode::Bc1`], [`BlockDecode::Bc2`],
//!   [`BlockDecode::Bc3`]): each channel the sum rounded half up to an 8-bit code,
//!   `(sum + 32) >> 6`. BC1's block is four-color where its first endpoint's 16 bits exceed its
//!   second's, else three-color, its fourth index transparent black (codes 0, alpha 0); BC2's and
//!   BC3's color blocks are four-color. BC1's alpha is 255; BC2's its 4-bit code times 17; BC3's
//!   its channel block's sum rounded so, a six-value block's sixth and seventh indices 0 and 255.
//!   The texel is the four 8-bit codes, red in the low byte (an `R8G8B8A8_UNORM` texel, whose
//!   read is each code over 255). The sRGB forms decode to the same codes: their color is the code
//!   through Metal's twelve-bit table (`metal::texel_format::srgb_raw`), alpha the code.
//! - **BC4 and BC5** ([`BlockDecode::Bc4Unorm`] to [`BlockDecode::Bc5Snorm`]): each channel
//!   the sum unrounded, as a `float` correctly rounded: `sum / 16320` unsigned, `sum / 8128`
//!   signed, where an endpoint of -128 enters the sum as it is and the value is clamped to -1
//!   after; a six-value block's sixth and seventh indices 0 and 1 (signed -1 and 1). The texel is
//!   the channels' `float`s (an `R32_SFLOAT` or `R32G32_SFLOAT` texel). A 16-bit normalized
//!   texel would not hold them: of the 16,077 sums BC4's weights make, 15,821 are no `R16_UNORM`
//!   code's value (`no_sixteen_bit_texel_holds_the_channel_values`).
//!
//! **The module** (one per device and decode). One `GLCompute` entry `main` with local size
//! ([`BlockDecodeModule::LOCAL_SIZE`], [`BlockDecodeModule::LOCAL_SIZE`], 1), no push constants,
//! and in set 0 three storage buffers of 4-byte words:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`BlockDecodeModule::SHAPE_BINDING`] | the region's shape ([`BlockDecodeModule::shape`]) |
//! | [`BlockDecodeModule::BLOCKS_BINDING`] | the region's blocks, read: tight, as a copy of the region out of the image lays them (each layer's rows of blocks, its first row first) |
//! | [`BlockDecodeModule::TEXELS_BINDING`] | the region's texels, written: tight, as a copy into an image of [`BlockDecode::texel_format`] reads them |
//!
//! Invocation `(x, y, z)` inside the region's width, height and depth (layers, or a 3D level's
//! slices) writes texel `(x, y)` of layer `z` from its block; one outside it writes nothing. The
//! consumer dispatches `⌈w / 8⌉ x ⌈h / 8⌉ x d` workgroups and keeps each binding within
//! `maxStorageBufferRange`, splitting a larger region into bands of whole block rows.
//!
//! **Synchronization** (consumer): the writes of the shape and the blocks (`TRANSFER_WRITE`) →
//! `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its texel writes
//! (`SHADER_STORAGE_WRITE`) → `TRANSFER_READ` before the texels are copied into the image.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::lower::arith::exact_division::div_exact;
use crate::lower::arith::legal::{Form, Held};
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, StorageClass, Type,
    TypedModule, UnOp,
};
use crate::vk::Format;

/// How a BC1 to BC5 format's blocks decode (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlockDecode {
    /// `BC1_RGBA` and its sRGB form: one color block.
    Bc1,
    /// `BC2_RGBA` and its sRGB form: explicit 4-bit alpha, then a four-color block.
    Bc2,
    /// `BC3_RGBA` and its sRGB form: an interpolated alpha block, then a four-color block.
    Bc3,
    /// `BC4_RUnorm`: one unsigned channel block.
    Bc4Unorm,
    /// `BC4_RSnorm`: one signed channel block.
    Bc4Snorm,
    /// `BC5_RGUnorm`: two unsigned channel blocks, red then green.
    Bc5Unorm,
    /// `BC5_RGSnorm`: two signed channel blocks, red then green.
    Bc5Snorm,
}

/// The weights of 64 of a four-color block's indices.
const FOUR_COLOR: [u32; 4] = [0, 64, 21, 43];
/// The weights of a three-color block's first three indices (its fourth is transparent black).
const THREE_COLOR: [u32; 3] = [0, 64, 32];
/// The weights of an eight-value channel block's indices, then of a six-value one's first six
/// (its seventh and eighth are constants), at 8 past them.
const CHANNEL_WEIGHTS: [u32; 16] = [0, 64, 9, 18, 27, 37, 46, 55, 0, 64, 13, 26, 38, 51, 0, 0];

impl BlockDecode {
    /// The decode of `pf`'s blocks: the BC1 to BC5 formats, their sRGB forms decoding to the
    /// unorm forms' codes; `None` for every other format.
    pub fn of(pf: PixelFormat) -> Option<BlockDecode> {
        use PixelFormat as P;
        Some(match pf {
            P::BC1RGBA | P::BC1RGBASRGB => BlockDecode::Bc1,
            P::BC2RGBA | P::BC2RGBASRGB => BlockDecode::Bc2,
            P::BC3RGBA | P::BC3RGBASRGB => BlockDecode::Bc3,
            P::BC4RUnorm => BlockDecode::Bc4Unorm,
            P::BC4RSnorm => BlockDecode::Bc4Snorm,
            P::BC5RGUnorm => BlockDecode::Bc5Unorm,
            P::BC5RGSnorm => BlockDecode::Bc5Snorm,
            _ => return None,
        })
    }

    /// Every decode.
    pub const ALL: [BlockDecode; 7] = [
        BlockDecode::Bc1,
        BlockDecode::Bc2,
        BlockDecode::Bc3,
        BlockDecode::Bc4Unorm,
        BlockDecode::Bc4Snorm,
        BlockDecode::Bc5Unorm,
        BlockDecode::Bc5Snorm,
    ];

    /// The bytes of a block, over 4 x 4 texels: 8 for BC1 and BC4, 16 for the others.
    pub fn block_bytes(self) -> u32 {
        match self {
            BlockDecode::Bc1 | BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => 8,
            _ => 16,
        }
    }

    /// The bytes of a decoded texel: 4, or 8 for BC5's two `float`s.
    pub fn texel_bytes(self) -> u32 {
        match self {
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => 8,
            _ => 4,
        }
    }

    /// The Vulkan format of the decoded texels: `R8G8B8A8_UNORM` of BC1 to BC3's codes,
    /// `R32_SFLOAT` of BC4's values, `R32G32_SFLOAT` of BC5's.
    pub fn texel_format(self) -> Format {
        match self {
            BlockDecode::Bc1 | BlockDecode::Bc2 | BlockDecode::Bc3 => Format::R8G8B8A8_UNORM,
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => Format::R32_SFLOAT,
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => Format::R32G32_SFLOAT,
        }
    }

    /// The Metal pixel format whose texels the decoded ones are: `RGBA8Unorm`, `R32Float` or
    /// `RG32Float`.
    pub fn texel_pixel_format(self) -> PixelFormat {
        match self {
            BlockDecode::Bc1 | BlockDecode::Bc2 | BlockDecode::Bc3 => PixelFormat::RGBA8Unorm,
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => PixelFormat::R32Float,
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => PixelFormat::RG32Float,
        }
    }

    /// Whether the channel blocks are signed.
    fn signed(self) -> bool {
        matches!(self, BlockDecode::Bc4Snorm | BlockDecode::Bc5Snorm)
    }

    /// The words of texel `j` (0 to 15, row-major in its block) of `block` as Metal decodes it
    /// (the module docs): the packed codes, or the channels' `float` bits; the second word is
    /// BC5's green, 0 elsewhere.
    pub fn texel(self, block: &[u8], j: usize) -> [u32; 2] {
        let codes = |[r, g, b]: [u32; 3], a: u32| r | g << 8 | b << 16 | a << 24;
        match self {
            BlockDecode::Bc1 => {
                let (rgb, black) = color(block, j, false);
                [codes(rgb, if black { 0 } else { 255 }), 0]
            }
            BlockDecode::Bc2 => {
                let (rgb, _) = color(&block[8..], j, true);
                let alpha =
                    (u64::from_le_bytes(block[..8].try_into().expect("8 bytes")) >> (4 * j)) & 15;
                [codes(rgb, alpha as u32 * 17), 0]
            }
            BlockDecode::Bc3 => {
                let (rgb, _) = color(&block[8..], j, true);
                let alpha = match channel(&block[..8], j, false) {
                    Ok(sum) => (sum as u32 + 32) >> 6,
                    Err(high) => u32::from(high) * 255,
                };
                [codes(rgb, alpha), 0]
            }
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => {
                [channel_value(&block[..8], j, self.signed()).to_bits(), 0]
            }
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => [
                channel_value(&block[..8], j, self.signed()).to_bits(),
                channel_value(&block[8..16], j, self.signed()).to_bits(),
            ],
        }
    }

    /// The words of the 16 texels of `block`, row-major: [`texel`](Self::texel)'s answers, its
    /// endpoints and weights worked out once for the block (the runtime decodes a write on the
    /// CPU with them, `decode`).
    fn block_texels(self, block: &[u8]) -> [[u32; 2]; 16] {
        let codes = |[r, g, b]: [u32; 3], a: u32| r | g << 8 | b << 16 | a << 24;
        let signed = self.signed();
        let floats =
            |block: &[u8]| channel_palette(block, signed).map(|t| channel_float(t, signed));
        match self {
            BlockDecode::Bc1 => {
                let palette = color_palette(block, false)
                    .map(|(rgb, black)| codes(rgb, if black { 0 } else { 255 }));
                std::array::from_fn(|j| [palette[color_index(block, j)], 0])
            }
            BlockDecode::Bc2 => {
                let palette = color_palette(&block[8..], true);
                let alphas = u64::from_le_bytes(block[..8].try_into().expect("8 bytes"));
                std::array::from_fn(|j| {
                    let alpha = ((alphas >> (4 * j)) & 15) as u32 * 17;
                    [codes(palette[color_index(&block[8..], j)].0, alpha), 0]
                })
            }
            BlockDecode::Bc3 => {
                let palette = color_palette(&block[8..], true);
                let alphas = channel_palette(&block[..8], false).map(|t| match t {
                    Ok(sum) => (sum as u32 + 32) >> 6,
                    Err(high) => u32::from(high) * 255,
                });
                std::array::from_fn(|j| {
                    let alpha = alphas[channel_index(&block[..8], j)];
                    [codes(palette[color_index(&block[8..], j)].0, alpha), 0]
                })
            }
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => {
                let red = floats(&block[..8]);
                std::array::from_fn(|j| [red[channel_index(&block[..8], j)].to_bits(), 0])
            }
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => {
                let (red, green) = (floats(&block[..8]), floats(&block[8..16]));
                std::array::from_fn(|j| {
                    [
                        red[channel_index(&block[..8], j)].to_bits(),
                        green[channel_index(&block[8..16], j)].to_bits(),
                    ]
                })
            }
        }
    }

    /// The texels of a region `width` x `height` x `depth` whose `blocks` are laid out as the
    /// module reads them (tight, row-major, each layer's after the last), laid out as it writes
    /// them: the CPU decode the module answers. (block by block, each block's palette worked out
    /// once.)
    pub fn decode(self, blocks: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let mut texels = vec![0; width * height * depth * self.texel_bytes() as usize];
        self.decode_into(blocks, size, &mut texels);
        texels
    }

    /// [`decode`](Self::decode) into `texels`, which holds exactly the region's texels (`width`
    /// x `height` x `depth`, each [`texel_bytes`](Self::texel_bytes)): so bands of a region's
    /// block rows can be decoded on several threads into one buffer.
    ///
    /// # Panics
    ///
    /// Where `blocks` holds fewer blocks than the region, or `texels` another length.
    pub fn decode_into(self, blocks: &[u8], [width, height, depth]: [u32; 3], texels: &mut [u8]) {
        let size = self.block_bytes() as usize;
        let (width, height) = (width as usize, height as usize);
        let (across, down) = (width.div_ceil(4), height.div_ceil(4));
        let words = (self.texel_bytes() / 4) as usize;
        let row = width * words * 4;
        assert_eq!(
            texels.len(),
            row * height * depth as usize,
            "the region's texels"
        );
        for b in 0..across * down * depth as usize {
            let decoded = self.block_texels(&blocks[b * size..(b + 1) * size]);
            let (z, by, bx) = (b / (across * down), b / across % down, b % across);
            for (ty, y) in (by * 4..(by * 4 + 4).min(height)).enumerate() {
                for (tx, x) in (bx * 4..(bx * 4 + 4).min(width)).enumerate() {
                    let at = (z * height + y) * row + x * words * 4;
                    for (k, word) in decoded[ty * 4 + tx][..words].iter().enumerate() {
                        texels[at + 4 * k..at + 4 * k + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// `bits` bits widened to 8 by replicating them.
fn widen(v: u32, bits: u32) -> u32 {
    (v << (8 - bits)) | (v >> (2 * bits - 8))
}

/// A 565 color's three channels widened to 8 bits.
fn rgb565(c: u32) -> [u32; 3] {
    [
        widen((c >> 11) & 31, 5),
        widen((c >> 5) & 63, 6),
        widen(c & 31, 5),
    ]
}

/// Texel `j` of a color block: its three 8-bit codes and whether it is BC1's transparent black.
/// `four` forces the four-color mode (BC2 and BC3).
fn color(block: &[u8], j: usize, four: bool) -> ([u32; 3], bool) {
    color_palette(block, four)[color_index(block, j)]
}

/// A color block's four texels by index ([`color`]'s answers): its codes and whether each is
/// BC1's transparent black.
fn color_palette(block: &[u8], four: bool) -> [([u32; 3], bool); 4] {
    let c0 = u32::from(u16::from_le_bytes([block[0], block[1]]));
    let c1 = u32::from(u16::from_le_bytes([block[2], block[3]]));
    let (e0, e1) = (rgb565(c0), rgb565(c1));
    let mix = |w: u32| std::array::from_fn(|k| ((64 - w) * e0[k] + w * e1[k] + 32) >> 6);
    std::array::from_fn(|i| match (i, four || c0 > c1) {
        (_, true) => (mix(FOUR_COLOR[i]), false),
        (3, false) => ([0; 3], true),
        (_, false) => (mix(THREE_COLOR[i]), false),
    })
}

/// The index of texel `j` of a color block.
fn color_index(block: &[u8], j: usize) -> usize {
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    ((indices >> (2 * j)) & 3) as usize
}

/// Texel `j` of a channel block (BC3's alpha, BC4's and BC5's channels): the sum of 64ths of its
/// endpoints, or a six-value block's constant as `Err(high)`.
fn channel(block: &[u8], j: usize, signed: bool) -> Result<i32, bool> {
    channel_palette(block, signed)[channel_index(block, j)]
}

/// A channel block's eight texels by index ([`channel`]'s answers).
fn channel_palette(block: &[u8], signed: bool) -> [Result<i32, bool>; 8] {
    let end = |b: u8| match signed {
        true => i32::from(b as i8),
        false => i32::from(b),
    };
    let (a0, a1) = (end(block[0]), end(block[1]));
    let eight = a0 > a1;
    std::array::from_fn(|i| {
        if !eight && i >= 6 {
            return Err(i == 7);
        }
        let w = CHANNEL_WEIGHTS[i + if eight { 0 } else { 8 }] as i32;
        Ok((64 - w) * a0 + w * a1)
    })
}

/// The index of texel `j` of a channel block.
fn channel_index(block: &[u8], j: usize) -> usize {
    let mut bits = [0u8; 8];
    bits[..6].copy_from_slice(&block[2..8]);
    ((u64::from_le_bytes(bits) >> (3 * j)) & 7) as usize
}

/// A BC4 or BC5 channel's value: the sum over 16320 (signed 8128, at least -1), correctly
/// rounded, or the constant.
fn channel_value(block: &[u8], j: usize, signed: bool) -> f32 {
    channel_float(channel(block, j, signed), signed)
}

/// A channel texel's value ([`channel_value`]) from its sum or constant ([`channel`]).
fn channel_float(texel: Result<i32, bool>, signed: bool) -> f32 {
    let (low, d) = match signed {
        true => (-1.0f32, 8128.0),
        false => (0.0, 16320.0),
    };
    match texel {
        // `d` is 64 times an odd number: no quotient lies on an `f32` tie, so rounding its `f64`
        // once more answers the correctly rounded `f32`.
        Ok(sum) => ((f64::from(sum) / d) as f32).max(low),
        Err(high) => [low, 1.0][usize::from(high)],
    }
}

/// The crate-owned block decode module of one device and decode, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDecodeModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl BlockDecodeModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The region's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 0;
    /// The region's blocks: a storage buffer, read.
    pub const BLOCKS_BINDING: u32 = 1;
    /// The region's texels: a storage buffer, written.
    pub const TEXELS_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = 5;
    /// Invocations of one workgroup along x, and along y: one texel each.
    pub const LOCAL_SIZE: u32 = 8;

    /// The shape's bytes of a region `width` x `height` x `depth` (layers, or a 3D level's
    /// slices): its size, and its blocks across and down.
    pub fn shape([width, height, depth]: [u32; 3]) -> [u8; Self::SHAPE_WORDS as usize * 4] {
        let words = [width, height, depth, width.div_ceil(4), height.div_ceil(4)];
        let mut bytes = [0; Self::SHAPE_WORDS as usize * 4];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the shape in the typed module.
pub(super) const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the blocks in the typed module.
pub(super) const BLOCKS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the texels in the typed module.
pub(super) const TEXELS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// A `u32[n]` `Private` table initialized to `values`.
pub(super) fn table(b: &mut ModuleBuilder, u32_: TypeId, values: &[u32]) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(values.len() as u64),
    })?;
    let arr = b.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(values.len());
    for &v in values {
        parts.push(b.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(v)),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, arr, Some(init))
}

/// What the decode of one texel reads.
#[derive(Clone, Copy)]
struct Cx {
    u32_: TypeId,
    zero: Value,
    one: Value,
    blocks: Value,
    /// The texel's index in its block, `4 (y mod 4) + (x mod 4)`.
    j: Value,
    /// The weights of a four-color block's indices ([`FOUR_COLOR`]).
    four_color: GlobalId,
    /// [`THREE_COLOR`], with a fourth weight of 0.
    three_color: GlobalId,
    /// [`CHANNEL_WEIGHTS`].
    channel_weights: GlobalId,
}

impl Cx {
    fn uint(&self, f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        let c = f.module().constant(Constant::Scalar {
            ty: self.u32_,
            bits: ConstBits(u64::from(v)),
        })?;
        f.module().const_value(c)
    }

    /// Word `k` of the block whose first word is `base`.
    fn word(&self, f: &mut FunctionBuilder<'_>, base: Value, k: u32) -> BResult<Value> {
        let k = self.uint(f, k)?;
        let index = f.binary(BinOp::IAdd, base, k)?;
        let p = f.access_chain(self.blocks, &[self.zero, index])?;
        f.load(p, MemAccess::default())
    }

    /// Entry `index` of the `Private` table `table`.
    fn entry(&self, f: &mut FunctionBuilder<'_>, table: GlobalId, index: Value) -> BResult<Value> {
        let table = f.module().global_value(table)?;
        let p = f.access_chain(table, &[index])?;
        f.load(p, MemAccess::default())
    }

    /// `(word >> shift) & mask`.
    fn bits(
        &self,
        f: &mut FunctionBuilder<'_>,
        word: Value,
        shift: u32,
        mask: u32,
    ) -> BResult<Value> {
        let shift = self.uint(f, shift)?;
        let shifted = f.binary(BinOp::ShiftRightLogical, word, shift)?;
        let mask = self.uint(f, mask)?;
        f.binary(BinOp::BitwiseAnd, shifted, mask)
    }

    /// `bits` bits of `v` widened to 8 by replicating them.
    fn widen(&self, f: &mut FunctionBuilder<'_>, v: Value, bits: u32) -> BResult<Value> {
        let up = self.uint(f, 8 - bits)?;
        let down = self.uint(f, 2 * bits - 8)?;
        let high = f.binary(BinOp::ShiftLeftLogical, v, up)?;
        let low = f.binary(BinOp::ShiftRightLogical, v, down)?;
        f.binary(BinOp::BitwiseOr, high, low)
    }

    /// The texel's three 8-bit codes of the color block whose endpoints are word `w0` and whose
    /// indices are word `w1`, and whether it is BC1's transparent black: `four` forces the
    /// four-color mode ([`color`]).
    fn color(
        &self,
        f: &mut FunctionBuilder<'_>,
        w0: Value,
        w1: Value,
        four: bool,
    ) -> BResult<([Value; 3], Value)> {
        let c0 = self.bits(f, w0, 0, 0xffff)?;
        let c1 = self.bits(f, w0, 16, 0xffff)?;
        let shift = f.binary(BinOp::ShiftLeftLogical, self.j, self.one)?;
        let i = f.binary(BinOp::ShiftRightLogical, w1, shift)?;
        let three = self.uint(f, 3)?;
        let i = f.binary(BinOp::BitwiseAnd, i, three)?;
        let four_w = self.entry(f, self.four_color, i)?;
        let (w, black) = if four {
            let no = f.binary(BinOp::INotEqual, self.zero, self.zero)?;
            (four_w, no)
        } else {
            let three_w = self.entry(f, self.three_color, i)?;
            let four_mode = f.binary(BinOp::UGreaterThan, c0, c1)?;
            let w = f.select(four_mode, four_w, three_w)?;
            let last = f.binary(BinOp::IEqual, i, three)?;
            let three_mode = f.unary(UnOp::LogicalNot, four_mode)?;
            let black = f.binary(BinOp::LogicalAnd, three_mode, last)?;
            (w, black)
        };
        let sixty_four = self.uint(f, 64)?;
        let rest = f.binary(BinOp::ISub, sixty_four, w)?;
        let half = self.uint(f, 32)?;
        let six = self.uint(f, 6)?;
        let mut rgb = [self.zero; 3];
        for (k, (shift, bits)) in [(11, 5), (5, 6), (0, 5)].into_iter().enumerate() {
            let mask = (1 << bits) - 1;
            let a = self.bits(f, c0, shift, mask)?;
            let a = self.widen(f, a, bits)?;
            let b = self.bits(f, c1, shift, mask)?;
            let b = self.widen(f, b, bits)?;
            let a = f.binary(BinOp::IMul, rest, a)?;
            let b = f.binary(BinOp::IMul, w, b)?;
            let sum = f.binary(BinOp::IAdd, a, b)?;
            let sum = f.binary(BinOp::IAdd, sum, half)?;
            let code = f.binary(BinOp::ShiftRightLogical, sum, six)?;
            rgb[k] = f.select(black, self.zero, code)?;
        }
        Ok((rgb, black))
    }

    /// The texel's sum of the channel block of words `w0` and `w1` (two's complement), whether
    /// the block is six-value with an index of 6 or 7 (a constant), and whether that index is 7
    /// ([`channel`]).
    fn channel(
        &self,
        f: &mut FunctionBuilder<'_>,
        w0: Value,
        w1: Value,
        signed: bool,
    ) -> BResult<(Value, Value, Value)> {
        let (a0, a1) = if signed {
            let k24 = self.uint(f, 24)?;
            let k16 = self.uint(f, 16)?;
            let a0 = f.binary(BinOp::ShiftLeftLogical, w0, k24)?;
            let a0 = f.binary(BinOp::ShiftRightArithmetic, a0, k24)?;
            let a1 = f.binary(BinOp::ShiftLeftLogical, w0, k16)?;
            let a1 = f.binary(BinOp::ShiftRightArithmetic, a1, k24)?;
            (a0, a1)
        } else {
            (self.bits(f, w0, 0, 0xff)?, self.bits(f, w0, 8, 0xff)?)
        };
        // Index `j`'s three bits from bit `16 + 3 j` of the block's 64: in the first word below
        // 32 (its bits and the second word's past them, the shifts within 0 to 31), else in the
        // second word alone.
        let three = self.uint(f, 3)?;
        let sixteen = self.uint(f, 16)?;
        let thirty_one = self.uint(f, 31)?;
        let thirty_two = self.uint(f, 32)?;
        let pos = f.binary(BinOp::IMul, self.j, three)?;
        let pos = f.binary(BinOp::IAdd, pos, sixteen)?;
        let low_shift = f.binary(BinOp::BitwiseAnd, pos, thirty_one)?;
        let low = f.binary(BinOp::ShiftRightLogical, w0, low_shift)?;
        let up = f.binary(BinOp::ISub, thirty_two, pos)?;
        let up = f.binary(BinOp::BitwiseAnd, up, thirty_one)?;
        let carried = f.binary(BinOp::ShiftLeftLogical, w1, up)?;
        let first = f.binary(BinOp::BitwiseOr, low, carried)?;
        let second = f.binary(BinOp::ShiftRightLogical, w1, low_shift)?;
        let in_first = f.binary(BinOp::ULessThan, pos, thirty_two)?;
        let i = f.select(in_first, first, second)?;
        let seven = self.uint(f, 7)?;
        let i = f.binary(BinOp::BitwiseAnd, i, seven)?;
        let eight = match signed {
            true => f.binary(BinOp::SGreaterThan, a0, a1)?,
            false => f.binary(BinOp::UGreaterThan, a0, a1)?,
        };
        let k8 = self.uint(f, 8)?;
        let offset = f.select(eight, self.zero, k8)?;
        let at = f.binary(BinOp::IAdd, i, offset)?;
        let w = self.entry(f, self.channel_weights, at)?;
        let sixty_four = self.uint(f, 64)?;
        let rest = f.binary(BinOp::ISub, sixty_four, w)?;
        let a = f.binary(BinOp::IMul, rest, a0)?;
        let b = f.binary(BinOp::IMul, w, a1)?;
        let sum = f.binary(BinOp::IAdd, a, b)?;
        let k6 = self.uint(f, 6)?;
        let high_index = f.binary(BinOp::UGreaterThanEqual, i, k6)?;
        let six_value = f.unary(UnOp::LogicalNot, eight)?;
        let constant = f.binary(BinOp::LogicalAnd, six_value, high_index)?;
        let high = f.binary(BinOp::IEqual, i, seven)?;
        Ok((sum, constant, high))
    }

    /// A BC4 or BC5 channel's `float` bits of the channel block of words `w0` and `w1`
    /// ([`channel_value`]): `sum / 16320` (signed `sum / 8128`, at least -1) correctly rounded by
    /// integer operations alone (`exact_division::div_exact`), or the constant.
    fn channel_value(
        &self,
        f: &mut FunctionBuilder<'_>,
        w0: Value,
        w1: Value,
        signed: bool,
    ) -> BResult<Value> {
        let (sum, constant, high) = self.channel(f, w0, w1, signed)?;
        let u = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        };
        let one_bits = self.uint(f, 1.0f32.to_bits())?;
        let (value, low_bits) = if signed {
            // The magnitude, at most 8128: an endpoint of -128 sums past -8128, which reads -1.
            let negative = f.binary(BinOp::SLessThan, sum, self.zero)?;
            let negated = f.binary(BinOp::ISub, self.zero, sum)?;
            let magnitude = f.select(negative, negated, sum)?;
            let d = self.uint(f, 8128)?;
            let past = f.binary(BinOp::UGreaterThan, magnitude, d)?;
            let magnitude = f.select(past, d, magnitude)?;
            let q = div_exact(f, u, magnitude, 8128)?;
            let q = f.convert(ConvOp::Bitcast, q, self.u32_)?;
            let sign = self.uint(f, 0x8000_0000)?;
            let signed_q = f.binary(BinOp::BitwiseOr, q, sign)?;
            let value = f.select(negative, signed_q, q)?;
            (value, self.uint(f, (-1.0f32).to_bits())?)
        } else {
            let q = div_exact(f, u, sum, 16320)?;
            (f.convert(ConvOp::Bitcast, q, self.u32_)?, self.zero)
        };
        let fixed = f.select(high, one_bits, low_bits)?;
        f.select(constant, fixed, value)
    }
}

/// The module of `decode` as a typed module, before serialization: the module
/// [`block_decode_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `SHAPE`, `BLOCKS` and `TEXELS` (the module docs).
fn typed_module(decode: BlockDecode) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = super::ycbcr_convert::words_buffer(&mut b, SHAPE, u32_)?;
    let blocks = super::ycbcr_convert::words_buffer(&mut b, BLOCKS, u32_)?;
    let texels = super::ycbcr_convert::words_buffer(&mut b, TEXELS, u32_)?;
    let four_color = table(&mut b, u32_, &FOUR_COLOR)?;
    let three_color = table(
        &mut b,
        u32_,
        &[THREE_COLOR[0], THREE_COLOR[1], THREE_COLOR[2], 0],
    )?;
    let channel_weights = table(&mut b, u32_, &CHANNEL_WEIGHTS)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = super::ycbcr_convert::scalar(&mut f, u32_, 0)?;
    let one = super::ycbcr_convert::scalar(&mut f, u32_, 1)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = super::ycbcr_convert::scalar(f, u32_, u64::from(index))?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, 0)?;
    let height = field(&mut f, 1)?;
    let depth = field(&mut f, 2)?;
    let across = field(&mut f, 3)?;
    let down = field(&mut f, 4)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let in_z = f.binary(BinOp::ULessThan, z, depth)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    let blocks = f.module().global_value(blocks)?;
    let texels = f.module().global_value(texels)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let two = super::ycbcr_convert::scalar(f, u32_, 2)?;
        let three = super::ycbcr_convert::scalar(f, u32_, 3)?;
        // The texel's block, `(z · down + y / 4) · across + x / 4`, and its index in it.
        let row = f.binary(BinOp::IMul, z, down)?;
        let block_y = f.binary(BinOp::ShiftRightLogical, y, two)?;
        let row = f.binary(BinOp::IAdd, row, block_y)?;
        let block = f.binary(BinOp::IMul, row, across)?;
        let block_x = f.binary(BinOp::ShiftRightLogical, x, two)?;
        let block = f.binary(BinOp::IAdd, block, block_x)?;
        let in_y = f.binary(BinOp::BitwiseAnd, y, three)?;
        let in_y = f.binary(BinOp::ShiftLeftLogical, in_y, two)?;
        let in_x = f.binary(BinOp::BitwiseAnd, x, three)?;
        let j = f.binary(BinOp::BitwiseOr, in_y, in_x)?;
        let block_words =
            super::ycbcr_convert::scalar(f, u32_, u64::from(decode.block_bytes() / 4))?;
        let base = f.binary(BinOp::IMul, block, block_words)?;
        let cx = Cx {
            u32_,
            zero,
            one,
            blocks,
            j,
            four_color,
            three_color,
            channel_weights,
        };
        let pack = |f: &mut FunctionBuilder<'_>, rgb: [Value; 3], a: Value| -> BResult<Value> {
            let mut packed = rgb[0];
            for (k, c) in [rgb[1], rgb[2], a].into_iter().enumerate() {
                let at = cx.uint(f, 8 * (k as u32 + 1))?;
                let placed = f.binary(BinOp::ShiftLeftLogical, c, at)?;
                packed = f.binary(BinOp::BitwiseOr, packed, placed)?;
            }
            Ok(packed)
        };
        let words: Vec<Value> = match decode {
            BlockDecode::Bc1 => {
                let w0 = cx.word(f, base, 0)?;
                let w1 = cx.word(f, base, 1)?;
                let (rgb, black) = cx.color(f, w0, w1, false)?;
                let opaque = cx.uint(f, 255)?;
                let a = f.select(black, zero, opaque)?;
                vec![pack(f, rgb, a)?]
            }
            BlockDecode::Bc2 => {
                let w0 = cx.word(f, base, 0)?;
                let w1 = cx.word(f, base, 1)?;
                let w2 = cx.word(f, base, 2)?;
                let w3 = cx.word(f, base, 3)?;
                let (rgb, _) = cx.color(f, w2, w3, true)?;
                // The alpha code of texel `j`: 4 bits at `4 j` of the first two words.
                let k7 = cx.uint(f, 7)?;
                let second = f.binary(BinOp::UGreaterThan, j, k7)?;
                let word = f.select(second, w1, w0)?;
                let lane = f.binary(BinOp::BitwiseAnd, j, k7)?;
                let shift = f.binary(BinOp::ShiftLeftLogical, lane, two)?;
                let code = f.binary(BinOp::ShiftRightLogical, word, shift)?;
                let fifteen = cx.uint(f, 15)?;
                let code = f.binary(BinOp::BitwiseAnd, code, fifteen)?;
                let seventeen = cx.uint(f, 17)?;
                let a = f.binary(BinOp::IMul, code, seventeen)?;
                vec![pack(f, rgb, a)?]
            }
            BlockDecode::Bc3 => {
                let w0 = cx.word(f, base, 0)?;
                let w1 = cx.word(f, base, 1)?;
                let w2 = cx.word(f, base, 2)?;
                let w3 = cx.word(f, base, 3)?;
                let (rgb, _) = cx.color(f, w2, w3, true)?;
                let (sum, constant, high) = cx.channel(f, w0, w1, false)?;
                let half = cx.uint(f, 32)?;
                let six = cx.uint(f, 6)?;
                let rounded = f.binary(BinOp::IAdd, sum, half)?;
                let rounded = f.binary(BinOp::ShiftRightLogical, rounded, six)?;
                let opaque = cx.uint(f, 255)?;
                let fixed = f.select(high, opaque, zero)?;
                let a = f.select(constant, fixed, rounded)?;
                vec![pack(f, rgb, a)?]
            }
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => {
                let w0 = cx.word(f, base, 0)?;
                let w1 = cx.word(f, base, 1)?;
                vec![cx.channel_value(f, w0, w1, decode.signed())?]
            }
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => {
                let mut words = Vec::with_capacity(2);
                for k in [0, 2] {
                    let w0 = cx.word(f, base, k)?;
                    let w1 = cx.word(f, base, k + 1)?;
                    words.push(cx.channel_value(f, w0, w1, decode.signed())?);
                }
                words
            }
        };
        // The texel's first word, `((z · height + y) · width + x) · words`.
        let at = f.binary(BinOp::IMul, z, height)?;
        let at = f.binary(BinOp::IAdd, at, y)?;
        let at = f.binary(BinOp::IMul, at, width)?;
        let at = f.binary(BinOp::IAdd, at, x)?;
        let count = super::ycbcr_convert::scalar(f, u32_, words.len() as u64)?;
        let at = f.binary(BinOp::IMul, at, count)?;
        for (k, word) in words.into_iter().enumerate() {
            let k = super::ycbcr_convert::scalar(f, u32_, k as u64)?;
            let index = f.binary(BinOp::IAdd, at, k)?;
            let p = f.access_chain(texels, &[zero, index])?;
            f.store(p, word, MemAccess::default())?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, decoded, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: BlockDecodeModule::LOCAL_SIZE,
            y: BlockDecodeModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(BlockDecodeModule::SET),
        Symbol::Binding(s) if *s == SHAPE => Some(BlockDecodeModule::SHAPE_BINDING),
        Symbol::Binding(s) if *s == BLOCKS => Some(BlockDecodeModule::BLOCKS_BINDING),
        Symbol::Binding(s) if *s == TEXELS => Some(BlockDecodeModule::TEXELS_BINDING),
        _ => None,
    }
}

/// The block decode module of `device` for `decode`: built with the typed builder, checked by
/// the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed bindings of [`BlockDecodeModule`].
pub fn block_decode_module(
    device: &Device,
    decode: BlockDecode,
) -> Result<BlockDecodeModule, Error> {
    decode_module(device, typed_module(decode)?, "block decode")
}

/// The decode module of `device` of the typed module `m` (whose bindings are the symbols
/// `SHAPE`, `BLOCKS` and `TEXELS`, `what` its kind): checked by the owned verifier against the
/// device's profile (requirements included), serialized, and numbered with the fixed bindings of
/// [`BlockDecodeModule`].
pub(super) fn decode_module(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<BlockDecodeModule, Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(BlockDecodeModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
