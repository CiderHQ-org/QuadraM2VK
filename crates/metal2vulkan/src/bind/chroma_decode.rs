//! A 4:2:2 texture's texels as Metal reads them, from its pairs: the texels the decoded twin of a
//! `GBGR422` or `BGRG422` texture holds (the runtime's `chroma_pairs.rs`), which
//! [`chroma_decode_module`] writes on the GPU.
//!
//! **The decode.** Two texels share 4 bytes: their own green codes and one blue and one red code
//! (`G0 B G1 R` for `GBGR422`, `B G0 R G1` for `BGRG422`), sited between the two. Metal reads
//! texel `x` as its own green code and, for red and blue, the codes interpolated at its centre:
//! `(3 c + n + 2) >> 2` of its pair's code `c` and the code `n` of the pair beside it on its side
//! (the pair before an even texel, after an odd one; its own pair at the row's ends), each
//! `code / 255`, alpha 1. Our probe on an Apple M3 Max under macOS 27.0.1 (both formats, 16 x 4
//! texels of random bytes): every `float` and `half` read, nearest, linear and bicubic sample
//! (`float` and `half`) and gather is bit for bit the one of an `RGBA8Unorm` texture holding those
//! codes, whose treatment their class takes (`metal::texel_format`, `RGBX8Unorm`'s). The
//! twin holds them (`R8G8B8A8_UNORM`, alpha 255), exact, and the device samples and filters it as
//! it does any 8-bit format's texture, which is how Metal samples them.
//!
//! A texel's codes read its row's pairs alone, so a decode is of whole rows.

use super::block_decode::{decode_module, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use super::ycbcr_convert::{scalar, words_buffer};
use crate::device::Device;
use crate::error::Error;
use crate::metal::PixelFormat;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::error::BResult;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule};
use crate::vk::Format;

/// The 4:2:2 decode module of `device` for `decode`'s pairs: one invocation per texel on the block
/// decode's bindings and shape ([`BlockDecodeModule`]), built with the typed builder, checked by
/// the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed bindings.
pub fn chroma_decode_module(
    device: &Device,
    decode: ChromaDecode,
) -> Result<BlockDecodeModule, Error> {
    decode_module(device, typed_module(decode)?, "4:2:2 decode")
}

/// How a 4:2:2 format's pairs lie (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChromaDecode {
    /// `GBGR422`: `G0 B G1 R`.
    Gbgr,
    /// `BGRG422`: `B G0 R G1`.
    Bgrg,
}

impl ChromaDecode {
    /// The decode of a texture of `pf`; `None` for a format of no pairs.
    pub fn of(pf: PixelFormat) -> Option<ChromaDecode> {
        match pf {
            PixelFormat::GBGR422 => Some(ChromaDecode::Gbgr),
            PixelFormat::BGRG422 => Some(ChromaDecode::Bgrg),
            _ => None,
        }
    }

    /// Both decodes.
    pub const ALL: [ChromaDecode; 2] = [ChromaDecode::Gbgr, ChromaDecode::Bgrg];

    /// The texels across and down of a pair.
    pub fn block_extent(self) -> [u32; 2] {
        [2, 1]
    }

    /// The bytes of a pair.
    pub const BLOCK_BYTES: u32 = 4;

    /// The bytes of a decoded texel.
    pub const TEXEL_BYTES: u32 = 4;

    /// The twin's format: `R8G8B8A8_UNORM` codes.
    pub fn texel_format(self) -> Format {
        Format::R8G8B8A8_UNORM
    }

    /// The Metal format of the twin's texels.
    pub fn texel_pixel_format(self) -> PixelFormat {
        PixelFormat::RGBA8Unorm
    }

    /// The byte of a pair (its 4 bytes as a little-endian word, byte `k` at bits `8k`) holding
    /// red, blue, and the green of the pair's texel `t` (0 or 1).
    fn bytes(self) -> (u32, u32, [u32; 2]) {
        match self {
            ChromaDecode::Gbgr => (3, 1, [0, 2]),
            ChromaDecode::Bgrg => (2, 0, [1, 3]),
        }
    }

    /// Texel `t` (0 or 1) of `pair` beside `next` (the pair before an even texel, after an odd
    /// one), each its 4 bytes as a little-endian word: the twin's codes as a little-endian word
    /// (red lowest, alpha 255).
    fn texel(self, pair: u32, next: u32, t: usize) -> u32 {
        let (r, b, g) = self.bytes();
        let byte = |word: u32, k: u32| (word >> (8 * k)) & 0xff;
        let chroma = |k: u32| (3 * byte(pair, k) + byte(next, k) + 2) >> 2;
        chroma(r) | byte(pair, g[t]) << 8 | chroma(b) << 16 | 0xff << 24
    }

    /// The twin's texels of whole rows `width` x `height` x `depth` (images one after another)
    /// whose `pairs` are laid out as a copy of the rows out of the texture's carrier lays them
    /// (each texel's 2 bytes, tight rows), laid out as a copy into the twin reads them.
    pub fn decode(self, pairs: &[u8], size: [u32; 3]) -> Vec<u8> {
        let [width, height, depth] = size.map(|axis| axis as usize);
        let mut out = vec![0; width * height * depth * Self::TEXEL_BYTES as usize];
        self.decode_into(pairs, size, &mut out);
        out
    }

    /// [`decode`](Self::decode) into `out`, which holds exactly the rows' texels. A row of an odd
    /// width ends inside a pair, whose bytes it holds whole.
    ///
    /// # Panics
    ///
    /// Where `pairs` holds fewer pairs than the rows, or `out` another length.
    pub fn decode_into(self, pairs: &[u8], [width, height, depth]: [u32; 3], out: &mut [u8]) {
        let (width, rows) = (width as usize, height as usize * depth as usize);
        let across = width.div_ceil(2);
        assert_eq!(out.len(), width * rows * 4, "the rows' texels");
        let word = |at: usize| {
            u32::from_le_bytes([pairs[at], pairs[at + 1], pairs[at + 2], pairs[at + 3]])
        };
        for (y, row) in out.chunks_exact_mut(width * 4).enumerate() {
            for (x, texel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let p = x / 2;
                let next = match x % 2 {
                    0 => p.saturating_sub(1),
                    _ => (p + 1).min(across - 1),
                };
                let (pair, next) = (word((y * across + p) * 4), word((y * across + next) * 4));
                *texel = self.texel(pair, next, x % 2).to_le_bytes();
            }
        }
    }
}

/// The GPU's decode: the module [`chroma_decode_module`] serializes, the same decode as the CPU's
/// ([`ChromaDecode::decode_into`]). One invocation per texel of whole rows (the block decode's
/// dispatch and shape, `BlockDecodeModule::shape`) reads its pair's word, at `(z · height + y) ·
/// ceil(width / 2) + x / 2`, and the word of the pair beside it, and writes its texel's word.
fn typed_module(decode: ChromaDecode) -> BResult<TypedModule> {
    let mut b = crate::spirv::builder::ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let pairs = words_buffer(&mut b, BLOCKS, u32_)?;
    let texels = words_buffer(&mut b, TEXELS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let [x, y, z] = [0, 1, 2].map(|i| f.composite_extract(invocation, &[i]));
    let (x, y, z) = (x?, y?, z?);
    let shape = f.module().global_value(shape)?;
    let mut field = |index: u32| -> BResult<_> {
        let index = scalar(&mut f, u32_, u64::from(index))?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let (width, height, depth) = (field(0)?, field(1)?, field(2)?);
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let in_z = f.binary(BinOp::ULessThan, z, depth)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    let pairs = f.module().global_value(pairs)?;
    let texels = f.module().global_value(texels)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let k = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v: u32| {
            scalar(f, u32_, u64::from(v))
        };
        let one = k(f, 1)?;
        // The row's index through the region, and its pairs across.
        let row = f.binary(BinOp::IMul, z, height)?;
        let row = f.binary(BinOp::IAdd, row, y)?;
        let across = f.binary(BinOp::IAdd, width, one)?;
        let across = f.binary(BinOp::ShiftRightLogical, across, one)?;
        let half = f.binary(BinOp::ShiftRightLogical, x, one)?;
        let at = f.binary(BinOp::IMul, row, across)?;
        let at = f.binary(BinOp::IAdd, at, half)?;
        let p = f.access_chain(pairs, &[zero, at])?;
        let pair = f.load(p, MemAccess::default())?;
        // The pair beside it: before an even texel, after an odd one, its own at the row's ends.
        let odd = f.binary(BinOp::BitwiseAnd, x, one)?;
        let odd = f.binary(BinOp::INotEqual, odd, zero)?;
        let first = f.binary(BinOp::IEqual, half, zero)?;
        let before = f.binary(BinOp::ISub, at, one)?;
        let before = f.select(first, at, before)?;
        let last = f.binary(BinOp::ISub, across, one)?;
        let last = f.binary(BinOp::IEqual, half, last)?;
        let after = f.binary(BinOp::IAdd, at, one)?;
        let after = f.select(last, at, after)?;
        let beside = f.select(odd, after, before)?;
        let p = f.access_chain(pairs, &[zero, beside])?;
        let next = f.load(p, MemAccess::default())?;
        let (r, b, g) = decode.bytes();
        let byte = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, word, at: u32| {
            let shift = k(f, 8 * at)?;
            let mask = k(f, 0xff)?;
            let v = f.binary(BinOp::ShiftRightLogical, word, shift)?;
            f.binary(BinOp::BitwiseAnd, v, mask)
        };
        // `(3 c + n + 2) >> 2`.
        let chroma = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, at: u32| {
            let c = byte(f, pair, at)?;
            let n = byte(f, next, at)?;
            let three = k(f, 3)?;
            let two = k(f, 2)?;
            let v = f.binary(BinOp::IMul, c, three)?;
            let v = f.binary(BinOp::IAdd, v, n)?;
            let v = f.binary(BinOp::IAdd, v, two)?;
            f.binary(BinOp::ShiftRightLogical, v, two)
        };
        let red = chroma(f, r)?;
        let blue = chroma(f, b)?;
        let green = [byte(f, pair, g[0])?, byte(f, pair, g[1])?];
        let green = f.select(odd, green[1], green[0])?;
        let eight = k(f, 8)?;
        let sixteen = k(f, 16)?;
        let green = f.binary(BinOp::ShiftLeftLogical, green, eight)?;
        let blue = f.binary(BinOp::ShiftLeftLogical, blue, sixteen)?;
        let alpha = k(f, 0xff << 24)?;
        let word = f.binary(BinOp::BitwiseOr, red, green)?;
        let word = f.binary(BinOp::BitwiseOr, word, blue)?;
        let word = f.binary(BinOp::BitwiseOr, word, alpha)?;
        let t = f.binary(BinOp::IMul, row, width)?;
        let t = f.binary(BinOp::IAdd, t, x)?;
        let ptr = f.access_chain(texels, &[zero, t])?;
        f.store(ptr, word, MemAccess::default())?;
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

#[cfg(test)]
mod tests;
