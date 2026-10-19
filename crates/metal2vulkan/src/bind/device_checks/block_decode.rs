//! The checks of the block-compressed formats' decodes, which Vulkan leaves to the driver: the
//! driver facts `ColorBlockDecodeIsMetals` (BC1 to BC3), `ChannelBlockDecodeIsMetals` (BC4 and
//! BC5), `Bc6hBlockDecodeIsMetals`, `Bc7BlockDecodeIsMetals` and `Bc7ReservedBlockIsMetals`.
//!
//! **The fetch** ([`texel_fetch_check_module`]): one `GLCompute` entry `main` at local size
//! ([`CheckKernel::LOCAL_SIZE`](super::CheckKernel::LOCAL_SIZE), 1, 1), and in set 0 a sampled 2D image at
//! [`TexelFetchCheck::IMAGE_BINDING`] and a storage buffer of 4-byte words at
//! [`TexelFetchCheck::TEXELS_BINDING`]. Invocation `i` below [`TexelFetchCheck::TEXELS`] fetches
//! texel `(i % SIDE, i / SIDE)` of level 0 ([`TexelFetchCheck::SIDE`]; `OpImageFetch`, each
//! sample of a multisampled image in turn) and stores sample `s`'s four components' bits at words
//! `4 (i n + s)` to `4 (i n + s) + 3` of the buffer, `n` the image's samples. It is dispatched as
//! [`TexelFetchCheck::workgroups`] workgroups along x.
//!
//! **The cases** ([`block_decode_cases`]): for each unsigned-normalized or float BC format (the
//! sRGB forms decode to the unorm forms' codes), 64 fixed random blocks, laid out 8 to a row in a
//! 32 x 32 image, every texel fetched as `float`s and compared bit for bit with Metal's read of it:
//! - BC1 to BC5: the crate's fitted decode (`bind::BlockDecode::texel`), each 8-bit code read as
//!   `code / 255`, BC4 as `(r, 0, 0, 1)` and BC5 as `(r, g, 0, 1)`. A signed block of every eighth
//!   has its first endpoint -128 and one more its second, which Metal clamps to -1 after the sum.
//! - BC6H and BC7: Metal's texels of the same blocks, read on an Apple M3 Max with macOS 27.0.1
//!   (`block_answers.txt`). The BC6H blocks cycle through its 18 mode codes, the four reserved
//!   among them (Metal reads those as 0); the BC7 blocks through its 8 modes and the reserved
//!   mode byte 0 (Metal reads `(0, 0, 0, 1)`). BC7's reserved blocks decide
//!   `Bc7ReservedBlockIsMetals`, its others `Bc7BlockDecodeIsMetals`
//!   ([`BlockDecodeCase::block_fact`]).

use std::sync::Arc;

use super::kernel::{finish, LOCAL_SIZE};
use super::{u32_value, words_buffer};
use crate::bind::block_decode::BlockDecode;
use crate::device::driver_facts::DriverFact;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::lower::arith::legal::f16_to_f64;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, IoDir,
    MemAccess, StorageClass, Type, TypedModule,
};
use crate::vk::Format;

/// What a [`texel_fetch_check_module`] fetches: its image's sampled type and samples.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TexelFetch {
    /// Whether the image's texels are unsigned integers (else `float`s).
    pub unsigned: bool,
    /// The image's samples: 1, or a multisampled image's count.
    pub samples: u32,
}

impl TexelFetch {
    /// A single-sampled image of `float` texels.
    pub const FLOAT: TexelFetch = TexelFetch {
        unsigned: false,
        samples: 1,
    };

    /// An image of `samples` samples of unsigned integer texels (`unsigned`) or `float`s.
    pub fn new(unsigned: bool, samples: u32) -> TexelFetch {
        TexelFetch { unsigned, samples }
    }
}

/// The texel fetch's module on one device (the module docs), validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TexelFetchCheck {
    /// SPIR-V 1.5 words, entry point `main`, the bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device.
    pub requirements: Requirements,
    /// What it fetches.
    pub fetch: TexelFetch,
}

impl TexelFetchCheck {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The image fetched, a sampled image.
    pub const IMAGE_BINDING: u32 = 0;
    /// The words written, a storage buffer of [`TexelFetchCheck::words`] words.
    pub const TEXELS_BINDING: u32 = 1;
    /// The image's side, in texels: its fetched region is `SIDE x SIDE` from the origin.
    pub const SIDE: u32 = 32;
    /// The texels fetched.
    pub const TEXELS: u32 = Self::SIDE * Self::SIDE;

    /// The words the fetch writes: four for each sample of each texel.
    pub fn words(&self) -> u32 {
        Self::TEXELS * self.fetch.samples * 4
    }

    /// The workgroups the fetch dispatches along x.
    pub fn workgroups(&self) -> u32 {
        Self::TEXELS.div_ceil(LOCAL_SIZE)
    }
}

const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: TexelFetchCheck::IMAGE_BINDING,
};
const TEXELS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: TexelFetchCheck::TEXELS_BINDING,
};

/// The texel fetch of `fetch` as a typed module (the module docs).
fn texel_fetch_typed_module(fetch: TexelFetch) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type: if fetch.unsigned { u32_ } else { f32_ },
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: fetch.samples > 1,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_ty)?;
    let texels = words_buffer(&mut b, TEXELS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let side = u32_value(&mut f, TexelFetchCheck::SIDE)?;
    let count = u32_value(&mut f, TexelFetchCheck::TEXELS)?;
    let stride = u32_value(&mut f, 4 * fetch.samples)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let image = f.module().global_value(image)?;
    let texels = f.module().global_value(texels)?;
    let inside = f.binary(BinOp::ULessThan, x, count)?;
    let store: ArmFn<'_, ()> = Box::new(move |f, _| {
        let column = f.binary(BinOp::UMod, x, side)?;
        let row = f.binary(BinOp::UDiv, x, side)?;
        let pixel = f.composite_construct(uvec2, &[column, row])?;
        let image = f.load(image, MemAccess::default())?;
        let first = f.binary(BinOp::IMul, x, stride)?;
        for s in 0..fetch.samples {
            let ops = match fetch.samples {
                1 => ImageOperands {
                    lod: Some(zero),
                    ..ImageOperands::default()
                },
                _ => ImageOperands {
                    sample: Some(u32_value(f, s)?),
                    ..ImageOperands::default()
                },
            };
            let texel = f.fetch(image, pixel, &ops)?;
            for c in 0..4 {
                let component = f.composite_extract(texel, &[c])?;
                let bits = match fetch.unsigned {
                    true => component,
                    false => f.convert(ConvOp::Bitcast, component, u32_)?,
                };
                let at = u32_value(f, 4 * s + c)?;
                let at = f.binary(BinOp::IAdd, first, at)?;
                let to = f.access_chain(texels, &[zero, at])?;
                f.store(to, bits, MemAccess::default())?;
            }
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, store, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The texel fetch of `fetch` on `device` (the module docs), built, checked against the device's
/// profile and numbered as the other checks' kernels are.
pub fn texel_fetch_check_module(
    device: &Device,
    fetch: TexelFetch,
) -> Result<TexelFetchCheck, Error> {
    let module = texel_fetch_typed_module(fetch)?;
    let (spirv, requirements) = finish(device, module, "texel fetch check")?;
    Ok(TexelFetchCheck {
        spirv,
        requirements,
        fetch,
    })
}

/// The blocks of a case: in a 32 x 32 image, 8 to a row of blocks.
const BLOCKS: usize = 64;

/// The blocks a row of the image holds.
const BLOCKS_A_ROW: usize = TexelFetchCheck::SIDE as usize / 4;

/// One format's blocks, and the texels Metal reads of them (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDecodeCase {
    /// The block-compressed format of the image.
    pub format: Format,
    /// The fact the case decides, with the other cases of it: of every block but those of a
    /// reserved mode [`reserved`](Self::reserved) decides.
    pub fact: DriverFact,
    /// The fact the case's blocks of a reserved mode decide apart from the others (BC7's, its
    /// first byte 0: `Bc7ReservedBlockIsMetals`); `None` where [`fact`](Self::fact) decides every
    /// block.
    pub reserved: Option<DriverFact>,
    /// The blocks, the image's rows of blocks first to last, tight: what a copy into the image's
    /// level 0 of `TexelFetchCheck::SIDE` texels a side reads.
    pub blocks: Vec<u8>,
    /// The words Metal's reads of the texels leave in the fetch's buffer
    /// ([`TexelFetchCheck`] of [`TexelFetch::FLOAT`]): each texel's four `float`s' bits.
    pub expected: Vec<u32>,
}

impl BlockDecodeCase {
    /// The fact block `block` of the case decides (its blocks numbered row by row, 8 to a row):
    /// [`reserved`](Self::reserved)'s for a block whose first byte is 0, where the case has
    /// one, else [`fact`](Self::fact)'s.
    pub fn block_fact(&self, block: usize) -> DriverFact {
        let bytes = self.blocks.len() / BLOCKS;
        match self.reserved {
            Some(reserved) if self.blocks[block * bytes] == 0 => reserved,
            _ => self.fact,
        }
    }

    /// Whether the case decides `fact`: of its blocks, or of its reserved blocks.
    pub fn decides(&self, fact: DriverFact) -> bool {
        self.fact == fact || self.reserved == Some(fact)
    }

    /// How `words`, the fetch's buffer after it, differ from Metal's reads at the texels of the
    /// blocks that decide `fact` ([`block_fact`](Self::block_fact)), in a few words for the
    /// other answer's statement: how many texels differ of how many, and the first, its block
    /// (and the block's first byte, which holds its mode) and the words read and Metal's. `None`
    /// for Metal's answer.
    pub fn difference(&self, fact: DriverFact, words: &[u32]) -> Option<String> {
        if words.len() != self.expected.len() {
            return Some(format!(
                "{:?}: {} words read where Metal's reads are {}",
                self.format,
                words.len(),
                self.expected.len()
            ));
        }
        let side = TexelFetchCheck::SIDE as usize;
        let block_of = |t: usize| (t / side / 4) * BLOCKS_A_ROW + t % side / 4;
        let texels: Vec<usize> = (0..words.len() / 4)
            .filter(|&t| self.block_fact(block_of(t)) == fact)
            .collect();
        let differs = |t: &&usize| words[*t * 4..*t * 4 + 4] != self.expected[*t * 4..*t * 4 + 4];
        let differing: Vec<usize> = texels.iter().filter(differs).copied().collect();
        let &first = differing.first()?;
        let (x, y) = (first % side, first / side);
        let block = block_of(first);
        let mode = self.blocks[block * (self.blocks.len() / BLOCKS)];
        Some(format!(
            "{:?}: {} of {} texels differ, the first ({x}, {y}, of block {block}, its first byte \
             {mode:#04x}) read {:08x?}, Metal's {:08x?}",
            self.format,
            differing.len(),
            texels.len(),
            &words[first * 4..first * 4 + 4],
            &self.expected[first * 4..first * 4 + 4]
        ))
    }
}

/// The xorshift generator of the blocks' bytes, from `seed`.
struct Blocks(u32);

impl Blocks {
    /// The next 4 bytes, little-endian.
    fn next(&mut self) -> [u8; 4] {
        let s = &mut self.0;
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        s.to_le_bytes()
    }

    /// `count` blocks of `bytes` each.
    fn take(mut self, count: usize, bytes: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|_| (0..bytes / 4).flat_map(|_| self.next()).collect())
            .collect()
    }
}

/// The BC6H mode codes, low bits first: the two 2-bit modes, then every 5-bit code whose low
/// two bits are 2 or 3, the last four reserved.
const BC6H_MODES: [u8; 18] = [
    0b00, 0b01, 0b00010, 0b00110, 0b01010, 0b01110, 0b10010, 0b10110, 0b11010, 0b11110, 0b00011,
    0b00111, 0b01011, 0b01111, 0b10011, 0b10111, 0b11011, 0b11111,
];

/// The BC6H blocks of `seed`: block `i` of mode code `BC6H_MODES[i % 18]`.
fn bc6h_blocks(seed: u32) -> Vec<Vec<u8>> {
    let mut blocks = Blocks(seed).take(BLOCKS, 16);
    for (i, block) in blocks.iter_mut().enumerate() {
        let code = BC6H_MODES[i % BC6H_MODES.len()];
        let mask = if code < 0b10 { 0b11 } else { 0b11111 };
        block[0] = (block[0] & !mask) | code;
    }
    blocks
}

/// The BC7 blocks: block `i` of mode `i % 9`, the ninth the reserved mode byte 0.
fn bc7_blocks() -> Vec<Vec<u8>> {
    let mut blocks = Blocks(0xBC7B_0001).take(BLOCKS, 16);
    for (i, block) in blocks.iter_mut().enumerate() {
        let mode = (i % 9) as u32;
        block[0] = match mode {
            8 => 0,
            mode => (block[0] & !((2u32 << mode) - 1) as u8) | 1 << mode,
        };
    }
    blocks
}

/// The blocks of a BC1 to BC5 `decode` of `seed`: a signed channel block of every eighth with
/// its first endpoint -128, and of the one four after it with its second.
fn bc1_to_bc5_blocks(decode: BlockDecode, seed: u32) -> Vec<Vec<u8>> {
    let bytes = decode.block_bytes() as usize;
    let mut blocks = Blocks(seed).take(BLOCKS, bytes);
    if matches!(decode, BlockDecode::Bc4Snorm | BlockDecode::Bc5Snorm) {
        for (i, block) in blocks.iter_mut().enumerate() {
            let endpoint = match i % 8 {
                3 => 0,
                7 => 1,
                _ => continue,
            };
            for channel in (0..bytes).step_by(8) {
                block[channel + endpoint] = 0x80;
            }
        }
    }
    blocks
}

/// The image's texel words of `blocks`, each block's 16 texels' words from `texel(block, j)`
/// (texel `j` row-major in its block): the blocks laid 8 to a row.
fn laid_out(blocks: &[Vec<u8>], texel: impl Fn(&[u8], usize) -> [u32; 4]) -> Vec<u32> {
    let side = TexelFetchCheck::SIDE as usize;
    let mut words = vec![0; side * side * 4];
    for (b, block) in blocks.iter().enumerate() {
        let (bx, by) = (b % BLOCKS_A_ROW * 4, b / BLOCKS_A_ROW * 4);
        for j in 0..16 {
            let at = ((by + j / 4) * side + bx + j % 4) * 4;
            words[at..at + 4].copy_from_slice(&texel(block, j));
        }
    }
    words
}

/// The `f32` bits of 8-bit code `code` read as a normalized value, `code / 255`.
fn unorm8(code: u32) -> u32 {
    (code as f32 / 255.0).to_bits()
}

/// `1.0`'s bits.
const ONE: u32 = 0x3F80_0000;

/// The case of a BC1 to BC5 `decode` of `format`: its texels the crate's decode's.
fn bc1_to_bc5_case(decode: BlockDecode, format: Format, seed: u32) -> BlockDecodeCase {
    let blocks = bc1_to_bc5_blocks(decode, seed);
    let expected = laid_out(&blocks, |block, j| {
        let [word, green] = decode.texel(block, j);
        match decode {
            BlockDecode::Bc1 | BlockDecode::Bc2 | BlockDecode::Bc3 => {
                [0, 8, 16, 24].map(|shift| unorm8(word >> shift & 0xFF))
            }
            BlockDecode::Bc4Unorm | BlockDecode::Bc4Snorm => [word, 0, 0, ONE],
            BlockDecode::Bc5Unorm | BlockDecode::Bc5Snorm => [word, green, 0, ONE],
        }
    });
    let fact = match decode {
        BlockDecode::Bc1 | BlockDecode::Bc2 | BlockDecode::Bc3 => {
            DriverFact::ColorBlockDecodeIsMetals
        }
        _ => DriverFact::ChannelBlockDecodeIsMetals,
    };
    BlockDecodeCase {
        format,
        fact,
        reserved: None,
        blocks: blocks.concat(),
        expected,
    }
}

/// Metal's texels of the BC6H and BC7 blocks, `block_answers.txt`'s format stated at its top.
const ANSWERS: &str = include_str!("block_answers.txt");

/// `block_answers.txt`'s format `name`: each block and its texels' bytes, in the file's order.
fn answers(name: &str) -> Vec<(Vec<u8>, Vec<u8>)> {
    let hex = |text: &str| -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap_or_default())
            .collect()
    };
    let heading = format!("format {name}");
    (ANSWERS.lines())
        .filter(|line| !line.starts_with('#'))
        .skip_while(|line| *line != heading)
        .skip(1)
        .take_while(|line| !line.starts_with("format "))
        .filter_map(|line| line.split_once(' '))
        .map(|(block, texels)| (hex(block), hex(texels)))
        .collect()
}

/// The case of `format` deciding `facts` (its blocks', its reserved blocks'), whose blocks are
/// `blocks` and Metal's texels `name`'s of `block_answers.txt`, each texel's bytes read by
/// `texel`. Metal's texels are those of the file's blocks, which a test holds equal to `blocks`.
fn recorded_case(
    format: Format,
    (fact, reserved): (DriverFact, Option<DriverFact>),
    name: &str,
    blocks: Vec<Vec<u8>>,
    texel: fn(&[u8]) -> [u32; 4],
) -> BlockDecodeCase {
    let answers = answers(name);
    let size = answers.first().map_or(0, |(_, texels)| texels.len() / 16);
    let read: Vec<Vec<u8>> = answers.into_iter().map(|(_, texels)| texels).collect();
    let expected = laid_out(&read, |texels, j| texel(&texels[j * size..(j + 1) * size]));
    BlockDecodeCase {
        format,
        fact,
        reserved,
        blocks: blocks.concat(),
        expected,
    }
}

/// A BC6H texel's words of its red, green and blue `half`s, little-endian: each widened, and
/// alpha 1.
fn bc6h_texel(bytes: &[u8]) -> [u32; 4] {
    let half = |c: usize| {
        let h = u16::from_le_bytes([bytes[2 * c], bytes[2 * c + 1]]);
        (f16_to_f64(h) as f32).to_bits()
    };
    [half(0), half(1), half(2), ONE]
}

/// A BC7 texel's words of its four 8-bit codes.
fn bc7_texel(bytes: &[u8]) -> [u32; 4] {
    [0, 1, 2, 3].map(|c| unorm8(u32::from(bytes[c])))
}

/// Every case of the block decodes' checks (the module docs), the cases of each fact together,
/// in the facts' order: BC1 to BC3, BC4 and BC5, BC6H, BC7.
pub fn block_decode_cases() -> Vec<BlockDecodeCase> {
    let channel = |decode, format, seed| bc1_to_bc5_case(decode, format, seed);
    vec![
        channel(BlockDecode::Bc1, Format::BC1_RGBA_UNORM_BLOCK, 0xBC1B_0001),
        channel(BlockDecode::Bc2, Format::BC2_UNORM_BLOCK, 0xBC2B_0001),
        channel(BlockDecode::Bc3, Format::BC3_UNORM_BLOCK, 0xBC3B_0001),
        channel(BlockDecode::Bc4Unorm, Format::BC4_UNORM_BLOCK, 0xBC4B_0001),
        channel(BlockDecode::Bc4Snorm, Format::BC4_SNORM_BLOCK, 0xBC4B_0002),
        channel(BlockDecode::Bc5Unorm, Format::BC5_UNORM_BLOCK, 0xBC5B_0001),
        channel(BlockDecode::Bc5Snorm, Format::BC5_SNORM_BLOCK, 0xBC5B_0002),
        recorded_case(
            Format::BC6H_UFLOAT_BLOCK,
            (DriverFact::Bc6hBlockDecodeIsMetals, None),
            "bc6h_ufloat",
            bc6h_blocks(0xBC6B_0001),
            bc6h_texel,
        ),
        recorded_case(
            Format::BC6H_SFLOAT_BLOCK,
            (DriverFact::Bc6hBlockDecodeIsMetals, None),
            "bc6h_sfloat",
            bc6h_blocks(0xBC6B_0002),
            bc6h_texel,
        ),
        recorded_case(
            Format::BC7_UNORM_BLOCK,
            (
                DriverFact::Bc7BlockDecodeIsMetals,
                Some(DriverFact::Bc7ReservedBlockIsMetals),
            ),
            "bc7_unorm",
            bc7_blocks(),
            bc7_texel,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        answers, bc6h_blocks, bc7_blocks, block_decode_cases, texel_fetch_typed_module, TexelFetch,
        TexelFetchCheck, BLOCKS,
    };
    use crate::device::driver_facts::DriverFact;
    use crate::spirv::testing::{assert_valid, committed_profiles};
    use crate::vk::Format;

    /// BC7's reserved blocks' fact.
    const RESERVED: DriverFact = DriverFact::Bc7ReservedBlockIsMetals;

    #[test]
    fn the_fetch_modules_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(
                texel_fetch_typed_module(TexelFetch::FLOAT).expect("module"),
                &p,
            );
            let multisampled = TexelFetch::new(true, 4);
            assert_valid(texel_fetch_typed_module(multisampled).expect("module"), &p);
        }
    }

    /// The recorded blocks are the generator's: Metal's texels are of the blocks the check
    /// uploads.
    #[test]
    fn the_recorded_blocks_are_the_generated_ones() {
        for (name, blocks) in [
            ("bc6h_ufloat", bc6h_blocks(0xBC6B_0001)),
            ("bc6h_sfloat", bc6h_blocks(0xBC6B_0002)),
            ("bc7_unorm", bc7_blocks()),
        ] {
            let recorded: Vec<Vec<u8>> = answers(name).into_iter().map(|(b, _)| b).collect();
            assert_eq!(recorded, blocks, "{name}");
        }
    }

    /// Every case fills the fetched image whole: its blocks cover it, and its expected words are
    /// four for each texel.
    #[test]
    fn every_case_fills_the_image() {
        let cases = block_decode_cases();
        assert_eq!(cases.len(), 10);
        for case in &cases {
            let block = case.blocks.len() / BLOCKS;
            assert!(block == 8 || block == 16, "{:?}", case.format);
            assert_eq!(case.blocks.len() % BLOCKS, 0, "{:?}", case.format);
            assert_eq!(
                case.expected.len() as u32,
                TexelFetchCheck::TEXELS * 4,
                "{:?}",
                case.format
            );
            assert_eq!(case.difference(case.fact, &case.expected), None);
        }
        let case = &cases[9];
        assert_eq!(case.format, Format::BC7_UNORM_BLOCK);
        let mut words = case.expected.clone();
        words[4 * 37] ^= 1;
        let difference = case.difference(case.fact, &words).expect("a difference");
        let first = format!(
            "{:?}: 1 of 912 texels differ, the first (5, 1, of block 1, its first byte 0xda)",
            case.format
        );
        assert!(difference.starts_with(&first), "{difference}");
        assert_eq!(case.difference(RESERVED, &words), None);
    }

    /// BC7's blocks of the reserved mode, the ninth of each nine (its first byte 0), decide
    /// `Bc7ReservedBlockIsMetals` apart from its others, and Metal reads each of their texels as
    /// opaque black: a driver's transparent black there differs for that fact alone.
    #[test]
    fn bc7_s_reserved_blocks_decide_their_own_fact() {
        let cases = block_decode_cases();
        let case = &cases[9];
        let reserved: Vec<usize> = (0..BLOCKS)
            .filter(|&b| case.block_fact(b) == RESERVED)
            .collect();
        assert_eq!(reserved, vec![8, 17, 26, 35, 44, 53, 62]);
        assert!(case.decides(RESERVED) && case.decides(case.fact));
        assert!(cases[..9].iter().all(|other| !other.decides(RESERVED)));
        let side = TexelFetchCheck::SIDE as usize;
        let mut words = case.expected.clone();
        for &block in &reserved {
            let (bx, by) = (block % 8 * 4, block / 8 * 4);
            for j in 0..16 {
                let at = ((by + j / 4) * side + bx + j % 4) * 4;
                assert_eq!(words[at..at + 4], [0, 0, 0, 0x3F80_0000], "block {block}");
                words[at + 3] = 0;
            }
        }
        assert_eq!(case.difference(case.fact, &words), None);
        let difference = case.difference(RESERVED, &words).expect("a difference");
        let first = format!(
            "{:?}: 112 of 112 texels differ, the first (0, 4, of block 8, its first byte 0x00)",
            case.format
        );
        assert!(difference.starts_with(&first), "{difference}");
    }
}
