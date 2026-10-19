//! The check of the driver fact `SampledFetchRecoversCodes`: a sampled view's fetch of each
//! normalized or 16-bit float format the mipmap filter reads in place returns, for every code, a
//! value its recovery reads that code back from (`bind::mipmap_filter`'s `fetch.rs`).
//!
//! **The module** ([`sampled_fetch_check_module`]): one `GLCompute` entry `main` at local size
//! ([`CheckKernel::LOCAL_SIZE`](super::CheckKernel::LOCAL_SIZE), 1, 1), and in set 0 a sampled
//! 2D image of `float` texels at [`SampledFetchCheck::IMAGE_BINDING`], a storage buffer of 4-byte
//! words written at [`SampledFetchCheck::CODES_BINDING`] and one of the case's parameters
//! ([`SampledFetchCase::parameters`]) at [`SampledFetchCheck::PARAMETERS_BINDING`]. Invocation `i`
//! below the case's texels fetches texel `(i % width, i / width)` of level 0, recovers each
//! component's code by its rule as the filter's fetch does (a unorm code `RoundEven(v (2^b - 1))`,
//! a snorm one `RoundEven(v (2^(b-1) - 1))` in `b` two's-complement bits, a half's bits exactly,
//! a NaN quieted), places it at its bit offset of the texel's bytes, and stores the texel's 64
//! bits at words `2 i` and `2 i + 1`.
//!
//! **The cases** ([`sampled_fetch_cases`]): each format of the filter's fetch whose components
//! are converted (no integer carrier), an image of `2^b` texels for its widest component of `b`
//! bits, texel `i` holding code `i` in each component (its low bits), 256 to a row. Metal's
//! answer: every code read back, but for what the filter reads alike: an snorm component's code
//! `-2^(b-1)` reads as `-(2^(b-1) - 1)` (both -1.0), and a half NaN as any NaN.

use std::sync::Arc;

use super::kernel::{finish, LOCAL_SIZE};
use super::{u32_value, words_buffer};
use crate::bind::mipmap_filter::fetch::{Component, Recover, COMPONENTS};
use crate::bind::mipmap_filter::kernel::half_bits;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType, ImageUsage, IoDir,
    MemAccess, StorageClass, Type, TypedModule,
};
use crate::vk::Format;

/// The sampled fetch check's module on one device (the module docs), validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledFetchCheck {
    /// SPIR-V 1.5 words, entry point `main`, the bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device.
    pub requirements: Requirements,
}

impl SampledFetchCheck {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The image fetched, a sampled image.
    pub const IMAGE_BINDING: u32 = 0;
    /// The recovered texels, a storage buffer of [`SampledFetchCase::words`] words.
    pub const CODES_BINDING: u32 = 1;
    /// The case's parameters, a storage buffer of [`SampledFetchCase::parameters`].
    pub const PARAMETERS_BINDING: u32 = 2;
}

const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: SampledFetchCheck::IMAGE_BINDING,
};
const CODES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: SampledFetchCheck::CODES_BINDING,
};
const PARAMETERS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: SampledFetchCheck::PARAMETERS_BINDING,
};

/// A component's rule in the parameters: none, unorm, snorm, half.
const NONE: u32 = 0;
const UNORM: u32 = 1;
const SNORM: u32 = 2;
const HALF: u32 = 3;

/// The parameters' words: the width and the texels, then each component's rule, bits and bit
/// offset.
const PARAMETER_WORDS: usize = 2 + 3 * 4;

/// `v` as a value.
fn uint(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    u32_value(f, v)
}

/// The sampled fetch check as a typed module (the module docs).
fn sampled_fetch_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_ty)?;
    let codes = words_buffer(&mut b, CODES, u32_)?;
    let parameters = words_buffer(&mut b, PARAMETERS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = uint(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let image = f.module().global_value(image)?;
    let codes = f.module().global_value(codes)?;
    let parameters = f.module().global_value(parameters)?;
    let parameter = move |f: &mut FunctionBuilder<'_>, at: u32| -> BResult<Value> {
        let at = uint(f, at)?;
        let from = f.access_chain(parameters, &[zero, at])?;
        f.load(from, MemAccess::default())
    };
    let width = parameter(&mut f, 0)?;
    let texels = parameter(&mut f, 1)?;
    let inside = f.binary(BinOp::ULessThan, x, texels)?;
    let store: ArmFn<'_, ()> = Box::new(move |f, _| {
        let column = f.binary(BinOp::UMod, x, width)?;
        let row = f.binary(BinOp::UDiv, x, width)?;
        let pixel = f.composite_construct(uvec2, &[column, row])?;
        let image = f.load(image, MemAccess::default())?;
        let ops = ImageOperands {
            lod: Some(zero),
            ..ImageOperands::default()
        };
        let texel = f.fetch(image, pixel, &ops)?;
        let (one, five, thirty_one) = (uint(f, 1)?, uint(f, 5)?, uint(f, 31)?);
        let mut words = [zero, zero];
        for lane in 0..4u32 {
            let rule = parameter(f, 2 + 3 * lane)?;
            let bits = parameter(f, 3 + 3 * lane)?;
            let offset = parameter(f, 4 + 3 * lane)?;
            let v = f.composite_extract(texel, &[lane])?;
            let mask = f.binary(BinOp::ShiftLeftLogical, one, bits)?;
            let mask = f.binary(BinOp::ISub, mask, one)?;
            let scaled = |f: &mut FunctionBuilder<'_>, scale: Value| -> BResult<Value> {
                let scale = f.convert(ConvOp::ConvertUToF, scale, f32_)?;
                let product = f.binary(BinOp::FMul, v, scale)?;
                f.glsl(GlslOp::RoundEven, &[product])
            };
            let unorm = scaled(f, mask)?;
            let unorm = f.convert(ConvOp::ConvertFToU, unorm, u32_)?;
            let half_mask = f.binary(BinOp::ShiftRightLogical, mask, one)?;
            let snorm = scaled(f, half_mask)?;
            let snorm = f.convert(ConvOp::ConvertFToS, snorm, i32_)?;
            let snorm = f.convert(ConvOp::Bitcast, snorm, u32_)?;
            let snorm = f.binary(BinOp::BitwiseAnd, snorm, mask)?;
            let v_bits = f.convert(ConvOp::Bitcast, v, u32_)?;
            let half = half_bits(f, v_bits)?;
            let mut code = zero;
            for (kind, value) in [(UNORM, unorm), (SNORM, snorm), (HALF, half)] {
                let kind = uint(f, kind)?;
                let is = f.binary(BinOp::IEqual, rule, kind)?;
                code = f.select(is, value, code)?;
            }
            let shift = f.binary(BinOp::BitwiseAnd, offset, thirty_one)?;
            let shifted = f.binary(BinOp::ShiftLeftLogical, code, shift)?;
            let word = f.binary(BinOp::ShiftRightLogical, offset, five)?;
            for (w, at) in words.iter_mut().zip([0u32, 1]) {
                let at = uint(f, at)?;
                let here = f.binary(BinOp::IEqual, word, at)?;
                let part = f.select(here, shifted, zero)?;
                *w = f.binary(BinOp::BitwiseOr, *w, part)?;
            }
        }
        let two = uint(f, 2)?;
        let first = f.binary(BinOp::IMul, x, two)?;
        for (w, at) in words.into_iter().zip([0u32, 1]) {
            let at = uint(f, at)?;
            let at = f.binary(BinOp::IAdd, first, at)?;
            let to = f.access_chain(codes, &[zero, at])?;
            f.store(to, w, MemAccess::default())?;
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

/// The sampled fetch check's module on `device` (the module docs), built, checked against the
/// device's profile and numbered as the other checks' kernels are.
pub fn sampled_fetch_check_module(device: &Device) -> Result<SampledFetchCheck, Error> {
    let module = sampled_fetch_typed_module()?;
    let (spirv, requirements) = finish(device, module, "sampled fetch check")?;
    Ok(SampledFetchCheck {
        spirv,
        requirements,
    })
}

/// The texels a row of a case's image holds, at most.
const ROW: u32 = 256;

/// One format's image of every code, and the parameters of its fetch (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampledFetchCase {
    /// The image's format.
    pub format: Format,
    /// The image's width and height, in texels.
    pub extent: [u32; 2],
    /// Its texels' bytes, tight: what a copy into its level 0 reads.
    pub texels: Vec<u8>,
    /// The module's parameters for it.
    pub parameters: Vec<u32>,
    components: [Option<Component>; 4],
}

impl SampledFetchCase {
    /// Its texels.
    pub fn texel_count(&self) -> u32 {
        self.extent[0] * self.extent[1]
    }

    /// The words the module writes: two a texel.
    pub fn words(&self) -> u32 {
        2 * self.texel_count()
    }

    /// The workgroups the module dispatches along x.
    pub fn workgroups(&self) -> u32 {
        self.texel_count().div_ceil(LOCAL_SIZE)
    }

    /// How `words`, the module's buffer after it, differ from Metal's answer (the module docs),
    /// in a few words for the other answer's statement: how many texels differ, and the first
    /// component that does, its code and the code read. `None` for Metal's answer.
    pub fn difference(&self, words: &[u32]) -> Option<String> {
        if words.len() != self.words() as usize {
            return Some(format!(
                "{:?}: {} words read of {}",
                self.format,
                words.len(),
                self.words()
            ));
        }
        let mut differing = 0;
        let mut first = None;
        for i in 0..self.texel_count() {
            let read =
                u64::from(words[2 * i as usize]) | u64::from(words[2 * i as usize + 1]) << 32;
            let wrong = (self.components.iter().enumerate())
                .filter_map(|(lane, c)| Some((lane, (*c)?)))
                .find(|&(_, c)| !reads_back(c, code_of(i, c), field(read, c)));
            if let Some((lane, c)) = wrong {
                differing += 1;
                first.get_or_insert((i, lane, code_of(i, c), field(read, c)));
            }
        }
        let (texel, lane, code, read) = first?;
        Some(format!(
            "{:?}: {differing} of {} texels differ, the first texel {texel}'s component {lane} \
             read code {read:#x} for {code:#x}",
            self.format,
            self.texel_count()
        ))
    }
}

/// The low `bits` bits.
fn mask(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

/// Component `c`'s field of a texel's 64 bits.
fn field(texel: u64, c: Component) -> u32 {
    ((texel >> c.offset) & mask(c.bits)) as u32
}

/// Component `c`'s code in texel `i`: `i`'s low bits.
fn code_of(i: u32, c: Component) -> u32 {
    (u64::from(i) & mask(c.bits)) as u32
}

/// Whether a fetch's recovery read `read` for component `c`'s code `code` as Metal's answer
/// has it (the module docs).
fn reads_back(c: Component, code: u32, read: u32) -> bool {
    match c.recover {
        Recover::Snorm if code == 1 << (c.bits - 1) => read == code + 1,
        Recover::Half if code & 0x7C00 == 0x7C00 && code & 0x3FF != 0 => {
            read & 0x7C00 == 0x7C00 && read & 0x3FF != 0
        }
        _ => read == code,
    }
}

/// The case of `format` of `components` (the module docs).
fn case(format: Format, components: [Option<Component>; 4]) -> SampledFetchCase {
    let present = components.iter().flatten();
    let widest = present.clone().map(|c| c.bits).max().unwrap_or(1);
    let bits = present.map(|c| c.offset + c.bits).max().unwrap_or(8);
    let bytes = bits.div_ceil(8) as usize;
    let count = 1u32 << widest;
    let width = count.min(ROW);
    let mut texels = Vec::with_capacity(count as usize * bytes);
    for i in 0..count {
        let texel = (components.iter().flatten())
            .fold(0u64, |t, c| t | u64::from(code_of(i, *c)) << c.offset);
        texels.extend_from_slice(&texel.to_le_bytes()[..bytes]);
    }
    let mut parameters = vec![width, count];
    for c in components {
        parameters.extend(match c {
            None => [NONE, 0, 0],
            Some(c) => {
                let rule = match c.recover {
                    Recover::Unorm => UNORM,
                    Recover::Snorm => SNORM,
                    Recover::Half => HALF,
                    Recover::Bits => NONE,
                };
                [rule, c.bits, c.offset]
            }
        });
    }
    debug_assert_eq!(parameters.len(), PARAMETER_WORDS);
    SampledFetchCase {
        format,
        extent: [width, count / width],
        texels,
        parameters,
        components,
    }
}

/// Every case of the check (the module docs): each format of the filter's fetch whose components
/// are converted, in the fetch's table's order.
pub fn sampled_fetch_cases() -> Vec<SampledFetchCase> {
    (COMPONENTS.iter())
        .filter(|(_, components)| (components.iter().flatten()).all(|c| c.recover != Recover::Bits))
        .map(|&(format, components)| case(format, components))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{mask, sampled_fetch_cases, sampled_fetch_typed_module, Recover};
    use crate::bind::MipmapFilterChain;
    use crate::spirv::testing::{assert_valid, committed_profiles};
    use crate::vk::Format;

    #[test]
    fn the_module_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(sampled_fetch_typed_module().expect("module"), &p);
        }
    }

    /// The cases are the formats whose fetch converts, each image holding every code of each
    /// component, and the identity read back is Metal's answer.
    #[test]
    fn the_cases_hold_every_code_of_each_converted_format() {
        let cases = sampled_fetch_cases();
        assert_eq!(cases.len(), 23);
        for case in &cases {
            assert!(
                MipmapFilterChain::fetch_converts(case.format),
                "{:?}",
                case.format
            );
            let texels = case.texel_count() as usize;
            assert_eq!(case.texels.len() % texels, 0, "{:?}", case.format);
            let bytes = case.texels.len() / texels;
            let mut words = Vec::new();
            for texel in case.texels.chunks_exact(bytes) {
                let mut t = [0u8; 8];
                t[..bytes].copy_from_slice(texel);
                let mut t = u64::from_le_bytes(t);
                // An snorm component's code -2^(b-1) reads as -(2^(b-1) - 1).
                for c in case.components.iter().flatten() {
                    let least = 1u64 << (c.bits - 1);
                    if c.recover == Recover::Snorm && (t >> c.offset) & mask(c.bits) == least {
                        t += 1 << c.offset;
                    }
                }
                words.extend([t as u32, (t >> 32) as u32]);
            }
            assert_eq!(case.difference(&words), None, "{:?}", case.format);
        }
        let r16 = cases
            .iter()
            .find(|c| c.format == Format::R16_SNORM)
            .expect("R16_SNORM");
        assert_eq!(r16.extent, [256, 256]);
        let mut words: Vec<u32> = (0..65536u32).flat_map(|i| [i, 0]).collect();
        assert!(r16.difference(&words).is_some(), "0x8000 reads as 0x8001");
        words[2 * 0x8000] = 0x8001;
        assert_eq!(r16.difference(&words), None);
        words[2 * 7] = 6;
        let difference = r16.difference(&words).expect("a difference");
        let first = format!(
            "{:?}: 1 of 65536 texels differ, the first texel 7's component 0 read code 0x6 for 0x7",
            r16.format
        );
        assert!(difference.starts_with(&first), "{difference}");
    }
}
