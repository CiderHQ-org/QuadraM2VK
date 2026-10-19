//! The shaders of the device's own checks of its driver's answers (`device::checks`), which the
//! runtime runs on the device before it takes a fact a check decides.
//!
//! A compute check is a [`CheckKernel`]: a module, the words its storage buffers start with, the
//! workgroups it dispatches, and the words Metal's answer leaves in one of them. The runtime runs
//! any such kernel alike and compares the words.
//!
//! **Bounds.** A check runs on devices without robust buffer access too (an adopted device may
//! enable none), where an access past a buffer is undefined, so every check module stays inside
//! its buffers for every invocation its draw or dispatch launches. A fragment stage's helper
//! invocations shade quads (llvmpipe 4x4 blocks) past the target's last row and column: its reads
//! of a buffer take a constant index or one below the buffer's own length (`OpArrayLength`), and
//! its image reads a constant coordinate. A kernel's every access lies inside the buffers it
//! describes over its dispatch of whole workgroups. A crate test reads each module these builders
//! and `raster_checks`'s finish and holds them to this (`tests::device_check_bounds`, which a new
//! check's builder joins).
//!
//! **The half quantize** ([`half_quantize_check`]): one `GLCompute` entry `main` with local size
//! (64, 1, 1) and in set 0 two storage buffers of 4-byte words, the inputs at binding 0 and the
//! results at binding 1. Invocation `x` reads input word `x` as an `f32`'s bits, rounds it to
//! `half` exactly as translation does where the profile takes `DriverDeviation::MetalHalfQuantize`
//! (the device's `OpQuantizeToF16`, a zero it answers taking the denormal grid's rounding,
//! `lower::arith::narrow`'s `QuantizeForm::Device`), and stores the rounded `f32`'s bits at result
//! word `x`. Its inputs are every rounding edge of every exponent the conversion meets (ties to
//! even and odd, the denormal grid, the overflow bound at 65520) and a fixed spread of other
//! values, no NaN (Vulkan leaves a NaN's bits to the device, and the deviation is the rounding's);
//! the expected words are Metal's conversion of each.
//!
//! **The index fetch** ([`index_fetch_check_module`]): one `Vertex` entry `main` and in set 0 one
//! storage buffer of [`IndexFetchCheck::FLAGS`] 4-byte words at
//! [`IndexFetchCheck::FLAGS_BINDING`]. Each invocation stores 1 at the word its `VertexIndex`
//! names, where it names one of them, and writes nothing else: a draw's flags are the vertex
//! indices it shaded. It writes no position, so a pipeline of it draws with the rasterizer off.
//!
//! The submodules hold the checks of translation's defects, each a [`CheckKernel`] of the
//! instructions translation writes where the profile does not take the workaround:
//!
//! - `converter`: the shader converter's (`CompositeInsertsWrittenInPlace`,
//!   [`composite_inserts_check`]; `LoadsForwardedIntoLoops`, [`loads_forwarded_check`];
//!   `LocalizedLoadsForwardedPastAtomicsAndCalls`, [`localized_loads_check`]).
//! - `compiler`: the shader compiler's (`NarrowMaskAfterSignedClamp`, [`narrow_mask_check`];
//!   `HalfConversionPairFolded`, [`half_round_trip_check`]).
//!
//! **The color writes** (`color`): six color-write deviations Metal's rule decides
//! (`MetalColorWrites` and its siblings, [`COLOR_WRITE_FACTS`]), each a set of draws
//! ([`ColorDraw`]) of a fragment that writes its input as computed ([`color_write_modules`]),
//! against Metal's texels.
//!
//! **The storage writes** (`storage`): five storage-write deviations Metal's rule decides
//! (`MetalHalfStorageWrites` and its siblings, [`STORAGE_WRITE_FACTS`]), each a set of writes
//! ([`StorageWrite`]) of a kernel that writes its input as computed into a storage image through
//! each declaration translation gives the format ([`storage_write_modules`]), against Metal's
//! texels.
//!
//! **The arithmetic** (`arith`): two arithmetic deviations Metal's rule decides
//! (`MetalCanonicalNans`, [`canonical_nans_check`]; `MetalFloatDenormals`,
//! [`float_denormals_check`]), each a [`CheckKernel`] of the device's own instructions over rows of
//! operands, one per math mode the device has ([`MathMode`]).
//!
//! **The block decodes** (`block_decode`): a texel fetch ([`texel_fetch_check_module`]) of images
//! of fixed blocks of each BC format against Metal's reads of them ([`block_decode_cases`]).
//!
//! **The sampled fetch** (`sampled_fetch`): every code of each format the mipmap filter reads in
//! place, fetched and recovered as the filter recovers it ([`sampled_fetch_check_module`],
//! [`sampled_fetch_cases`]).
//!
//! **The color outputs** (`occlusion`): a fragment of 1 to 4 color outputs
//! ([`color_outputs_check_fragment`]), drawn under an occlusion query.
//!
//! **The depth writes** (`fragment_depth`): a per-sample fragment that writes chosen words as its
//! `FragDepth` ([`depth_writes_check_fragment`]), the write of the runtime's depth copy draws.
//!
//! **The descriptors** (`descriptors`): the shaders of the checks of the runtime's descriptor
//! defects ([`sampler_clamp_check_module`], [`array_length_check_module`],
//! [`pushed_uniform_check_module`]).
//!
//! **The sweeps** (`sweep`): three deviations measured on every input of a 32-bit domain
//! (`MetalHalfConversion`, `MetalMinMax` and `MetalHalfArithmetic`; [`Sweep`]), each a module
//! that compares the device's form with the exact form on the GPU and counts the inputs where
//! they differ ([`SweepKernel`]), one per math mode.

pub(crate) mod arith;
pub(crate) mod block_decode;
pub(crate) mod color;
pub(crate) mod compiler;
pub(crate) mod converter;
pub(crate) mod descriptors;
pub(crate) mod fragment_depth;
pub(crate) mod kernel;
pub(crate) mod occlusion;
pub(crate) mod sampled_fetch;
pub(crate) mod storage;
pub(crate) mod sweep;

pub use arith::{canonical_nans_check, float_denormals_check};
pub use block_decode::{
    block_decode_cases, texel_fetch_check_module, BlockDecodeCase, TexelFetch, TexelFetchCheck,
};
pub use color::{
    color_write_draws, color_write_modules, ColorDraw, ColorOutput, ColorWriteModules,
    COLOR_WRITE_FACTS,
};
pub use compiler::{half_round_trip_check, half_round_trip_unsupported, narrow_mask_check};
pub use converter::{composite_inserts_check, loads_forwarded_check, localized_loads_check};
pub use descriptors::{
    array_length_check_module, pushed_uniform_check_module, sampler_clamp_check_module,
    DescriptorCheck,
};
pub use fragment_depth::depth_writes_check_fragment;
pub use kernel::MathMode;
pub use occlusion::color_outputs_check_fragment;
pub use sampled_fetch::{
    sampled_fetch_cases, sampled_fetch_check_module, SampledFetchCase, SampledFetchCheck,
};
pub use storage::{
    storage_write_modules, storage_writes, StorageWrite, StorageWriteModule, STORAGE_WRITE_FACTS,
};
pub use sweep::{Sweep, SweepKernel};

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::narrow;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder, QuantizeForm};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

/// A compute check: one `GLCompute` entry `main` at local size `(local, 1, 1)` (the kernel's own,
/// [`CheckKernel::LOCAL_SIZE`] or 1), storage buffer `i` of [`buffers`](Self::buffers) at set 0,
/// binding `i`, no push constants, dispatched as [`workgroups`](Self::workgroups) workgroups along
/// x.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckKernel {
    /// SPIR-V 1.5 words, validated against the device's profile.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within its profile).
    pub requirements: Requirements,
    /// Workgroups along x.
    pub workgroups: u32,
    /// Each storage buffer's words before the dispatch, binding by binding.
    pub buffers: Vec<Vec<u32>>,
    /// The binding read back after the dispatch.
    pub read: u32,
    /// The words Metal's answer leaves in it.
    pub expected: Vec<u32>,
}

impl CheckKernel {
    /// The local size along x of a kernel of more than one invocation a workgroup.
    pub const LOCAL_SIZE: u32 = 64;

    /// The kernel of the module `spirv` needing `requirements`, dispatching `workgroups`, over
    /// `buffers`, whose binding `read` Metal leaves as `expected`.
    fn new(
        (spirv, requirements): (Arc<[u32]>, Requirements),
        workgroups: u32,
        buffers: Vec<Vec<u32>>,
        read: u32,
        expected: Vec<u32>,
    ) -> CheckKernel {
        CheckKernel {
            spirv,
            requirements,
            workgroups,
            buffers,
            read,
            expected,
        }
    }

    /// Whether `words`, read back from binding [`read`](Self::read), are Metal's answer.
    pub fn answers_metal(&self, words: &[u32]) -> bool {
        words == self.expected.as_slice()
    }

    /// How `words`, read back from binding [`read`](Self::read), differ from Metal's answer, in a
    /// few words for the other answer's statement (`CheckAnswer::NotMetals`): how many words
    /// differ, and the first, its index, the word read and Metal's. `None` for Metal's answer.
    pub fn difference(&self, words: &[u32]) -> Option<String> {
        if self.answers_metal(words) {
            return None;
        }
        if words.len() != self.expected.len() {
            return Some(format!(
                "{} words read where Metal's answer has {}",
                words.len(),
                self.expected.len()
            ));
        }
        let differing: Vec<usize> = (0..words.len())
            .filter(|&i| words[i] != self.expected[i])
            .collect();
        let first = differing[0];
        Some(format!(
            "{} of {} words differ, the first word {first} read {:#010x}, Metal's {:#010x}",
            differing.len(),
            words.len(),
            words[first],
            self.expected[first]
        ))
    }
}

/// How `texels`, an image of `width` texels a row of `texel_bytes` bytes each read back row by
/// row, tightly packed, differ from Metal's `expected`, in a few words for the other answer's
/// statement (`CheckAnswer::NotMetals`): how many texels differ, and the first, its position, the
/// bytes read and Metal's. `None` for Metal's texels.
fn texel_difference(
    texels: &[u8],
    expected: &[u8],
    texel_bytes: u32,
    width: u32,
) -> Option<String> {
    if texels == expected {
        return None;
    }
    if texels.len() != expected.len() {
        return Some(format!(
            "{} bytes read where Metal's texels have {}",
            texels.len(),
            expected.len()
        ));
    }
    let size = (texel_bytes as usize).max(1);
    let differing: Vec<usize> = (texels.chunks(size).zip(expected.chunks(size)))
        .enumerate()
        .filter(|(_, (read, metal))| read != metal)
        .map(|(at, _)| at)
        .collect();
    let first = differing[0];
    let width = (width as usize).max(1);
    let bytes = |of: &[u8]| of[first * size..(first * size + size).min(of.len())].to_vec();
    Some(format!(
        "{} of {} texels differ, the first at ({}, {}) read {:02x?}, Metal's {:02x?}",
        differing.len(),
        texels.len().div_ceil(size),
        first % width,
        first / width,
        bytes(texels),
        bytes(expected)
    ))
}

/// The words the half quantize check rounds.
const HALF_QUANTIZE_WORDS: usize = 1024;

/// The index fetch check's vertex module on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexFetchCheck {
    /// SPIR-V 1.5 words, entry point `main`, the binding numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: a vertex stage's storage writes
    /// (`vertexPipelineStoresAndAtomics`).
    pub requirements: Requirements,
}

impl IndexFetchCheck {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The flags, a storage buffer of [`FLAGS`](Self::FLAGS) words the draw writes.
    pub const FLAGS_BINDING: u32 = 0;
    /// The words of the flags: a `VertexIndex` from 0 below it marks its own.
    pub const FLAGS: u32 = 512;
}

const INPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
const OUTPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
const FLAGS: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The half quantize check as a typed module (the module docs).
fn half_quantize_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    b.set_quantize_form(QuantizeForm::Device);
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = words_buffer(&mut b, INPUT, u32_)?;
    let output = words_buffer(&mut b, OUTPUT, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let input = f.module().global_value(input)?;
    let output = f.module().global_value(output)?;
    let from = f.access_chain(input, &[zero, x])?;
    let word = f.load(from, MemAccess::default())?;
    let value = f.convert(ConvOp::Bitcast, word, f32_)?;
    let half = Form {
        held: Held::Float(FloatFmt::Half),
        lanes: 1,
    };
    let rounded = narrow::round(&mut f, half, value)?;
    let rounded = f.convert(ConvOp::Bitcast, rounded, u32_)?;
    let to = f.access_chain(output, &[zero, x])?;
    f.store(to, rounded, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: CheckKernel::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The index fetch check as a typed module (the module docs).
fn index_fetch_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let flags = words_buffer(&mut b, FLAGS, u32_)?;
    let index = b.builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, i32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let count = u32_value(&mut f, IndexFetchCheck::FLAGS)?;
    let index = f.module().global_value(index)?;
    let index = f.load(index, MemAccess::default())?;
    let index = f.convert(ConvOp::Bitcast, index, u32_)?;
    let flags = f.module().global_value(flags)?;
    let inside = f.binary(BinOp::ULessThan, index, count)?;
    let mark: ArmFn<'_, ()> = Box::new(move |f, _| {
        let to = f.access_chain(flags, &[zero, index])?;
        f.store(to, one, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, mark, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Vertex, id, Vec::new())?;
    b.finalize()
}

/// The fixed binding numbers of the modules' symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s) if *s == INPUT => Some(0),
        Symbol::Binding(s) if *s == OUTPUT => Some(1),
        Symbol::Binding(s) if *s == FLAGS => Some(IndexFetchCheck::FLAGS_BINDING),
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed bindings; `what` names it in the error where the device lacks what it needs.
fn finish(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    #[cfg(test)]
    kernel::record(&m);
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The half quantize check of `device` (the module docs): its module built with the typed
/// builder, checked by the owned verifier against the device's profile, serialized, and numbered
/// with its fixed bindings; its inputs; and Metal's conversion of each as the expected words. It
/// rounds through the device's quantize whatever the profile takes: the check is of that form.
pub fn half_quantize_check(device: &Device) -> Result<CheckKernel, Error> {
    let module = finish(device, half_quantize_typed_module()?, "half quantize check")?;
    let inputs = half_quantize_inputs();
    let expected = inputs.iter().map(|&bits| metal_half_of_f32(bits)).collect();
    let groups = HALF_QUANTIZE_WORDS as u32 / CheckKernel::LOCAL_SIZE;
    let buffers = vec![inputs, vec![0; HALF_QUANTIZE_WORDS]];
    Ok(CheckKernel::new(module, groups, buffers, 1, expected))
}

/// The half quantize check's inputs (the module docs): each edge and each exponent's rounding
/// edges of both signs, then a fixed spread of values of exponents near `half`'s range, no NaN,
/// [`HALF_QUANTIZE_WORDS`] in all.
fn half_quantize_inputs() -> Vec<u32> {
    let edges = [
        0x0000_0000u32,
        0x3880_0000,
        0x387F_FFFF,
        0x3380_0000,
        0x3380_0001,
        0x337F_FFFF,
        0x477F_E000,
        0x477F_EFFF,
        0x477F_F000,
        0x4780_0000,
        0x7F80_0000,
        0x3F80_0800,
        0x3F80_1800,
        0x3F80_07FF,
        0x3F80_0801,
    ];
    // Below the kept bits of a normal `half`: an exact tie to an even and to an odd neighbour,
    // either side of a tie, a tie that carries into the exponent, the largest significand, a
    // middle one and an exact value.
    let significands = [
        0x0000_1000u32,
        0x0000_3000,
        0x0000_0FFF,
        0x0000_1001,
        0x007F_F000,
        0x007F_FFFF,
        0x0040_0000,
        0x0000_2000,
    ];
    let mut words: Vec<u32> = edges.to_vec();
    for exponent in 0x66u32..=0x8F {
        words.extend(significands.iter().map(|s| exponent << 23 | s));
    }
    let signed: Vec<u32> = words.iter().map(|w| w | 0x8000_0000).collect();
    words.extend(signed);
    let mut state = 0x9E37_79B9u32;
    while words.len() < HALF_QUANTIZE_WORDS {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let exponent = 0x60 + (state >> 23 & 0x3F);
        words.push(state & 0x807F_FFFF | exponent << 23);
    }
    words
}

/// The index fetch check's vertex module of `device` (the module docs), built and checked as
/// [`half_quantize_check`]'s is. An internal error where the device lacks a vertex stage's
/// storage writes (`vertexPipelineStoresAndAtomics`).
pub fn index_fetch_check_module(device: &Device) -> Result<IndexFetchCheck, Error> {
    let (spirv, requirements) = finish(device, index_fetch_typed_module()?, "index fetch check")?;
    Ok(IndexFetchCheck {
        spirv,
        requirements,
    })
}

/// The `f32` bits `bits` rounded to `half` as Metal's conversion of a `float` to `half` rounds
/// it, widened back: to nearest even on the half grid, half denormals kept on the grid of
/// `2^-24`, the sign of a zero kept, infinity of the sign from 65520 up, and a NaN to `0x7E00`'s
/// widening, `0x7FC00000` (the M3 Max, every `f32`: `DriverDeviation::MetalHalfConversion`'s
/// measurements). The half quantize check's expected answer.
fn metal_half_of_f32(bits: u32) -> u32 {
    let magnitude = bits & 0x7FFF_FFFF;
    let sign = bits & 0x8000_0000;
    if magnitude > 0x7F80_0000 {
        return 0x7FC0_0000;
    }
    if magnitude > 0x477F_EFFF {
        return sign | 0x7F80_0000;
    }
    if magnitude < 0x3880_0000 {
        let x = f32::from_bits(bits);
        let k = (x * 16_777_216.0).round_ties_even();
        return (k * (1.0 / 16_777_216.0)).to_bits();
    }
    let lsb = (bits >> 13) & 1;
    (bits + 0xFFF + lsb) & 0xFFFF_E000
}

#[cfg(test)]
mod tests {
    use super::{
        half_quantize_inputs, half_quantize_typed_module, index_fetch_typed_module,
        metal_half_of_f32, texel_difference, CheckKernel, HALF_QUANTIZE_WORDS,
    };
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_check_modules_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(half_quantize_typed_module().expect("module"), &p);
        }
        for (stem, p) in committed_profiles().expect("profiles") {
            let stores = p
                .enabled()
                .features
                .contains(&crate::device::VkFeature::VertexPipelineStoresAndAtomics);
            if stores {
                assert_valid(index_fetch_typed_module().expect("module"), &p);
            } else {
                eprintln!("{stem}: no vertex stores, the index fetch check does not run");
            }
        }
    }

    /// A kernel's difference from Metal's answer names how many words differ and the first;
    /// Metal's answer has none.
    #[test]
    fn a_kernel_s_difference_names_the_first_differing_word() {
        let module = (Vec::new().into(), Default::default());
        let kernel = CheckKernel::new(module, 1, Vec::new(), 0, vec![1, 2, 3]);
        assert_eq!(kernel.difference(&[1, 2, 3]), None);
        assert_eq!(
            kernel.difference(&[1, 5, 7]).as_deref(),
            Some("2 of 3 words differ, the first word 1 read 0x00000005, Metal's 0x00000002")
        );
        assert_eq!(
            kernel.difference(&[1, 2]).as_deref(),
            Some("2 words read where Metal's answer has 3")
        );
    }

    /// An image's difference from Metal's texels names how many texels differ and the first, by
    /// its position in the rows; Metal's texels have none.
    #[test]
    fn an_image_s_difference_names_the_first_differing_texel() {
        let expected = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        assert_eq!(texel_difference(&expected, &expected, 2, 3), None);
        let mut read = expected;
        read[8] = 0xff;
        read[11] = 0xee;
        assert_eq!(
            texel_difference(&read, &expected, 2, 3).as_deref(),
            Some("2 of 6 texels differ, the first at (1, 1) read [ff, 09], Metal's [08, 09]")
        );
        assert_eq!(
            texel_difference(&read[..4], &expected, 2, 3).as_deref(),
            Some("4 bytes read where Metal's texels have 12")
        );
    }

    /// The expected answer is the IEEE `half` of every input, which holds no NaN, and fills
    /// whole workgroups; a NaN answers as Metal's `0x7E00`.
    #[test]
    fn the_expected_half_is_metal_s() {
        use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
        let inputs = half_quantize_inputs();
        assert_eq!(inputs.len(), HALF_QUANTIZE_WORDS);
        assert_eq!(inputs.len() % CheckKernel::LOCAL_SIZE as usize, 0);
        for bits in inputs {
            let x = f32::from_bits(bits);
            assert!(!x.is_nan(), "{bits:#010x}");
            let want = f16_to_f64(f64_to_f16(f64::from(x))) as f32;
            assert_eq!(metal_half_of_f32(bits), want.to_bits(), "{bits:#010x}");
        }
        assert_eq!(metal_half_of_f32(0x7FC0_0001), 0x7FC0_0000);
        assert_eq!(metal_half_of_f32(0xFF80_0001), 0x7FC0_0000);
    }
}
