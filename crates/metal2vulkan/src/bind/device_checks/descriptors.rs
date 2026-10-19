//! The shaders of the checks of the runtime's descriptor defects, each a module the runtime binds
//! as its draws and dispatches bind theirs (pushed where the device pushes set 0):
//!
//! - **The clamped sample** ([`sampler_clamp_check_module`]): one `GLCompute` entry `main` at
//!   local size ([`CheckKernel::LOCAL_SIZE`](super::CheckKernel::LOCAL_SIZE), 1, 1), and in set 0
//!   a combined image sampler of a 2D `float` image at [`DescriptorCheck::IMAGE_BINDING`], a
//!   storage buffer of coordinates at [`DescriptorCheck::INPUT_BINDING`] and one of results at
//!   [`DescriptorCheck::OUTPUT_BINDING`]. Invocation `i` below
//!   [`DescriptorCheck::SAMPLES`] samples the image at `(u, 0.5)` at level 0, `u` the `float`
//!   of coordinate word `i`, and stores the red component's bits at result word `i`
//!   (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`).
//! - **The array length** ([`array_length_check_module`]): one `GLCompute` entry `main` at local
//!   size (1, 1, 1), a storage buffer of 4-byte words at [`DescriptorCheck::INPUT_BINDING`] and
//!   one of results at [`DescriptorCheck::OUTPUT_BINDING`]: its one invocation stores the first
//!   buffer's runtime array's length (`OpArrayLength`) at result word 0
//!   (`RuntimeDriverDefect::PushedBufferSizesUnset`).
//! - **The pushed uniform** ([`pushed_uniform_check_module`]): one `Vertex` entry `main` per draw
//!   `d` of [`DescriptorCheck::UNIFORM_READS`], a uniform buffer of
//!   [`DescriptorCheck::UNIFORM_WORDS`] 4-byte words (`uvec4`s at a 16-byte stride) at
//!   [`DescriptorCheck::INPUT_BINDING`] and a storage buffer of results at
//!   [`DescriptorCheck::OUTPUT_BINDING`]: it stores uniform word `UNIFORM_READS[d]` at result
//!   word `d`. Draws 0 and 2 read their word at a constant offset, the reads a driver may load
//!   ahead of the draw as constants; draw 1 at an offset of its `VertexIndex` (0 in a draw from
//!   vertex 0), which it reads through the descriptor, bounded by the array's length (0 past
//!   it). It writes no position, so a pipeline of it draws with the rasterizer off
//!   (`RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`).

use std::sync::Arc;

use super::kernel::{finish, LOCAL_SIZE};
use super::{u32_value, words_buffer};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{CombinedSampler, ResourceKind, ResourceSlot, SamplerRef, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageOperands,
    ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

/// One of the descriptor checks' modules on one device (the module docs), validated like any
/// output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorCheck {
    /// SPIR-V 1.5 words, entry point `main`, the bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device.
    pub requirements: Requirements,
}

impl DescriptorCheck {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The clamped sample's combined image sampler.
    pub const IMAGE_BINDING: u32 = 0;
    /// What the module reads: the coordinates, the sized buffer, or the uniform buffer.
    pub const INPUT_BINDING: u32 = 1;
    /// What it writes: a storage buffer of 4-byte words.
    pub const OUTPUT_BINDING: u32 = 2;
    /// The clamped sample's samples.
    pub const SAMPLES: u32 = 8;
    /// The pushed uniform's uniform buffer's words.
    pub const UNIFORM_WORDS: u32 = 8;
    /// The uniform word each draw of the pushed uniform stores, by draw: the second's at an offset
    /// of its `VertexIndex`, the others' at constant offsets (the module docs).
    pub const UNIFORM_READS: [u32; 3] = [0, 4, 1];
}

/// A resource of the descriptor checks of `kind` at `binding`, of `stage`.
const fn slot(stage: Stage, kind: ResourceKind, binding: u32) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind,
        index: binding,
    }
}

/// The combined image sampler: the texture of [`DescriptorCheck::IMAGE_BINDING`] through the
/// sampler of the same slot, as the runtime's set binds them (one combined binding).
const SAMPLED: ResourceSlot = slot(
    Stage::Kernel,
    ResourceKind::CombinedTextureSampler(CombinedSampler {
        sampler: SamplerRef::Slot(DescriptorCheck::IMAGE_BINDING),
        compare: false,
        integer: false,
        cube_gradient: false,
        raw: false,
    }),
    DescriptorCheck::IMAGE_BINDING,
);
const INPUT: ResourceSlot = slot(
    Stage::Kernel,
    ResourceKind::Buffer,
    DescriptorCheck::INPUT_BINDING,
);
const OUTPUT: ResourceSlot = slot(
    Stage::Kernel,
    ResourceKind::Buffer,
    DescriptorCheck::OUTPUT_BINDING,
);
const UNIFORM: ResourceSlot = slot(
    Stage::Vertex,
    ResourceKind::Buffer,
    DescriptorCheck::INPUT_BINDING,
);
const RESULTS: ResourceSlot = slot(
    Stage::Vertex,
    ResourceKind::Buffer,
    DescriptorCheck::OUTPUT_BINDING,
);

/// The scalar constant of `ty` of `bits` as a value.
fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// The clamped sample as a typed module (the module docs).
fn sampler_clamp_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let combined = b.ty(Type::SampledImage { image })?;
    let sampled = b.resource(SAMPLED, StorageClass::UniformConstant, combined)?;
    let coordinates = words_buffer(&mut b, INPUT, u32_)?;
    let results = words_buffer(&mut b, OUTPUT, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let count = u32_value(&mut f, DescriptorCheck::SAMPLES)?;
    let half = scalar(&mut f, f32_, u64::from(0.5f32.to_bits()))?;
    let lod = scalar(&mut f, f32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let sampled = f.module().global_value(sampled)?;
    let coordinates = f.module().global_value(coordinates)?;
    let results = f.module().global_value(results)?;
    let inside = f.binary(BinOp::ULessThan, x, count)?;
    let store: ArmFn<'_, ()> = Box::new(move |f, _| {
        let from = f.access_chain(coordinates, &[zero, x])?;
        let u = f.load(from, MemAccess::default())?;
        let u = f.convert(ConvOp::Bitcast, u, f32_)?;
        let coord = f.composite_construct(vec2, &[u, half])?;
        let sampled = f.load(sampled, MemAccess::default())?;
        let ops = ImageOperands {
            lod: Some(lod),
            ..ImageOperands::default()
        };
        let texel = f.sample(sampled, coord, None, false, &ops)?;
        let red = f.composite_extract(texel, &[0])?;
        let red = f.convert(ConvOp::Bitcast, red, u32_)?;
        let to = f.access_chain(results, &[zero, x])?;
        f.store(to, red, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, store, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    let size = ExecMode::LocalSize {
        x: LOCAL_SIZE,
        y: 1,
        z: 1,
    };
    b.entry(ExecModel::GLCompute, id, vec![size])?;
    b.finalize()
}

/// The array length as a typed module (the module docs).
fn array_length_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let sized = words_buffer(&mut b, INPUT, u32_)?;
    let results = words_buffer(&mut b, OUTPUT, u32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let sized = f.module().global_value(sized)?;
    let results = f.module().global_value(results)?;
    let length = f.array_length(sized, 0)?;
    let to = f.access_chain(results, &[zero, zero])?;
    f.store(to, length, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    let size = ExecMode::LocalSize { x: 1, y: 1, z: 1 };
    b.entry(ExecModel::GLCompute, id, vec![size])?;
    b.finalize()
}

/// The pushed uniform of draw `draw` as a typed module (the module docs).
fn pushed_uniform_typed_module(draw: u32) -> BResult<TypedModule> {
    let word = DescriptorCheck::UNIFORM_READS[draw as usize];
    let dynamic = draw == 1;
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let vectors = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(DescriptorCheck::UNIFORM_WORDS / 4)),
    })?;
    let array = b.ty(Type::Array {
        elem: uvec4,
        len: vectors,
        stride: Some(16),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std140, true, &[array])?;
    let block = b.ty(Type::Struct {
        members: [array].into(),
        layout: Some(layout),
    })?;
    let uniform = b.resource(UNIFORM, StorageClass::Uniform, block)?;
    let results = words_buffer(&mut b, RESULTS, u32_)?;
    let index = b.builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, i32_)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let vector = u32_value(&mut f, word / 4)?;
    let to = u32_value(&mut f, draw)?;
    let uniform = f.module().global_value(uniform)?;
    let results = f.module().global_value(results)?;
    let vector = match dynamic {
        false => vector,
        true => {
            let index = f.module().global_value(index)?;
            let index = f.load(index, MemAccess::default())?;
            let index = f.convert(ConvOp::Bitcast, index, u32_)?;
            f.binary(BinOp::IAdd, index, vector)?
        }
    };
    // The read is bounded by the array's length, else 0 is stored: a device without robustness
    // reads nothing past the buffer whatever `VertexIndex` holds.
    let vectors = u32_value(&mut f, DescriptorCheck::UNIFORM_WORDS / 4)?;
    let inside = f.binary(BinOp::ULessThan, vector, vectors)?;
    let to = f.access_chain(results, &[zero, to])?;
    let read: ArmFn<'_, ()> = Box::new(move |f, _| {
        let from = f.access_chain(uniform, &[zero, vector])?;
        let value = f.load(from, MemAccess::default())?;
        let value = f.composite_extract(value, &[word % 4])?;
        f.store(to, value, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let past: ArmFn<'_, ()> = Box::new(move |f, _| {
        f.store(to, zero, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, read, Some(past))?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Vertex, id, Vec::new())?;
    b.finalize()
}

/// `m` of `device`, built, checked against its profile and numbered as the other checks' modules
/// are.
fn checked(device: &Device, m: BResult<TypedModule>, what: &str) -> Result<DescriptorCheck, Error> {
    let (spirv, requirements) = finish(device, m?, what)?;
    Ok(DescriptorCheck {
        spirv,
        requirements,
    })
}

/// The clamped sample's module on `device` (the module docs).
pub fn sampler_clamp_check_module(device: &Device) -> Result<DescriptorCheck, Error> {
    checked(device, sampler_clamp_typed_module(), "sampler clamp check")
}

/// The array length's module on `device` (the module docs).
pub fn array_length_check_module(device: &Device) -> Result<DescriptorCheck, Error> {
    checked(device, array_length_typed_module(), "array length check")
}

/// The pushed uniform's module of draw `draw` (an index of [`DescriptorCheck::UNIFORM_READS`])
/// on `device` (the module docs). An internal error where the device lacks a vertex stage's
/// storage writes (`vertexPipelineStoresAndAtomics`).
pub fn pushed_uniform_check_module(device: &Device, draw: u32) -> Result<DescriptorCheck, Error> {
    if DescriptorCheck::UNIFORM_READS.get(draw as usize).is_none() {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Build,
            format!("the pushed uniform check has no draw {draw}"),
        )));
    }
    checked(
        device,
        pushed_uniform_typed_module(draw),
        "pushed uniform check",
    )
}

#[cfg(test)]
mod tests {
    use super::{
        array_length_typed_module, pushed_uniform_typed_module, sampler_clamp_typed_module,
    };
    use crate::device::VkFeature;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_descriptor_check_modules_are_valid_on_every_committed_profile() {
        for (stem, p) in committed_profiles().expect("profiles") {
            assert_valid(sampler_clamp_typed_module().expect("module"), &p);
            assert_valid(array_length_typed_module().expect("module"), &p);
            let stores = &p.enabled().features;
            if stores.contains(&VkFeature::VertexPipelineStoresAndAtomics) {
                for draw in 0..3 {
                    assert_valid(pushed_uniform_typed_module(draw).expect("module"), &p);
                }
            } else {
                eprintln!("{stem}: no vertex stores, the pushed uniform check does not run");
            }
        }
    }
}
