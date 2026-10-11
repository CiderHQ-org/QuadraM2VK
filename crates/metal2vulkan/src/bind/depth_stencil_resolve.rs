//! The draws of a multisampled depth/stencil resolve the device's resolve modes do not state.
//!
//! Metal resolves a depth aspect by `Sample0`, `Min` or `Max`, and a stencil aspect by `Sample0`
//! or `DepthResolvedSample`: the stencil of the sample whose depth the depth filter took. Vulkan's
//! resolve modes state `DepthResolvedSample` after `Sample0` (`SAMPLE_ZERO`) but not after `Min`
//! or `Max`, and a device may lack `MIN` and `MAX` for depth. These crate-owned modules resolve
//! such an aspect by a draw over the single-sampled resolve target, after the pass, reading the
//! attachment's samples through sampled views:
//!
//! - **The pick** ([`DepthPick`]): each pixel's samples' depths, read in sample order from the
//!   depth aspect's view at [`DepthStencilResolveModules::DEPTH_BINDING`], and the last sample
//!   whose depth is the smallest ([`DepthPick::Min`]) or the first whose depth is the largest
//!   ([`DepthPick::Max`]) under ordered comparison: of equal depths, the highest sample after
//!   `Min` and the lowest after `Max`, as Metal's `DepthResolvedSample` takes them (a probe on an
//!   Apple M3 Max and an M4: every pattern of two and four samples from four depths, both chips
//!   alike).
//! - **The write** ([`ResolveWrite`]): the picked depth as `FragDepth` ([`ResolveWrite::Depth`],
//!   under depth compare `ALWAYS` with depth writes on: exact only where the device stores the
//!   float `FragDepth` holds unchanged, which the consumer establishes per device); or the picked
//!   sample's stencil, read from the stencil aspect's view at
//!   [`DepthStencilResolveModules::STENCIL_BINDING`], as `FragStencilRefEXT`
//!   ([`ResolveWrite::StencilExport`], one draw under compare `ALWAYS`, pass op `REPLACE` and write
//!   mask 0xFF, where the device exports stencil) or as one bit plane
//!   ([`ResolveWrite::StencilPlane`]: the invocation is killed where the stencil's bit is clear;
//!   eight draws over a stencil cleared to 0, each under write mask [`StencilBit::mask`],
//!   reference 0xFF, compare `ALWAYS` and pass op `REPLACE`, exact without stencil export).
//!
//! The vertex stage draws one triangle over the whole viewport from `VertexIndex` 0 to 2 (the
//! sample planes merge's). The fragment stage runs once per pixel of the single-sampled target,
//! whose `FragCoord` is the pixel's center, truncated to the pixel. Both views are multisampled
//! 2D images of one sample count, the resolve's ([`DepthStencilResolveDraw::samples`]): float for
//! the depth aspect, unsigned for the stencil aspect.
//!
//! **Synchronization** (consumer): the attachment's writes before the draw's reads, and the
//! target's earlier accesses before its attachment writes; the draw's writes before whatever reads
//! the target next.

use std::sync::Arc;

use super::depth_stencil_copy::{finish, multisampled_image, u32_ty, uint, StencilBit};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, KillForm};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConvOp, ExecMode, ExecModel, ImageOperands, ImageUsage, IoDir, MemAccess, StorageClass,
    Type, TypedModule,
};

/// Which sample of each pixel a resolve draw takes (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DepthPick {
    /// The last sample of the smallest depth: Metal's `Min` depth filter.
    Min,
    /// The first sample of the largest depth: Metal's `Max` depth filter.
    Max,
}

/// What a resolve draw writes from the sample it takes (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResolveWrite {
    /// The sample's depth, as `FragDepth`.
    Depth,
    /// The sample's stencil, as `FragStencilRefEXT` (`VK_EXT_shader_stencil_export`).
    StencilExport,
    /// One bit plane of the sample's stencil: the invocation is killed where its bit is clear.
    StencilPlane(StencilBit),
}

impl ResolveWrite {
    /// Whether the draw reads the stencil aspect.
    pub fn reads_stencil(self) -> bool {
        !matches!(self, ResolveWrite::Depth)
    }
}

/// One resolve draw: the pick, the samples of each pixel, and the write.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthStencilResolveDraw {
    pub pick: DepthPick,
    /// The attachment's samples: a power of two from 2 to 64.
    pub samples: u32,
    pub write: ResolveWrite,
}

impl DepthStencilResolveDraw {
    /// The draw that takes `pick` of each pixel's `samples` samples and writes `write`.
    pub fn new(pick: DepthPick, samples: u32, write: ResolveWrite) -> DepthStencilResolveDraw {
        DepthStencilResolveDraw {
            pick,
            samples,
            write,
        }
    }
}

/// The crate-owned modules of one resolve draw on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthStencilResolveModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the depth aspect at
    /// [`DepthStencilResolveModules::DEPTH_BINDING`] and, where it writes stencil, the stencil
    /// aspect at [`DepthStencilResolveModules::STENCIL_BINDING`], of set
    /// [`DepthStencilResolveModules::SET`].
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device: for the export, the extension.
    pub requirements: Requirements,
}

impl DepthStencilResolveModules {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The depth aspect, a sampled image.
    pub const DEPTH_BINDING: u32 = 0;
    /// The stencil aspect, a sampled image.
    pub const STENCIL_BINDING: u32 = 1;
}

/// The binding symbol of the depth aspect ([`DepthStencilResolveModules::DEPTH_BINDING`]).
pub(super) const DEPTH_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 16,
};
/// The binding symbol of the stencil aspect ([`DepthStencilResolveModules::STENCIL_BINDING`]).
pub(super) const STENCIL_SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 17,
};

/// Whether a resolve draw takes `samples` samples a pixel: a power of two from 2 to 64.
fn valid_samples(samples: u32) -> bool {
    samples.is_power_of_two() && (2..=64).contains(&samples)
}

/// The builder's refusal of a draw it cannot state (no samples, or no stencil to write).
fn malformed() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ImageFetch,
        rule: OperandRule::Arity,
    }
}

/// Of `depths` (one per sample, in sample order), the depth `pick` takes and its sample: the
/// last of the smallest or the first of the largest under ordered comparison (the module docs).
pub(super) fn pick_sample(
    f: &mut FunctionBuilder<'_>,
    pick: DepthPick,
    depths: &[Value],
) -> BResult<(Value, Value)> {
    let (&first, rest) = depths.split_first().ok_or_else(malformed)?;
    let mut best = first;
    let mut index = uint(f, 0)?;
    for (s, &depth) in (1u32..).zip(rest) {
        let op = match pick {
            DepthPick::Min => BinOp::FOrdLessThanEqual,
            DepthPick::Max => BinOp::FOrdGreaterThan,
        };
        let better = f.binary(op, depth, best)?;
        best = f.select(better, depth, best)?;
        let s = uint(f, s)?;
        index = f.select(better, s, index)?;
    }
    Ok((best, index))
}

/// The fragment stage of `draw` as a typed module (the module docs). Its bindings are the symbols
/// `DEPTH_SOURCE` and, where it reads the stencil, `STENCIL_SOURCE`.
pub fn fragment_typed_module(draw: DepthStencilResolveDraw) -> BResult<TypedModule> {
    if !valid_samples(draw.samples) {
        return Err(malformed());
    }
    let mut b = ModuleBuilder::new();
    let u32_ = u32_ty(&mut b)?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let unknown = crate::spv::ImageFormat::Unknown;
    let depth_ty = multisampled_image(&mut b, f32_, ImageUsage::Sampled, unknown)?;
    let depth_image = b.resource(DEPTH_SOURCE, StorageClass::UniformConstant, depth_ty)?;
    let stencil_image = if draw.write.reads_stencil() {
        let ty = multisampled_image(&mut b, u32_, ImageUsage::Sampled, unknown)?;
        Some(b.resource(STENCIL_SOURCE, StorageClass::UniformConstant, ty)?)
    } else {
        None
    };
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let pixel = f.composite_construct(uvec2, &[x, y])?;
    let image = f.module().global_value(depth_image)?;
    let image = f.load(image, MemAccess::default())?;
    let mut depths = Vec::with_capacity(draw.samples as usize);
    for s in 0..draw.samples {
        let ops = ImageOperands {
            sample: Some(uint(&mut f, s)?),
            ..ImageOperands::default()
        };
        let texel = f.fetch(image, pixel, &ops)?;
        depths.push(f.composite_extract(texel, &[0])?);
    }
    let (depth, index) = pick_sample(&mut f, draw.pick, &depths)?;
    let stencil = match stencil_image {
        Some(stencil_image) => {
            let image = f.module().global_value(stencil_image)?;
            let image = f.load(image, MemAccess::default())?;
            let ops = ImageOperands {
                sample: Some(index),
                ..ImageOperands::default()
            };
            let texel = f.fetch(image, pixel, &ops)?;
            Some(f.composite_extract(texel, &[0])?)
        }
        None => None,
    };
    match (draw.write, stencil) {
        (ResolveWrite::Depth, _) => {
            let output = f
                .module()
                .builtin(crate::spv::BuiltIn::FragDepth, IoDir::Output, f32_)?;
            let output = f.module().global_value(output)?;
            f.store(output, depth, MemAccess::default())?;
        }
        (ResolveWrite::StencilExport, Some(stencil)) => {
            let reference = f.convert(ConvOp::Bitcast, stencil, i32_)?;
            let output =
                f.module()
                    .builtin(crate::spv::BuiltIn::FragStencilRefEXT, IoDir::Output, i32_)?;
            let output = f.module().global_value(output)?;
            f.store(output, reference, MemAccess::default())?;
        }
        (ResolveWrite::StencilPlane(bit), Some(stencil)) => {
            let mask = uint(&mut f, bit.mask())?;
            let set = f.binary(BinOp::BitwiseAnd, stencil, mask)?;
            let zero = uint(&mut f, 0)?;
            let clear = f.binary(BinOp::IEqual, set, zero)?;
            // A pixel whose picked sample's bit is clear writes nothing.
            let kill: ArmFn<'_, ()> = Box::new(|f, _| {
                f.kill(KillForm::Kill)?;
                Ok(ArmEnd::Exits)
            });
            f.if_(&mut (), clear, kill, None)?;
        }
        (_, None) => return Err(malformed()),
    }
    f.ret(None)?;
    let id = f.finish()?;
    let mut modes = vec![ExecMode::OriginUpperLeft];
    match draw.write {
        ResolveWrite::Depth => modes.push(ExecMode::DepthReplacing),
        ResolveWrite::StencilExport => modes.push(ExecMode::StencilRefReplacing),
        ResolveWrite::StencilPlane(_) => {}
    }
    b.entry(ExecModel::Fragment, id, modes)?;
    b.finalize()
}

/// The fixed binding numbers of the modules' symbols.
pub(super) fn binding_of(slot: &ResourceSlot) -> Option<u32> {
    match *slot {
        s if s == DEPTH_SOURCE => Some(DepthStencilResolveModules::DEPTH_BINDING),
        s if s == STENCIL_SOURCE => Some(DepthStencilResolveModules::STENCIL_BINDING),
        _ => None,
    }
}

/// The modules of the resolve draw `draw` on `device`: built with the typed builder (the vertex
/// stage is the sample planes merge's, `bind::sample_planes`), checked by the owned verifier
/// against the device's profile (requirements included), serialized, and numbered with the fixed
/// bindings of [`DepthStencilResolveModules`]. Refused, `Unsupported` naming `feature` (what the
/// consumer resolves for) and what the device lacks, where it lacks what the draw needs.
pub fn depth_stencil_resolve_modules(
    device: &Device,
    draw: DepthStencilResolveDraw,
    feature: Feature,
) -> Result<DepthStencilResolveModules, Error> {
    if !valid_samples(draw.samples) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Invariant,
            format!("a depth/stencil resolve of {} samples", draw.samples),
        )));
    }
    let (vertex, vertex_needs) = finish(
        device,
        super::sample_planes::merge_vertex_typed_module()?,
        &feature,
        "depth/stencil resolve's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        fragment_typed_module(draw)?,
        &feature,
        "depth/stencil resolve",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(DepthStencilResolveModules {
        vertex,
        fragment,
        requirements,
    })
}

#[cfg(test)]
mod tests;
