//! The fragment stage of the checks of the runtime's fragment-depth draws
//! (`DriverFact::FragmentDepthStoresUnchanged`, `DriverFact::FragmentDepthKeepsDepth16Codes`,
//! `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`), [`depth_writes_check_fragment`]:
//! one `Fragment` entry `main` (`OriginUpperLeft`, `DepthReplacing`) that runs per sample (it reads
//! `SampleId`) and writes as its `FragDepth` the word at slot `4 (y · width + x) + s` of the
//! storage buffer at [`RasterCheck::PARAMS_BINDING`] of set [`RasterCheck::SET`], bit-cast to
//! `float`: `(x, y)` its pixel in a target `width` wide, `s` its sample. It is the write the
//! runtime's copy draws make of a fetched depth (`bind::DepthStencilCopy::Depth`), of words a
//! check chooses. A slot past the buffer's end (a helper invocation's, past the target) reads the
//! buffer's first word.

use super::kernel::finish;
use super::{u32_value, words_buffer};
use crate::bind::RasterCheck;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{BinOp, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule};

/// The words the fragment writes as its depths.
const DEPTHS: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Buffer,
    index: RasterCheck::PARAMS_BINDING,
};

/// The fragment for a target `width` wide as a typed module (the module docs).
fn depth_writes_typed_module(width: u32) -> BResult<TypedModule> {
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
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let void = b.ty(Type::Void)?;
    let depths = words_buffer(&mut b, DEPTHS, u32_)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let sample = b.builtin(crate::spv::BuiltIn::SampleId, IoDir::Input, i32_)?;
    let depth = b.builtin(crate::spv::BuiltIn::FragDepth, IoDir::Output, f32_)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let width = u32_value(&mut f, width)?;
    let row = f.binary(BinOp::IMul, y, width)?;
    let pixel = f.binary(BinOp::IAdd, row, x)?;
    let slots = u32_value(&mut f, RasterCheck::SAMPLE_SLOTS)?;
    let sample = f.module().global_value(sample)?;
    let sample = f.load(sample, MemAccess::default())?;
    let sample = f.convert(ConvOp::Bitcast, sample, u32_)?;
    let slot = f.binary(BinOp::IMul, pixel, slots)?;
    let slot = f.binary(BinOp::IAdd, slot, sample)?;
    let depths = f.module().global_value(depths)?;
    let len = f.array_length(depths, 0)?;
    let inside = f.binary(BinOp::ULessThan, slot, len)?;
    let slot = f.select(inside, slot, zero)?;
    let word = f.access_chain(depths, &[zero, slot])?;
    let word = f.load(word, MemAccess::default())?;
    let value = f.convert(ConvOp::Bitcast, word, f32_)?;
    let depth = f.module().global_value(depth)?;
    f.store(depth, value, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::Fragment,
        id,
        vec![ExecMode::OriginUpperLeft, ExecMode::DepthReplacing],
    )?;
    b.finalize()
}

/// The fragment stage that writes chosen depths, for a target `width` wide, on `device` (the
/// module docs), checked against its profile; an internal error where the device lacks
/// `sampleRateShading`.
pub fn depth_writes_check_fragment(device: &Device, width: u32) -> Result<RasterCheck, Error> {
    let (spirv, requirements) = finish(
        device,
        depth_writes_typed_module(width)?,
        "depth writes check",
    )?;
    Ok(RasterCheck {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::depth_writes_typed_module;
    use crate::device::VkFeature;
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Valid on every committed profile with `sampleRateShading`; each profile without it is
    /// printed.
    #[test]
    fn the_depth_writes_modules_are_valid_on_every_committed_profile() {
        for (stem, p) in committed_profiles().expect("profiles") {
            if !p.enabled().features.contains(&VkFeature::SampleRateShading) {
                eprintln!("{stem}: no SampleRateShading, the depth writes check skipped");
                continue;
            }
            for width in [8, 128] {
                assert_valid(depth_writes_typed_module(width).expect("module"), &p);
            }
        }
    }
}
