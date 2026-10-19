//! The fragment stage of the check of `RuntimeDriverDefect::OcclusionCountPerColorOutput`
//! ([`color_outputs_check_fragment`]): one `Fragment` entry `main` (`OriginUpperLeft`) that writes
//! `(1, 1, 1, 1)` to each of its `outputs` color outputs, at locations `0..outputs`, as a fragment
//! function writing that many color attachments writes them. It reads nothing, so a draw of it
//! under an occlusion query counts what its coverage passes alone.

use super::kernel::finish;
use crate::bind::RasterCheck;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{
    ConstBits, Constant, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule,
};

/// The most color outputs the fragment writes: Vulkan's least `maxColorAttachments`.
const MOST_COLOR_OUTPUTS: u32 = 4;

/// The fragment of `outputs` color outputs as a typed module (the module docs).
fn color_outputs_typed_module(outputs: u32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let void = b.ty(Type::Void)?;
    let colors = (0..outputs)
        .map(|location| b.fixed_io(location, IoDir::Output, vec4, false))
        .collect::<BResult<Vec<_>>>()?;
    let mut f = b.function(void)?;
    let one = f.module().constant(Constant::Scalar {
        ty: f32_,
        bits: ConstBits(u64::from(1f32.to_bits())),
    })?;
    let one = f.module().const_value(one)?;
    let value = f.composite_construct(vec4, &[one, one, one, one])?;
    for color in colors {
        let color = f.module().global_value(color)?;
        f.store(color, value, MemAccess::default())?;
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The fragment stage of `outputs` color outputs (1 to 4, Vulkan's least `maxColorAttachments`) on
/// `device` (the module docs), checked against its profile; an internal error for another count.
pub fn color_outputs_check_fragment(device: &Device, outputs: u32) -> Result<RasterCheck, Error> {
    if !(1..=MOST_COLOR_OUTPUTS).contains(&outputs) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Build,
            format!(
                "the color outputs check writes 1 to {MOST_COLOR_OUTPUTS} outputs, not {outputs}"
            ),
        )));
    }
    let (spirv, requirements) = finish(
        device,
        color_outputs_typed_module(outputs)?,
        "color outputs check",
    )?;
    Ok(RasterCheck {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{color_outputs_typed_module, MOST_COLOR_OUTPUTS};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_color_outputs_modules_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for outputs in 1..=MOST_COLOR_OUTPUTS {
                assert_valid(color_outputs_typed_module(outputs).expect("module"), &p);
            }
        }
    }
}
