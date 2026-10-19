//! What the check modules of the submodules share: their local size, their math modes and their
//! serialization.

use std::sync::Arc;

use super::CheckKernel;
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::{Device, DeviceProfile};
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, Symbol};
use crate::spirv::ir::{ExecMode, TypedModule};

/// [`CheckKernel::LOCAL_SIZE`], for the kernels.
pub(crate) const LOCAL_SIZE: u32 = CheckKernel::LOCAL_SIZE;

/// The math mode of a check's entry. Translation writes a device's arithmetic in entries with and
/// without `SignedZeroInfNanPreserve` for 32 bits (an operation without every fast-math flag
/// declares it), so a check of arithmetic runs in each such mode the device has.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MathMode {
    /// No float-control mode: the entry of operations with every fast-math flag.
    Relaxed,
    /// `SignedZeroInfNanPreserve` for 32 bits: the entry of an operation without every fast-math
    /// flag, on a device with `shaderSignedZeroInfNanPreserveFloat32`.
    Preserving,
}

impl MathMode {
    /// The math modes `device` has: [`MathMode::Relaxed`], and [`MathMode::Preserving`] where it
    /// has `shaderSignedZeroInfNanPreserveFloat32`.
    pub fn of(device: &Device) -> Vec<MathMode> {
        MathMode::of_profile(device.profile())
    }

    /// [`MathMode::of`] on a profile.
    pub(crate) fn of_profile(profile: &DeviceProfile) -> Vec<MathMode> {
        match profile
            .fields()
            .float_controls
            .f32
            .signed_zero_inf_nan_preserve
        {
            true => vec![MathMode::Relaxed, MathMode::Preserving],
            false => vec![MathMode::Relaxed],
        }
    }

    /// The execution modes the mode adds to an entry's.
    pub(crate) fn exec_modes(self) -> Vec<ExecMode> {
        match self {
            MathMode::Relaxed => Vec::new(),
            MathMode::Preserving => vec![ExecMode::SignedZeroInfNanPreserve { width: 32 }],
        }
    }
}

/// `m` checked by the owned verifier against `device`'s profile (requirements included),
/// serialized as a profile without the driver workarounds writes it (`spirv::serialize::serialize`:
/// no converter encoding, no function control), and numbered: each buffer, image and combined image
/// sampler at set 0, binding its slot's index. `what` names the module in an error.
pub(crate) fn finish(
    device: &Device,
    m: TypedModule,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    #[cfg(test)]
    record(&m);
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let binding_of = |sym: &Symbol| match sym {
        Symbol::DescriptorSet(_) => Some(0),
        Symbol::Binding(s)
            if matches!(
                s.kind,
                ResourceKind::Buffer
                    | ResourceKind::Texture
                    | ResourceKind::CombinedTextureSampler(_)
            ) =>
        {
            Some(s.index)
        }
        _ => None,
    };
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

#[cfg(test)]
thread_local! {
    /// The check modules finished on this thread inside a [`finished`] call: those of [`finish`],
    /// of the half quantize's and the index fetch's (`device_checks`) and of the raster checks
    /// (`raster_checks`), each as it was verified. The bounds guard
    /// (`tests::device_check_bounds`) reads every module a check builder makes through it.
    static FINISHED: std::cell::RefCell<Option<Vec<TypedModule>>> =
        const { std::cell::RefCell::new(None) };
}

/// Keep a copy of the verified check module `m` where a [`finished`] call on this thread watches.
#[cfg(test)]
pub(crate) fn record(m: &TypedModule) {
    FINISHED.with(|f| {
        if let Some(modules) = f.borrow_mut().as_mut() {
            modules.push(m.clone());
        }
    });
}

/// `build`'s result, and a copy of each check module it finished, in order.
#[cfg(test)]
pub(crate) fn finished<T>(build: impl FnOnce() -> T) -> (T, Vec<TypedModule>) {
    FINISHED.with(|f| *f.borrow_mut() = Some(Vec::new()));
    let built = build();
    let modules = FINISHED.with(|f| f.borrow().clone().unwrap_or_default());
    FINISHED.with(|f| *f.borrow_mut() = None);
    (built, modules)
}
