//! How each kind of degradation's answer stands to Metal's.
//!
//! [`Degradation::category`] says what a kind costs; [`Degradation::exactness`] says whether a
//! program still gets Metal's answer, which a consumer that answers as Metal needs at each
//! announcement. It follows each kind's documentation, which states the evidence. A precision
//! kind is a near-equivalent unless its documentation says Metal's own answer is not one value,
//! or that the work fails instead. A performance kind is exact unless its cost says a pool of the
//! command buffer's can fail the work.

use crate::degradation::{Degradation, DegradationCategory};
use crate::error::CommandBufferError;

/// How a degradation's answer stands to Metal's ([`Degradation::exactness`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exactness {
    /// Metal's answer, exactly: at a cost Metal does not pay
    /// ([`DegradationCategory::Performance`]), or through a use of the device its driver does not
    /// advertise but is measured to run as Metal does ([`DegradationCategory::Validity`]).
    Exact,
    /// Metal's own answer is not one value there (it reads memory-dependent bytes, keeps or
    /// loses texels as its tiler runs, races, or differs from run to run or between Apple's
    /// GPUs), and the runtime gives one such answer.
    OneOfMetals,
    /// The runtime gives no answer where Metal's would differ: the command buffer whose work
    /// would show the difference fails with this error, of
    /// [`FailureOrigin::Runtime`](crate::FailureOrigin::Runtime), where Metal runs it.
    FailsInstead(CommandBufferError),
    /// Metal's answer, exactly, at a cost Metal does not pay, while a pool of the command buffer
    /// (a sixty-fourth of the device's largest device-local heap) holds the work: a draw the pool
    /// cannot hold draws nothing, and its command buffer fails with a [`CommandBufferError`] of
    /// [`FailureOrigin::Runtime`](crate::FailureOrigin::Runtime) where Metal draws it
    /// ([`CommandBufferError::IndexPoolOverflow`] for `indirect-index-bounds-pass`,
    /// [`CommandBufferError::MeshRecordsOverflow`] for `emulated-mesh-draws`,
    /// [`CommandBufferError::TessellationRecordsOverflow`] for `emulated-tessellation-draws`).
    ExactWithinPool,
    /// A recorded near-equivalent of Metal's answer, or an answer where none of Metal's is
    /// measured.
    NearEquivalent,
}

impl Degradation {
    /// How the kind's answer stands to Metal's (the module docs), the same for every
    /// occurrence.
    pub const fn exactness(self) -> Exactness {
        use CommandBufferError as E;
        use Degradation as D;
        match self.category() {
            DegradationCategory::Performance | DegradationCategory::Validity => match self {
                D::IndirectIndexBoundsPass
                | D::EmulatedMeshDraws
                | D::EmulatedTessellationDraws => Exactness::ExactWithinPool,
                _ => Exactness::Exact,
            },
            DegradationCategory::Precision => match self {
                // As each kind's documentation says, Metal reads the memory past the buffer;
                // keeps the earlier texels unless its tiler overflows; reads rows past the first
                // as memory-dependent bytes; runs no stable range; reads what its copy already
                // wrote, differently from run to run; answers a plain state's identifier, an
                // address that differs from run to run. Not `mipmaps-within-spread`: its levels lie
                // within a unit of an M3 Max's or an M4's, at neither's exactly, so they are a
                // near-equivalent, not one of Metal's. A removed argument's read: Metal gives it no
                // one value (its compiler drops or folds it, and its debug layer reads zero), and
                // the runtime gives zero. A color fill packed where fills ignore the color: Metal
                // writes a constant of its process's, and the runtime one constant, zero.
                D::ZeroedIndexFetch
                | D::DontCareStore
                | D::NarrowSurfaceRows
                | D::RepeatedExecutionStart
                | D::OverlappingTextureCopyAsMemmove
                | D::PlainStateUniqueIdentifier
                | D::AbsentArgumentRead
                | D::UnpackedFillColor
                // Points Metal sizes none of: what it draws is undefined, mostly nothing, and the
                // runtime draws nothing.
                | D::UnsizedPoints => Exactness::OneOfMetals,
                D::UnlistedArgumentSampler => Exactness::FailsInstead(E::UnlistedArgumentSampler),
                D::EncodedUseRebound => Exactness::FailsInstead(E::EncodedUseRebound),
                _ => Exactness::NearEquivalent,
            },
        }
    }
}
