//! `exactness`: how each degradation's answer stands to Metal's.

use crate::{CommandBufferError, Degradation, DegradationCategory, Exactness, FailureOrigin};

/// A kind that costs time and memory, or runs through an unadvertised use, gives Metal's answer
/// (within its pool, where one can fail the work); a precision kind never does, and a pipeline's
/// precision note is a near-equivalent.
#[test]
fn exactness_follows_the_category() {
    for &kind in Degradation::ALL {
        let exact = matches!(
            kind.exactness(),
            Exactness::Exact | Exactness::ExactWithinPool
        );
        match kind.category() {
            DegradationCategory::Performance | DegradationCategory::Validity => {
                assert!(exact, "{kind:?}")
            }
            DegradationCategory::Precision => assert!(!exact, "{kind:?}"),
        }
        if kind.precision_note().is_some() {
            assert_eq!(kind.exactness(), Exactness::NearEquivalent, "{kind:?}");
        }
    }
}

/// The kinds whose documentation says Metal's own answer is not one value, and those whose work
/// fails instead, each with a failure that is the runtime's, not Metal's.
#[test]
fn the_kinds_metal_answers_otherwise_are_named() {
    use Degradation as D;
    let one_of_metals: Vec<Degradation> = Degradation::ALL
        .iter()
        .copied()
        .filter(|k| k.exactness() == Exactness::OneOfMetals)
        .collect();
    assert_eq!(one_of_metals.len(), 9, "{one_of_metals:?}");
    for kind in [
        D::ZeroedIndexFetch,
        D::DontCareStore,
        D::NarrowSurfaceRows,
        D::RepeatedExecutionStart,
        D::OverlappingTextureCopyAsMemmove,
        D::PlainStateUniqueIdentifier,
        // Metal's compiler drops or folds a removed argument's read, and its debug layer reads
        // zero: the runtime's zero is one of Metal's answers, and the work completes.
        D::AbsentArgumentRead,
        // Metal writes one constant of its process's for a color packed where fills ignore the
        // color, which differs between processes and machines: the runtime's zero bytes are one.
        D::UnpackedFillColor,
        // Metal sizes no point outside the point class without a point size, and draws mostly
        // nothing there, sometimes a few points or a box: the runtime's nothing is one.
        D::UnsizedPoints,
    ] {
        assert!(one_of_metals.contains(&kind), "{kind:?}");
    }
    // The device's blit chain lies within a unit of an M3 Max's or an M4's levels, at neither's
    // exactly: a near-equivalent, not one of Metal's answers.
    assert_eq!(
        D::MipmapsWithinSpread.exactness(),
        Exactness::NearEquivalent
    );
    // A state's identifier with indirect command buffer support, or a buffer's after an encoder,
    // is one Metal answers alike in every run: the runtime's is a near-equivalent.
    assert_eq!(
        D::UnmeasuredUniqueIdentifier.exactness(),
        Exactness::NearEquivalent
    );
    let fails: Vec<(Degradation, CommandBufferError)> = Degradation::ALL
        .iter()
        .filter_map(|&k| match k.exactness() {
            Exactness::FailsInstead(error) => Some((k, error)),
            _ => None,
        })
        .collect();
    assert_eq!(fails.len(), 2, "{fails:?}");
    for pair in [
        (
            D::UnlistedArgumentSampler,
            CommandBufferError::UnlistedArgumentSampler,
        ),
        (D::EncodedUseRebound, CommandBufferError::EncodedUseRebound),
    ] {
        assert!(fails.contains(&pair), "{pair:?}");
    }
    for (kind, error) in fails {
        assert_eq!(error.origin(), FailureOrigin::Runtime, "{kind:?}");
    }
}

/// A performance kind whose cost says a pool of the runtime's can fail the work is exact only
/// within that pool, and no other kind is: the label tells a consumer what the cost text does.
#[test]
fn a_pool_that_can_fail_the_work_is_not_exact() {
    use Degradation as D;
    let within: Vec<Degradation> = Degradation::ALL
        .iter()
        .copied()
        .filter(|k| k.exactness() == Exactness::ExactWithinPool)
        .collect();
    assert_eq!(
        within,
        vec![
            D::IndirectIndexBoundsPass,
            D::EmulatedMeshDraws,
            D::EmulatedTessellationDraws
        ],
        "{within:?}"
    );
    for &kind in Degradation::ALL {
        if kind.category() != DegradationCategory::Precision {
            assert_eq!(
                kind.cost().contains("fails the command buffer"),
                within.contains(&kind),
                "{kind:?}: {}",
                kind.cost()
            );
        }
    }
}
