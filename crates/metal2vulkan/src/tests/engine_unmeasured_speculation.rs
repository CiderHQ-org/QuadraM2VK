//! A quad operation in a branch's arm of a shape no probe of Metal's speculation covers
//! (`air::canon::speculate`): the owned fixture `unmeasured_speculation`. On every committed profile
//! the pipeline of `quad_first_in_an_unmeasured_arm`, `quad_broadcast_first` beside `sin` in the
//! arm, records `PrecisionNote::UnmeasuredGroupSpeculation`. That of `sin_in_an_unmeasured_arm`,
//! the same arm without the broadcast, records none, nor does that of `gsm2_quad_ops`'
//! `quad_first_in_a_speculated_arm`, whose arm Metal was measured to speculate. `llvm-as` assembles
//! the fixtures; without it the test skips visibly.

use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, AssembleError};
use crate::pipeline::PrecisionNote;
use crate::spirv::testing::committed_profiles;

/// Fixture `name` assembled, or `None` (skipped visibly) without `llvm-as`.
fn assemble(name: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{name}.ll"));
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {name}: {e}"),
    }
}

#[test]
fn only_a_group_operation_in_an_unmeasured_arm_is_noted() {
    let (Some(unmeasured), Some(quad_ops)) = (
        assemble("unmeasured_speculation"),
        assemble("gsm2_quad_ops"),
    ) else {
        return;
    };
    let kernels = [
        (&unmeasured, "quad_first_in_an_unmeasured_arm", true),
        (&unmeasured, "sin_in_an_unmeasured_arm", false),
        (&quad_ops, "quad_first_in_a_speculated_arm", false),
    ];
    let profiles = committed_profiles().expect("profiles");
    assert!(!profiles.is_empty());
    for (stem, p) in &profiles {
        for (bc, entry, noted) in kernels {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                a.precision
                    .contains(&PrecisionNote::UnmeasuredGroupSpeculation),
                noted,
                "{what}"
            );
        }
    }
}
