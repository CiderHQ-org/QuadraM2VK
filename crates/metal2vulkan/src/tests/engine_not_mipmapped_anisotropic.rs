//! Constexpr samplers with anisotropy and no mip filter, sampled with gradients: the owned fixture
//! `not_mipmapped_anisotropic`. Through Metal's own sampler (`DriverDeviation::MetalSampler`) the
//! one whose minification and magnification filters differ (`anisotropic_mixed_filters`) records
//! `PrecisionNote::NotMipmappedAnisotropicFilter`: no level clamp the device's sampler takes reads
//! Metal's `NotMipmapped` footprint (`metal::sampler::vk_sampler`). Equal filters
//! (`anisotropic_equal_filters`), a mip filter (`anisotropic_mipmapped_mixed`), or a device that
//! does not sample through Metal's own sampler record none. `llvm-as` assembles the fixture;
//! without it the test skips visibly.

use crate::error::DriverDeviation;
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, AssembleError};
use crate::pipeline::PrecisionNote;
use crate::spirv::testing::committed_profiles;

#[test]
fn only_mixed_filters_without_mips_through_metals_sampler_are_noted() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/not_mipmapped_anisotropic.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            return;
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the fixture: {e}"),
    };
    let kernels = [
        ("anisotropic_mixed_filters", true),
        ("anisotropic_equal_filters", false),
        ("anisotropic_mipmapped_mixed", false),
    ];
    let (mut noted, mut quiet) = (0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let own = p.takes_driver_deviation(DriverDeviation::MetalSampler)
            && p.fields().images.sampler_anisotropy;
        for (entry, mixed) in kernels {
            let what = format!("{entry} on {stem}");
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let note = a
                .precision
                .contains(&PrecisionNote::NotMipmappedAnisotropicFilter);
            assert_eq!(note, own && mixed, "{what}");
            noted += usize::from(note);
            quiet += usize::from(mixed && !note);
        }
    }
    assert!(noted > 0, "no profile samples through Metal's own sampler");
    assert!(
        quiet > 0,
        "every profile samples through Metal's own sampler"
    );
}
