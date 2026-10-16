//! A sample whose level the device's level arithmetic takes (`lower::sampler::level::device_when`):
//! the owned fixture `device_level`. Under exact sampling, on every committed profile that samples
//! neither through Metal's own sampler nor natively, the pipeline records
//! `PrecisionNote::DeviceSampleLevel` for an implicit-level sample through nearest filters and a
//! nearest mip filter (with a bias within the limit or without), an integer texture's, one with
//! explicit gradients, an LOD query, and an integer texture's explicit level of `1.5` (which
//! Metal's level rule is not handed on); for a runtime sampler's, only where its word states
//! nearest filters and a nearest mip filter. It records none for a sample whose level chooses
//! nothing (nearest filters without a mip filter), a filtered one (its filter note records it), a
//! cube's (`CubeGradientFootprint`), one at an explicit level the rule snaps, or an integer
//! texture's whole level of `2`. A profile that samples through Metal's own sampler or natively
//! records none, nor does any profile under native sampling. `llvm-as` assembles the fixture;
//! without it the test skips visibly.

use crate::device::{DevicePolicy, SamplingPolicy};
use crate::metal::sampler::SamplerWord;
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, AssembleError};
use crate::pipeline::PrecisionNote::DeviceSampleLevel;
use crate::spirv::testing::committed_profiles_with;

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

/// Where an entry's pipeline holds the note.
#[derive(Clone, Copy, Debug)]
enum Held {
    Never,
    Always,
    /// Where the runtime sampler's word states nearest filters and a nearest mip filter.
    NearestMips,
}

#[test]
fn a_device_level_is_noted_where_it_chooses_and_no_other_note_records_it() {
    let Some(bc) = assemble("device_level") else {
        return;
    };
    let entries = [
        ("dl_nearest_mips", Held::Always),
        ("dl_nearest_mips_bias", Held::Always),
        ("dl_nearest_base", Held::Never),
        ("dl_linear_mips", Held::Never),
        ("dl_uint_mips", Held::Always),
        ("dl_runtime", Held::NearestMips),
        ("dl_query", Held::Always),
        ("dl_cube", Held::Never),
        ("dl_gradients", Held::Always),
        ("dl_level", Held::Never),
        ("dl_uint_level", Held::Always),
        ("dl_uint_whole_level", Held::Never),
    ];
    let exact = DevicePolicy {
        sampling: SamplingPolicy::Exact,
        ..DevicePolicy::default()
    };
    let mut reached = 0;
    for (policy, name) in [(exact, "exact"), (DevicePolicy::default(), "native")] {
        for (stem, p) in committed_profiles_with(policy).expect("profiles") {
            let device = p.device_samples();
            reached += usize::from(!device);
            for (entry, held) in entries {
                let what = format!("{entry} on {stem} under {name} sampling");
                let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let noted = a.precision.contains(&DeviceSampleLevel);
                let when = a.specialized.iter().find(|s| s.note == DeviceSampleLevel);
                match (device, held) {
                    (true, _) | (false, Held::Never) => assert!(!noted, "{what}"),
                    (false, Held::Always) => assert!(noted && when.is_none(), "{what}: {when:?}"),
                    (false, Held::NearestMips) => {
                        assert!(noted, "{what}");
                        let when = when.unwrap_or_else(|| panic!("{what}: unconditional"));
                        assert!(
                            when.when.iter().all(|c| c.all.iter().any(|t| {
                                t.mask == SamplerWord::LINEAR_FILTER
                                    && t.bits == 1 << SamplerWord::MIP_FILTER_SHIFT
                                    && t.equal
                            })),
                            "{what}: {when:?}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        reached > 0,
        "no profile under exact sampling takes the device's level arithmetic"
    );
}
