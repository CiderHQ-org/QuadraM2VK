//! Plain color writers a draw moves into the interlocked storage form.
//!
//! A render pass instance can keep a color out of its subpass for other draws' interlocked storage
//! form (`BindShape::interlocked_colors`), or hold an extended-range color's `half` shadow
//! (`BindShape::extended_shadows`); a draw there of a pipeline that writes the color as an
//! ordinary output runs the variant that writes it through the storage form, inside the fragment
//! function and so before the depth and stencil tests. The owned fixture `smw_writers` has one
//! plain writer of color 0 per way those tests bear on such a store: none (no discard, no other
//! write, or Metal's early tests), its discard alone (the early-tests variant where the tests
//! write nothing, the function's own depth or stencil test where they write), and a write of
//! other memory or a returned depth or sample mask (the function's own tests wherever the draw's
//! tests can drop a fragment).
//!
//! On every committed profile that orders fetch through the pixel interlock alone, in a pass with
//! a depth and stencil attachment, under each kind of draw tests: the plan of the pipeline as
//! compiled selects the variant the moved writer needs at once, and that variant, compiled, plans
//! itself (a fixed point, which the runtime requires of every variant it compiles), or both refuse
//! the draw. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, FragmentTests};
use crate::budget::Budget;
use crate::device::profile::FramebufferFetchPolicy;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::BindError;
use crate::library::Library;
use crate::lower::fragment::interlock_orders;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// How the depth and stencil tests bear on a writer's store before them.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// The variant declares early tests: no draw's tests constrain it.
    Free,
    /// Only its discard keeps it from early tests.
    EarlyVariant,
    /// It writes other memory or returns depth or a sample mask: only tests that drop nothing.
    Untested,
}

/// The writers of `smw_writers` and their classes.
const WRITERS: [(&str, Class); 6] = [
    ("smw_plain", Class::Free),
    ("smw_early", Class::Free),
    ("smw_discard", Class::EarlyVariant),
    ("smw_store", Class::Untested),
    ("smw_depth", Class::Untested),
    ("smw_mask", Class::Untested),
];

/// How a draw moves color 0 into the storage form: an interlocked `RGBA16Float` color, or a
/// `BGRA10_XR` color's `half` shadow.
#[derive(Clone, Copy, Debug)]
enum Move {
    Interlocked,
    Shadow,
}

const TESTS: [FragmentTests; 6] = [
    FragmentTests::Pass,
    FragmentTests::Drop,
    FragmentTests::Depth { writes: false },
    FragmentTests::Depth { writes: true },
    FragmentTests::Stencil,
    FragmentTests::Any,
];

fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../metal2vulkan-validation/profiles/{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The early-tests variant (`read_only_tests`) and the function's own tests (`shader_tests`) a
/// writer of `class` needs under `tests`; `None` where the draw is refused (tests that may count
/// what they keep, or that the draw does not state).
fn wanted(class: Class, tests: FragmentTests) -> Option<(bool, bool)> {
    match (class, tests) {
        (Class::Free, _) | (_, FragmentTests::Pass) => Some((false, false)),
        (Class::EarlyVariant, FragmentTests::Drop | FragmentTests::Depth { writes: false }) => {
            Some((true, false))
        }
        (_, FragmentTests::Any) => None,
        _ => Some((false, true)),
    }
}

#[test]
fn a_moved_writer_selects_its_tests_in_its_first_plan() {
    let bc = match llvm_assemble(&engine_dir().join("smw_writers.ll")) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected smw_writers: {why}"),
    };
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let profiles = committed_profiles().expect("profiles");
    // The profiles checked, and the writers' variants that planned themselves, by the tests
    // they run (`read_only_tests`, `shader_tests`).
    let mut checked = Vec::new();
    let mut fixed_points = std::collections::BTreeSet::<(&str, (bool, bool))>::new();
    for (stem, profile) in &profiles {
        if !interlock_orders(profile)
            || profile.fields().framebuffer_fetch != FramebufferFetchPolicy::RequireOrdered
        {
            continue;
        }
        let dev = device(stem);
        checked.push(stem.as_str());
        for (writer, class) in WRITERS {
            for moved in [Move::Interlocked, Move::Shadow] {
                let mut d = RenderPipelineDescriptor::new(function("smw_full"));
                d.fragment = Some(function(writer));
                d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(match moved {
                    Move::Interlocked => PixelFormat::RGBA16Float,
                    Move::Shadow => PixelFormat::BGRA10XR,
                }));
                d.depth_format = Some(PixelFormat::Depth32FloatStencil8);
                d.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
                let what = format!("{stem} {writer} {moved:?}");
                // A device short of what the writer needs (a fragment store) refuses it whole.
                let base = match dev.compile_render_pipeline(&d, &Budget::standard()) {
                    Ok(base) => base,
                    Err(e) => {
                        eprintln!("{what}: refused as compiled: {e}");
                        continue;
                    }
                };
                assert!(base.graphics.storage_attachments.is_empty(), "{what}");
                // The `early_variant` the function states, and its variant with it.
                let stated = match class {
                    Class::Free => None,
                    Class::EarlyVariant => Some(true),
                    Class::Untested => Some(false),
                };
                let movable = base.dependencies.iter().find_map(|d| match d.fact {
                    RuntimeFact::MovableStorageWrites { early_variant } => Some(early_variant),
                    _ => None,
                });
                assert_eq!(movable, stated, "{what}: the function's movable writes");
                for tests in TESTS {
                    let mut state = BindState {
                        fragment_tests: tests,
                        ..Default::default()
                    };
                    match moved {
                        Move::Interlocked => state.interlocked_colors.insert(0),
                        Move::Shadow => state.extended_shadows.insert(0),
                    };
                    let shape = state.shape();
                    let what = format!("{what} {tests:?}");
                    let planned = base.plan(&dev, &shape);
                    // The variant the plan names, compiled, plans itself: the runtime fails a draw
                    // whose compiled variant asks for another one.
                    let variant = planned.as_ref().ok().map(|plan| {
                        let variant = dev
                            .compile_render_variant(&d, &plan.variant, &Budget::standard())
                            .unwrap_or_else(|e| panic!("{what}: the variant: {e}"));
                        let again = variant.plan(&dev, &shape).unwrap_or_else(|e| {
                            panic!("{what}: the variant planned refuses the draw: {e}")
                        });
                        assert_eq!(again.variant, plan.variant, "{what}: a fixed point");
                        variant
                    });
                    let plan = match (planned, wanted(class, tests)) {
                        (Ok(plan), Some(_)) => plan,
                        (Err(BindError::Unsupported(_)), None) => continue,
                        // The function's own tests the device does not ship.
                        (Err(BindError::Unsupported(u)), Some((_, true))) => {
                            eprintln!("{what}: its own tests refused: {u:?}");
                            continue;
                        }
                        (planned, want) => {
                            panic!(
                                "{what}: planned {:?}, wanted {want:?}",
                                planned.map(|p| p.variant)
                            )
                        }
                    };
                    let (read_only, shader) = wanted(class, tests).expect("planned");
                    assert_eq!(
                        (plan.variant.read_only_tests, plan.variant.shader_tests),
                        (read_only, shader),
                        "{what}"
                    );
                    let variant = variant.expect("compiled for the plan");
                    assert!(
                        variant
                            .graphics
                            .storage_attachments
                            .iter()
                            .any(|a| a.color_index == 0 && a.writes),
                        "{what}: the variant writes color 0 through the storage form"
                    );
                    let stored_before = variant.dependencies.iter().find_map(|d| match d.fact {
                        RuntimeFact::TestedStorageWrites { early_variant, .. } => {
                            Some(early_variant)
                        }
                        _ => None,
                    });
                    assert_eq!(
                        stored_before, stated,
                        "{what}: the variant's store before the tests"
                    );
                    for m in &variant.stages {
                        check_words_valid(&m.spirv, &variant.requirements, dev.profile())
                            .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    }
                    fixed_points.insert((writer, (read_only, shader)));
                }
            }
        }
    }
    // Not vacuous: each writer planned itself under every kind of tests it takes.
    eprintln!("moved writers: checked on {checked:?}");
    for (writer, class) in WRITERS {
        let kinds: &[(bool, bool)] = match class {
            Class::Free => &[(false, false)],
            Class::EarlyVariant => &[(false, false), (true, false), (false, true)],
            Class::Untested => &[(false, false), (false, true)],
        };
        for kind in kinds {
            assert!(
                fixed_points.contains(&(writer, *kind)),
                "{writer}: no fixed point with tests {kind:?}"
            );
        }
    }
}
