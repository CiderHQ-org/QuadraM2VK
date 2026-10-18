//! `private_properties`: Apple's private device properties, as Metal answers them at each level
//! and as the runtime answers them on every committed device profile, read from what the runtime
//! acts on.

use std::collections::BTreeSet;

use super::capabilities::{committed_descriptions, created_at};
use super::families::{all_audits, format_capabilities};
use crate::argument_table::{check_index, BUFFER_SLOTS, SAMPLER_SLOTS, TEXTURE_SLOTS};
use crate::private_properties::MEASURED_FAMILY;
use crate::texture_descriptor::{Refused, TextureDescriptor, TextureProblem};
use crate::{MetalLevel, PrivateAnswer, PrivateCount, PrivateFlag, PrivateProperties};
use metal2vulkan::device::{Device, DevicePolicy};
use metal2vulkan::metal::TextureType;

/// The answers on the committed profile `stem` at `level`, as `Device::private_properties`
/// reads them.
fn answers(
    stem: &str,
    desc: &metal2vulkan::device::DeviceDescription,
    level: MetalLevel,
) -> PrivateProperties {
    let (enabled, properties) = created_at(stem, desc, level);
    let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
    let rows = all_audits(stem, desc).remove(&MEASURED_FAMILY).unwrap();
    let predicates = compile.profile().addressed_needs().is_empty();
    let formats = |pf| format_capabilities(&enabled, &compile, level, pf);
    PrivateProperties::of(
        &properties,
        compile.metal_properties(),
        &rows,
        predicates,
        &formats,
    )
}

/// `ALL` lists each property once, in declaration order, under a selector of its own.
#[test]
fn every_property_has_one_selector() {
    for (index, p) in PrivateCount::ALL.into_iter().enumerate() {
        assert_eq!(p as usize, index, "{p:?}");
    }
    for (index, p) in PrivateFlag::ALL.into_iter().enumerate() {
        assert_eq!(p as usize, index, "{p:?}");
    }
    let selectors: BTreeSet<&str> = PrivateCount::ALL
        .iter()
        .map(|p| p.selector())
        .chain(PrivateFlag::ALL.iter().map(|p| p.selector()))
        .collect();
    assert_eq!(
        selectors.len(),
        PrivateCount::ALL.len() + PrivateFlag::ALL.len()
    );
}

/// Spot values as the probes printed them on an Apple M3 Max at each release (an Apple M4 under
/// macOS 27.0.1 answers `gpuCoreCount` 10 and `supportsComputeCompressedTextureWrite` yes; the M3
/// Max 40 and no).
#[test]
fn the_answers_are_the_probes() {
    for level in MetalLevel::ALL {
        assert_eq!(level.private_count(PrivateCount::GpuCoreCount), 40);
        assert_eq!(level.private_count(PrivateCount::MaxComputeBuffers), 31);
        assert_eq!(
            level.private_count(PrivateCount::MaxComputeThreadgroupMemory),
            65536
        );
        assert_eq!(
            level.private_count(PrivateCount::MaxVertexAmplificationCount),
            2
        );
        assert_eq!(
            level.private_count(PrivateCount::MaxPredicatedNestingDepth),
            u64::from(u32::MAX)
        );
        assert!(level.private_flag(PrivateFlag::SupportsTileShaders));
        // `supportsPrimitiveType:`: the five public types and four private ones.
        assert!(
            (0..9).all(|raw| level.supports_primitive_type(raw)),
            "{level}"
        );
        assert!(
            !(9..13).any(|raw| level.supports_primitive_type(raw)),
            "{level}"
        );
        assert!(!level.private_flag(PrivateFlag::SupportsComputeCompressedTextureWrite));
    }
}

/// Which properties answer Metal's measured value: those no document describes.
#[test]
fn metals_answers_are_the_undocumented_properties() {
    use PrivateCount as C;
    use PrivateFlag as F;
    let metal_counts = BTreeSet::from([
        C::GpuCoreCount,
        C::MaxVertexAmplificationCount,
        C::MaxComputeThreadgroupMemory,
    ]);
    let metal_flags = BTreeSet::from([
        F::IsLargeMrtSupported,
        F::SupportsComputeCompressedTextureWrite,
        F::SupportsConditionalLoadStore,
    ]);
    for (stem, desc) in committed_descriptions() {
        for level in MetalLevel::ALL {
            let p = answers(&stem, &desc, level);
            for c in PrivateCount::ALL {
                let metal = level.private_count(c);
                match p.count(c) {
                    PrivateAnswer::Metal(v) => {
                        assert!(metal_counts.contains(&c), "{stem} {c:?}");
                        assert_eq!(v, metal, "{stem} {c:?}");
                        assert_eq!(p.count(c).value(), None, "{stem} {c:?}");
                        assert!(p.unknown().contains(&c.selector()), "{stem} {c:?}");
                    }
                    PrivateAnswer::Device { metal: m, .. } => {
                        assert!(!metal_counts.contains(&c), "{stem} {c:?}");
                        assert_eq!(m, metal, "{stem} {c:?}");
                    }
                }
            }
            for f in PrivateFlag::ALL {
                let metal = level.private_flag(f);
                match p.flag(f) {
                    PrivateAnswer::Metal(v) => {
                        assert!(metal_flags.contains(&f), "{stem} {f:?}");
                        assert_eq!(v, metal, "{stem} {f:?}");
                        assert_eq!(p.flag(f).value(), None, "{stem} {f:?}");
                        assert!(p.unknown().contains(&f.selector()), "{stem} {f:?}");
                    }
                    PrivateAnswer::Device { metal: m, .. } => {
                        assert!(!metal_flags.contains(&f), "{stem} {f:?}");
                        assert_eq!(m, metal, "{stem} {f:?}");
                    }
                }
            }
        }
    }
}

/// The per-stage table counts are what the encoders' tables take, and the 2D size and layers
/// what the descriptor takes, on every profile: the last index or size answered is taken, the
/// next refused. On MoltenVK on an M3 and an M4 every limit answer is Metal's.
#[test]
fn table_counts_and_sizes_are_what_the_runtime_takes() {
    use PrivateCount as C;
    for (stem, desc) in committed_descriptions() {
        let p = answers(&stem, &desc, MetalLevel::NEWEST);
        for (counts, slots) in [
            (
                [
                    C::MaxVertexBuffers,
                    C::MaxFragmentBuffers,
                    C::MaxComputeBuffers,
                    C::MaxTileBuffers,
                ],
                BUFFER_SLOTS,
            ),
            (
                [
                    C::MaxVertexTextures,
                    C::MaxFragmentTextures,
                    C::MaxComputeTextures,
                    C::MaxTileTextures,
                ],
                TEXTURE_SLOTS,
            ),
            (
                [
                    C::MaxVertexSamplers,
                    C::MaxFragmentSamplers,
                    C::MaxComputeSamplers,
                    C::MaxTileSamplers,
                ],
                SAMPLER_SLOTS,
            ),
        ] {
            for c in counts {
                let answer = u32::try_from(p.count(c).value().unwrap()).unwrap();
                assert!(answer <= slots, "{stem} {c:?}");
                if answer == slots {
                    assert!(check_index("t", answer - 1, slots).is_ok(), "{stem} {c:?}");
                    assert!(check_index("t", answer, slots).is_err(), "{stem} {c:?}");
                }
            }
        }
        let side = p.count(C::MaxTextureWidth2D).value().unwrap();
        assert_eq!(Some(side), p.count(C::MaxTextureHeight2D).value(), "{stem}");
        let square = |n: u64| TextureDescriptor {
            texture_type: TextureType::Type2D,
            width: n,
            height: n,
            ..TextureDescriptor::default()
        };
        let rules = super::texture_descriptor::APPLE;
        assert!(square(side).check(rules).is_ok(), "{stem} {side}");
        if side == 16384 {
            assert_eq!(
                square(side + 1).check(rules),
                Err(Refused::Stops(TextureProblem::PastLargestSide {
                    max: side
                })),
                "{stem}"
            );
        }
        let layers = p.count(C::MaxTextureLayers).value().unwrap();
        let array = |n: u64| TextureDescriptor {
            texture_type: TextureType::Type2DArray,
            array_length: n,
            ..TextureDescriptor::default()
        };
        assert!(array(layers).check(rules).is_ok(), "{stem} {layers}");
        if stem.starts_with("moltenvk-") {
            for c in PrivateCount::ALL {
                if let PrivateAnswer::Device { answer, metal } = p.count(c) {
                    assert_eq!(answer, metal, "{stem} {c:?}");
                }
            }
        }
    }
}

/// The runtime draws Metal's five public primitive types and none of the private ones.
#[test]
fn the_device_draws_the_public_primitive_types() {
    let (stem, desc) = committed_descriptions().into_iter().next().unwrap();
    let (_, properties) = created_at(&stem, &desc, MetalLevel::NEWEST);
    assert!((0..5).all(|raw| properties.supports_primitive_type(raw)));
    assert!(!(5..13).any(|raw| properties.supports_primitive_type(raw)));
    assert!(!properties.supports_primitive_type(u64::from(u32::MAX) + 1));
}

/// Command predication's properties answer as it runs: Metal's where the device has device
/// addresses, which its step module reaches memory by, and none elsewhere.
#[test]
fn predication_answers_where_it_runs() {
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created_at(&stem, &desc, MetalLevel::NEWEST);
        let compile = Device::new(enabled, DevicePolicy::UNTRUSTED).unwrap();
        let runs = compile.profile().addressed_needs().is_empty();
        let p = answers(&stem, &desc, MetalLevel::NEWEST);
        let jump = p.flag(PrivateFlag::SupportsCommandBufferJump).value();
        assert_eq!(jump, Some(runs), "{stem}");
        let depth = p.count(PrivateCount::MaxPredicatedNestingDepth).value();
        assert_eq!(
            depth,
            Some(if runs { u64::from(u32::MAX) } else { 0 }),
            "{stem}"
        );
    }
}

/// `gpuCoreCount` is the device's where its driver reports its cores, and none elsewhere: never
/// Apple's GPU's; `supportsNativeHardwareFP16` is the device's `shaderFloat16`.
#[test]
fn the_core_count_and_fp16_are_the_devices() {
    use crate::capabilities::DeviceProperties;
    for (stem, desc) in committed_descriptions() {
        let (enabled, properties) = created_at(&stem, &desc, MetalLevel::NEWEST);
        let compile = Device::new(enabled.clone(), DevicePolicy::UNTRUSTED).unwrap();
        let rows = all_audits(&stem, &desc).remove(&MEASURED_FAMILY).unwrap();
        let predicates = compile.profile().addressed_needs().is_empty();
        let formats = |pf| format_capabilities(&enabled, &compile, MetalLevel::NEWEST, pf);
        let of = |properties: &DeviceProperties| {
            PrivateProperties::of(
                properties,
                compile.metal_properties(),
                &rows,
                predicates,
                &formats,
            )
        };
        let unknown = of(&properties).count(PrivateCount::GpuCoreCount);
        assert_eq!(unknown.value(), None, "{stem}");
        assert_eq!(unknown.metal(), 40, "{stem}: Metal's, the M3 Max's, beside");
        let mut reported = properties.clone();
        reported.gpu_core_count = Some(84);
        assert_eq!(
            of(&reported).count(PrivateCount::GpuCoreCount).value(),
            Some(84),
            "{stem}"
        );
        assert_eq!(
            of(&properties)
                .flag(PrivateFlag::SupportsNativeHardwareFp16)
                .value(),
            Some(enabled.has(metal2vulkan::device::VkFeature::ShaderFloat16)),
            "{stem}"
        );
    }
}

/// The answers read from the device: the texture alignments are the 16 bytes the runtime takes a
/// linear texture and an IOSurface plane at; ordered atomics follow the Vulkan memory model at
/// device scope; the XR10 formats follow their family row; a half-float texture is filtered
/// bicubically everywhere; and `RGB10A2Unorm_sRGB`, which the runtime holds no texture of, is
/// filtered nowhere.
#[test]
fn the_alignments_atomics_and_formats_are_the_devices() {
    use crate::families::{FamilyRow, FeatureRow, RowState};
    use metal2vulkan::device::VkFeature;
    use PrivateCount as C;
    use PrivateFlag as F;
    for (stem, desc) in committed_descriptions() {
        let (enabled, _) = created_at(&stem, &desc, MetalLevel::NEWEST);
        let p = answers(&stem, &desc, MetalLevel::NEWEST);
        for c in [
            C::DeviceLinearTextureAlignmentBytes,
            C::LinearTextureAlignmentBytes,
            C::IosurfaceTextureAlignmentBytes,
            C::IosurfaceReadOnlyTextureAlignmentBytes,
        ] {
            assert_eq!(
                p.count(c),
                PrivateAnswer::Device {
                    answer: crate::linear_plan::LINEAR_ALIGNMENT,
                    metal: 16
                },
                "{stem} {c:?}"
            );
        }
        let model = enabled.has(VkFeature::VulkanMemoryModel)
            && enabled.has(VkFeature::VulkanMemoryModelDeviceScope);
        let flag = |f| p.flag(f).value();
        assert_eq!(flag(F::SupportsMemoryOrderAtomics), Some(model), "{stem}");
        let rows = all_audits(&stem, &desc).remove(&MEASURED_FAMILY).unwrap();
        let xr10 = rows
            .iter()
            .find(|a| a.row == FamilyRow::Feature(FeatureRow::ExtendedRangePixelFormats))
            .map(|a| a.state == RowState::Exact);
        assert_eq!(flag(F::SupportsExtendedXr10Formats), xr10, "{stem}");
        assert_eq!(
            flag(F::SupportsFloat16BcubicFiltering),
            Some(true),
            "{stem}"
        );
        assert_eq!(flag(F::IsRgb10a2GammaSupported), Some(false), "{stem}");
    }
}
