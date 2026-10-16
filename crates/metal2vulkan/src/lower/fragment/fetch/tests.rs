//! The storage form's choice: what blocks it, with which reason, and how it declares an attachment.

use super::*;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::lower::arith::extended_range::blend::BlendMath;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;

/// Committed profile `stem` with `features` removed from its description.
fn profile(stem: &str, features: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in features {
        desc.features.remove(f);
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The state of a pipeline of one color attachment of `format` at index 0, changed by `edit`.
fn state(
    format: PixelFormat,
    edit: impl FnOnce(&mut ColorAttachmentDescriptor),
) -> FramebufferFetchInputs {
    let mut c = ColorAttachmentDescriptor::new(format);
    edit(&mut c);
    FramebufferFetchInputs::of([0], &[Some(c)], false, false, 1)
}

const WRITES: ColorUse = ColorUse {
    writes: true,
    early_tests: true,
    draw_tested: false,
    math: BlendMath::Fast,
};
const READS: ColorUse = ColorUse {
    writes: false,
    ..WRITES
};

fn form(
    inputs: &FramebufferFetchInputs,
    multisampled: bool,
    use_: ColorUse,
    p: &DeviceProfile,
) -> Result<StorageView, Unsupported> {
    storage_form(0, inputs, multisampled, use_, p).expect("no descriptor error")
}

fn declared(
    r: Result<StorageView, Unsupported>,
) -> (vk::Format, crate::spv::ImageFormat, SampledType) {
    let v = r.expect("the storage form");
    (v.format, v.declared, v.numeric)
}

/// The image format a view of `viewed` for `use_` declares on `p`: none where the device grants
/// that access without a format, else `typed`.
fn expected_declaration(
    viewed: vk::Format,
    use_: ColorUse,
    typed: crate::spv::ImageFormat,
    p: &DeviceProfile,
) -> crate::spv::ImageFormat {
    match accessed_without_format(viewed, StorageUse::texture(true, use_.writes), p) {
        true => crate::spv::ImageFormat::Unknown,
        false => typed,
    }
}

/// A clean single-sample state takes the storage form on every device that orders fetch only
/// through the interlock, declared without a format where the device grants the format's storage
/// access without one, else with the format's SPIR-V image format.
#[test]
fn a_clean_state_takes_the_storage_form() {
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "intel-irisxe-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for (pf, want) in [
            (
                PixelFormat::R32Float,
                (
                    vk::Format::R32_SFLOAT,
                    crate::spv::ImageFormat::R32f,
                    SampledType::Float,
                ),
            ),
            (
                PixelFormat::RGBA8Unorm,
                (
                    vk::Format::R8G8B8A8_UNORM,
                    crate::spv::ImageFormat::Rgba8,
                    SampledType::Float,
                ),
            ),
            (
                PixelFormat::RGBA32Uint,
                (
                    vk::Format::R32G32B32A32_UINT,
                    crate::spv::ImageFormat::Rgba32ui,
                    SampledType::Uint,
                ),
            ),
            (
                PixelFormat::RG16Float,
                (
                    vk::Format::R16G16_SFLOAT,
                    crate::spv::ImageFormat::Rg16f,
                    SampledType::Float,
                ),
            ),
        ] {
            for use_ in [WRITES, READS] {
                let mut want = want;
                want.1 = expected_declaration(want.0, use_, want.1, &p);
                assert_eq!(
                    declared(form(&state(pf, |_| {}), false, use_, &p)),
                    want,
                    "{stem} {pf:?} {use_:?}"
                );
            }
        }
    }
    for stem in ["lavapipe", "vk12-floor", "swiftshader"] {
        assert!(!interlock_orders(&profile(stem, &[])), "{stem}");
    }
}

/// What the form does not model is `NotImplemented`: an amplified pipeline, a one- or two-channel
/// sRGB format, and,
/// for a written attachment only, blending (but one and zero, added), a signed normalized format under a write mask that
/// keeps some of its channels, alpha to coverage, a returned sample mask at one sample, and a depth
/// or stencil attachment without early fragment tests, but where each draw's plan checks its tests.
#[test]
fn unmodelled_states_are_not_implemented() {
    let p = profile("moltenvk-m3", &[]);
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    let r32 = state(PixelFormat::R32Float, |_| {});
    let mut amplified = r32.clone();
    amplified.amplified = true;
    assert_eq!(form(&amplified, false, READS, &p), ni);
    for pf in [PixelFormat::R8UnormSRGB, PixelFormat::RG8UnormSRGB] {
        let srgb = state(pf, |_| {});
        assert_eq!(form(&srgb, false, READS, &p), ni, "{pf:?}");
    }
    let blended = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
    });
    // Only the red channel of a one-channel format needs writing.
    let red = state(PixelFormat::R32Float, |c| {
        c.write_mask = ColorWriteMask::RED
    });
    let rg = state(PixelFormat::RGBA8Snorm, |c| {
        c.write_mask = ColorWriteMask::RED.union(ColorWriteMask::GREEN)
    });
    let mut coverage = r32.clone();
    coverage.alpha_to_coverage = true;
    let mut depth = r32.clone();
    depth.depth_or_stencil = true;
    let late = ColorUse {
        early_tests: false,
        ..WRITES
    };
    for (what, inputs, use_) in [
        ("blending", &blended, WRITES),
        ("signed normalized write mask", &rg, WRITES),
        ("alpha to coverage", &coverage, WRITES),
        ("depth without early tests", &depth, late),
    ] {
        assert_eq!(form(inputs, false, use_, &p), ni, "{what}");
        let read = ColorUse {
            writes: false,
            ..use_
        };
        assert!(form(inputs, false, read, &p).is_ok(), "{what}, read only");
    }
    assert!(form(&red, false, WRITES, &p).is_ok());
    assert!(form(&depth, false, WRITES, &p).is_ok());
    // A write before the tests whose draws' plans check them.
    let tested = ColorUse {
        draw_tested: true,
        ..late
    };
    assert!(form(&depth, false, tested, &p).is_ok());
}

/// An 8-bit four-channel sRGB attachment takes the storage form through a view in `R32_UINT`,
/// declared `R32ui` (without a format where the device grants that) with `uint` texels, which the
/// function reads and writes as floats it decodes and encodes itself; a blend into one, a replacing
/// blend too, is not measured, so a written attachment under it is `NotImplemented`, a read-only
/// one is not.
#[test]
fn a_four_channel_srgb_attachment_takes_a_uint_view() {
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        for (pf, format) in [
            (PixelFormat::RGBA8UnormSRGB, vk::Format::R8G8B8A8_SRGB),
            (PixelFormat::BGRA8UnormSRGB, vk::Format::B8G8R8A8_SRGB),
        ] {
            for use_ in [WRITES, READS] {
                let view = form(&state(pf, |_| {}), false, use_, &p).expect("the storage form");
                assert_eq!(
                    (view.format, view.declared, view.numeric, view.uint_view),
                    (
                        format,
                        expected_declaration(
                            vk::Format::R32_UINT,
                            use_,
                            crate::spv::ImageFormat::R32ui,
                            &p
                        ),
                        SampledType::Uint,
                        Some(vk::Format::R32_UINT)
                    ),
                    "{stem} {pf:?} {use_:?}"
                );
                assert_eq!(view.read_class(), SampledType::Float, "{stem} {pf:?}");
            }
            // A mask that keeps channels merges their bytes.
            let masked = state(pf, |c| c.write_mask = ColorWriteMask::RED);
            assert!(form(&masked, false, WRITES, &p).is_ok(), "{stem} {pf:?}");
            for replaces in [false, true] {
                let blended = state(pf, |c| {
                    c.blending_enabled = true;
                    if !replaces {
                        c.destination_rgb_blend_factor = crate::metal::BlendFactor::One;
                    }
                });
                assert_eq!(form(&blended, false, WRITES, &p), ni, "{stem} {pf:?}");
                assert!(form(&blended, false, READS, &p).is_ok(), "{stem} {pf:?}");
            }
        }
    }
}

/// A blend of one and zero, added, for color and alpha stores the source as no blend does, so a
/// written attachment under it takes the storage form, but for a signed normalized format (its NaN
/// stores another code under the blend); one and zero for color alone is a blend, which a format
/// whose blend the shader does not compute refuses. A write whose
/// mask covers none of the format's channels stores nothing: no blend, alpha to coverage, one
/// sample's returned mask, or late depth test bears on it, but an `Unspecialized` mask stays
/// refused.
#[test]
fn a_replacing_blend_and_a_write_that_stores_nothing_take_the_storage_form() {
    use crate::metal::{BlendFactor, BlendOperation};
    let p = profile("moltenvk-m3", &[]);
    let ni = Err(Unsupported::not_implemented(Feature::FramebufferFetch));
    let replacing = |pf: PixelFormat| state(pf, |c| c.blending_enabled = true);
    for pf in [
        PixelFormat::RG16Float,
        PixelFormat::R32Float,
        PixelFormat::RGBA8Unorm,
    ] {
        assert!(form(&replacing(pf), false, WRITES, &p).is_ok(), "{pf:?}");
    }
    assert_eq!(
        form(&replacing(PixelFormat::RG8Snorm), false, WRITES, &p),
        ni
    );
    let alpha_adds = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.destination_alpha_blend_factor = BlendFactor::One;
    });
    assert_eq!(form(&alpha_adds, false, WRITES, &p), ni);
    let subtracts = state(PixelFormat::R32Float, |c| {
        c.blending_enabled = true;
        c.rgb_blend_operation = BlendOperation::Subtract;
    });
    assert_eq!(form(&subtracts, false, WRITES, &p), ni);
    // A write that stores nothing, under each state that refuses a store.
    let none = |edit: fn(&mut ColorAttachmentDescriptor)| {
        state(PixelFormat::RG16Float, |c| {
            c.write_mask = ColorWriteMask::NONE;
            edit(c);
        })
    };
    let blended = none(|c| {
        c.blending_enabled = true;
        c.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    });
    let mut coverage = none(|_| {});
    coverage.alpha_to_coverage = true;
    let mut depth = none(|_| {});
    depth.depth_or_stencil = true;
    let late = ColorUse {
        early_tests: false,
        ..WRITES
    };
    for (what, inputs, use_) in [
        ("blending", &blended, WRITES),
        ("alpha to coverage", &coverage, WRITES),
        ("depth without early tests", &depth, late),
    ] {
        let view = form(inputs, false, use_, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert!(view.stores_nothing(), "{what}");
    }
    let unspecialized = state(PixelFormat::RG16Float, |c| {
        c.write_mask = ColorWriteMask::UNSPECIALIZED
    });
    assert_eq!(form(&unspecialized, false, WRITES, &p), ni);
    assert!(form(&unspecialized, false, READS, &p).is_ok());
}

/// A blend into a 16-bit float attachment that the write stores takes the storage form with the
/// blend the shader computes (`StorageView::blend`), under either blend math, on every device that
/// orders fetch through the pixel interlock alone: for each 16-bit float format, the macOS 27
/// window server's one over one minus the source alpha among the factors and operations Metal's
/// model holds, the blend color's difference with its one minus (under fast math Metal's form, the
/// minuend's product fused and the subtrahend's rounded), and a write mask that keeps some
/// channels. A factor of the second source is
/// refused by name. A read-only attachment, a replacing blend under fast math and a write that
/// stores nothing take the form without one; a replacing blend without fast math keeps the
/// destination's product by zero (EV FBL-zf1-m3), so the shader computes it.
#[test]
fn a_sixteen_bit_float_blend_is_computed_in_the_storage_form() {
    use crate::metal::{BlendFactor as F, BlendOperation as O};
    let descriptor = |pf: PixelFormat, (s, d, op): (F, F, O), mask: ColorWriteMask| {
        let mut c = ColorAttachmentDescriptor::new(pf);
        c.blending_enabled = true;
        c.source_rgb_blend_factor = s;
        c.destination_rgb_blend_factor = d;
        c.rgb_blend_operation = op;
        c.source_alpha_blend_factor = s;
        c.destination_alpha_blend_factor = d;
        c.alpha_blend_operation = op;
        c.write_mask = mask;
        c
    };
    let of =
        |c: ColorAttachmentDescriptor| FramebufferFetchInputs::of([0], &[Some(c)], false, false, 1);
    let equations = [
        (F::One, F::OneMinusSourceAlpha, O::Add),
        (F::SourceAlpha, F::OneMinusSourceAlpha, O::Subtract),
        (
            F::DestinationColor,
            F::OneMinusDestinationAlpha,
            O::ReverseSubtract,
        ),
        (F::BlendColor, F::OneMinusBlendAlpha, O::Add),
        (F::SourceAlphaSaturated, F::One, O::Add),
        (F::One, F::One, O::Max),
    ];
    let refused = |gap| {
        Err(Unsupported::not_implemented(Feature::ExtendedRangeBlend(
            gap,
        )))
    };
    let partial = ColorWriteMask::RED.union(ColorWriteMask::ALPHA);
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx50-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for pf in [
            PixelFormat::R16Float,
            PixelFormat::RG16Float,
            PixelFormat::RGBA16Float,
        ] {
            for math in [BlendMath::Fast, BlendMath::Rounded] {
                let writes = ColorUse { math, ..WRITES };
                let reads = ColorUse { math, ..READS };
                for eq in equations {
                    for mask in [ColorWriteMask::ALL, partial] {
                        let what = format!("{stem} {pf:?} {math:?} {eq:?} {mask:?}");
                        let c = descriptor(pf, eq, mask);
                        let view = form(&of(c), false, writes, &p)
                            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                        assert_eq!(view.blend, BlendEquation::of(&c), "{what}");
                        assert_eq!(view.extended, None, "{what}");
                        let view = form(&of(c), false, reads, &p)
                            .unwrap_or_else(|e| panic!("{what}: read: {e:?}"));
                        assert_eq!(view.blend, None, "{what}: read");
                    }
                }
                let what = format!("{stem} {pf:?} {math:?}");
                let dual = descriptor(pf, (F::Source1Color, F::Zero, O::Add), ColorWriteMask::ALL);
                assert_eq!(
                    form(&of(dual), false, writes, &p),
                    refused(ExtendedRangeBlend::DualSource),
                    "{what}"
                );
                let differs = descriptor(
                    pf,
                    (F::BlendColor, F::OneMinusBlendColor, O::Subtract),
                    ColorWriteMask::ALL,
                );
                // Under fast math too: Metal's form, the minuend's product fused, the
                // subtrahend's rounded.
                assert!(
                    form(&of(differs), false, writes, &p).is_ok_and(|v| v.blend.is_some()),
                    "{what}"
                );
                let replacing = descriptor(pf, (F::One, F::Zero, O::Add), ColorWriteMask::ALL);
                let view = form(&of(replacing), false, writes, &p).expect(&what);
                let kept_zero = math == BlendMath::Rounded;
                let want = BlendEquation::of(&replacing).filter(|_| kept_zero);
                assert_eq!(view.blend, want, "{what}: replacing");
                let none = descriptor(pf, equations[0], ColorWriteMask::NONE);
                let view = form(&of(none), false, writes, &p).expect(&what);
                assert!(
                    view.stores_nothing() && view.blend.is_none(),
                    "{what}: none"
                );
            }
        }
    }
}

/// A write mask short of the format's channels keeps the others: the view names them, an empty set
/// where the mask covers every channel (or the function only reads), all four where it covers none,
/// so that the write stores nothing. A signed normalized format whose mask covers none of its
/// channels stores nothing either, so nothing of it is lost. A mask that keeps some of a format's
/// channels compiles as its class: the view names the class's channels (every channel but the first
/// left out, so the first kept) and takes the pipeline's own from the draw (`kept_at_draw`),
/// whichever such mask the descriptor sets; an sRGB format's view names the mask's own and takes
/// none from the draw.
#[test]
fn a_write_mask_names_the_channels_the_write_keeps() {
    use crate::lower::fragment::kept_channels;
    use crate::vk::ColorComponentFlags as C;
    let p = profile("moltenvk-m3", &[]);
    let kept = |pf: PixelFormat, mask: ColorWriteMask, use_: ColorUse| {
        form(&state(pf, |c| c.write_mask = mask), false, use_, &p).map(|view| {
            (
                view.kept,
                view.kept_at_draw,
                view.merges(),
                view.stores_nothing(),
            )
        })
    };
    let rg = ColorWriteMask::RED.union(ColorWriteMask::GREEN);
    let ba = ColorWriteMask::BLUE.union(ColorWriteMask::ALPHA);
    let rgba8 = PixelFormat::RGBA8Unorm;
    assert_eq!(kept(rgba8, rg, WRITES), Ok((C::R, true, true, false)));
    // Another partial mask: the same class, the same view.
    assert_eq!(
        kept(rgba8, ColorWriteMask::BLUE, WRITES),
        kept(rgba8, rg, WRITES)
    );
    // The pipeline's own kept channels, which the draw names.
    assert_eq!(kept_channels(rgba8, rg, true), ba);
    assert_eq!(
        kept(rgba8, ColorWriteMask::ALL, WRITES),
        Ok((C::empty(), false, false, false))
    );
    assert_eq!(
        kept(rgba8, rg, READS),
        Ok((C::empty(), false, false, false))
    );
    let green = ColorWriteMask::GREEN;
    assert_eq!(
        kept(PixelFormat::RG16Float, green, WRITES),
        Ok((C::R, true, true, false))
    );
    // Blue and alpha are no channels of `RG16Float`: the write stores nothing.
    assert_eq!(
        kept(PixelFormat::RG16Float, ba, WRITES),
        Ok((C::known_bits(), false, false, true))
    );
    // An sRGB format's write merges its view's bytes under the mask's own channels.
    assert_eq!(
        kept(PixelFormat::BGRA8UnormSRGB, rg, WRITES),
        Ok((C::B | C::A, false, true, false))
    );
    assert_eq!(
        kept(PixelFormat::RG8Snorm, ColorWriteMask::NONE, WRITES),
        Ok((C::known_bits(), false, false, true))
    );
    assert_eq!(
        kept(PixelFormat::RG8Snorm, ColorWriteMask::ALL, WRITES),
        Ok((C::empty(), false, false, false))
    );
}

/// A multisampled pipeline takes the per-sample form: a multisampled storage image where the device
/// has `shaderStorageImageMultisample` (RADV, NVIDIA), planes, one single-sample layer per sample,
/// where it does not (ANV, MoltenVK); a pipeline of one sample takes one sample's image. The
/// `SampleId` read needs `sampleRateShading`, named where the device lacks it. Its returned sample
/// mask is modelled (the write is gated on the invocation's own sample's bit), one sample's is not.
/// Beside a depth attachment the module declares early tests, whose image writes a driver with
/// `DriverWorkaround::EarlyTestsDropMultisampledImageWrites` (MoltenVK on Apple's GPU) drops at
/// more than one sample: the writes are refused there, naming the defect, and taken elsewhere.
#[test]
fn a_multisampled_state_takes_the_per_sample_form_or_names_what_it_needs() {
    use crate::device::profile::DriverWorkaround;
    use crate::error::Needs;
    let r32 = state(PixelFormat::R32Float, |_| {});
    // Without a depth or stencil attachment the module declares no early tests.
    let writes = ColorUse {
        early_tests: false,
        ..WRITES
    };
    let mut depth = r32.clone();
    depth.depth_or_stencil = true;
    for (stem, samples) in [
        ("amd-rdna3-radv", StorageSamples::Multisampled),
        ("nvidia-rtx50-linux", StorageSamples::Multisampled),
        ("moltenvk-m3", StorageSamples::Planes),
        ("intel-arl-anv", StorageSamples::Planes),
        ("intel-irisxe-anv", StorageSamples::Planes),
    ] {
        let p = profile(stem, &[]);
        for use_ in [READS, writes] {
            let r = form(&r32, true, use_, &p);
            assert_eq!(r.clone().map(|v| v.samples), Ok(samples), "{stem} {use_:?}");
            assert_eq!(
                declared(r),
                (
                    vk::Format::R32_SFLOAT,
                    expected_declaration(
                        vk::Format::R32_SFLOAT,
                        use_,
                        crate::spv::ImageFormat::R32f,
                        &p
                    ),
                    SampledType::Float
                ),
                "{stem} {use_:?}"
            );
        }
        assert_eq!(
            form(&r32, false, READS, &p).map(|v| v.samples),
            Ok(StorageSamples::One),
            "{stem}"
        );
        let early = form(&depth, true, WRITES, &p).map(|v| v.samples);
        if stem == "moltenvk-m3" {
            assert_eq!(
                early,
                Err(Unsupported::driver_defect(
                    Feature::FramebufferFetch,
                    DriverWorkaround::EarlyTestsDropMultisampledImageWrites
                ))
            );
        } else {
            assert_eq!(early, Ok(samples), "{stem}");
        }
        assert_eq!(
            form(&depth, true, READS, &p).map(|v| v.samples),
            Ok(samples),
            "{stem}"
        );
        assert!(form(&depth, false, WRITES, &p).is_ok(), "{stem}");
    }
    let no_storage_ms = profile(
        "amd-rdna3-radv",
        &[VkFeature::ShaderStorageImageMultisample],
    );
    assert_eq!(
        form(&r32, true, READS, &no_storage_ms).map(|v| v.samples),
        Ok(StorageSamples::Planes)
    );
    for stem in ["amd-rdna3-radv", "moltenvk-m3"] {
        let bare = profile(stem, &[VkFeature::SampleRateShading]);
        let needs = form(&r32, true, READS, &bare).expect_err("blocked").needs;
        assert_eq!(
            needs,
            Needs::AnyOf(vec![vec![Capability::Feature(
                VkFeature::SampleRateShading
            )]]),
            "{stem}"
        );
        assert!(form(&r32, false, READS, &bare).is_ok(), "{stem}");
    }
}

/// What the device lacks blocks the form with that capability: the fragment-stage store for a
/// written attachment, and, for an extended format without `shaderStorageImageExtendedFormats`,
/// that feature unless the device reads and writes the format without a format, which it then
/// declares.
#[test]
fn device_gaps_name_what_the_device_lacks() {
    use crate::error::Needs;
    let r32 = state(PixelFormat::R32Float, |_| {});
    let no_stores = profile("moltenvk-m3", &[VkFeature::FragmentStoresAndAtomics]);
    assert_eq!(
        form(&r32, false, WRITES, &no_stores)
            .expect_err("blocked")
            .needs,
        Needs::AnyOf(vec![vec![Capability::Feature(
            VkFeature::FragmentStoresAndAtomics
        )]])
    );
    assert!(form(&r32, false, READS, &no_stores).is_ok());
    let rg16 = state(PixelFormat::RG16Float, |_| {});
    let formatless = profile(
        "moltenvk-m3",
        &[VkFeature::ShaderStorageImageExtendedFormats],
    );
    assert_eq!(
        declared(form(&rg16, false, WRITES, &formatless)).1,
        crate::spv::ImageFormat::Unknown
    );
    assert!(form(&rg16, false, WRITES, &formatless)
        .expect("formatless")
        .formatless());
    let neither = profile(
        "moltenvk-m3",
        &[
            VkFeature::ShaderStorageImageExtendedFormats,
            VkFeature::ShaderStorageImageReadWithoutFormat,
        ],
    );
    assert_eq!(
        form(&rg16, false, READS, &neither)
            .expect_err("blocked")
            .needs,
        Needs::AnyOf(vec![vec![Capability::Feature(
            VkFeature::ShaderStorageImageExtendedFormats
        )]])
    );
}

/// No attachment at the color read refuses the pipeline whatever the form.
#[test]
fn a_missing_attachment_is_a_descriptor_error() {
    let p = profile("moltenvk-m3", &[]);
    let none = FramebufferFetchInputs::of([0], &[None], false, false, 1);
    assert_eq!(
        storage_form(0, &none, false, READS, &p),
        Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
            index: 0
        }))
    );
}

/// An extended-range format's blend (`extended_range_blend`): taken where the device orders fetch
/// through the pixel interlock alone and the storage form holds, for every form Metal's model
/// blends and its `_sRGB` twin, and where it orders fetch through rasterization-order input
/// attachments, over which the function blends; the interlock named where it has neither; refused
/// for a padded form, whose blends are not measured, and for any other format.
#[test]
fn an_extended_range_blend_follows_the_ordered_storage_form() {
    let blended = [
        PixelFormat::BGR10XR,
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        PixelFormat::BGRA10XRSRGB,
        PixelFormat::RGB10A82PXR10,
        PixelFormat::RGB10A82PXR10SRGB,
    ];
    for stem in [
        "moltenvk-m3",
        "moltenvk-m4",
        "amd-rdna3-radv",
        "amd-780m-radv",
        "intel-arl-anv",
        "intel-irisxe-anv",
        "nvidia-rtx5080-laptop",
        "nvidia-pascal-linux",
    ] {
        let p = profile(stem, &[]);
        assert!(interlock_orders(&p), "{stem}");
        for pf in blended {
            assert_eq!(extended_range_blend(pf, &p), Ok(()), "{stem}: {pf:?}");
        }
        for pf in [PixelFormat::R10UnormX6, PixelFormat::RG10UnormX12] {
            assert!(extended_range_blend(pf, &p).is_err(), "{stem}: {pf:?}");
        }
        assert_eq!(
            extended_range_blend(PixelFormat::RGBA16Float, &p),
            Err(Unsupported::not_implemented(Feature::Format)),
            "{stem}"
        );
    }
    for stem in ["lavapipe", "gate-lavapipe-x86_64", "swiftshader"] {
        let p = profile(stem, &[]);
        assert!(!interlock_orders(&p), "{stem}");
        for pf in blended {
            assert_eq!(extended_range_blend(pf, &p), Ok(()), "{stem}: {pf:?}");
        }
        for pf in [PixelFormat::R10UnormX6, PixelFormat::RG10UnormX12] {
            assert_eq!(
                extended_range_blend(pf, &p),
                Err(Unsupported::not_implemented(Feature::Format)),
                "{stem}: {pf:?}"
            );
        }
    }
    let interlock = Unsupported::needs(
        Feature::FramebufferFetch,
        Capability::Feature(VkFeature::FragmentShaderPixelInterlock),
    );
    for stem in ["vk12-floor", "roadmap-2022"] {
        let p = profile(stem, &[]);
        assert_eq!(
            extended_range_blend(PixelFormat::BGRA10XR, &p),
            Err(interlock.clone()),
            "{stem}"
        );
    }
    // The storage form's own device needs, named where the device lacks one.
    let p = profile("moltenvk-m4", &[VkFeature::FragmentStoresAndAtomics]);
    assert!(extended_range_blend(PixelFormat::BGRA10XR, &p)
        .is_err_and(|u| u.feature == Feature::FramebufferFetch));
}
