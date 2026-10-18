//! Which color render targets may take `STORAGE` usage for framebuffer fetch's interlock form, or
//! the sample planes' copies' usage, and the answers remembered for them (`framebuffer_fetch.rs`).

use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use crate::framebuffer_fetch::{
    fetch_candidate, planes_candidate, planes_copies, planes_copies_gap, storage_candidate,
    stores_samples, through_inputs, through_multisampled_storage, through_planes, through_storage,
    UsageAnswers,
};
use crate::image_plan::{plan_image, ImagePlan};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags};

/// A checked layout of a 2D texture of `pf`, `width` by 8, with `samples` (a multisample type
/// above 1).
fn layout(pf: PixelFormat, width: u64, samples: u64) -> TextureLayout {
    let checked = TextureDescriptor {
        texture_type: if samples > 1 {
            TextureType::Type2DMultisample
        } else {
            TextureType::Type2D
        },
        pixel_format: pf,
        width,
        height: 8,
        sample_count: samples,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .expect("a valid descriptor");
    TextureLayout {
        pixel_format: pf,
        ..checked
    }
}

/// On every committed profile, a device's fetch is one form or the other or neither, and only a
/// device whose fetch is the interlock form offers `STORAGE` to a color render target that is
/// not memoryless, of a format the device stores, without it yet: a single-sampled one, and a
/// 4x one where the fetch reads the attachment's own multisampled image and the device stores
/// images of 4 samples, never where it reads sample planes. The candidate is the plan with
/// `STORAGE` added and nothing else changed.
#[test]
fn storage_is_offered_to_the_interlock_forms_color_targets() {
    let (mut offered, mut multisampled) = (0, 0);
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        let fragment = profile.fragment();
        assert!(
            !(through_inputs(profile) && through_storage(profile)),
            "{name}"
        );
        assert_eq!(
            through_storage(profile),
            fragment.pixel_interlock && !fragment.rasterization_order_color_access,
            "{name}"
        );
        let plan = |pf, samples, usage, memoryless| {
            plan_image(&layout(pf, 8, samples), usage, memoryless, &facts).ok()
        };
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA16Float,
            PixelFormat::R32Float,
            PixelFormat::RGBA32Uint,
        ] {
            let Some(target) = plan(pf, 1, TextureUsage::RENDER_TARGET, false) else {
                continue;
            };
            let stores = profile.supports_format(target.format, FormatUsage::StorageImage);
            let candidate = storage_candidate(&target, profile);
            assert_eq!(
                candidate.is_some(),
                through_storage(profile)
                    && stores
                    && !target.usage.contains(ImageUsageFlags::STORAGE),
                "{name} {pf:?}"
            );
            if let Some(candidate) = candidate {
                offered += 1;
                assert_eq!(
                    candidate,
                    ImagePlan {
                        usage: target.usage | ImageUsageFlags::STORAGE,
                        ..target.clone()
                    },
                    "{name} {pf:?}"
                );
            }
            if let Some(target) = plan(pf, 4, TextureUsage::RENDER_TARGET, false) {
                let candidate = storage_candidate(&target, profile);
                assert_eq!(
                    candidate.is_some(),
                    through_multisampled_storage(profile, 4)
                        && stores_samples(profile, 4)
                        && stores
                        && !target.usage.contains(ImageUsageFlags::STORAGE),
                    "{name} {pf:?} 4x"
                );
                assert!(
                    candidate.is_none() || !through_planes(profile, 4),
                    "{name} {pf:?} 4x"
                );
                if let Some(candidate) = candidate {
                    multisampled += 1;
                    assert_eq!(
                        candidate,
                        ImagePlan {
                            usage: target.usage | ImageUsageFlags::STORAGE,
                            ..target.clone()
                        },
                        "{name} {pf:?} 4x"
                    );
                }
            }
            let refused = [
                plan(pf, 4, TextureUsage::RENDER_TARGET, true),
                plan(pf, 1, TextureUsage::RENDER_TARGET, true),
                plan(pf, 1, TextureUsage::SHADER_READ, false),
            ];
            for other in refused.iter().flatten() {
                assert_eq!(storage_candidate(other, profile), None, "{name} {pf:?}");
            }
        }
    }
    assert!(offered > 0, "an interlock profile offers storage");
    assert!(
        multisampled > 0,
        "a profile with multisampled storage images offers it at 4x"
    );
}

/// A multisampled `A8Unorm` render target held as `A8_UNORM`, whose class ("8-bit alpha") no
/// unsigned copy format shares, takes the planes copies in its own format
/// (`PlanesCopy::Alpha8`), on a device whose fetch reads planes: its candidate is the plan sampled,
/// neither mutable nor listing another format, which the copies then read and write directly.
#[test]
fn an_a8_target_takes_the_planes_copies_in_its_own_format() {
    let mut planes = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        let l = layout(PixelFormat::A8Unorm, 8, 4);
        let Ok(target) = plan_image(&l, TextureUsage::RENDER_TARGET, false, &facts) else {
            continue;
        };
        if target.format != Format::A8_UNORM {
            continue;
        }
        assert_eq!(PlanesCopy::for_format(target.format), None, "{name}");
        assert_eq!(
            PlanesCopy::for_planes(target.format),
            Some(PlanesCopy::Alpha8),
            "{name}"
        );
        if !through_planes(profile, 4) || planes_copies_gap(target.format, profile).is_some() {
            continue;
        }
        planes += 1;
        let candidate = planes_candidate(&target, profile).expect("the planes' usage");
        assert_eq!(
            candidate,
            ImagePlan {
                usage: target.usage | ImageUsageFlags::SAMPLED,
                ..target.clone()
            },
            "{name}"
        );
        assert!(planes_copies(&candidate), "{name}");
        assert_eq!(planes_candidate(&candidate, profile), None, "{name}");
    }
    assert!(planes > 0, "a planes profile holds A8Unorm as A8_UNORM");
}

/// Only a device whose fetch reads sample planes (the interlock form without multisampled storage
/// images) offers the planes' copies' usage to a color render target: a multisampled one that is
/// not memoryless, of a format whose copy format the device samples and renders and which it
/// samples itself. The candidate is the plan sampled and mutable, listing its own format first
/// and the copy format, with nothing else changed; the copies then read and write it directly,
/// and it is offered nothing more. A single-sampled target's fetch candidate is its `STORAGE`
/// one, and so is a 4x one's on a device that reads no sample planes.
#[test]
fn the_planes_usage_is_offered_to_multisampled_color_targets_where_the_fetch_reads_planes() {
    let mut offered = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        assert_eq!(
            through_planes(profile, 4),
            through_storage(profile) && !profile.images().storage_multisample,
            "{name}"
        );
        assert!(!through_planes(profile, 1), "{name}");
        let plan = |pf, samples, usage, memoryless| {
            plan_image(&layout(pf, 8, samples), usage, memoryless, &facts).ok()
        };
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::RG8Unorm,
            PixelFormat::RGBA16Float,
            PixelFormat::R32Float,
            PixelFormat::RGBA32Uint,
        ] {
            if let Some(single) = plan(pf, 1, TextureUsage::RENDER_TARGET, false) {
                assert_eq!(
                    fetch_candidate(&single, profile),
                    storage_candidate(&single, profile),
                    "{name} {pf:?}"
                );
                assert_eq!(planes_candidate(&single, profile), None, "{name} {pf:?}");
            }
            if let Some(memoryless) = plan(pf, 4, TextureUsage::RENDER_TARGET, true) {
                assert_eq!(
                    planes_candidate(&memoryless, profile),
                    None,
                    "{name} {pf:?}"
                );
            }
            let Some(target) = plan(pf, 4, TextureUsage::RENDER_TARGET, false) else {
                continue;
            };
            let copy = PlanesCopy::for_format(target.format).expect("a copy format");
            let candidate = planes_candidate(&target, profile);
            assert_eq!(
                candidate.is_some(),
                through_planes(profile, 4)
                    && planes_copies_gap(target.format, profile).is_none()
                    && !planes_copies(&target),
                "{name} {pf:?}"
            );
            assert_eq!(
                fetch_candidate(&target, profile),
                if through_planes(profile, 4) {
                    candidate.clone()
                } else {
                    storage_candidate(&target, profile)
                },
                "{name} {pf:?}"
            );
            let Some(candidate) = candidate else {
                continue;
            };
            offered += 1;
            let mut formats = target.view_formats.clone();
            if formats.is_empty() {
                formats.push(target.format);
            }
            if !formats.contains(&copy.format()) {
                formats.push(copy.format());
            }
            assert_eq!(
                candidate,
                ImagePlan {
                    usage: target.usage | ImageUsageFlags::SAMPLED,
                    flags: target.flags | ImageCreateFlags::MUTABLE_FORMAT,
                    view_formats: formats,
                    ..target.clone()
                },
                "{name} {pf:?}"
            );
            assert_eq!(candidate.view_formats[0], target.format, "{name} {pf:?}");
            assert!(planes_copies(&candidate), "{name} {pf:?}");
            assert_eq!(planes_candidate(&candidate, profile), None, "{name} {pf:?}");
        }
    }
    assert!(offered > 0, "a planes profile offers the usage");
}

/// The answers keep the latest 64 image descriptions: the oldest goes first, and an answer is
/// found by the whole description and the candidate's: one description answers each usage added
/// to it apart.
#[test]
fn storage_answers_keep_the_latest_descriptions() {
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(_, _, device)| through_storage(device.profile()))
        .expect("an interlock profile");
    let facts = facts(&desc, &device);
    let plan = |width| {
        plan_image(
            &layout(PixelFormat::RGBA8Unorm, width, 1),
            TextureUsage::RENDER_TARGET,
            false,
            &facts,
        )
        .expect("a render target")
    };
    let with = |width, usage| {
        let mut candidate = plan(width);
        candidate.usage |= usage;
        candidate
    };
    let storage = |width| with(width, ImageUsageFlags::STORAGE);
    let input = |width| with(width, ImageUsageFlags::INPUT_ATTACHMENT);
    let mut answers = UsageAnswers::default();
    for width in 1..=64 {
        answers.remember(plan(width), storage(width), width % 2 == 0);
    }
    assert_eq!(answers.answer(&plan(1), &storage(1)), Some(false));
    assert_eq!(answers.answer(&plan(64), &storage(64)), Some(true));
    assert_eq!(answers.answer(&plan(65), &storage(65)), None);
    assert_eq!(answers.answer(&plan(64), &input(64)), None);
    answers.remember(plan(65), storage(65), true);
    assert_eq!(answers.answer(&plan(1), &storage(1)), None);
    assert_eq!(answers.answer(&plan(2), &storage(2)), Some(true));
    assert_eq!(answers.answer(&plan(65), &storage(65)), Some(true));
    // One description's answers for two usages are two answers: the oldest, its `STORAGE`
    // one, goes for its input attachment one.
    answers.remember(plan(2), input(2), false);
    assert_eq!(answers.answer(&plan(2), &input(2)), Some(false));
    assert_eq!(answers.answer(&plan(2), &storage(2)), None);
    assert_eq!(answers.answer(&plan(3), &storage(3)), Some(false));
}

/// A multisampled depth/stencil render target takes `SAMPLED` as a candidate exactly where the
/// device's colour copies cannot move its aspects and the device samples its format, so the
/// copies without colour copies may read it (`depth_stencil_scratch::sampled_candidate`): never
/// at one sample, never transient (memoryless), never where the copies move it; the candidate is
/// the plan with `SAMPLED` added and nothing else changed.
#[test]
fn multisampled_depth_stencil_targets_are_offered_sampled_where_the_copies_read_them() {
    use crate::depth_stencil_scratch::sampled_candidate_on;
    let mut offered = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        for pf in [
            PixelFormat::Depth32Float,
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
        ] {
            for (samples, memoryless) in [(1, false), (4, false), (4, true)] {
                let usage = TextureUsage::RENDER_TARGET;
                let Ok(target) = plan_image(&layout(pf, 8, samples), usage, memoryless, &facts)
                else {
                    continue;
                };
                assert!(sampled_candidate_on(&target, profile, false).is_none());
                let candidate = sampled_candidate_on(&target, profile, true);
                let samples_it = profile.supports_format(target.format, FormatUsage::SampledImage);
                let vk_usage = target.usage;
                let open = vk_usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    && !vk_usage.intersects(
                        ImageUsageFlags::TRANSIENT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    );
                assert_eq!(
                    candidate.is_some(),
                    samples > 1 && open && samples_it,
                    "{name} {pf:?} {samples}x memoryless {memoryless}: {vk_usage:?}"
                );
                if let Some(candidate) = candidate {
                    let mut expected = target.clone();
                    expected.usage |= ImageUsageFlags::SAMPLED;
                    assert_eq!(candidate, expected, "{name} {pf:?}");
                    offered += 1;
                }
                let sampled = plan_image(
                    &layout(pf, 8, samples),
                    usage | TextureUsage::SHADER_READ,
                    memoryless,
                    &facts,
                );
                if let Ok(sampled) = sampled {
                    assert!(
                        sampled_candidate_on(&sampled, profile, true).is_none(),
                        "{name} {pf:?}: already sampled"
                    );
                }
            }
        }
    }
    assert!(offered > 0);
}

/// A single-sample depth render target is offered sampled usage where its driver keeps a depth
/// image's hierarchical depth in `GENERAL` only for a sampled image
/// (`image_plan::hiz_candidate`, `DriverFact::SampledDepthKeepsHiZ`): never without the fact,
/// never multisampled, never transient (memoryless), never a stencil alone, never already
/// sampled; the candidate is the plan with `SAMPLED` added and nothing else changed. The committed
/// Iris Xe of RPL-P's profile has the fact, and its `Depth32Float` target takes the usage.
#[test]
fn single_sample_depth_targets_are_offered_sampled_where_the_driver_keeps_hiz_so() {
    use crate::image_plan::hiz_candidate;
    use metal2vulkan::device::DriverFact;
    let mut offered = Vec::new();
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let profile = device.profile();
        let keeps = desc
            .driver_facts()
            .contains(&DriverFact::SampledDepthKeepsHiZ);
        for pf in [
            PixelFormat::Depth16Unorm,
            PixelFormat::Depth32Float,
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
        ] {
            for (samples, memoryless) in [(1, false), (1, true), (4, false)] {
                let usage = TextureUsage::RENDER_TARGET;
                let Ok(target) = plan_image(&layout(pf, 8, samples), usage, memoryless, &facts)
                else {
                    continue;
                };
                assert!(hiz_candidate(&target, profile, false).is_none());
                let candidate = hiz_candidate(&target, profile, keeps);
                let samples_it = profile.supports_format(target.format, FormatUsage::SampledImage);
                let vk_usage = target.usage;
                let open = vk_usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    && !vk_usage.intersects(
                        ImageUsageFlags::TRANSIENT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    );
                let depth = pf != PixelFormat::Stencil8;
                assert_eq!(
                    candidate.is_some(),
                    keeps && samples == 1 && depth && open && samples_it,
                    "{name} {pf:?} {samples}x memoryless {memoryless}: {vk_usage:?}"
                );
                if let Some(candidate) = candidate {
                    let mut expected = target.clone();
                    expected.usage |= ImageUsageFlags::SAMPLED;
                    assert_eq!(candidate, expected, "{name} {pf:?}");
                    offered.push((name.clone(), pf));
                }
                let sampled = plan_image(
                    &layout(pf, 8, samples),
                    usage | TextureUsage::SHADER_READ,
                    memoryless,
                    &facts,
                );
                if let Ok(sampled) = sampled {
                    assert!(
                        hiz_candidate(&sampled, profile, true).is_none(),
                        "{name} {pf:?}: already sampled"
                    );
                }
            }
        }
    }
    assert!(offered.contains(&(
        "intel-irisxe-rplp-anv".to_owned(),
        PixelFormat::Depth32Float
    )));
}
