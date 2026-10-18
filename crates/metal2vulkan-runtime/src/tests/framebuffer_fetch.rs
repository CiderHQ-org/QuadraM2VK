//! Which framebuffer-fetch pipelines the runtime makes: the compile layer's ordered forms, never
//! the unordered one `AcceptUnordered` admits; and which color render targets may take `STORAGE`
//! usage for the interlock form, or the sample planes' copies' usage, and the answers remembered
//! for them (`framebuffer_fetch.rs`).
//!
//! The graphics states are the compile layer's three fetch outputs as
//! `pipeline::compile::tests::a_framebuffer_fetch_pipeline_is_ordered_exactly_where_the_device_orders_it`
//! compiles them from `fragment_fetch.ll`'s `fetch_scalar` into an `R32Float` attachment:
//! ordered input attachments on a device with rasterization-order access, the interlocked
//! storage attachment on one with the pixel interlock alone, and, under `AcceptUnordered`,
//! unordered input attachments on a synthetic profile with neither feature (or a blending
//! attachment on the interlock alone). Compiling them here would assemble the fixture through
//! test infrastructure the runtime may not reach (`api_surface`'s public-layer check).

use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use crate::error::Error;
use crate::framebuffer_fetch::{
    fetch_candidate, ordered, planes_candidate, planes_copies, planes_copies_gap,
    storage_candidate, stores_samples, through_inputs, through_multisampled_storage,
    through_planes, through_storage, UsageAnswers,
};
use crate::image_plan::{plan_image, ImagePlan};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::device::{FormatUsage, VkFeature};
use metal2vulkan::error::{Capability, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentState, GraphicsState, InputAttachmentUse, SampledType, StorageAttachmentUse,
    StorageAttachmentView,
};
use metal2vulkan::vk::ImageCreateFlags;
use metal2vulkan::vk::{
    ColorComponentFlags, Format, ImageUsageFlags, PrimitiveTopology, SampleCountFlags,
};

/// The graphics state of a fetching pipeline: `input` for an input attachment, ordered by
/// rasterization-order access where `access`; otherwise the storage attachment.
fn fetching(input: bool, access: bool) -> GraphicsState {
    let write_mask = ColorComponentFlags::R
        | ColorComponentFlags::G
        | ColorComponentFlags::B
        | ColorComponentFlags::A;
    let color = ColorAttachmentState::new(Format::R32_SFLOAT, write_mask, None);
    let mut state = GraphicsState::new(PrimitiveTopology::TRIANGLE_LIST, SampleCountFlags::COUNT_1);
    state.color_attachments = vec![input.then_some(color)];
    state.input_attachments = input
        .then_some(InputAttachmentUse::new(0, 0, SampledType::Float))
        .into_iter()
        .collect();
    state.rasterization_order_attachment_access = input && access;
    state.storage_attachments = (!input)
        .then_some(StorageAttachmentUse::new(
            0,
            Format::R32_SFLOAT,
            true,
            false,
            StorageAttachmentView::Attachment,
        ))
        .into_iter()
        .collect();
    state
}

/// The ordered forms are made, as is a pipeline that fetches nothing; the unordered input
/// attachment is refused, naming the access that orders it.
#[test]
fn only_ordered_framebuffer_fetch_is_made() {
    let mut plain = fetching(true, false);
    plain.input_attachments.clear();
    assert_eq!(ordered(&plain), Ok(()));
    assert_eq!(ordered(&fetching(true, true)), Ok(()));
    assert_eq!(ordered(&fetching(false, false)), Ok(()));
    match ordered(&fetching(true, false)) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::FramebufferFetch,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::RasterizationOrderColorAttachmentAccess
                )]])
            )
        ),
        other => panic!("the unordered fetch is made: {other:?}"),
    }
}

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
/// copy format shares, is offered no planes copies' usage, on a device whose fetch reads planes
/// too: no view in `R8_UINT` is listed for it.
#[test]
fn an_a8_target_is_offered_no_planes_copies() {
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
        planes += usize::from(through_planes(profile, 4));
        assert_eq!(PlanesCopy::for_format(target.format), None, "{name}");
        assert_eq!(planes_candidate(&target, profile), None, "{name}");
        assert!(!planes_copies(&target), "{name}");
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
