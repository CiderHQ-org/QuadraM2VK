//! The order of a pass's sample-planes steps (`sample_planes.rs`): which copies run before each
//! render pass instance and after the last, from the instances' left-out attachments and the
//! attachments' load and store actions; which attachment the fetching draws' subpass keeps as
//! its anchor (`render_target.rs`); and which formats' copy views keep their components.

use super::image_plan::committed;
use crate::color_list::ColorList;
use crate::framebuffer_fetch::{through_multisampled_storage, through_planes};
use crate::render_target::PassShape;
use crate::render_values::{LoadAction, StoreAction};
use crate::sample_planes::{ordered_steps, PlanesAttachment, PlanesStep};
use crate::view_plan::keeps_components;
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::vk::Format;
use PlanesStep::{Clear, Merge, Split};

/// Color attachment `index` with sample planes, loaded and stored by `load` and `store`, which
/// the device resolves.
fn attachment(index: u32, load: LoadAction, store: StoreAction) -> PlanesAttachment {
    PlanesAttachment {
        index,
        load,
        store,
        exact_resolve: false,
    }
}

/// A pass of fetching draws alone: its first instance runs the load action on the planes (a
/// split, a clear, or nothing for `DontCare`), and the merge after the last runs for every store
/// action but `DontCare`.
#[test]
fn the_first_instance_loads_the_planes_and_the_last_stores_them() {
    for (load, before) in [
        (LoadAction::Load, vec![Split(0)]),
        (LoadAction::Clear, vec![Clear(0)]),
        (LoadAction::DontCare, vec![]),
    ] {
        for (store, after) in [
            (StoreAction::Store, vec![Merge(0)]),
            (StoreAction::MultisampleResolve, vec![Merge(0)]),
            (StoreAction::StoreAndMultisampleResolve, vec![Merge(0)]),
            (StoreAction::DontCare, vec![]),
        ] {
            let steps = ordered_steps(&[attachment(0, load, store)], &[&[0]]);
            assert_eq!(steps, vec![before.clone(), after], "{load:?} {store:?}");
        }
    }
}

/// An attachment resolved by the exact resolve, which reads the planes, merges after the last
/// instance only where Metal keeps its samples: not for `MultisampleResolve`, for
/// `StoreAndMultisampleResolve`. Where the last instance draws it, nothing merges after.
#[test]
fn an_exact_resolve_reads_the_planes_without_a_merge() {
    for (store, after) in [
        (StoreAction::MultisampleResolve, vec![]),
        (StoreAction::StoreAndMultisampleResolve, vec![Merge(0)]),
    ] {
        let exact = PlanesAttachment {
            exact_resolve: true,
            ..attachment(0, LoadAction::Clear, store)
        };
        let steps = ordered_steps(&[exact], &[&[0]]);
        assert_eq!(steps, vec![vec![Clear(0)], after], "{store:?}");
        let steps = ordered_steps(&[exact], &[&[0], &[]]);
        assert_eq!(
            steps,
            vec![vec![Clear(0)], vec![Merge(0)], vec![]],
            "{store:?}"
        );
    }
}

/// An attachment moves into the planes before each instance that leaves it out after one that
/// did not, and back before each that includes it after one that left it out; only the first
/// instance's load action applies, and the last instance's steps after it.
#[test]
fn the_planes_follow_the_instances_that_leave_their_attachment_out() {
    let fetched = attachment(0, LoadAction::Clear, StoreAction::MultisampleResolve);
    // Fetch, draw, fetch twice: the 4x memoryless pass of `runtime_lane::framebuffer_fetch`.
    let steps = ordered_steps(&[fetched], &[&[0], &[], &[0]]);
    assert_eq!(
        steps,
        vec![
            vec![Clear(0)],
            vec![Merge(0)],
            vec![Split(0)],
            vec![Merge(0)]
        ]
    );
    // Draw, then fetch: the first instance's clear is its own load operation.
    let steps = ordered_steps(&[fetched], &[&[], &[0]]);
    assert_eq!(steps, vec![vec![], vec![Split(0)], vec![Merge(0)]]);
    // Fetch, then draw: the last instance stores the attachment itself.
    let steps = ordered_steps(&[fetched], &[&[0], &[]]);
    assert_eq!(steps, vec![vec![Clear(0)], vec![Merge(0)], vec![]]);
}

/// Each attachment's steps follow its own index among the instances' left-out attachments, in
/// attachment order at each position.
#[test]
fn each_attachment_has_its_own_steps() {
    let first = attachment(0, LoadAction::Load, StoreAction::Store);
    let second = attachment(2, LoadAction::DontCare, StoreAction::DontCare);
    let steps = ordered_steps(&[first, second], &[&[0, 2], &[2], &[0]]);
    assert_eq!(
        steps,
        vec![
            vec![Split(0)],
            vec![Merge(0)],
            vec![Split(0), Merge(2)],
            vec![Merge(0)],
        ]
    );
    // An attachment no instance leaves out has no step.
    let steps = ordered_steps(&[first, second], &[&[0], &[0]]);
    assert_eq!(steps, vec![vec![Split(0)], vec![], vec![Merge(0)]]);
}

/// A pass of `samples` samples with `colors`, and a depth/stencil attachment where `depth`, whose
/// draws' subpass leaves out `storage`.
fn shape(samples: u32, colors: &[Option<Format>], depth: bool, storage: &[u32]) -> PassShape {
    PassShape {
        colors: ColorList::of(colors.iter().copied()).unwrap(),
        depth_stencil: depth.then_some(Format::D32_SFLOAT),
        samples,
        color_resolves: ColorList::of(colors.iter().map(|_| None)).unwrap(),
        depth_stencil_resolve: None,
        fetch_inputs: false,
        no_inputs: ColorList::new(),
        storage: ColorList::of(storage.iter().copied()).unwrap(),
        view_mask: 0,
        locations: ColorList::new(),
    }
}

/// A multisampled subpass that leaves out every attachment it has keeps the first as its anchor
/// where they are sample planes (a device without multisampled storage images); one with another
/// attachment, of one sample, or reading input attachments keeps none, and so does one whose
/// draws write the attachments' own multisampled images (a device with multisampled storage
/// images), on every committed profile.
#[test]
fn a_subpass_left_without_attachments_keeps_its_first_planes() {
    let float = Some(Format::R32_SFLOAT);
    let quad = Some(Format::R8G8B8A8_UNORM);
    let (mut planes, mut own) = (0, 0);
    for (name, _, device) in committed() {
        let profile = device.profile();
        let anchors = through_planes(profile, 4);
        planes += usize::from(anchors);
        own += usize::from(through_multisampled_storage(profile, 4));
        let first = |index: u32| anchors.then_some(index);
        let anchor = |shape: PassShape| shape.anchor(profile);
        assert_eq!(anchor(shape(4, &[float], false, &[0])), first(0), "{name}");
        assert_eq!(
            anchor(shape(4, &[None, float, quad], false, &[1, 2])),
            first(1),
            "{name}"
        );
        assert_eq!(
            anchor(shape(4, &[float, quad], false, &[1])),
            None,
            "{name}"
        );
        assert_eq!(anchor(shape(4, &[float], true, &[0])), None, "{name}");
        assert_eq!(anchor(shape(1, &[float], false, &[0])), None, "{name}");
        assert_eq!(anchor(shape(4, &[float], false, &[])), None, "{name}");
        let inputs = PassShape {
            fetch_inputs: true,
            ..shape(4, &[float], false, &[0])
        };
        assert_eq!(anchor(inputs), None, "{name}");
    }
    assert!(planes > 0, "a planes profile");
    assert!(own > 0, "a profile with multisampled storage images");
}

/// The copies' views keep the components of a format of one 8-, 16- or 32-bit component, or of
/// two or four 32-bit ones, and reinterpret every other's, whose planes a device without view
/// reinterpretation refuses (`sample_planes.rs`).
#[test]
fn the_copy_views_keep_the_components_of_one_component_or_of_32_bit_ones() {
    let keeps = |format: Format| {
        let copy = PlanesCopy::for_format(format).expect("a copy format");
        keeps_components(format, copy.format())
    };
    for format in [
        Format::R8_UNORM,
        Format::R16_SFLOAT,
        Format::R32_SFLOAT,
        Format::R32G32_SFLOAT,
        Format::R32G32B32A32_SFLOAT,
    ] {
        assert!(keeps(format), "{format:?}");
    }
    for format in [
        Format::R8G8_UNORM,
        Format::R8G8B8A8_UNORM,
        Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT,
        Format::A2B10G10R10_UNORM_PACK32,
        Format::B10G11R11_UFLOAT_PACK32,
    ] {
        assert!(!keeps(format), "{format:?}");
    }
}
