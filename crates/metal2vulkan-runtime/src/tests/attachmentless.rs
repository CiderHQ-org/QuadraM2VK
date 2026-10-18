use super::image_plan::committed;
use crate::attachmentless::keeps;
use crate::color_list::ColorList;
use crate::framebuffer_fetch::through_planes;
use crate::render_target::PassShape;
use metal2vulkan::vk::Format;

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

/// A device keeps the pass of a shape without any attachment, at every sample count and view
/// mask (a multiview instance's), and of no other: not of one with a color or a depth/stencil
/// attachment, and not of the sample planes' pass, whose subpass leaves out every attachment the
/// pass still describes (it keeps its anchor instead, `render_target.rs`, on a device that reads
/// sample planes).
#[test]
fn only_a_pass_without_attachments_is_kept() {
    for samples in [1, 2, 4, 8] {
        assert!(keeps(&shape(samples, &[], false, &[])), "{samples}");
        for view_mask in [0b11, 0b1111] {
            let viewed = shape(samples, &[], false, &[]).with_view_mask(view_mask);
            assert!(keeps(&viewed), "{samples} {view_mask:#b}");
        }
    }
    assert!(!keeps(
        &shape(4, &[Some(Format::R32_SFLOAT)], false, &[]).with_view_mask(0b11)
    ));
    let float = Some(Format::R32_SFLOAT);
    assert!(!keeps(&shape(4, &[float], false, &[])));
    assert!(!keeps(&shape(4, &[], true, &[])));
    let planes = shape(4, &[float], false, &[0]);
    let (_, _, device) = committed()
        .into_iter()
        .find(|(_, _, device)| through_planes(device.profile(), 4))
        .expect("a committed profile that reads sample planes");
    assert_eq!(planes.anchor(device.profile()), Some(0));
    assert!(!keeps(&planes));
    let fetching = PassShape {
        fetch_inputs: true,
        ..shape(1, &[float], false, &[])
    };
    assert!(!keeps(&fetching));
}

/// The draws' subpass references an attachment at a location where its routing names a color
/// the pass has (each location its own color where it names none), and none at a color its
/// subpass leaves out: a pipeline keeps such a color unwritten there (`pipeline/graphics.rs`).
#[test]
fn a_routed_subpass_references_the_color_each_location_names() {
    let float = Some(Format::R8G8B8A8_UNORM);
    let own = shape(1, &[None, float], false, &[]);
    assert_eq!(
        (0..3).map(|l| own.references(l)).collect::<Vec<_>>(),
        [false, true, false]
    );
    let routed = shape(1, &[None, float], false, &[])
        .with_locations(&[Some(1), None])
        .unwrap();
    assert_eq!(
        (0..3).map(|l| routed.references(l)).collect::<Vec<_>>(),
        [true, false, false]
    );
    assert_eq!(routed.subpass_colors(), 2);
    let left_out = shape(1, &[float, float], false, &[1]);
    assert_eq!(
        (0..2).map(|l| left_out.references(l)).collect::<Vec<_>>(),
        [true, false]
    );
}
