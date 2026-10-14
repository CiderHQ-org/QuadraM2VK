//! The box a bound linear attachment rendered apart is copied in and merged back over
//! (`linear_attachment::choose`): what a draw writes of each color ([`writes`],
//! [`pass_writes`]) and the box the pass's draws write each in ([`Coverage`]).

use super::super::color_reformat::Routing;
use super::super::linear_attachment::{pass_writes, writes, ColorWrites, Coverage};
use ash::vk;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::ColorAttachmentState;
use metal2vulkan::vk::{ColorComponentFlags as Mask, Format};

const BLENDED: Option<bool> = Some(true);
const OPAQUE: Option<bool> = Some(false);

/// `writes` to its last color written.
fn listed(writes: ColorWrites) -> Vec<Option<bool>> {
    let count = writes
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |at| at + 1);
    writes[..count].to_vec()
}

fn rect(x: i32, y: i32, width: u32, height: u32) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D { x, y },
        extent: vk::Extent2D { width, height },
    }
}

#[test]
fn a_mask_keeping_channels_reads_and_an_empty_one_writes_nothing() {
    let color = |mask| {
        Some(ColorAttachmentState::new(
            Format::B8G8R8A8_UNORM,
            mask,
            None,
        ))
    };
    let rgb = Mask::R.union(Mask::G).union(Mask::B);
    let states = [
        color(Mask::known_bits()),
        color(rgb),
        color(Mask::empty()),
        None,
    ];
    assert_eq!(listed(writes(&states)), [OPAQUE, BLENDED]);
    assert!(writes(&states)[2..].iter().all(Option::is_none));
}

#[test]
fn every_write_widens_its_colors_box_alone() {
    let mut coverage = Coverage::default();
    // A quarter layer over color 0, then a full-screen one over color 1.
    coverage.add(&[BLENDED], &[rect(0, 0, 960, 540)]);
    coverage.add(&[None, OPAQUE], &[rect(0, 0, 1920, 1080)]);
    assert_eq!(coverage.of(0), Some([0, 0, 960, 540]));
    assert_eq!(coverage.of(1), Some([0, 0, 1920, 1080]));
    // A color past the draws' colors: nothing written, no box.
    assert_eq!(coverage.of(2), None);
    // Two rectangles of one draw (two viewports) box both.
    let mut two = Coverage::default();
    two.add(
        &[BLENDED],
        &[rect(100, 50, 200, 100), rect(1000, 600, 10, 20)],
    );
    assert_eq!(two.of(0), Some([100, 50, 1010, 620]));
    // An empty scissor writes nothing and leaves the box.
    two.add(&[BLENDED], &[rect(5, 5, 0, 10)]);
    assert_eq!(two.of(0), Some([100, 50, 1010, 620]));
}

#[test]
fn a_draw_writes_the_pass_colors_its_routing_places_and_its_variant_zeroes() {
    let (bgra, rg) = (Some(PixelFormat::BGRA8Unorm), Some(PixelFormat::RG16Float));
    // Each state color at its own index; one the pass lacks is dropped.
    let same = Routing::by_index(&[bgra, rg], &[bgra, rg]);
    assert_eq!(
        listed(pass_writes(&[BLENDED, OPAQUE], &same, false)),
        [BLENDED, OPAQUE]
    );
    let dropped = Routing::by_index(&[bgra, rg], &[bgra]);
    assert_eq!(
        listed(pass_writes(&[OPAQUE, BLENDED], &dropped, false)),
        [OPAQUE]
    );
    // A state lacking color 0 beside its color 1 stores its color 1 into the pass's color 0.
    let mut moved = Routing::default();
    moved.placed.insert(1, 0);
    assert_eq!(
        listed(pass_writes(&[None, BLENDED], &moved, false)),
        [BLENDED]
    );
    // A state of no color whose variant zeroes the pass's color 0 writes it, reading nothing;
    // one Metal keeps the color under writes none.
    let bare = Routing::by_index(&[], &[bgra]);
    assert_eq!(listed(pass_writes(&[], &bare, true)), [OPAQUE]);
    assert_eq!(listed(pass_writes(&[], &bare, false)), []);
    // Its writes widen the pass's box: copies of the box would otherwise drop them.
    let mut coverage = Coverage::default();
    coverage.add(
        &pass_writes(&[BLENDED], &same, false),
        &[rect(0, 0, 64, 64)],
    );
    coverage.add(&pass_writes(&[], &bare, true), &[rect(448, 64, 64, 64)]);
    assert_eq!(coverage.of(0), Some([0, 0, 512, 128]));
}

#[test]
fn an_unbounded_draw_writes_everywhere_in_no_box() {
    let mut coverage = Coverage::default();
    coverage.add(&[OPAQUE], &[rect(0, 0, 64, 64)]);
    coverage.everywhere();
    assert_eq!(coverage.of(0), None);
    assert_eq!(coverage.of(3), None);
    assert_eq!(Coverage::unbounded().of(0), None);
}
