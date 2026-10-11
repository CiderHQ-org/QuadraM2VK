//! A render pipeline's color in a pass attachment of another pixel format: the view each measured
//! pair draws through, the records that agree, and the formats held as their own. The bytes each
//! comment quotes are the Metal probes's (an M4), the fragment's output 0.25/0.5/0.75/1 (signed
//! 0.25/-0.5/0.75/-1, integers 0x11/0x22/0x33/0x44).

use super::super::color_reformat::{
    held, misplaced, routing, uncovered, view_format, Routing, Uncovered,
};
use super::image_plan::committed;
use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan::metal::{BlendFactor, BlendOperation, ColorWriteMask};
use metal2vulkan::pipeline::ColorAttachmentDescriptor;
use metal2vulkan::vk::Format as F;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn channel_order_twins_draw_through_the_pass_s_own_format() {
    let cases = [
        // `a_rgba8_in_bgra8`: bf8040ff, the color in the pass's channel order.
        (F::R8G8B8A8_UNORM, F::B8G8R8A8_UNORM, F::B8G8R8A8_UNORM),
        // `a_bgra8_in_rgba8`: 4080bfff.
        (F::B8G8R8A8_UNORM, F::R8G8B8A8_UNORM, F::R8G8B8A8_UNORM),
        // `a_rgba8s_in_bgra8s`: e1bc89ff.
        (F::R8G8B8A8_SRGB, F::B8G8R8A8_SRGB, F::B8G8R8A8_SRGB),
        // `a_bgra8s_in_rgba8s`: 89bce1ff.
        (F::B8G8R8A8_SRGB, F::R8G8B8A8_SRGB, F::R8G8B8A8_SRGB),
        // `p_rgb10a2_in_bgr10a2`: the word d00802ff, blue in bits 0-9.
        (
            F::A2B10G10R10_UNORM_PACK32,
            F::A2R10G10B10_UNORM_PACK32,
            F::A2R10G10B10_UNORM_PACK32,
        ),
        // `p_bgr10a2_in_rgb10a2`: the word eff80100, red in bits 0-9.
        (
            F::A2R10G10B10_UNORM_PACK32,
            F::A2B10G10R10_UNORM_PACK32,
            F::A2B10G10R10_UNORM_PACK32,
        ),
    ];
    for (state, pass, view) in cases {
        assert_eq!(
            view_format(state, pass),
            Some(view),
            "{state:?} in {pass:?}"
        );
    }
}

#[test]
fn other_encodings_draw_through_a_view_of_the_state_s() {
    let cases = [
        // `a_rgba8_in_bgra8s`: bf8040ff, the state's plain encoding.
        (F::R8G8B8A8_UNORM, F::B8G8R8A8_SRGB, F::B8G8R8A8_UNORM),
        // `a_rgba8s_in_bgra8`: e1bc89ff, the state's sRGB encoding.
        (F::R8G8B8A8_SRGB, F::B8G8R8A8_UNORM, F::B8G8R8A8_SRGB),
        // `a_bgra8s_in_rgba8`: 89bce1ff.
        (F::B8G8R8A8_SRGB, F::R8G8B8A8_UNORM, F::R8G8B8A8_SRGB),
        // `n_rgba8snorm_in_bgra8`: 5fc02081.
        (F::R8G8B8A8_SNORM, F::B8G8R8A8_UNORM, F::B8G8R8A8_SNORM),
        // `e_rgba8uint_in_bgra8`: 33221144.
        (F::R8G8B8A8_UINT, F::B8G8R8A8_UNORM, F::B8G8R8A8_UINT),
        // `e_bgra8_in_r32f`: 4080bfff, red first.
        (F::B8G8R8A8_UNORM, F::R32_SFLOAT, F::R8G8B8A8_UNORM),
        // `p_bgr10a2_in_r32f`: the word eff80100, red first.
        (
            F::A2R10G10B10_UNORM_PACK32,
            F::R32_SFLOAT,
            F::A2B10G10R10_UNORM_PACK32,
        ),
    ];
    for (state, pass, view) in cases {
        assert_eq!(
            view_format(state, pass),
            Some(view),
            "{state:?} in {pass:?}"
        );
    }
}

#[test]
fn bytes_no_view_stores_have_none() {
    let cases = [
        // `e_r32f_in_bgra8`: 8000003e, the float's bytes 0 and 2 exchanged.
        (F::R32_SFLOAT, F::B8G8R8A8_UNORM),
        // `e_rgb10a2_in_bgra8`: f80100ef.
        (F::A2B10G10R10_UNORM_PACK32, F::B8G8R8A8_UNORM),
        // `p_rgba8_in_bgr10a2`: fb830fc4, red and blue fields exchanged.
        (F::R8G8B8A8_UNORM, F::A2R10G10B10_UNORM_PACK32),
        // `n_rgba8_in_rgba8snorm`: 4081bfff, 0x80 stored as 0x81.
        (F::R8G8B8A8_UNORM, F::R8G8B8A8_SNORM),
        // A packed 16-bit format, which Metal's record holds otherwise.
        (F::R5G6B5_UNORM_PACK16, F::R8G8_UNORM),
        (F::R8G8_UNORM, F::R5G6B5_UNORM_PACK16),
        // Texels of different sizes (unbuilt).
        (F::R16G16B16A16_SFLOAT, F::R8G8B8A8_UNORM),
        (F::R8G8B8A8_UNORM, F::R16G16B16A16_SFLOAT),
    ];
    for (state, pass) in cases {
        assert_eq!(view_format(state, pass), None, "{state:?} in {pass:?}");
    }
}

/// A routing from its placed colors, uncovered colors by location, dropped colors and subpass
/// locations.
fn routed(
    placed: &[(u32, u32)],
    uncovered: &[(u32, u32)],
    dropped: &[u32],
    locations: &[Option<u32>],
) -> Routing {
    Routing {
        locations: locations.to_vec(),
        placed: placed.iter().copied().collect::<BTreeMap<_, _>>(),
        uncovered: uncovered.iter().copied().collect::<BTreeMap<_, _>>(),
        dropped: dropped.iter().copied().collect::<BTreeSet<_>>(),
    }
}

#[test]
fn each_state_color_is_routed_to_the_pass_color_at_its_place() {
    let rgba8 = Some(P::RGBA8Unorm);
    let cases = [
        // Every color at its own index: no locations.
        (
            vec![rgba8],
            vec![Some(P::BGRA8Unorm)],
            routed(&[(0, 0)], &[], &[], &[]),
        ),
        (
            vec![rgba8],
            vec![rgba8, rgba8],
            routed(&[(0, 0)], &[(1, 1)], &[], &[]),
        ),
        (
            vec![rgba8, rgba8],
            vec![rgba8],
            routed(&[(0, 0)], &[], &[1], &[]),
        ),
        // Probe 2 `h_none_rgba8_in_rgba8_rgba8`: the state's color 1 into the pass's color 0,
        // the pass's color 1 uncovered at the highest free location.
        (
            vec![None, rgba8],
            vec![rgba8, rgba8],
            routed(
                &[(1, 0)],
                &[(7, 1)],
                &[],
                &[None, Some(0), None, None, None, None, None, Some(1)],
            ),
        ),
        // Probe 3 `g_rgba8_rgba8_in_none_rgba8`: the state's color 0 into the pass's color 1,
        // its color 1 dropped.
        (
            vec![rgba8, rgba8],
            vec![None, rgba8],
            routed(&[(0, 1)], &[], &[1], &[Some(1)]),
        ),
        // Probe 3 `g_rgba8_rgba8_in_none_rgba8_rgba8`: colors 0 and 1 into 1 and 2.
        (
            vec![rgba8, rgba8],
            vec![None, rgba8, rgba8],
            routed(&[(0, 1), (1, 2)], &[], &[], &[Some(1), Some(2)]),
        ),
        // Probe 4 `b_plain`: the state's color 2 into the pass's color 1, the pass's color 2
        // uncovered at the highest free location.
        (
            vec![rgba8, None, rgba8],
            vec![rgba8, rgba8, rgba8],
            routed(
                &[(0, 0), (2, 1)],
                &[(7, 2)],
                &[],
                &[Some(0), None, Some(1), None, None, None, None, Some(2)],
            ),
        ),
        // A color of its own index at its own start, of another size: the format check decides.
        (
            vec![Some(P::RGBA16Float)],
            vec![rgba8],
            routed(&[(0, 0)], &[], &[], &[]),
        ),
    ];
    for (state, pass, want) in cases {
        assert_eq!(routing(&state, &pass), Some(want), "{state:?} in {pass:?}");
    }
    // Probe 3 `g_r8_rgba8_in_none_rgba8`: the pass's color 1 holds the state's colors 0 and part
    // of 1 (`4020609f`): no whole color.
    let partial = [
        (vec![Some(P::R8Unorm), rgba8], vec![None, rgba8]),
        (vec![Some(P::RGBA16Float)], vec![rgba8, rgba8]),
        (vec![rgba8], vec![Some(P::R8Unorm), Some(P::R8Unorm)]),
    ];
    for (state, pass) in partial {
        assert_eq!(routing(&state, &pass), None, "{state:?} in {pass:?}");
    }
}

/// An `RGBA8Unorm` color writing `mask`, blended by `blend` (source and destination factors of
/// RGB and alpha) where given.
fn color(mask: ColorWriteMask, blend: Option<[BlendFactor; 4]>) -> ColorAttachmentDescriptor {
    let mut color = ColorAttachmentDescriptor::new(P::RGBA8Unorm);
    color.write_mask = mask;
    if let Some([src, dst, src_alpha, dst_alpha]) = blend {
        color.blending_enabled = true;
        color.source_rgb_blend_factor = src;
        color.destination_rgb_blend_factor = dst;
        color.source_alpha_blend_factor = src_alpha;
        color.destination_alpha_blend_factor = dst_alpha;
    }
    color
}

#[test]
fn an_uncovered_color_is_zeroed_where_metal_s_code_reads_no_attachment() {
    use BlendFactor as F;
    let all = ColorWriteMask::ALL;
    let alpha_over = [
        F::SourceAlpha,
        F::OneMinusSourceAlpha,
        F::SourceAlpha,
        F::OneMinusSourceAlpha,
    ];
    let cases = [
        // Probe 4 `z_plain`, probe 5 `y_dst_zero`.
        (vec![color(all, None)], Uncovered::Zeroed),
        (
            vec![color(all, Some([F::One, F::Zero, F::One, F::Zero]))],
            Uncovered::Zeroed,
        ),
        // Probe 4 `z_blend_add`, `z_mask_red`, `z_mask_none`; probe 5 `y_dst_one`,
        // `y_mask_all_but_alpha`.
        (
            vec![color(all, Some([F::One, F::One, F::One, F::One]))],
            Uncovered::Kept,
        ),
        (vec![color(ColorWriteMask::RED, None)], Uncovered::Kept),
        (vec![color(ColorWriteMask::NONE, None)], Uncovered::Kept),
        (
            vec![color(all, Some([F::Zero, F::One, F::Zero, F::One]))],
            Uncovered::Kept,
        ),
        (
            vec![color(
                ColorWriteMask::RED | ColorWriteMask::GREEN | ColorWriteMask::BLUE,
                None,
            )],
            Uncovered::Kept,
        ),
        // A source factor of the destination reads it.
        (
            vec![color(
                all,
                Some([F::DestinationColor, F::Zero, F::One, F::Zero]),
            )],
            Uncovered::Kept,
        ),
        // Probe 5 `y_alpha_buffer1`, `y_alpha_varying1` (kept) and `y_alpha_const1` (zeroed: the
        // literal 1 folds the factor): a near-equivalent.
        (
            vec![color(all, Some(alpha_over))],
            Uncovered::KeptUnlessFolded,
        ),
        // Probe 4 `m_blend1`: any color reading the attachment keeps the record.
        (
            vec![
                color(all, None),
                color(all, Some([F::One, F::One, F::One, F::One])),
            ],
            Uncovered::Kept,
        ),
        (
            vec![color(all, None), color(all, Some(alpha_over))],
            Uncovered::KeptUnlessFolded,
        ),
    ];
    for (colors, want) in cases {
        assert_eq!(uncovered(&colors), want, "{colors:?}");
    }
    // `Min` and `Max` read the destination whatever the factors.
    let mut min = color(all, Some([F::One, F::Zero, F::One, F::Zero]));
    min.rgb_blend_operation = BlendOperation::Min;
    assert_eq!(uncovered([&min]), Uncovered::Kept);
    // Probe 5 `y_r8_mask_red` (zeroed) and `y_r8_mask_green` (kept): the mask over the format's
    // own channels.
    let mut red = ColorAttachmentDescriptor::new(P::R8Unorm);
    red.write_mask = ColorWriteMask::RED;
    assert_eq!(uncovered([&red]), Uncovered::Zeroed);
    red.write_mask = ColorWriteMask::GREEN;
    assert_eq!(uncovered([&red]), Uncovered::Kept);
}

#[test]
fn held_formats_are_metal_s_own() {
    let measured = [
        "lavapipe",
        "amd-rdna3-radv",
        "intel-arl-anv",
        "nvidia-rtx50-linux",
        "moltenvk-m4",
    ];
    let devices = committed().into_iter();
    for (name, _, device) in devices.filter(|(n, ..)| measured.contains(&n.as_str())) {
        let profile = device.profile();
        assert_eq!(
            held(P::BGRA8Unorm, profile),
            Some(F::B8G8R8A8_UNORM),
            "{name}"
        );
        assert_eq!(
            held(P::RGB10A2Unorm, profile),
            Some(F::A2B10G10R10_UNORM_PACK32),
            "{name}"
        );
        // An extended-range format's texels are held in words of another layout.
        assert_eq!(held(P::BGR10XR, profile), None, "{name}");
    }
}

#[test]
fn a_misplaced_color_is_the_lowest_that_moves() {
    let rgba8 = Some(P::RGBA8Unorm);
    let cases = [
        // Probe 2 `h_none_rgba8_in_rgba8_rgba8`: the state's color 1 lies where the pass holds
        // its color 0.
        (vec![None, rgba8], vec![rgba8, rgba8], Some(0)),
        // Probe 3 `g_rgba8_rgba8_in_none_rgba8`: the state's color 0 where the pass holds its
        // color 1.
        (vec![rgba8, rgba8], vec![None, rgba8], Some(0)),
        // Probe 3 `g_none_rgba8_in_none_none_rgba8`: color 1 where the pass holds color 2.
        (vec![None, rgba8], vec![None, None, rgba8], Some(1)),
        // A state of colors 0 and 2 in a pass of colors 0, 1 and 2: color 2 where the pass
        // holds color 1.
        (vec![rgba8, None, rgba8], vec![rgba8, rgba8, rgba8], Some(1)),
        // Every color at its place: a color only one side has, past the other's.
        (vec![rgba8], vec![rgba8, rgba8], None),
        (vec![rgba8, rgba8], vec![rgba8], None),
        (vec![None, rgba8], vec![None, rgba8], None),
        (vec![rgba8], vec![Some(P::BGRA8Unorm)], None),
        // `Invalid` is no color.
        (vec![Some(P::Invalid), rgba8], vec![None, rgba8], None),
        // A format without a place in the record decides nothing.
        (vec![Some(P::BC1RGBA), rgba8], vec![rgba8], None),
    ];
    for (state, pass, index) in cases {
        assert_eq!(misplaced(&state, &pass), index, "{state:?} in {pass:?}");
    }
}
