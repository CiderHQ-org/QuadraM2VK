//! Two-plane extended-range textures over a surface's two planes (`texture/surface_two_planes.rs`).

use crate::linear_plan::BufferLayout;
use crate::texture::surface_two_planes::{alpha_plane, surface_layout};
use crate::ycbcr::IoSurfaceYCbCr;
use crate::Error;
use metal2vulkan::metal::PixelFormat as P;

/// `'b3a8'`, the OSType Metal's answers were measured over.
const B3A8: u32 = u32::from_be_bytes(*b"b3a8");

/// Plane 0's GPU layout of a `[w, h]` texture from `offset`, rows `pitch` apart.
fn color(offset: u64, pitch: u64, [w, h]: [u64; 2]) -> BufferLayout {
    BufferLayout {
        offset,
        row_pitch: pitch,
        slice_pitch: pitch * h,
        row_bytes: 4 * w,
        rows: h,
        slices: 1,
    }
}

/// Metal's GPU finds each texel's color word in plane 0 and its alpha byte in plane 1, each plane
/// from its offset rounded down to 16 bytes in rows its `bytesPerRow` rounded down to 16 apart (an
/// Apple M4: CoreVideo `'b3a8'` surfaces of 64 x 64 and 37 x 5, and `IOSurfaceCreate` ones at
/// 260/68 bytes a row with plane 1 4 bytes past plane 0, read at 0/256
/// and 1296/64, and at 272/80 with plane 1 48 bytes past it), and in the consumer's shape: the
/// layout from plane 0's first byte, plane 1's GPU rows, and the span the moves stage.
#[test]
fn a_surface_two_plane_texture_finds_its_planes_where_metal_does() {
    // (extent, plane 0's GPU offset and rows, plane 1's offset and bytesPerRow, the buffer's
    // length) -> (plane 1 past plane 0's first byte, its GPU rows, its GPU offset, the span).
    let cases: [([u64; 2], (u64, u64), [u64; 2], u64, (u64, u64, u64, u64)); 5] = [
        (
            [64, 64],
            (512, 256),
            [16896, 64],
            21056,
            (16384, 64, 16896, 16384 + 63 * 64 + 64),
        ),
        (
            [37, 5],
            (512, 192),
            [2048, 64],
            2624,
            (1536, 64, 2048, 1536 + 4 * 64 + 37),
        ),
        (
            [37, 5],
            (0, 256),
            [1304, 68],
            1644,
            (1296, 64, 1296, 1296 + 4 * 64 + 37),
        ),
        (
            [61, 9],
            (0, 272),
            [2496, 80],
            3216,
            (2496, 80, 2496, 2496 + 8 * 80 + 61),
        ),
        (
            [384, 256],
            (0, 1536),
            [393216, 512],
            524288,
            (393216, 512, 393216, 393216 + 255 * 512 + 384),
        ),
    ];
    for (extent, (offset, pitch), alpha, length, want) in cases {
        let (layout, plane) = surface_layout(&color(offset, pitch, extent), alpha, length)
            .unwrap_or_else(|why| panic!("{extent:?}: {why}"));
        assert_eq!(
            (
                layout.alpha_at,
                layout.alpha_pitch,
                plane.offset,
                layout.span()
            ),
            want,
            "{extent:?}"
        );
        assert_eq!(
            (layout.width, layout.height, layout.color_pitch),
            (extent[0], extent[1], pitch)
        );
        assert_eq!(
            (plane.row_pitch, plane.row_bytes, plane.rows),
            (want.1, extent[0], extent[1])
        );
        assert_eq!(plane.offset, offset + layout.alpha_at);
    }
}

/// A plane 1 whose rows overlap on Metal's GPU (37 texels at 40 bytes a row, 32 on the GPU), that
/// starts before plane 0's last texel, or that runs past the buffer is refused.
#[test]
fn a_surface_two_plane_texture_refuses_planes_metal_was_not_measured_over() {
    let plane0 = color(0, 256, [37, 5]);
    for (alpha, length) in [([2048, 40], 4096), ([1100, 64], 4096), ([2048, 64], 2200)] {
        assert!(
            surface_layout(&plane0, alpha, length).is_err(),
            "{alpha:?} in {length}"
        );
    }
    // One row of 37 alpha bytes at 40 bytes a row overlaps nothing.
    assert!(surface_layout(&color(0, 256, [37, 1]), [2048, 40], 4096).is_ok());
}

/// The texture takes plane 1 from the surface's facts, over plane 0 of a two-plane `'b3a8'`
/// surface; without the facts, without planes, over another OSType and over another plane it is
/// refused, naming the format.
#[test]
fn a_surface_two_plane_texture_takes_plane_1_from_the_surface_s_facts() {
    let facts = IoSurfaceYCbCr::new(B3A8, 0, [384, 256], 393216, 512);
    for pf in [P::RGB10A82PXR10, P::RGB10A82PXR10SRGB] {
        assert_eq!(alpha_plane(pf, true, Some(facts)).unwrap(), [393216, 512]);
        let refusals = [
            alpha_plane(pf, true, None),
            alpha_plane(pf, false, Some(facts)),
            alpha_plane(
                pf,
                true,
                Some(IoSurfaceYCbCr::new(
                    u32::from_be_bytes(*b"BGRA"),
                    0,
                    [384, 256],
                    393216,
                    512,
                )),
            ),
            alpha_plane(
                pf,
                true,
                Some(IoSurfaceYCbCr::new(B3A8, 1, [384, 256], 393216, 512)),
            ),
        ];
        for refusal in refusals {
            match refusal {
                Err(Error::Unsupported(gap)) => {
                    assert!(gap.to_string().contains(pf.name()), "{gap}")
                }
                other => panic!("{pf:?}: {other:?}"),
            }
        }
    }
}
