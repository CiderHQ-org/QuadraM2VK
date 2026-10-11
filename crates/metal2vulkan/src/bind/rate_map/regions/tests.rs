use super::*;
use crate::bind::rate_map::{
    RateMapAxis, RateMapDescriptor, RateMapLayerDescriptor, PHYSICAL_GRANULARITY,
};
use crate::pipeline::PrecisionNote;

fn axis(tiles: &[(u32, u32)]) -> RateMapAxis {
    RateMapAxis::from_tiles(PHYSICAL_GRANULARITY, tiles).expect("tiles")
}

fn map(screen: [u32; 2], x: &[(u32, u32)], y: &[(u32, u32)]) -> RasterizationRateMap {
    RasterizationRateMap::from_layers(screen, vec![[axis(x), axis(y)]]).expect("map")
}

/// A map of quality 1 everywhere, which Apple compiles to the identity.
fn full_quality(screen: [u32; 2]) -> RateMapDescriptor {
    let mut desc = RateMapDescriptor::new(screen);
    desc.layers
        .push(RateMapLayerDescriptor::new(vec![1.0], vec![1.0]));
    desc
}

fn limits() -> RegionLimits {
    RegionLimits::new([16384, 16384], [-32768.0, 32767.0])
}

/// Apple M3's compile of quality 0.5 on a 256-pixel axis: four tiles of 63 screen pixels to 32
/// physical, and the last 4 one to one.
const HALF_256: [(u32, u32); 5] = [(63, 32), (63, 32), (63, 32), (63, 32), (4, 4)];

#[test]
fn an_identity_map_is_one_region_with_the_guests_viewport() {
    let m = RasterizationRateMap::new(&full_quality([70, 40])).expect("identity");
    let plan = m
        .regions(0, [0.0, 0.0, 70.0, 40.0], None, &limits())
        .expect("plan");
    assert_eq!(plan.physical_size(), [70, 40]);
    assert_eq!(plan.regions().len(), 1);
    let r = plan.regions()[0];
    assert_eq!(r.viewport(), [0.0, 0.0, 70.0, 40.0]);
    assert_eq!((r.scissor_offset(), r.scissor_extent()), ([0, 0], [70, 40]));
    assert_eq!(plan.precision(), None);
}

/// A pass is Metal's only over the identity: on any other axis its lines are this crate's tiles',
/// and its last physical line may be shaded where Metal leaves it, so the plan carries the
/// decoder's note, whatever its region count.
#[test]
fn a_pass_over_a_map_that_is_not_the_identity_is_a_near_equivalent() {
    let half = map([256, 40], &HALF_256, &[(32, 32), (8, 8)]);
    let plan = half
        .regions(0, [0.0, 0.0, 256.0, 40.0], None, &limits())
        .expect("plan");
    assert_eq!(plan.precision(), Some(PrecisionNote::RasterizationRateMap));
    // One tile of 33 screen pixels to 32 physical ones and the identity down: one region, and
    // still the note.
    let one = map([33, 16], &[(33, 32)], &[(16, 16)]);
    let plan = one
        .regions(0, [0.0, 0.0, 33.0, 16.0], None, &limits())
        .expect("plan");
    assert_eq!(plan.regions().len(), 1);
    assert_eq!(plan.precision(), Some(PrecisionNote::RasterizationRateMap));
    // Explicit tiles of the identity are the identity.
    let ident = map([40, 16], &[(32, 32), (8, 8)], &[(16, 16)]);
    let plan = ident
        .regions(0, [0.0, 0.0, 40.0, 16.0], None, &limits())
        .expect("plan");
    assert_eq!(plan.precision(), None);
}

#[test]
fn tiles_on_one_line_merge_and_each_line_is_a_region() {
    let m = map([256, 40], &HALF_256, &[(32, 32), (8, 8)]);
    let plan = m
        .regions(0, [0.0, 0.0, 256.0, 40.0], None, &limits())
        .expect("plan");
    assert_eq!(plan.physical_size(), [132, 40]);
    let regions = plan.regions();
    assert_eq!(regions.len(), 2);
    // The four 63-to-32 tiles are one line from screen 0: viewport [0, 256 * 32/63).
    assert_eq!(
        regions[0].viewport(),
        [0.0, 0.0, (256.0f64 * 32.0 / 63.0) as f32, 40.0]
    );
    assert_eq!(regions[0].scissor_offset(), [0, 0]);
    assert_eq!(regions[0].scissor_extent(), [128, 40]);
    // The last tile, one to one from screen 252 at physical 128.
    assert_eq!(regions[1].viewport(), [-124.0, 0.0, 256.0, 40.0]);
    assert_eq!(regions[1].scissor_offset(), [128, 0]);
    assert_eq!(regions[1].scissor_extent(), [4, 40]);
}

#[test]
fn regions_tile_the_physical_image_row_by_row() {
    let m = map([37, 100], &[(33, 32), (4, 4)], &[(64, 32), (36, 32)]);
    let plan = m
        .regions(0, [0.0, 0.0, 37.0, 100.0], None, &limits())
        .expect("plan");
    let cells: Vec<_> = plan
        .regions()
        .iter()
        .map(|r| (r.scissor_offset(), r.scissor_extent()))
        .collect();
    assert_eq!(
        cells,
        [
            ([0, 0], [32, 32]),
            ([32, 0], [4, 32]),
            ([0, 32], [32, 32]),
            ([32, 32], [4, 32]),
        ]
    );
}

/// The exactness argument, to Apple's fixed point: inside each region's scissor, the viewport's
/// inverse sends a physical pixel center to the screen point the map's decoder gives it, for the
/// guest's screen-space viewport (here a smaller one than the screen), within what the decoder's
/// 1/256-pixel offsets move it (the regions draw with the tiles' exact lines).
#[test]
fn a_regions_viewport_is_the_decoders_line() {
    let m = map([256, 100], &HALF_256, &[(64, 32), (36, 32)]);
    let guest = [10.0, 5.0, 200.0, 80.0];
    let plan = m.regions(0, guest, None, &limits()).expect("plan");
    for r in plan.regions() {
        let [x, y, w, h] = r.viewport().map(f64::from);
        let [ox, oy] = r.scissor_offset();
        let [ex, ey] = r.scissor_extent();
        for py in oy..oy + ey {
            for px in ox..ox + ex {
                let center = [px as f32 + 0.5, py as f32 + 0.5];
                let decoded = m.physical_to_screen(center, 0);
                // Viewport inverse: from physical to the guest's NDC, then to screen.
                let sx = guest[0] + (f64::from(center[0]) - x) / w * guest[2];
                let sy = guest[1] + (f64::from(center[1]) - y) / h * guest[3];
                // Half a 1/256-pixel step, times a tile's at most 100 screen pixels per
                // physical one here (2), plus the arithmetic's rounding.
                let bound = 2.0 / 512.0 + 1e-4;
                assert!(
                    (sx - f64::from(decoded[0])).abs() < bound
                        && (sy - f64::from(decoded[1])).abs() < bound,
                    "pixel {px},{py}: viewport {sx},{sy} decoder {decoded:?}"
                );
            }
        }
    }
}

#[test]
fn a_plan_outside_the_devices_viewport_limits_is_refused() {
    let m = map([256, 40], &HALF_256, &[(32, 32), (8, 8)]);
    let whole = [0.0, 0.0, 256.0, 40.0];
    assert_eq!(
        m.regions(1, whole, None, &limits()),
        Err(RegionError::Layer { layer: 1 })
    );
    assert_eq!(
        m.regions(0, [0.0, 0.0, 0.0, 40.0], None, &limits()),
        Err(RegionError::EmptyViewport { axis: 0 })
    );
    // The last tile's region is 256 wide: over a 200-pixel maximum.
    assert_eq!(
        m.regions(
            0,
            whole,
            None,
            &RegionLimits::new([200, 200], [-400.0, 399.0])
        ),
        Err(RegionError::ViewportDimensions {
            axis: 0,
            extent: 256.0,
            max: 200
        })
    );
    // ... and its origin -124 leaves a range from -100.
    assert!(matches!(
        m.regions(
            0,
            whole,
            None,
            &RegionLimits::new([4096, 4096], [-100.0, 4095.0])
        ),
        Err(RegionError::ViewportBounds { axis: 0, .. })
    ));
}

/// The guest's scissor: over the identity, screen and physical pixels are one, so it clips the
/// region's scissor; over any other map, where Metal applies it is not measured, so only one
/// that covers the screen (and so the image, in either space) is admitted.
#[test]
fn a_guest_scissor_clips_an_identity_pass_and_must_cover_the_screen_otherwise() {
    let ident = RasterizationRateMap::new(&full_quality([64, 64])).expect("identity");
    let whole = [0.0, 0.0, 64.0, 64.0];
    let cell = |scissor| {
        let plan = ident
            .regions(0, whole, Some(scissor), &limits())
            .expect("plan");
        assert_eq!(plan.regions().len(), 1);
        assert_eq!(plan.precision(), None);
        let r = plan.regions()[0];
        (r.scissor_offset(), r.scissor_extent())
    };
    assert_eq!(cell([0, 0, 16, 16]), ([0, 0], [16, 16]));
    // Past the image it is clipped to it, and outside it no pixel is left.
    assert_eq!(cell([60, 8, 100, 4]), ([60, 8], [4, 4]));
    assert_eq!(cell([100, 0, 5, 5]), ([64, 0], [0, 5]));
    assert_eq!(cell([0, 0, u32::MAX, u32::MAX]), ([0, 0], [64, 64]));

    let half = map([256, 40], &HALF_256, &[(32, 32), (8, 8)]);
    let screen = [0.0, 0.0, 256.0, 40.0];
    let unclipped = half.regions(0, screen, None, &limits()).expect("plan");
    assert_eq!(
        half.regions(0, screen, Some([0, 0, 16, 16]), &limits()),
        Err(RegionError::Scissor {
            scissor: [0, 0, 16, 16]
        })
    );
    // The physical size is not the screen's: a scissor of it is refused too.
    assert_eq!(
        half.regions(0, screen, Some([0, 0, 132, 40]), &limits()),
        Err(RegionError::Scissor {
            scissor: [0, 0, 132, 40]
        })
    );
    assert_eq!(
        half.regions(0, screen, Some([0, 0, 256, 40]), &limits()),
        Ok(unclipped)
    );
}

/// A guest viewport that is not finite, has no extent, or is flipped is refused by name, as is
/// one too small for a region's line to keep its extent in `f32` (Vulkan requires a positive
/// width and height).
#[test]
fn degenerate_guest_viewports_are_refused_by_name() {
    let m = map([256, 40], &HALF_256, &[(32, 32), (8, 8)]);
    let refused = |viewport| m.regions(0, viewport, None, &limits()).unwrap_err();
    assert!(matches!(
        refused([f64::NAN, 0.0, 256.0, 40.0]),
        RegionError::ViewportNotFinite { .. }
    ));
    assert!(matches!(
        refused([0.0, 0.0, f64::INFINITY, 40.0]),
        RegionError::ViewportNotFinite { .. }
    ));
    assert_eq!(
        refused([0.0, 0.0, 256.0, -40.0]),
        RegionError::NegativeViewport {
            axis: 1,
            extent: -40.0
        }
    );
    assert_eq!(
        refused([0.0, 0.0, 256.0, 0.0]),
        RegionError::EmptyViewport { axis: 1 }
    );
    assert!(matches!(
        refused([0.0, 0.0, 1e-300, 40.0]),
        RegionError::RegionExtent { axis: 0, .. }
    ));
}

/// A plan holds at most `MAX_REGIONS` regions, so a map of many lines is refused before its
/// regions are built: tiles of alternating ratios make one segment each.
#[test]
fn a_plan_of_more_than_the_most_regions_is_refused() {
    let lines = |n: usize| -> Vec<(u32, u32)> {
        (0..n)
            .map(|i| if i % 2 == 0 { (32, 32) } else { (64, 32) })
            .collect()
    };
    let screen = |tiles: &[(u32, u32)]| tiles.iter().map(|&(s, _)| s).sum::<u32>();
    let (x, y) = (lines(128), lines(128));
    let at_most = map([screen(&x), screen(&y)], &x, &y);
    let viewport = [0.0, 0.0, f64::from(screen(&x)), f64::from(screen(&y))];
    let plan = at_most.regions(0, viewport, None, &limits()).expect("plan");
    assert_eq!(plan.regions().len(), RateMapRegions::MAX_REGIONS);
    let x = lines(129);
    let over = map([screen(&x), screen(&y)], &x, &y);
    let viewport = [0.0, 0.0, f64::from(screen(&x)), f64::from(screen(&y))];
    assert_eq!(
        over.regions(0, viewport, None, &limits()),
        Err(RegionError::TooManyRegions {
            regions: 129 * 128,
            max: RateMapRegions::MAX_REGIONS
        })
    );
}
