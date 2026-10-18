//! `texture/surface_depth`: the bytes a pass on a depth or stencil texture over a plane of a
//! surface with planes writes back, the runs of Metal's elements' bytes, against every element's
//! bytes taken one by one.

use crate::texture::heap_tiled::{Elements, TiledLayout};
use crate::texture::surface_depth::element_runs;
use metal2vulkan::metal::TextureType;
use std::ops::Range;

/// Metal's tiled layout of a 2D texture of `[w, h]` elements of `unit` bytes, one level.
fn layout(unit: u64, [w, h]: [u64; 2]) -> TiledLayout {
    TiledLayout::of(unit, Elements::Texels, TextureType::Type2D, [w, h, 1], 1, 1).unwrap()
}

/// The runs of `layout`'s elements' bytes, each element's taken alone, in order and joined.
fn one_by_one(layout: &TiledLayout) -> Vec<Range<u64>> {
    let [w, h] = layout.levels[0].extent;
    let mut at: Vec<u64> = (0..h)
        .flat_map(|y| (0..w).map(move |x| layout.element(0, 0, x, y)))
        .collect();
    at.sort_unstable();
    let mut runs: Vec<Range<u64>> = Vec::new();
    for first in at {
        match runs.last_mut() {
            Some(run) if run.end == first => run.end = first + layout.unit,
            _ => runs.push(first..first + layout.unit),
        }
    }
    runs
}

#[test]
fn a_pass_writes_back_the_runs_of_metal_s_elements_alone() {
    for unit in [1, 2, 4] {
        for extent in [
            [1, 1],
            [4, 4],
            [19, 7],
            [65, 3],
            [128, 1],
            [64, 64],
            [33, 65],
            [100, 70],
            [130, 67],
            [129, 129],
            [200, 150],
        ] {
            let layout = layout(unit, extent);
            assert_eq!(
                element_runs(&layout),
                one_by_one(&layout),
                "{unit} bytes, {extent:?}"
            );
        }
    }
    // Whole tiles lie end to end from the first byte: one run.
    assert_eq!(element_runs(&layout(4, [128, 64])), vec![0..128 * 64 * 4]);
    // The end of Metal's last element at 19x7 ("the rule's span"), short of
    // the tile's 32x8.
    for (unit, end) in [(4, 692), (2, 346), (1, 173)] {
        let runs = element_runs(&layout(unit, [19, 7]));
        assert_eq!(runs.last().map(|run| run.end), Some(end), "{unit} bytes");
    }
}
