//! The capture assembly's two placeholders (`bind::capture_assembly`'s module docs) under the
//! rule and the kernel: the triangle of three `(0, 0, 0, 1)` vertices, which a restart writes,
//! reaches no tile and is flagged only where its state's viewport flags every primitive drawn
//! under it; the triangle of three NaN vertices, which an unknown parity writes, reaches no tile
//! and is counted unmeasured.

use std::collections::BTreeMap;

use super::kernel::{random_pass, Drawn, Pass, Rng};
use super::reference::{reach, snap_clamped, Flags, Primitive, Raster};
use super::{measured, points};
use crate::bind::capture_assembly::{NOTHING, UNKNOWN};
use crate::bind::tile_reach::kernel::typed_module;

/// Under every measured case's state and the states of 64 seeded random passes (viewports,
/// scissors, culls, windings, depth clipped and clamped, 1, 2, 4 and other sample counts), the
/// rule reaches no tile with either placeholder and flags them as the module docs say, and the
/// kernel bins them as the rule does.
#[test]
fn the_capture_assembly_s_placeholders_reach_nothing() {
    let nothing = Primitive::Triangle([NOTHING.map(f32::from_bits); 3]);
    let unknown = Primitive::Triangle([UNKNOWN.map(f32::from_bits); 3]);
    let mut rows = measured::rows();
    rows.extend(measured::later_rows());
    rows.extend(points::rows());
    let mut states: Vec<Raster> = rows.iter().map(|r| r.raster).collect();
    let mut rng = Rng(0xa55e_b1e0);
    let areas = [[128, 128], [96, 64], [33, 31], [1, 1], [16384, 200]];
    for _ in 0..64 {
        let area = rng.pick(&areas);
        let samples = rng.pick(&[1, 2, 4, 3]);
        let pass = random_pass(&mut rng, area, samples, 1);
        states.extend(pass.states.into_iter().map(|mut s| {
            s.area = area;
            s.samples = samples;
            s
        }));
    }
    let mut wrong = Vec::new();
    // The kernel's passes: one per render area and samples, both placeholders under each state.
    let mut passes: BTreeMap<([u32; 2], u32), Pass> = BTreeMap::new();
    for state in &states {
        let [ox, oy, w, h] = state.viewport;
        let viewport = [ox, ox + w, oy, oy + h]
            .into_iter()
            .all(|c| snap_clamped(c).1);
        for (prim, unmeasured) in [(nothing, !viewport), (unknown, true)] {
            let mut flags = Flags::default();
            let tiles = reach(state, &prim, &mut flags);
            let expected = Flags {
                near_far: false,
                unmeasured,
            };
            if !tiles.is_empty() || flags != expected {
                wrong.push(format!("{prim:?} under {state:?}: {tiles:?}, {flags:?}"));
            }
        }
        let pass = passes
            .entry((state.area, state.samples))
            .or_insert_with(|| Pass {
                area: state.area,
                samples: state.samples,
                layers: 1,
                states: Vec::new(),
                drawn: Vec::new(),
            });
        let at = pass.states.len() as u32;
        pass.states.push(*state);
        pass.drawn.push(Drawn::of(nothing, at, 0));
        pass.drawn.push(Drawn::of(unknown, at, 0));
    }
    let m = typed_module().expect("module");
    for pass in passes.values() {
        wrong.extend(pass.differences(&m));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
