//! A `Depth16Unorm` depth as the fragment stage writes Metal's code (`stage_io::depth16`), on the
//! interpreter, against Metal's rule (probes `depth16` and `depthout` on the M3 Max under macOS
//! 27.0.1; `runtime_lane::unorm_depth_bias` draws the measured rows): a depth `d` of `[0, 1]` stores
//! the exact product `65535 d` rounded to nearest, a tie up; a biased fragment adds
//! `slope · m + units`, each operation rounded on its own, bounded by a nonzero clamp from its
//! side, to its depth rounding toward zero, and clamps the sum to `[0, 1]` first. The stage writes
//! code `c` as `c` times the `f32` nearest 1/65535, which a further check shows a device that
//! rounds to nearest, as Vulkan asks, stores as `c`.

use super::*;
use crate::lower::stage_io::depth16::{code_depth, offset_depth, side_bounded};

/// Metal's code of a depth `d` of `[0, 1]`: the product with 65535, exact in `f64`, rounded to
/// nearest, a tie up.
fn metal_code(d: f32) -> u32 {
    (f64::from(d) * 65535.0 + 0.5).floor() as u32
}

/// The `f32` the stage writes for code `c`, as bits: `c` times the `f32` nearest 1/65535.
fn written(c: u32) -> u64 {
    u64::from((c as f32 * (1.0f32 / 65535.0)).to_bits())
}

/// Depths of `[0, 1]` beside every `step`-th code and its midpoint, `n` `f32` steps either side.
fn around_codes(step: usize, n: u32) -> Vec<f32> {
    let mut out = vec![0.0, 1.0, 0.5, f32::MIN_POSITIVE, 1e-40];
    for k in (0..=65535u32).step_by(step) {
        for c in [f64::from(k), f64::from(k) + 0.5] {
            let x = (c / 65535.0) as f32;
            for d in 0..=n {
                out.extend([
                    f32::from_bits(x.to_bits().wrapping_add(d)),
                    f32::from_bits(x.to_bits().wrapping_sub(d)),
                ]);
            }
        }
    }
    out.retain(|x| (0.0..=1.0).contains(x));
    out
}

/// Every code's written value converts back to the code: its product with 65535 lies within
/// 2^-7 of it, so a conversion that rounds to nearest stores the code.
#[test]
fn every_code_written_converts_back_to_the_code() {
    for c in 0..=65535u32 {
        let v = f64::from(f32::from_bits(written(c) as u32));
        let off = (v * 65535.0 - f64::from(c)).abs();
        assert!(off < 1.0 / 128.0, "code {c}: {off}");
    }
}

/// `code_depth` writes Metal's code of each depth beside the codes and their midpoints, 0.5's
/// tie up among them.
#[test]
fn a_depth_is_written_as_metal_s_code() {
    let f = float(32);
    let cases: Vec<Case> = around_codes(17, 2)
        .into_iter()
        .map(|x| vec![vec![u64::from(x.to_bits())]])
        .collect();
    check(
        "code_depth",
        &[f],
        f,
        &|fb, xs| code_depth(fb, xs[0]),
        &cases,
        &|c| Some(vec![written(metal_code(f32v(c[0][0])))]),
        false,
    );
}

/// Metal's biased depth of `z` at slope `m` under `[units, slope, clamp]`: the offset
/// `slope · m + units` in `f32`, each operation rounded, a nonzero clamp bounding it from its side,
/// added to `z` rounding toward zero, clamped to `[0, 1]`, and its code. `None` where the sum is
/// not exact in `f64` (no case here).
fn metal_biased_code(z: f32, m: f32, [units, slope, clamp]: [f32; 3]) -> Option<u32> {
    let offset = slope * m + units;
    let offset = if (clamp > 0.0 && offset > clamp) || (clamp < 0.0 && offset < clamp) {
        clamp
    } else {
        offset
    };
    let exact = f64::from(z) + f64::from(offset);
    if exact - f64::from(z) != f64::from(offset) {
        return None;
    }
    let nearest = exact as f32;
    let toward_zero = if f64::from(nearest).abs() > exact.abs() {
        f32::from_bits(nearest.to_bits() - 1)
    } else {
        nearest
    };
    Some(metal_code(toward_zero.clamp(0.0, 1.0)))
}

/// Metal's `[units, slope, clamp]` for constant `c`, slope factor `s` and clamp `k`
/// (`metal::state::depth16_bias`).
fn factors(c: f32, s: f32, k: f32) -> [f32; 3] {
    crate::metal::state::depth16_bias(c, s, k)
}

/// The cases of part `part` of `of`: depths beside codes and midpoints (and the midpoints the
/// offsets land the sums on), under constants, slope factors, slopes and clamps a draw takes.
fn biased_cases(part: usize, of: usize) -> Vec<Case> {
    let constants = [
        -1000.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 100.0, 30000.0,
    ];
    let slopes = [0.0, 0.5, -1.0, 2.0, 1000.0];
    let ms = [0.0, 1.0 / 4096.0, 1.0 / 1080.0, 0.0123, 0.5];
    let clamps = [0.0, 1.0 / 8192.0, -1.0 / 8192.0];
    let depths = around_codes(8191, 1);
    let mut cases = Vec::new();
    for c in constants {
        for s in slopes {
            for k in clamps {
                let [units, slope, clamp] = factors(c, s, k);
                for m in ms {
                    // The depths whose biased sums land beside a code's midpoint.
                    let offset = slope * m + units;
                    let landed = depths
                        .iter()
                        .map(|&z| (f64::from(z) - f64::from(offset)) as f32);
                    for z in depths.iter().copied().chain(landed) {
                        if !(0.0..=1.0).contains(&z) {
                            continue;
                        }
                        let bits = |v: f32| vec![u64::from(v.to_bits())];
                        cases.push(vec![
                            bits(z),
                            bits(m),
                            bits(units),
                            bits(slope),
                            bits(clamp),
                        ]);
                    }
                }
            }
        }
    }
    part_of(cases, part, of)
}

/// `offset_depth` writes Metal's code of each biased depth.
fn biased_grid(part: usize, of: usize) {
    let f = float(32);
    check(
        "offset_depth",
        &[f, f, f, f, f],
        f,
        &|fb, xs| offset_depth(fb, xs[0], xs[1], [xs[2], xs[3], xs[4]]),
        &biased_cases(part, of),
        &|c| {
            let v = |i: usize| f32v(c[i][0]);
            metal_biased_code(v(0), v(1), [v(2), v(3), v(4)]).map(|code| vec![written(code)])
        },
        false,
    );
}

parts!(biased_grid, 4;
    a_biased_depth_is_written_as_metal_s_code_0 = 0,
    a_biased_depth_is_written_as_metal_s_code_1 = 1,
    a_biased_depth_is_written_as_metal_s_code_2 = 2,
    a_biased_depth_is_written_as_metal_s_code_3 = 3,
);

/// The cases of part `part` of `of` whose bias has side `raises` (`metal::state::depth16_bias_side`):
/// depths beside codes and midpoints, 0 and 1 among them, and the depths whose biased sums land
/// beside a midpoint, under constants, slope factors and slopes, and clamps of 0, past the units,
/// at them and short of them on either side.
fn sided_cases(raises: bool, part: usize, of: usize) -> Vec<Case> {
    let constants = [
        -30000.0, -1000.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 100.0, 30000.0,
    ];
    let slopes = [0.0, -0.0, 0.5, -1.0, 2.0, -1000.0, 1000.0];
    let ms = [0.0, 1.0 / 4096.0, 1.0 / 1080.0, 0.0123, 0.5, 64.0];
    let depths = around_codes(8191, 1);
    let mut cases = Vec::new();
    for c in constants {
        for s in slopes {
            let units = factors(c, s, 0.0)[0];
            let clamps = [
                0.0,
                1.0 / 8192.0,
                -1.0 / 8192.0,
                units,
                -units,
                units / 2.0,
                -units / 2.0,
                units * 2.0,
            ];
            for k in clamps {
                if crate::metal::state::depth16_bias_side(c, s, k) != Some(raises) {
                    continue;
                }
                let [units, slope, clamp] = factors(c, s, k);
                for m in ms {
                    let offset = slope * m + units;
                    let landed = depths
                        .iter()
                        .map(|&z| (f64::from(z) - f64::from(offset)) as f32);
                    for z in depths.iter().copied().chain(landed) {
                        if !(0.0..=1.0).contains(&z) {
                            continue;
                        }
                        let bits = |v: f32| vec![u64::from(v.to_bits())];
                        cases.push(vec![
                            bits(z),
                            bits(m),
                            bits(units),
                            bits(slope),
                            bits(clamp),
                        ]);
                    }
                }
            }
        }
    }
    part_of(cases, part, of)
}

/// A bias of known side bounds the written depth by `FragCoord.z` on that side
/// (`side_bounded`), which a stage declaring `DepthGreater` or `DepthLess` must: on every case of
/// that side the bound writes Metal's code unchanged, and that code's written value lies on the
/// side of the fragment's depth (the proof in `side_bounded`'s docs, step 3).
fn sided_grid(raises: bool, part: usize, of: usize) {
    let f = float(32);
    let cases = sided_cases(raises, part, of);
    assert!(cases.len() > 1000, "too few cases: {}", cases.len());
    check(
        "side_bounded",
        &[f, f, f, f, f],
        f,
        &|fb, xs| {
            let stored = offset_depth(fb, xs[0], xs[1], [xs[2], xs[3], xs[4]])?;
            side_bounded(fb, stored, xs[0], raises)
        },
        &cases,
        &|c| {
            let v = |i: usize| f32v(c[i][0]);
            let code = metal_biased_code(v(0), v(1), [v(2), v(3), v(4)])?;
            let w = f32::from_bits(written(code) as u32);
            let z = v(0);
            assert!(
                if raises { w >= z } else { w <= z },
                "code {code} of {z:e} under {:?} writes {w:e}, past the fragment's depth",
                [v(1), v(2), v(3), v(4)]
            );
            Some(vec![written(code)])
        },
        false,
    );
}

fn raising_grid(part: usize, of: usize) {
    sided_grid(true, part, of);
}

fn lowering_grid(part: usize, of: usize) {
    sided_grid(false, part, of);
}

parts!(raising_grid, 4;
    a_raising_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_0 = 0,
    a_raising_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_1 = 1,
    a_raising_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_2 = 2,
    a_raising_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_3 = 3,
);

parts!(lowering_grid, 4;
    a_lowering_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_0 = 0,
    a_lowering_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_1 = 1,
    a_lowering_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_2 = 2,
    a_lowering_bias_bounded_by_the_fragment_s_depth_writes_metal_s_code_3 = 3,
);

/// The sides `metal::state::depth16_bias_side` gives: a bias of units above 0 and a slope factor
/// not below 0 raises the depth, one of units below 0 and a slope factor not above 0 lowers it, a
/// clamp short of the units on their side leaves it unknown, and so do opposite signs and no bias.
#[test]
fn a_bias_has_a_side_only_where_every_triangle_s_depth_takes_it() {
    use crate::metal::state::depth16_bias_side as side;
    let unit = 1.0 / 65535.0;
    assert_eq!(side(0.0, 0.0, 0.0), None);
    assert_eq!(side(2.0, 0.0, 0.0), Some(true));
    assert_eq!(side(0.0, 1.0, 0.0), Some(true));
    assert_eq!(side(0.0, -0.0, 0.0), None);
    assert_eq!(side(-0.0, 0.5, 0.0), Some(true));
    assert_eq!(side(-2.0, 0.0, 0.0), Some(false));
    assert_eq!(side(-0.5, -3.0, 0.0), Some(false));
    assert_eq!(side(2.0, -1.0, 0.0), None);
    assert_eq!(side(-2.0, 1.0, 0.0), None);
    assert_eq!(side(0.0, -1.0, 0.0), None);
    // A clamp on the bias's side keeps it only at or past the units (3 of them for 2).
    assert_eq!(side(2.0, 0.0, 3.0 * unit), Some(true));
    assert_eq!(side(2.0, 0.0, 2.0 * unit), None);
    assert_eq!(side(2.0, 0.0, -unit), Some(true));
    assert_eq!(side(-2.0, 0.0, -3.0 * unit), Some(false));
    assert_eq!(side(-2.0, 0.0, -2.0 * unit), None);
    assert_eq!(side(-2.0, 0.0, unit), Some(false));
    // Infinite or NaN factors have none; a NaN or infinite clamp bounds nothing.
    assert_eq!(side(f32::INFINITY, 0.0, 0.0), None);
    assert_eq!(side(1.0, f32::NAN, 0.0), None);
    assert_eq!(side(1.0, f32::INFINITY, 0.0), None);
    assert_eq!(side(1.0, 0.0, f32::NAN), Some(true));
    assert_eq!(side(1.0, 0.0, f32::INFINITY), Some(true));
    assert_eq!(side(-1.0, 0.0, f32::NEG_INFINITY), Some(false));
}
