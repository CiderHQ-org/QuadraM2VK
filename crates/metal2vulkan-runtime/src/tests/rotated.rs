//! `rotated`: where a rotated texture's turns put each texel, against Metal's measured rule
//! (the loads).

use super::super::rotated::{loads_apart, quarter, source_map, stored_extent, Turn};

/// Where Metal stores a pass's texel `(x, y)` of a `w x h` frame at `rotation` (measured).
fn metal_stores(rotation: u8, (w, h): (i32, i32), (x, y): (i32, i32)) -> (i32, i32) {
    match rotation {
        1 | 4 => (h - 1 - y, x),
        2 => (w - 1 - x, h - 1 - y),
        3 => (y, w - 1 - x),
        _ => (x, y),
    }
}

/// The plane's texel a pass that draws loads into its texel `(x, y)` of a frame at `rotation`
/// (measured): the inverse of the store, but at rotation 4 the plane's transpose.
fn metal_loads(rotation: u8, frame: (i32, i32), (x, y): (i32, i32)) -> (i32, i32) {
    match rotation {
        4 => (y, x),
        _ => metal_stores(rotation, frame, (x, y)),
    }
}

fn apply(m: [i32; 6], (x, y): (i32, i32)) -> (i32, i32) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

/// For every rotation and three frames, the load takes each pass texel from the plane's texel
/// Metal loads it from, the keep from the stored texel Metal puts it at, the store takes each
/// stored texel from the pass texel Metal puts there, and the store is the extent Metal makes
/// it: a quarter turn's (1, 3 and 4) the frame's height wide. The load and the keep differ at
/// rotation 4 alone.
#[test]
fn turns_take_each_texel_from_where_metal_stores_it() {
    for rotation in 0..=4u8 {
        for (w, h) in [(64u32, 16u32), (16, 64), (7, 5)] {
            let [sw, sh] = stored_extent(rotation, [w, h]);
            assert_eq!(
                quarter(rotation),
                (sw, sh) == (h, w) && w != h,
                "{rotation}"
            );
            let frame = (w as i32, h as i32);
            let load = source_map(rotation, [w, h], Turn::Load);
            let keep = source_map(rotation, [w, h], Turn::Keep);
            let back = source_map(rotation, [w, h], Turn::Store);
            assert_eq!(loads_apart(rotation, [w, h]), rotation == 4, "{rotation}");
            let mut seen = vec![false; (sw * sh) as usize];
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let (sx, sy) = metal_stores(rotation, frame, (x, y));
                    assert!(sx >= 0 && sy >= 0 && sx < sw as i32 && sy < sh as i32);
                    assert_eq!(
                        apply(keep, (x, y)),
                        (sx, sy),
                        "{rotation} {w}x{h} ({x}, {y})"
                    );
                    assert_eq!(
                        apply(load, (x, y)),
                        metal_loads(rotation, frame, (x, y)),
                        "{rotation} {w}x{h} ({x}, {y})"
                    );
                    assert_eq!(
                        apply(back, (sx, sy)),
                        (x, y),
                        "{rotation} {w}x{h} ({sx}, {sy})"
                    );
                    seen[(sy as u32 * sw + sx as u32) as usize] = true;
                }
            }
            assert!(
                seen.iter().all(|&s| s),
                "every stored texel is a pass texel's"
            );
        }
    }
}

/// A pass that draws, loading a frame of rotation 4 and storing it again, leaves the plane's
/// texel `(x, y)` holding the texel it held at `(W' - 1 - x, y)`, `W'` the plane's width: mirrored
/// left to right, Metal's answer (the consumer's 32x32 frame's texel (0, 0) takes (31, 0), and
/// (20, 30) takes (11, 30)). Every other rotation's leaves each texel in place, as the keep leaves
/// rotation 4's.
#[test]
fn a_loaded_and_stored_frame_lands_where_metal_puts_it() {
    for rotation in 0..=4u8 {
        for (w, h) in [(32u32, 32u32), (64, 16), (24, 40)] {
            let [sw, sh] = stored_extent(rotation, [w, h]);
            let back = source_map(rotation, [w, h], Turn::Store);
            for turn in [Turn::Load, Turn::Keep] {
                let into = source_map(rotation, [w, h], turn);
                for sy in 0..sh as i32 {
                    for sx in 0..sw as i32 {
                        let held = apply(into, apply(back, (sx, sy)));
                        let metal = match (rotation, turn) {
                            (4, Turn::Load) => (sw as i32 - 1 - sx, sy),
                            _ => (sx, sy),
                        };
                        assert_eq!(held, metal, "{rotation} {w}x{h} {turn:?} ({sx}, {sy})");
                    }
                }
            }
        }
    }
    let load = source_map(4, [32, 32], Turn::Load);
    let back = source_map(4, [32, 32], Turn::Store);
    assert_eq!(apply(load, apply(back, (0, 0))), (31, 0));
    assert_eq!(apply(load, apply(back, (20, 30))), (11, 30));
}
