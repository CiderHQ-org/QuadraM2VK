//! The tile reach rule (the module docs) as a plain function over one primitive's clip-space
//! vertices: the reference every measured case checks, and the kernel answers as.
//!
//! Arithmetic: a vertex's window position in `f32` (`x_w = (w / 2) * x_ndc + (o_x + w / 2)`, y
//! down, each operation rounded on its own), snapped to 1/256 of a pixel ties to even (the cases
//! measured cannot tell even from up), and every test after that in integers: a primitive's
//! closed area against a half-open rectangle by the separating axes of both (the rectangle's far
//! edges approached from inside, [`meets`]), a triangle's sample cull row by row of samples
//! ([`covers_sample`]), a point's square from its integer extent ([`point_extent`]).

use std::collections::BTreeSet;

/// Units of a pixel a vertex snaps to.
pub(super) const SUBPIXEL: i64 = 256;

/// The largest snapped coordinate the rule is measured near (1e5 pixels) is far below this (2^21
/// pixels): past it a coordinate is clamped and the answer flagged unmeasured. Below it every
/// product the kernel's integer tests form fits an `i64`.
pub(super) const COORDINATE_LIMIT: i64 = 1 << 29;

/// The largest point size Metal draws, in pixels: a larger size draws as this one.
pub(super) const POINT_SIZE_LIMIT: f32 = 511.0;

/// Which faces a draw culls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cull {
    /// None.
    None,
    /// Front faces.
    Front,
    /// Back faces.
    Back,
}

/// Which winding faces front.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Winding {
    /// Clockwise in window coordinates (y down), Metal's default.
    Clockwise,
    /// Counter-clockwise.
    CounterClockwise,
}

/// A draw's rasterization state: the pass's render area and samples, the encoder's viewport,
/// scissor, cull mode, winding and depth clip mode.
#[derive(Clone, Copy, Debug)]
pub(super) struct Raster {
    /// The render area's width and height, in pixels.
    pub area: [u32; 2],
    /// The viewport's origin and size, in pixels.
    pub viewport: [f32; 4],
    /// The scissor rectangle's origin and size, in pixels.
    pub scissor: [u32; 4],
    /// The cull mode.
    pub cull: Cull,
    /// The front-facing winding.
    pub winding: Winding,
    /// Whether depth clamps rather than clips.
    pub clamp: bool,
    /// The pass's samples per pixel: 1, 2 or 4.
    pub samples: u32,
}

impl Raster {
    /// Metal's defaults over a render area of `area`: the viewport and scissor of the area, no
    /// cull, clockwise front faces, depth clipped, one sample.
    pub(super) fn over(area: [u32; 2]) -> Raster {
        Raster {
            area,
            viewport: [0.0, 0.0, area[0] as f32, area[1] as f32],
            scissor: [0, 0, area[0], area[1]],
            cull: Cull::None,
            winding: Winding::Clockwise,
            clamp: false,
            samples: 1,
        }
    }

    /// A tile's width and height in pixels: 32 by 32, 32 by 16 at 4 samples.
    pub(super) fn tile(&self) -> [u32; 2] {
        match self.samples {
            4 => [32, 16],
            _ => [32, 32],
        }
    }

    /// The sample positions of a pixel in 1/256 units: Metal's defaults, Vulkan's standard ones
    /// (a triangle holding each one alone reaches, at 2 and at 4 samples).
    fn sample_offsets(&self) -> &'static [[i64; 2]] {
        match self.samples {
            2 => &[[192, 192], [64, 64]],
            4 => &[[96, 32], [224, 96], [32, 160], [160, 224]],
            _ => &[[128, 128]],
        }
    }

    /// The half-open rectangle `[x0, x1) x [y0, y1)`, in 1/256 units, that clips every
    /// primitive: the viewport's, the scissor's and the render area's.
    fn region(&self) -> [i64; 4] {
        let [ox, oy, w, h] = self.viewport;
        let along = |o: f32, e: f32| {
            let (a, b) = (snap_clamped(o).0, snap_clamped(o + e).0);
            (a.min(b), a.max(b))
        };
        let (vx0, vx1) = along(ox, w);
        let (vy0, vy1) = along(oy, h);
        let [sx, sy, sw, sh] = self.scissor.map(i64::from);
        let [aw, ah] = self.area.map(i64::from);
        [
            vx0.max(sx * SUBPIXEL).max(0),
            vx1.min((sx + sw) * SUBPIXEL).min(aw * SUBPIXEL),
            vy0.max(sy * SUBPIXEL).max(0),
            vy1.min((sy + sh) * SUBPIXEL).min(ah * SUBPIXEL),
        ]
    }

    /// Whether the viewport's edges lie within the coordinates measured.
    fn viewport_measured(&self) -> bool {
        let [ox, oy, w, h] = self.viewport;
        [ox, ox + w, oy, oy + h]
            .into_iter()
            .all(|c| snap_clamped(c).1)
    }

    /// The window position of the clip-space vertex `v`, unsnapped.
    fn window(&self, v: [f32; 4]) -> [f32; 2] {
        let [ox, oy, w, h] = self.viewport;
        let (xn, yn) = (v[0] / v[3], v[1] / v[3]);
        let (px, py) = (w * 0.5, h * 0.5);
        [px * xn + (ox + px), -py * yn + (oy + py)]
    }
}

/// One primitive's clip-space vertices.
#[derive(Clone, Copy, Debug)]
pub(super) enum Primitive {
    /// A point and its size.
    Point([f32; 4], f32),
    /// A line's two ends.
    Line([[f32; 4]; 2]),
    /// A triangle's three vertices.
    Triangle([[f32; 4]; 3]),
}

/// Why the tiles a primitive reached may not be Metal's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Flags {
    /// A primitive crossed the near or far plane under depth clipping: its reach is that of the
    /// clipped polygon, whose new vertices Apple's clipper may place otherwise (unmeasured).
    pub near_far: bool,
    /// A case the measurements do not cover: a vertex not finite, with a subnormal component, at
    /// or behind the eye, or past the coordinates measured; a viewport past them; a line of no
    /// length; a point past the depth range, or of a size negative, not a number or subnormal.
    pub unmeasured: bool,
}

/// The tiles `[x, y]` that `prim`, drawn under `raster`, reaches.
pub(super) fn reach(raster: &Raster, prim: &Primitive, flags: &mut Flags) -> BTreeSet<[u32; 2]> {
    let mut tiles = BTreeSet::new();
    if !raster.viewport_measured() {
        flags.unmeasured = true;
    }
    match *prim {
        Primitive::Point(v, size) => point(raster, v, size, flags, &mut tiles),
        Primitive::Line(ends) => line(raster, ends, flags, &mut tiles),
        Primitive::Triangle(vertices) => triangle(raster, vertices, flags, &mut tiles),
    }
    tiles
}

/// Every tile of the render area, as a pass's clear reaches it.
pub(super) fn every_tile(raster: &Raster) -> BTreeSet<[u32; 2]> {
    let [tw, th] = raster.tile();
    let [w, h] = raster.area;
    (0..h.div_ceil(th))
        .flat_map(|y| (0..w.div_ceil(tw)).map(move |x| [x, y]))
        .collect()
}

/// The coordinate `c` snapped to 1/256, and whether it lay within the coordinates measured.
pub(super) fn snap_clamped(c: f32) -> (i64, bool) {
    let scaled = f64::from(c) * SUBPIXEL as f64;
    if !scaled.is_finite() {
        return (0, false);
    }
    let limit = COORDINATE_LIMIT as f64;
    if scaled.abs() >= limit {
        return (scaled.clamp(-limit, limit) as i64, false);
    }
    (scaled.round_ties_even() as i64, true)
}

/// The snapped window position of the clip-space vertex `v`.
fn snapped(raster: &Raster, v: [f32; 4], flags: &mut Flags) -> [i64; 2] {
    let [x, y] = raster.window(v);
    let (x, xi) = snap_clamped(x);
    let (y, yi) = snap_clamped(y);
    if !(xi && yi) {
        flags.unmeasured = true;
    }
    [x, y]
}

/// The planes a primitive is clipped by, as signed distances (inside where not negative):
/// `z >= 0` and `z <= w` under depth clipping.
fn depth_planes(raster: &Raster) -> Vec<fn(&[f32; 4]) -> f32> {
    let near: fn(&[f32; 4]) -> f32 = |v| v[2];
    let far: fn(&[f32; 4]) -> f32 = |v| v[3] - v[2];
    match raster.clamp {
        true => Vec::new(),
        false => vec![near, far],
    }
}

/// `polygon` clipped by the plane `distance` (Sutherland-Hodgman, in `f32`); a polygon of two
/// vertices is a segment.
pub(super) fn clip(polygon: &[[f32; 4]], distance: fn(&[f32; 4]) -> f32) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    let n = polygon.len();
    let edges = if n == 2 { 1 } else { n };
    for i in 0..edges {
        let (a, b) = (polygon[i], polygon[(i + 1) % n]);
        let (da, db) = (distance(&a), distance(&b));
        if da >= 0.0 {
            out.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            out.push([0, 1, 2, 3].map(|k| a[k] + t * (b[k] - a[k])));
        }
        if n == 2 && db >= 0.0 {
            out.push(b);
        }
    }
    out
}

/// `vertices` clipped by the depth planes (flagging a crossing) and by `w > 0` (flagging a
/// vertex at or behind the eye); empty where wholly past one plane.
fn depth_clipped(raster: &Raster, vertices: &[[f32; 4]], flags: &mut Flags) -> Vec<[f32; 4]> {
    if vertices.iter().flatten().any(|c| !c.is_finite()) {
        flags.unmeasured = true;
        return Vec::new();
    }
    if vertices.iter().flatten().any(|c| c.is_subnormal()) {
        flags.unmeasured = true;
    }
    let mut polygon = vertices.to_vec();
    for plane in depth_planes(raster) {
        if polygon.iter().all(|v| plane(v) < 0.0) {
            return Vec::new();
        }
        if polygon.iter().any(|v| plane(v) < 0.0) {
            flags.near_far = true;
            polygon = clip(&polygon, plane);
        }
    }
    if polygon.iter().any(|v| v[3] <= 0.0) {
        flags.unmeasured = true;
        polygon = clip(&polygon, |v| v[3]);
    }
    polygon
}

/// A triangle's reach (the module docs).
fn triangle(
    raster: &Raster,
    vertices: [[f32; 4]; 3],
    flags: &mut Flags,
    tiles: &mut BTreeSet<[u32; 2]>,
) {
    let polygon = depth_clipped(raster, &vertices, flags);
    if polygon.len() < 3 {
        return;
    }
    let points: Vec<[i64; 2]> = polygon.iter().map(|&v| snapped(raster, v, flags)).collect();
    for i in 1..points.len() - 1 {
        let t = [points[0], points[i], points[i + 1]];
        let area = cross(t[0], t[1], t[2]);
        if area == 0 {
            continue;
        }
        // Clockwise in window coordinates, y down, is a positive area.
        let front = (area > 0) == (raster.winding == Winding::Clockwise);
        let culled = match raster.cull {
            Cull::None => false,
            Cull::Front => front,
            Cull::Back => !front,
        };
        if culled || sample_culled(raster, t, area.signum()) {
            continue;
        }
        let polygon: Vec<[i128; 2]> = t.iter().map(|p| p.map(i128::from)).collect();
        bin(raster, &polygon, 1, tiles);
    }
}

/// Twice the signed area of the triangle `a`, `b`, `c`.
fn cross(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i128 {
    let d = |p: [i64; 2], q: [i64; 2]| [i128::from(q[0] - p[0]), i128::from(q[1] - p[1])];
    let (u, v) = (d(a, b), d(a, c));
    u[0] * v[1] - u[1] * v[0]
}

/// `n / d` rounded down.
fn floor_div(n: i128, d: i128) -> i128 {
    let q = n / d;
    if (n % d != 0) && ((n < 0) != (d < 0)) {
        q - 1
    } else {
        q
    }
}

/// `n / d` rounded up.
fn ceil_div(n: i128, d: i128) -> i128 {
    -floor_div(-n, d)
}

/// Whether the small-primitive cull drops the triangle `t` of orientation `sign`: one whose box
/// (closed) holds few sample positions and whose closed area holds none of them. The cull tests a
/// box of at most 12 positions at 1 and 2 samples and of at most 24 at 4, in every shape measured
/// (1 by 12 to 12 by 1 pixel centres; slopes from 1 in 3 to 3 in 1 at 2 and 4 samples), and
/// keeps one of more.
fn sample_culled(raster: &Raster, t: [[i64; 2]; 3], sign: i128) -> bool {
    let tested = match raster.samples {
        4 => 24,
        _ => 12,
    };
    box_samples(raster, t) <= tested && !covers_sample(raster, t, sign)
}

/// The sample positions the closed box of `t` holds.
fn box_samples(raster: &Raster, t: [[i64; 2]; 3]) -> i128 {
    let unit = i128::from(SUBPIXEL);
    let span = |axis: usize, offset: i64| {
        let lo = i128::from(t.iter().map(|p| p[axis]).min().unwrap_or(0));
        let hi = i128::from(t.iter().map(|p| p[axis]).max().unwrap_or(0));
        let o = i128::from(offset);
        (floor_div(hi - o, unit) - ceil_div(lo - o, unit) + 1).max(0)
    };
    raster
        .sample_offsets()
        .iter()
        .map(|&[ox, oy]| span(0, ox) * span(1, oy))
        .sum()
}

/// Whether the closed triangle `t`, of orientation `sign`, holds a sample of the plane: row by
/// row of sample positions, the span the three edges leave holds one.
fn covers_sample(raster: &Raster, t: [[i64; 2]; 3], sign: i128) -> bool {
    let unit = i128::from(SUBPIXEL);
    let min_y = i128::from(t.iter().map(|p| p[1]).min().unwrap_or(0));
    let max_y = i128::from(t.iter().map(|p| p[1]).max().unwrap_or(0));
    for &[ox, oy] in raster.sample_offsets() {
        let (ox, oy) = (i128::from(ox), i128::from(oy));
        for row in ceil_div(min_y - oy, unit)..=floor_div(max_y - oy, unit) {
            let y = row * unit + oy;
            let (mut lo, mut hi) = (i128::MIN, i128::MAX);
            let mut empty = false;
            for k in 0..3 {
                let (a, b) = (t[k].map(i128::from), t[(k + 1) % 3].map(i128::from));
                // sign * cross(b - a, (x, y) - a) >= 0, as alpha - beta * x >= 0.
                let beta = sign * (b[1] - a[1]);
                let alpha = sign * ((b[0] - a[0]) * (y - a[1]) + (b[1] - a[1]) * a[0]);
                match beta.signum() {
                    1 => hi = hi.min(floor_div(alpha, beta)),
                    -1 => lo = lo.max(ceil_div(alpha, beta)),
                    _ => empty |= alpha < 0,
                }
            }
            if !empty && lo <= hi && ceil_div(lo - ox, unit) <= floor_div(hi - ox, unit) {
                return true;
            }
        }
    }
    false
}

/// Add each tile the closed convex `polygon` meets, its vertices in 1/256 units times `scale`,
/// within the region of `raster`.
fn bin(raster: &Raster, polygon: &[[i128; 2]], scale: i128, tiles: &mut BTreeSet<[u32; 2]>) {
    let [x0, x1, y0, y1] = raster.region().map(i128::from);
    let [tw, th] = raster.tile().map(|t| i128::from(t) * i128::from(SUBPIXEL));
    let [w, h] = raster.area;
    for ty in 0..h.div_ceil(raster.tile()[1]) {
        for tx in 0..w.div_ceil(raster.tile()[0]) {
            let (i, j) = (i128::from(tx), i128::from(ty));
            let rect = [
                x0.max(i * tw),
                x1.min((i + 1) * tw),
                y0.max(j * th),
                y1.min((j + 1) * th),
            ];
            if rect[0] < rect[1] && rect[2] < rect[3] && meets(polygon, scale, rect) {
                tiles.insert([tx, ty]);
            }
        }
    }
}

/// Whether the closed convex `polygon` (in 1/256 units times `scale`, of non-zero area) meets
/// the half-open rectangle `[x0, x1) x [y0, y1)` (in 1/256 units): no separating axis among the
/// rectangle's two and the polygon's edge normals, the rectangle's far edges taken an
/// infinitesimal inside it (a tie on one of them separates).
fn meets(polygon: &[[i128; 2]], scale: i128, [x0, x1, y0, y1]: [i128; 4]) -> bool {
    let [x0, x1, y0, y1] = [x0, x1, y0, y1].map(|c| c * scale);
    let xs = polygon.iter().map(|p| p[0]);
    let ys = polygon.iter().map(|p| p[1]);
    let (min_x, max_x) = (xs.clone().min(), xs.max());
    let (min_y, max_y) = (ys.clone().min(), ys.max());
    let (Some(min_x), Some(max_x), Some(min_y), Some(max_y)) = (min_x, max_x, min_y, max_y) else {
        return false;
    };
    if !(min_x < x1 && max_x >= x0 && min_y < y1 && max_y >= y0) {
        return false;
    }
    let n = polygon.len();
    let doubled: i128 = (0..n)
        .map(|i| {
            let (p, q) = (polygon[i], polygon[(i + 1) % n]);
            p[0] * q[1] - p[1] * q[0]
        })
        .sum();
    let sign = doubled.signum();
    for i in 0..n {
        let (p, q) = (polygon[i], polygon[(i + 1) % n]);
        // f(x, y) = sign * cross(q - p, (x, y) - p), not negative inside.
        let gx = -sign * (q[1] - p[1]);
        let gy = sign * (q[0] - p[0]);
        let x = if gx > 0 { x1 } else { x0 };
        let y = if gy > 0 { y1 } else { y0 };
        let f = gx * (x - p[0]) + gy * (y - p[1]);
        if f < 0 || (f == 0 && (gx > 0 || gy > 0)) {
            return false;
        }
    }
    true
}

/// A line's reach (the module docs).
fn line(raster: &Raster, ends: [[f32; 4]; 2], flags: &mut Flags, tiles: &mut BTreeSet<[u32; 2]>) {
    let segment = depth_clipped(raster, &ends, flags);
    if segment.len() < 2 {
        return;
    }
    let [p, q] = [segment[0], segment[1]].map(|v| snapped(raster, v, flags).map(i128::from));
    let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
    let half = i128::from(SUBPIXEL / 2);
    if dx == 0 && dy == 0 {
        flags.unmeasured = true;
        let square = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            .map(|[sx, sy]| [p[0] + sx * half, p[1] + sy * half]);
        bin(raster, &square, 1, tiles);
        return;
    }
    // Scaled by the major axis's extent: the ends move half a pixel along the major axis, along
    // the line, and the sides lie half a pixel each way across it.
    let x_major = dx.abs() > dy.abs();
    let scale = if x_major { dx.abs() } else { dy.abs() };
    let along = [half * dx, half * dy];
    let across = match x_major {
        true => [0, half * scale],
        false => [half * scale, 0],
    };
    let (p, q) = (p.map(|c| c * scale), q.map(|c| c * scale));
    let start = [p[0] - along[0], p[1] - along[1]];
    let end = [q[0] + along[0], q[1] + along[1]];
    let quad = [
        [start[0] - across[0], start[1] - across[1]],
        [end[0] - across[0], end[1] - across[1]],
        [end[0] + across[0], end[1] + across[1]],
        [start[0] + across[0], start[1] + across[1]],
    ];
    bin(raster, &quad, scale, tiles);
}

/// Where a point of `size` pixels (not negative, at most [`POINT_SIZE_LIMIT`]) reaches about its
/// snapped centre, in 1/256 units: how far back from the centre its square starts, along x and
/// along y, and its extent from there. It starts half its size back, the start rounded half up
/// (the left and top edges reach from more than half a unit past a tile's edge), and spans its
/// size rounded to the nearest 1/16 of a pixel, a half up (the right and bottom edges follow from
/// the left and top ones).
pub(super) fn point_extent(size: f32) -> (i64, i64) {
    // Exact in f64: an f32 size below 512 times 256.
    let units = f64::from(size) * SUBPIXEL as f64;
    let back = (units / 2.0 - 0.5).ceil() as i64;
    let extent = ((units / 16.0 + 0.5).floor() * 16.0) as i64;
    (back, extent)
}

/// A point's reach (the module docs).
fn point(
    raster: &Raster,
    v: [f32; 4],
    size: f32,
    flags: &mut Flags,
    tiles: &mut BTreeSet<[u32; 2]>,
) {
    if v.iter().any(|c| !c.is_finite()) || v[3] <= 0.0 {
        flags.unmeasured = true;
        return;
    }
    if v.iter().any(|c| c.is_subnormal()) {
        flags.unmeasured = true;
    }
    if depth_planes(raster).iter().any(|plane| plane(&v) < 0.0) {
        flags.unmeasured = true;
        return;
    }
    let size = match size >= 0.0 {
        true => size,
        false => {
            flags.unmeasured = true;
            0.0
        }
    };
    if size.is_subnormal() {
        flags.unmeasured = true;
    }
    let (back, extent) = point_extent(size.min(POINT_SIZE_LIMIT));
    let [cx, cy] = snapped(raster, v, flags);
    let [x0, y0] = [cx - back, cy - back].map(i128::from);
    let [x1, y1] = [x0, y0].map(|c| c + i128::from(extent));
    bin(raster, &[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 1, tiles);
}
