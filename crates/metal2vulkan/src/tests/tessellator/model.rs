//! A plain model of the emulated tessellator, written apart from its SPIR-V emitters
//! (`lower::tessellation::tessellator`) for the tests to hold them to.
//!
//! It is the D3D11 reference tessellator's factor processing, point placement and stitching
//! (Microsoft's `CHWTessellator`, MIT), reordered as Metal's tessellator on the M3 Max orders a
//! patch's triangles and their first corners, with Metal's batch of points across a draw's
//! patches. It holds no crate item, so the same file builds the offline comparison against the
//! probe model that was checked on the M3.

/// 1.0 in the reference's 16.16 fixed point.
pub const FXP_ONE: u32 = 0x1_0000;
const FXP_HALF: u32 = 0x8000;
/// The points one batch holds.
pub const BATCH: u32 = 512;

/// `MTLTessellationPartitionMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Pow2,
    Integer,
    FractionalOdd,
    FractionalEven,
}

/// The pipeline state the tessellator bakes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Model {
    pub quad: bool,
    pub part: Part,
    pub cw: bool,
    pub max_factor: u32,
    pub scale_enabled: bool,
}

/// A 1D factor's placement context (`TESS_FACTOR_CONTEXT`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ctx {
    pub odd: bool,
    pub frac: u32,
    pub num_half: u32,
    pub split: u32,
    pub inv_floor: u32,
    pub inv_ceil: u32,
}

/// `s_fixedReciprocal[n]`: `65536 / n` rounded to nearest (no tie arises for `n <= 64`).
pub fn reciprocal(n: u32) -> u32 {
    (131_072 + n) / (2 * n)
}

fn remove_msb(v: u32) -> u32 {
    if v == 0 {
        0
    } else {
        v & !(1u32 << (31 - v.leading_zeros()))
    }
}

fn fxp_ceil(x: u32) -> u32 {
    if x & 0xffff != 0 {
        (x & 0x7fff_0000) + FXP_ONE
    } else {
        x
    }
}

/// `ComputeTessFactorContext`.
pub fn context(f: u32, odd: bool) -> Ctx {
    let mut half = f.div_ceil(2);
    if odd || half == FXP_HALF {
        half += FXP_HALF;
    }
    let floor = half & 0x7fff_0000;
    let ceil = fxp_ceil(half);
    let num_half = ceil >> 16;
    let split = if ceil == floor {
        num_half + 1
    } else if odd {
        if floor == FXP_ONE {
            0
        } else {
            (remove_msb((floor >> 16) - 1) << 1) + 1
        }
    } else {
        (remove_msb(floor >> 16) << 1) + 1
    };
    let mut floor_segments = (floor * 2) >> 16;
    let mut ceil_segments = (ceil * 2) >> 16;
    if odd {
        floor_segments -= 1;
        ceil_segments -= 1;
    }
    Ctx {
        odd,
        frac: half - floor,
        num_half,
        split,
        inv_floor: reciprocal(floor_segments),
        inv_ceil: reciprocal(ceil_segments),
    }
}

/// `NumPointsForTessFactor`.
pub fn num_points(f: u32, odd: bool) -> u32 {
    if odd {
        (fxp_ceil(FXP_HALF + f.div_ceil(2)) * 2) >> 16
    } else {
        ((fxp_ceil(f.div_ceil(2)) * 2) >> 16) + 1
    }
}

/// `PlacePointIn1D`: point `point`'s location, 16.16.
pub fn place_1d(c: &Ctx, point: u32) -> u32 {
    let (mut p, flip) = if point >= c.num_half {
        let mut p = (c.num_half << 1) - point;
        if c.odd {
            p -= 1;
        }
        (p, true)
    } else {
        (point, false)
    };
    if p == c.num_half {
        return FXP_HALF;
    }
    let on_ceil = p;
    if p > c.split {
        p -= 1;
    }
    let lf = p * c.inv_floor;
    let lc = on_ceil * c.inv_ceil;
    let loc = lf
        .wrapping_mul(FXP_ONE - c.frac)
        .wrapping_add(lc.wrapping_mul(c.frac));
    let loc = (loc + FXP_HALF) >> 16;
    if flip {
        FXP_ONE - loc
    } else {
        loc
    }
}

/// `half` bits as `f32`, exactly.
pub fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    if exp == 31 {
        if h & 0x3ff != 0 {
            f32::NAN
        } else {
            sign * f32::INFINITY
        }
    } else if exp == 0 {
        sign * mant * f32::from_bits(0x3380_0000) // 2^-24
    } else {
        sign * (1024.0 + mant) * 2f32.powi(exp - 25)
    }
}

fn is_nan(a: f32) -> bool {
    a.is_nan()
}

fn flush(a: f32) -> f32 {
    if a.is_finite() && a != 0.0 && a.abs() < f32::MIN_POSITIVE {
        if a.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        a
    }
}

/// The reference's `tess_fmin`.
fn tess_fmin(a: f32, b: f32) -> f32 {
    let (fa, fb) = (flush(a), flush(b));
    if is_nan(fb) {
        a
    } else if fa == 0.0 && fb == 0.0 {
        if fa.is_sign_negative() {
            a
        } else {
            b
        }
    } else if fa < fb {
        a
    } else {
        b
    }
}

/// The reference's `tess_fmax`.
fn tess_fmax(a: f32, b: f32) -> f32 {
    let (fa, fb) = (flush(a), flush(b));
    if is_nan(fb) {
        a
    } else if fa == 0.0 && fb == 0.0 {
        if fb.is_sign_negative() {
            a
        } else {
            b
        }
    } else if fa >= fb {
        a
    } else {
        b
    }
}

/// `floatToIDotF<15, 16, false>`.
pub fn float_to_fixed(input: f32) -> u32 {
    let bits = input.to_bits() as i32;
    let mantissa_mask = (1i32 << 23) - 1;
    let unbiased = ((bits >> 23) & 0xff) - 127;
    if unbiased == 128 && bits & mantissa_mask != 0 {
        0
    } else if bits >= (127 + 15) << 23 {
        u32::MAX
    } else if bits < 0 || unbiased < -16 - 1 {
        0
    } else {
        let mut out = (bits & mantissa_mask) | (1 << 23);
        let extra = 23 - 16 - unbiased;
        if extra >= 0 {
            let lsb = 1i32 << extra;
            let half = lsb >> 1;
            if out & lsb != 0 || (out & (lsb - 1)) > half {
                out += half;
            }
            out >>= extra;
        } else {
            out <<= -extra;
        }
        out as u32
    }
}

/// A point of a patch: what `place` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Point {
    /// A domain corner: `u` and `v` 0 or 1.
    Corner { u: bool, v: bool },
    /// A triangle's center, `(1/3, 1/3)` (an even inside factor's innermost ring).
    Center,
    /// Point `q` of outer edge `edge`'s factor.
    Outer { edge: u32, q: u32 },
    /// A quad's interior point: point `a` of the `u` inside factor, `b` of the `v` one.
    Grid { a: u32, b: u32 },
    /// A triangle's interior point: ring `ring`, edge `edge`, point `q` of the inside factor.
    Ring { edge: u32, ring: u32, q: u32 },
}

/// A patch's derived structure.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    pub culled: bool,
    pub minimum: bool,
    /// Outer edges' contexts and point counts.
    pub outer: Vec<(Ctx, u32)>,
    /// Inside factors' contexts and point counts (quad: `u`, `v`; triangle: one).
    pub inside: Vec<(Ctx, u32)>,
}

impl Model {
    fn edges(&self) -> usize {
        if self.quad {
            4
        } else {
            3
        }
    }

    fn insides(&self) -> usize {
        if self.quad {
            2
        } else {
            1
        }
    }

    fn integer(&self) -> bool {
        matches!(self.part, Part::Pow2 | Part::Integer)
    }

    /// The processed factors of a patch whose factors are the `half`s `halves`, at encoder scale
    /// `scale` (a `half`'s value; an infinite one as `2^32`): 16.16, each 0 for a culled patch.
    pub fn process(&self, halves: &[u16], scale: f32) -> [u32; 6] {
        let (ne, ni) = (self.edges(), self.insides());
        let mut x: Vec<f32> = halves[..ne + ni]
            .iter()
            .map(|&h| half_value(h) * if self.scale_enabled { scale } else { 1.0 })
            .collect();
        // Metal's own steps: the pipeline's maximum, `pow2`'s rounding.
        let upper = if self.part == Part::FractionalOdd {
            self.max_factor - 1
        } else {
            self.max_factor
        } as f32;
        for v in x.iter_mut() {
            if *v > upper {
                *v = upper;
            }
            if self.part == Part::Pow2 && *v > 0.0 {
                let c = v.clamp(1.0, 64.0).ceil() as u32;
                *v = c.next_power_of_two() as f32;
            }
        }
        // The reference's `ProcessTessFactors`.
        if x[..ne].iter().any(|v| v.is_nan() || *v <= 0.0) {
            return [0; 6];
        }
        let (lo, hi) = match self.part {
            Part::Pow2 | Part::Integer => (1.0f32, 64.0f32),
            Part::FractionalEven => (2.0, 64.0),
            Part::FractionalOdd => (1.0, 63.0),
        };
        for v in x[..ne].iter_mut() {
            *v = tess_fmin(hi, tess_fmax(lo, *v));
            if self.integer() {
                *v = v.ceil();
            }
        }
        let mut lo_in = lo;
        if self.part == Part::FractionalOdd {
            let above = 1.0 + f32::from_bits(0x3700_0000); // 1 + 2^-17
            let edges = x[..ne].iter().any(|v| *v > above);
            let insides = self.quad && x[ne..].iter().any(|v| *v > above);
            if edges || insides {
                lo_in = 1.0 + f32::from_bits(0x3780_0000); // 1 + 2^-16
            }
        }
        for v in x[ne..].iter_mut() {
            *v = tess_fmin(hi, tess_fmax(lo_in, *v));
            if self.integer() {
                *v = v.ceil();
            }
        }
        let mut out = [0u32; 6];
        for (o, v) in out.iter_mut().zip(x.iter()) {
            *o = float_to_fixed(*v);
        }
        out
    }

    /// The structure of a patch of processed factors `p`.
    pub fn patch(&self, p: &[u32; 6]) -> Patch {
        let (ne, ni) = (self.edges(), self.insides());
        if p[..ne].contains(&0) {
            return Patch {
                culled: true,
                ..Patch::default()
            };
        }
        let odd_of = |f: u32, inside: bool| match self.part {
            Part::FractionalOdd => true,
            Part::FractionalEven => false,
            Part::Pow2 | Part::Integer => (f >> 16) & 1 == 1 && !(inside && f == FXP_ONE),
        };
        let minimum =
            self.part != Part::FractionalEven && p[..ne + ni].iter().all(|f| *f == FXP_ONE);
        let outer = p[..ne]
            .iter()
            .map(|&f| {
                let odd = odd_of(f, false);
                (context(f, odd), num_points(f, odd))
            })
            .collect();
        let inside = p[ne..ne + ni]
            .iter()
            .map(|&f| {
                let odd = odd_of(f, true);
                let least = if odd { 4 } else { 3 };
                (context(f, odd), num_points(f, odd).max(least))
            })
            .collect();
        Patch {
            culled: false,
            minimum,
            outer,
            inside,
        }
    }

    /// A quad's interior grid: its cells along `u` and `v`.
    fn grid(pt: &Patch) -> (u32, u32) {
        (pt.inside[0].1 - 3, pt.inside[1].1 - 3)
    }

    /// The inside points of the transition ring's edge `k`.
    fn inside_edge_points(&self, pt: &Patch, k: usize) -> u32 {
        if self.quad {
            pt.inside[(k + 1) & 1].1 - 2
        } else {
            pt.inside[0].1 - 2
        }
    }

    /// The triangles of the transition ring's edge `k`.
    fn ring_edge_triangles(&self, pt: &Patch, k: usize) -> u32 {
        self.inside_edge_points(pt, k) + pt.outer[k].1 - 2
    }

    /// A triangle's regular rings and whether it has a center triangle.
    fn tri_rings(pt: &Patch) -> (u32, bool) {
        let n = pt.inside[0].1;
        let rings = (n + 1) >> 1;
        (rings.saturating_sub(2), pt.inside[0].0.odd)
    }

    /// The interior's regions (rows, columns), in Metal's order.
    pub fn regions(&self, pt: &Patch) -> Vec<(u32, u32)> {
        if self.quad {
            let (nu, nv) = Self::grid(pt);
            let (hu, hv, cu, cv) = (nu / 2, nv / 2, nu.div_ceil(2), nv.div_ceil(2));
            vec![(hu, nv - hv), (hv, cu), (nu - cu, cv), (nv - cv, nu - hu)]
        } else {
            let (k, odd) = Self::tri_rings(pt);
            vec![(k, k + u32::from(odd)); 3]
        }
    }

    /// Whether the interior has a center (a quad's middle cell, a triangle's center triangle).
    fn center(&self, pt: &Patch) -> bool {
        if self.quad {
            let (nu, nv) = Self::grid(pt);
            nu & 1 == 1 && nv & 1 == 1
        } else {
            Self::tri_rings(pt).1
        }
    }

    /// The triangles of a patch.
    pub fn triangle_count(&self, p: &[u32; 6]) -> u32 {
        let pt = self.patch(p);
        if pt.culled {
            return 0;
        }
        if pt.minimum {
            return if self.quad { 2 } else { 1 };
        }
        let ring: u32 = (0..self.edges())
            .map(|k| self.ring_edge_triangles(&pt, k))
            .sum();
        let cells: u32 = self.regions(&pt).iter().map(|(r, c)| r * c).sum();
        let center = match (self.center(&pt), self.quad) {
            (false, _) => 0,
            (true, true) => 2,
            (true, false) => 1,
        };
        ring + 2 * cells + center
    }

    /// The batch's inputs of a patch: its points, its interior's, its transition ring's and each
    /// of the ring's edges'.
    fn batch_inputs(&self, pt: &Patch) -> (u32, u32, u32, Vec<u32>) {
        let outer: u32 = pt.outer.iter().map(|(_, n)| n).sum::<u32>() - self.edges() as u32;
        if pt.minimum {
            return (outer, 0, outer, vec![outer]);
        }
        let interior = if self.quad {
            (pt.inside[0].1 - 2) * (pt.inside[1].1 - 2)
        } else {
            let n = pt.inside[0].1;
            let r = (n >> 1) - 1;
            if pt.inside[0].0.odd {
                3 * (r * (r + 1) - r)
            } else {
                3 * r * (r + 1) + 1
            }
        };
        let ring_inside = if self.quad {
            let (nu, nv) = Self::grid(pt);
            match (nu, nv) {
                (0, 0) => 1,
                (0, n) | (n, 0) => n + 1,
                (a, b) => 2 * (a + b),
            }
        } else {
            let e = pt.inside[0].1 - 2;
            if e == 1 {
                1
            } else {
                3 * (e - 1)
            }
        };
        let sides = (0..self.edges())
            .map(|k| pt.outer[k].1 + self.inside_edge_points(pt, k))
            .collect();
        let cells = self.center(pt) || self.regions(pt).iter().any(|(r, c)| *r > 0 && *c > 0);
        let interior_charged = if cells { interior } else { 0 };
        (
            outer + interior,
            interior_charged,
            outer + ring_inside,
            sides,
        )
    }

    /// The batch fill after a patch of processed factors `p` drawn at fill `used`, and the
    /// patch's tile word: per region (5 bits each, region 0 lowest), 0 for a region drawn whole,
    /// else `(T - 1) / 2` for a region drawn in tiles of side `T`.
    pub fn stream_step(&self, used: u32, p: &[u32; 6]) -> (u32, u32) {
        let pt = self.patch(p);
        if pt.culled {
            return (used, 0);
        }
        let (points, interior, ring, sides) = self.batch_inputs(&pt);
        let mut used = if used >= BATCH { 0 } else { used };
        if used + points <= BATCH {
            return (used + points, 0);
        }
        if used + ring <= BATCH {
            used += ring;
        } else {
            for k in 0..sides.len() {
                if used + sides[k] > BATCH {
                    let rest = interior + sides[k..].iter().sum::<u32>();
                    if rest <= BATCH {
                        return (rest, 0);
                    }
                    used = 0;
                }
                used += sides[k];
            }
        }
        let mut word = 0u32;
        let mut first = true;
        let regions = if pt.minimum {
            Vec::new()
        } else {
            self.regions(&pt)
        };
        for (index, (r, c)) in regions.into_iter().enumerate() {
            if r == 0 || c == 0 {
                continue;
            }
            let mut t = side(BATCH - used);
            if t < 5 {
                if first && interior <= BATCH {
                    return (interior, word);
                }
                used = 0;
                t = side(BATCH);
            }
            first = false;
            word |= ((t - 1) / 2) << (5 * index);
            let (th, tw) = tile_shape(r, c, t);
            for ti in (0..r).step_by(th as usize) {
                for tj in (0..c).step_by(tw as usize) {
                    let cost = (th.min(r - ti) + 1) * (tw.min(c - tj) + 1);
                    if used + cost > BATCH {
                        used = 0;
                    }
                    used += cost;
                }
            }
        }
        (used, word)
    }

    /// The points of triangle `l` of a patch of processed factors `p` and tile word `tiles`, in
    /// Metal's order, its first corner first.
    pub fn decode(&self, l: u32, p: &[u32; 6], tiles: u32) -> [Point; 3] {
        self.decode_in(l, &self.patch(p), tiles)
    }

    /// [`Model::decode`] in a patch's structure.
    pub fn decode_in(&self, l: u32, pt: &Patch, tiles: u32) -> [Point; 3] {
        let t = self.decode_cw(l, pt, tiles);
        if self.cw {
            t
        } else {
            [t[0], t[2], t[1]]
        }
    }

    fn decode_cw(&self, mut l: u32, pt: &Patch, tiles: u32) -> [Point; 3] {
        let corner = |u: bool, v: bool| Point::Corner { u, v };
        if pt.minimum {
            return if self.quad {
                if l == 0 {
                    [
                        corner(false, false),
                        corner(false, true),
                        corner(true, false),
                    ]
                } else {
                    [corner(false, true), corner(true, true), corner(true, false)]
                }
            } else {
                [
                    corner(false, false),
                    corner(false, true),
                    corner(true, false),
                ]
            };
        }
        for k in 0..self.edges() {
            let n = self.ring_edge_triangles(pt, k);
            if l < n {
                return self.transition(pt, k, l);
            }
            l -= n;
        }
        if self.center(pt) {
            let n = if self.quad { 2 } else { 1 };
            if l < n {
                return if self.quad {
                    let (nu, nv) = Self::grid(pt);
                    self.quad_cell(0, nu / 2, nv / 2, l)
                } else {
                    self.tri_center(pt)
                };
            }
            l -= n;
        }
        let regions = self.regions(pt);
        for (index, &(r, c)) in regions.iter().enumerate() {
            let n = 2 * r * c;
            if l < n {
                let code = (tiles >> (5 * index)) & 0x1f;
                let (th, tw) = if code == 0 {
                    (r, c)
                } else {
                    tile_shape(r, c, 2 * code + 1)
                };
                let (i, j) = tiled_cell(r, c, th, tw, l / 2);
                return self.region_cell(pt, index as u32, i, j, l & 1);
            }
            l -= n;
        }
        unreachable!("triangle past the patch")
    }

    /// Triangle `l` of the transition ring's edge `k`.
    fn transition(&self, pt: &Patch, k: usize, l: u32) -> [Point; 3] {
        let axis = if self.quad { (k + 1) & 1 } else { 0 };
        let (ictx, _) = pt.inside[axis];
        let (octx, _) = pt.outer[k];
        let in_half = ictx.num_half - u32::from(ictx.odd);
        let out_half = octx.num_half - u32::from(octx.odd);
        let (kind, o, i) = stitch_transition(l, in_half, ictx.odd, out_half, octx.odd);
        let op = |w: u32| self.outer_point(pt, k, w);
        let ip = |w: u32| self.inner_point(pt, k, 1, w);
        match kind {
            1 | 5 => [op(o), ip(i), op(o + 1)],
            4 => [op(o), ip(i + 1), op(o + 1)],
            _ => [ip(i), ip(i + 1), op(o)],
        }
    }

    /// Walk point `w` of outer edge `k`.
    fn outer_point(&self, pt: &Patch, k: usize, w: u32) -> Point {
        let last = pt.outer[k].1 - 1;
        let forward = if self.quad {
            k == 1 || k == 2
        } else {
            k & 1 == 1
        };
        Point::Outer {
            edge: k as u32,
            q: if forward { w } else { last - w },
        }
    }

    /// Walk point `w` of edge `k` of inside ring `ring` (1 for the transition ring's inside).
    fn inner_point(&self, pt: &Patch, k: usize, ring: u32, w: u32) -> Point {
        if self.quad {
            let end_u = pt.inside[0].1 - 1 - ring;
            let end_v = pt.inside[1].1 - 1 - ring;
            let (a, b) = match k {
                0 => (ring, end_v - w),
                1 => (ring + w, ring),
                2 => (end_u, ring + w),
                _ => (end_u - w, end_v),
            };
            Point::Grid { a, b }
        } else {
            let n = pt.inside[0].1;
            let points = n - 2 * ring;
            if points == 1 {
                return Point::Center;
            }
            let (k, w) = if w == points - 1 {
                ((k + 1) % 3, 0)
            } else {
                (k, w)
            };
            let end = n - 1 - ring;
            Point::Ring {
                edge: k as u32,
                ring,
                q: if k & 1 == 1 { ring + w } else { end - w },
            }
        }
    }

    /// Triangle `half` (0 or 1) of a quad interior cell of quadrant `q` whose lower corner is
    /// grid `(cu, cv)` (interior cell coordinates).
    fn quad_cell(&self, q: u32, cu: u32, cv: u32, half: u32) -> [Point; 3] {
        let (p00, p10, p01, p11) = match q {
            0 => ((cu + 1, cv), (cu, cv), (cu + 1, cv + 1), (cu, cv + 1)),
            1 => ((cu + 1, cv + 1), (cu + 1, cv), (cu, cv + 1), (cu, cv)),
            2 => ((cu, cv + 1), (cu + 1, cv + 1), (cu, cv), (cu + 1, cv)),
            _ => ((cu, cv), (cu, cv + 1), (cu + 1, cv), (cu + 1, cv + 1)),
        };
        let g = |(x, y): (u32, u32)| Point::Grid { a: x + 1, b: y + 1 };
        if half == 0 {
            [g(p10), g(p11), g(p00)]
        } else {
            [g(p00), g(p11), g(p01)]
        }
    }

    fn tri_center(&self, pt: &Patch) -> [Point; 3] {
        let (k, _) = Self::tri_rings(pt);
        let n = pt.inside[0].1;
        let ring = (n >> 1) - 1;
        let c = |e: usize| self.inner_point(pt, e, ring, 0);
        if k == 0 {
            [c(1), c(0), c(2)]
        } else {
            [c(2), c(1), c(0)]
        }
    }

    /// Triangle `half` of cell `(i, j)` of region `index`.
    fn region_cell(&self, pt: &Patch, index: u32, i: u32, j: u32, half: u32) -> [Point; 3] {
        if self.quad {
            let (nu, nv) = Self::grid(pt);
            let (hu, hv, cu, cv) = (nu / 2, nv / 2, nu.div_ceil(2), nv.div_ceil(2));
            match index {
                0 => self.quad_cell(0, hu - 1 - i, hv + j, half),
                1 => self.quad_cell(1, cu - 1 - j, hv - 1 - i, half),
                2 => self.quad_cell(2, cu + i, cv - 1 - j, half),
                _ => self.quad_cell(3, hu + j, cv + i, half),
            }
        } else {
            self.sector_cell(pt, index as usize, i, j, half)
        }
    }

    /// Triangle `half` of cell `(i, j)` of a triangle's sector `s`.
    fn sector_cell(&self, pt: &Patch, s: usize, i: u32, j: u32, half: u32) -> [Point; 3] {
        let (rings, odd) = Self::tri_rings(pt);
        let n = pt.inside[0].1;
        let numring = (n + 1) >> 1;
        let prev = (s + 2) % 3;
        let shift = u32::from(odd);
        let k = i.max(j.saturating_sub(shift));
        debug_assert!(k < rings);
        // Metal's ring `k` (innermost first) stitches the reference's ring `r` (inside) to `r - 1`.
        let r = numring - 1 - k;
        let o = |e: usize, w: u32| self.inner_point(pt, e, r - 1, w);
        let inn = |e: usize, w: u32| self.inner_point(pt, e, r, w);
        let quads = 2 * k + shift;
        if i == k && j == k + shift {
            // The corner: edge `s`'s first triangle and edge `s - 1`'s last.
            let last = n - 2 * (r - 1) - 1;
            let (p10, p11, p00, p01) = (o(s, 1), o(s, 0), inn(s, 0), o(prev, last - 1));
            return if half == 0 {
                [p10, p11, p00]
            } else {
                [p00, p11, p01]
            };
        }
        let (start, e, m) = if i == k {
            (true, s, k + shift - 1 - j)
        } else {
            (false, prev, quads - 1 - (k - 1 - i))
        };
        let (olo, ohi, ilo, ihi) = (o(e, m + 1), o(e, m + 2), inn(e, m), inn(e, m + 1));
        match (start, half) {
            (true, 0) => [ohi, olo, ihi],
            (true, _) => [ihi, olo, ilo],
            (false, 0) => [ihi, ohi, ilo],
            (false, _) => [ilo, ohi, olo],
        }
    }

    /// The domain coordinates of `point` in a patch of processed factors `p`.
    pub fn place(&self, point: Point, p: &[u32; 6]) -> [f32; 3] {
        self.place_in(point, &self.patch(p))
    }

    /// [`Model::place`] in a patch's structure.
    pub fn place_in(&self, point: Point, pt: &Patch) -> [f32; 3] {
        let (u, v) = match point {
            Point::Corner { u, v } => (if u { FXP_ONE } else { 0 }, if v { FXP_ONE } else { 0 }),
            Point::Center => (0x5555, 0x5555),
            Point::Outer { edge, q } => {
                let loc = place_1d(&pt.outer[edge as usize].0, q);
                match (self.quad, edge) {
                    (true, 0) | (false, 0) => (0, loc),
                    (true, 1) | (false, 1) => (loc, 0),
                    (true, 2) => (FXP_ONE, loc),
                    (true, _) => (loc, FXP_ONE),
                    (false, _) => (loc, FXP_ONE - loc),
                }
            }
            Point::Grid { a, b } => (place_1d(&pt.inside[0].0, a), place_1d(&pt.inside[1].0, b)),
            Point::Ring { edge, ring, q } => {
                let ctx = &pt.inside[0].0;
                let perp = (place_1d(ctx, ring) * 0xaaaa + FXP_HALF) >> 16;
                let param = place_1d(ctx, q);
                let along = param - perp.div_ceil(2);
                match edge {
                    0 => (perp, along),
                    1 => (along, perp),
                    _ => (along, FXP_ONE - along - perp),
                }
            }
        };
        let uf = u as f32 / 65536.0;
        let vf = v as f32 / 65536.0;
        let w = if self.quad { 0.0 } else { (1.0 - uf) - vf };
        [uf, vf, w]
    }
}

/// The largest odd tile side `T` whose `(T + 1)^2` points fit `free`; 0 when none does.
pub fn side(free: u32) -> u32 {
    let mut t = 0;
    let mut s = 1;
    while s < 64 {
        if (s + 1) * (s + 1) <= free {
            t = s;
        }
        s += 2;
    }
    t
}

/// The tile shape (rows, columns) of a region of `r` rows and `c` columns at side `t`.
pub fn tile_shape(r: u32, c: u32, t: u32) -> (u32, u32) {
    let b = (t + 1) * (t + 1);
    if r < t {
        (r, c.min(b / (r + 1) - 1))
    } else if c < t {
        (r.min(b / (c + 1) - 1), c)
    } else {
        (t, t)
    }
}

/// Cell `x` of an `r` by `c` region drawn in `th` by `tw` tiles, row-major over tiles and over
/// each tile's cells.
pub fn tiled_cell(r: u32, c: u32, th: u32, tw: u32, x: u32) -> (u32, u32) {
    let band = x / (th * c);
    let ti = band * th;
    let rest = x - band * th * c;
    let h = th.min(r - ti);
    let col = rest / (h * tw);
    let tj = col * tw;
    let rest = rest - col * h * tw;
    let w = tw.min(c - tj);
    (ti + rest / w, tj + rest % w)
}

/// `finalPointPositionTable`.
pub const FINAL_POSITION: [u32; 33] = [
    0, 32, 16, 8, 17, 4, 18, 9, 19, 2, 20, 10, 21, 5, 22, 11, 23, 1, 24, 12, 25, 6, 26, 13, 27, 3,
    28, 14, 29, 7, 30, 15, 31,
];

/// Triangle `l` of `StitchTransition` between an inside row of `in_half` half points and an
/// outside row of `out_half`: its kind (1 advance outside, 2 advance inside, 3 and 4 the middle
/// quad's, 5 pointing inside, 6 pointing outside) and the walk positions of its outside and
/// inside points before it.
pub fn stitch_transition(
    l: u32,
    in_half: u32,
    in_odd: bool,
    out_half: u32,
    out_odd: bool,
) -> (u32, u32, u32) {
    let middle: &[u32] = match (in_odd, out_odd) {
        (true, true) => &[3, 4],
        (false, true) => &[5],
        (true, false) => &[6],
        (false, false) => &[],
    };
    let first = FINAL_POSITION[1..]
        .iter()
        .flat_map(|&f| [(f < in_half, 2), (f < out_half, 1)]);
    let second = FINAL_POSITION[1..]
        .iter()
        .rev()
        .flat_map(|&f| [(f < out_half, 1), (f < in_half, 2)]);
    let ends = [(FINAL_POSITION[0] < out_half, 1)];
    let steps = ends
        .into_iter()
        .chain(first)
        .chain(middle.iter().map(|&k| (true, k)))
        .chain(second)
        .chain(ends)
        .filter(|s| s.0)
        .map(|s| s.1);
    let (mut o, mut i) = (0, 0);
    for (n, kind) in steps.enumerate() {
        if n as u32 == l {
            return (kind, o, i);
        }
        match kind {
            1 | 5 => o += 1,
            2 | 6 => i += 1,
            4 => {
                o += 1;
                i += 1;
            }
            _ => {}
        }
    }
    unreachable!("transition triangle past the edge")
}

/// One draw: a pipeline, an encoder scale (`half` bits; `None` without the scale) and its
/// patches' factors (`half` bits), drawn in order through one batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draw {
    pub model: Model,
    pub scale: Option<u16>,
    pub patches: Vec<Vec<u16>>,
}

impl Draw {
    /// The scale as the binder pushes it: the `half`'s value, an infinite one as `2^32`.
    pub fn scale_value(&self) -> f32 {
        match self.scale {
            Some(0x7c00) => 4_294_967_296.0,
            Some(h) => half_value(h),
            None => 1.0,
        }
    }

    /// The draw as a line of the probe model's input: `<quad|tri> <part> <cw> <max> <scale|none>
    /// <patches> <halves...>`.
    pub fn line(&self) -> String {
        let m = &self.model;
        let part = match m.part {
            Part::Pow2 => 0,
            Part::Integer => 1,
            Part::FractionalOdd => 2,
            Part::FractionalEven => 3,
        };
        let scale = match self.scale {
            Some(h) => format!("{h:04x}"),
            None => "none".to_string(),
        };
        let mut s = format!(
            "{} {} {} {} {} {}",
            if m.quad { "quad" } else { "tri" },
            part,
            u32::from(m.cw),
            m.max_factor,
            scale,
            self.patches.len()
        );
        for h in self.patches.iter().flatten() {
            s.push_str(&format!(" {h:04x}"));
        }
        s
    }
}

/// FNV-1a over a word's 4 bytes, low first.
pub fn fnv(h: &mut u64, x: u32) {
    for b in x.to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

pub const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

/// Each patch's triangle count and the hash of its triangles' corners' `u`, `v`, `w` bits in
/// Metal's order, drawing `d` through one batch.
pub fn draw_patches(d: &Draw) -> Vec<(u32, u64)> {
    let m = &d.model;
    let mut used = 0;
    let mut out = Vec::new();
    for halves in &d.patches {
        let p = m.process(halves, d.scale_value());
        let count = m.triangle_count(&p);
        let (after, tiles) = m.stream_step(used, &p);
        used = after;
        let pt = m.patch(&p);
        let mut h = FNV_START;
        for l in 0..count {
            for point in m.decode_in(l, &pt, tiles) {
                for x in m.place_in(point, &pt) {
                    fnv(&mut h, x.to_bits());
                }
            }
        }
        out.push((count, h));
    }
    out
}

/// The digest of a set of draws: FNV over each patch's count and hash, in order.
pub fn digest(draws: &[Draw]) -> u64 {
    let mut h = FNV_START;
    for d in draws {
        for (count, ph) in draw_patches(d) {
            fnv(&mut h, count);
            fnv(&mut h, ph as u32);
            fnv(&mut h, (ph >> 32) as u32);
        }
    }
    h
}

/// The `half` bits of integer `n` in `1..=2048`.
pub fn half_of(n: u32) -> u16 {
    let e = 31 - n.leading_zeros();
    let mant = (n << (10 - e)) & 0x3ff;
    (((e + 15) << 10) | mant) as u16
}

/// The `half` bits of `n / 4` for `n` in `1..=8191`.
pub fn half_quarters(n: u32) -> u16 {
    let e = 31 - n.leading_zeros(); // n = 2^e · 1.m, value 2^(e - 2) · 1.m
    let mant = if e >= 10 {
        (n >> (e - 10)) & 0x3ff
    } else {
        (n << (10 - e)) & 0x3ff
    };
    (((e + 13) << 10) | mant) as u16
}

const PARTS: [Part; 4] = [
    Part::Pow2,
    Part::Integer,
    Part::FractionalOdd,
    Part::FractionalEven,
];

/// Single-patch draws at maximum 64 without the scale: every factor at `n / 4` for every third
/// `n` in `4..=256`; edges and insides on a grid of integers; a quad's mixed integer factors.
pub fn uniform_draws(quad: bool) -> Vec<Draw> {
    let per = if quad { 6 } else { 4 };
    let ni = if quad { 2 } else { 1 };
    let mut out = Vec::new();
    for part in PARTS {
        for cw in [false, true] {
            let model = Model {
                quad,
                part,
                cw,
                max_factor: 64,
                scale_enabled: false,
            };
            let one = |halves: Vec<u16>| Draw {
                model,
                scale: None,
                patches: vec![halves],
            };
            for n in (4..=256).step_by(3) {
                out.push(one(vec![half_quarters(n); per]));
            }
            for e in (1..65).step_by(7) {
                for i in (1..65).step_by(7) {
                    let mut h = vec![half_of(e); per - ni];
                    h.extend(vec![half_of(i); ni]);
                    out.push(one(h));
                }
            }
            if quad {
                for a in (1..65).step_by(9) {
                    for b in (1..65).step_by(9) {
                        out.push(one(vec![
                            half_of((a * 7 + b) % 64 + 1),
                            half_of(b),
                            half_of(a),
                            half_of((a + b * 3) % 64 + 1),
                            half_of(a),
                            half_of(b),
                        ]));
                    }
                }
            }
        }
    }
    out
}

/// SplitMix64.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u32) as usize]
    }

    /// `half` bits in `lo..=hi`.
    pub fn bits(&mut self, lo: u16, hi: u16) -> u16 {
        lo + self.below(u32::from(hi - lo) + 1) as u16
    }
}

/// Factors that cull, are not finite, are subnormal or sit at the range's edges.
pub const SPECIAL: [u16; 14] = [
    0x0000, 0x8000, 0xbc00, 0x7e00, 0x7c00, 0xfc00, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x4000, 0x4200,
    0x3c01, 0x7bff,
];

fn random_factor(rng: &mut Rng, max: u32) -> u16 {
    match rng.below(100) {
        0..=3 => rng.pick(&SPECIAL),
        4..=11 => half_of(1 + rng.below(5)),
        12..=29 => half_of(1 + rng.below(64)),
        30..=49 => rng.bits(0x3c00, 0x4800),
        50..=59 => rng.bits(0x0000, 0x3e00),
        60..=69 => half_of((max + rng.below(3)).saturating_sub(1).max(1)),
        _ => rng.bits(0x3c00, 0x5420),
    }
}

/// `n` random multi-patch draws: every domain, partition, winding, several maxima and scales.
pub fn random_draws(seed: u64, n: usize) -> Vec<Draw> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for _ in 0..n {
        let quad = rng.below(2) == 1;
        let part = PARTS[rng.below(4) as usize];
        let cw = rng.below(2) == 1;
        let mut max_factor = match part {
            Part::Pow2 => rng.pick(&[64, 64, 16, 32, 2, 48]),
            _ => rng.pick(&[64, 64, 64, 16, 32, 48, 63, 15, 2, 3]),
        };
        if part == Part::FractionalOdd && max_factor < 3 {
            max_factor = 3;
        }
        let scale = match rng.below(100) {
            0..=14 => Some(rng.bits(0x34cd, 0x4200)),
            15..=17 => Some(0x7c00),
            18..=19 => Some(rng.pick(&[0x0001, 0x3c00, 0x7bff, 0x1000])),
            _ => None,
        };
        let model = Model {
            quad,
            part,
            cw,
            max_factor,
            scale_enabled: scale.is_some(),
        };
        let per = if quad { 6 } else { 4 };
        let count = rng.pick(&[1, 1, 1, 2, 3, 4, 6, 10, 20]);
        let patches = (0..count)
            .map(|_| {
                if rng.below(5) == 0 {
                    vec![random_factor(&mut rng, max_factor); per]
                } else {
                    (0..per)
                        .map(|_| random_factor(&mut rng, max_factor))
                        .collect()
                }
            })
            .collect();
        out.push(Draw {
            model,
            scale,
            patches,
        });
    }
    out
}
