//! The Apple M3 Max's levels below the first of a small 2D chain, on the CPU
//! ([`super::MipmapFilterChain::expected_levels`]): its shader arithmetic, written plainly from
//! the module docs' "The M3 Max's tail" in `f64`, where every product and sum it rounds is exact
//! (the kernel does the same in integers).
//!
//! Also the values a level keeps for the next (a tail level's unrounded mixes; the first
//! level's `model.rs` computes beside its codes): the M3 Max reads a level's unrounded values,
//! not its stored codes, at the levels the chain says.

use super::layout::{den, Field, Layout, Rule};
use super::model::{bits_of, search};
use crate::lower::arith::srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::extended_srgb::{
    SRGB_DECODE_STEPS as XR_DECODE, SRGB_ENCODE_FIRST as XR_ENCODE,
};

/// The values a level keeps of each texel: one per field (`RGB9E5Float`'s three channels), at
/// most four.
pub(super) type Kept = [f64; 4];

/// The precision of the M3 Max's arithmetic for a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Precision {
    /// `f32`: 24 significant bits, the least normal exponent -126.
    F32,
    /// `f16`: 11 significant bits, the least normal exponent -14.
    F16,
}

impl Precision {
    fn bits(self) -> (i32, i32) {
        match self {
            Precision::F32 => (24, -126),
            Precision::F16 => (11, -14),
        }
    }

    /// The quantum of a value of exponent `e` (subnormals on the least normal's).
    fn ulp(self, e: i32) -> f64 {
        let (p, emin) = self.bits();
        pow2(e.max(emin) - p + 1)
    }

    /// `x` rounded to nearest even (subnormals kept). Every value here is far inside `f64`'s
    /// normal range, so `x / ulp` and the product back are exact.
    pub(super) fn round(self, x: f64) -> f64 {
        if x == 0.0 || !x.is_finite() {
            return x;
        }
        let ulp = self.ulp(exponent(x));
        (x / ulp).round_ties_even() * ulp
    }

    /// `x + y` (both values of this precision or products exact in `f64`) rounded to nearest
    /// even, exactly: `f64`'s sum and its error (Knuth's two-sum), the error deciding where the
    /// sum lands halfway between two values.
    fn sum(self, x: f64, y: f64) -> f64 {
        let s = x + y;
        let b = s - x;
        let error = (x - (s - b)) + (y - b);
        if s == 0.0 {
            return 0.0;
        }
        let ulp = self.ulp(exponent(s));
        let m = s / ulp;
        let r = if error != 0.0 && m - m.floor() == 0.5 {
            if error > 0.0 {
                m.ceil() * ulp
            } else {
                m.floor() * ulp
            }
        } else {
            m.round_ties_even() * ulp
        };
        if r == 0.0 {
            0.0
        } else {
            r
        }
    }
}

fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The exponent of a nonzero, normal `x`: `2^e <= |x| < 2^(e + 1)`.
fn exponent(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `n / d` rounded to nearest even at precision `p`, exactly (`d > 0`).
pub(super) fn ratio(n: i64, d: u64, p: Precision) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let (bits, emin) = p.bits();
    let a = u128::from(n.unsigned_abs());
    let d = u128::from(d);
    let width = |v: u128| 128 - v.leading_zeros() as i32;
    let mut e = width(a) - width(d);
    let below = |e: i32| if e >= 0 { a < d << e } else { a << -e < d };
    if below(e) {
        e -= 1;
    }
    let ue = e.max(emin) - bits + 1;
    let (num, den) = if ue >= 0 { (a, d << ue) } else { (a << -ue, d) };
    let (m, r) = (num / den, num % den);
    let m = if 2 * r > den || (2 * r == den && m & 1 == 1) {
        m + 1
    } else {
        m
    };
    let v = m as f64 * pow2(ue);
    if n < 0 {
        -v
    } else {
        v
    }
}

/// The M3 Max's mix of `a` and `b` at `t = w / 2048`: `fma(a, 1 - t, b t)` with `b t` rounded
/// first, every rounding at precision `p`; an infinity or a NaN as IEEE gives it (a NaN beside a
/// weight of 0 too).
pub(super) fn mix(a: f64, b: f64, w: u32, p: Precision) -> f64 {
    let t = f64::from(w) / 2048.0;
    let s = f64::from(2048 - w) / 2048.0;
    if !a.is_finite() || !b.is_finite() {
        return a * s + b * t;
    }
    p.sum(a * s, p.round(b * t))
}

/// The tail's coordinate along one axis: destination texel `i` of `small` from `big` reads
/// texels `x0` and `min(x0 + 1, big - 1)` at `w / 2048` of the second, from the centre `x =
/// (i + 0.5) big / small` rounded to a half float (nearest even) where `big` is at most
/// [`HALF_CENTRE`], to an `f32` past it, `c = x - 0.5`, `x0 = floor(c)`. An axis of one texel reads
/// it alone.
pub(super) fn axis(i: u32, big: u32, small: u32) -> (u32, u32) {
    if big == 1 {
        return (0, 0);
    }
    // x in units of 2^-11: x >= 1/2, so its half-float quantum is at least 2^-11. Past
    // HALF_CENTRE the chain's tail levels halve (a pass's second level), so x is a whole number.
    let precision = if big <= HALF_CENTRE {
        Precision::F16
    } else {
        Precision::F32
    };
    let x = ratio(
        i64::from(2 * i + 1) * i64::from(big),
        2 * u64::from(small),
        precision,
    );
    let c = (x * 2048.0) as u32 - 1024;
    ((c >> 11).min(big - 1), c & 2047)
}

/// The widest (or tallest) level whose tail centres the M3 Max rounds to a half float. From
/// levels of 4096, 8192 and 16384 texels along the axis it computes them in `f32`
/// (share-mips-m3b). Past this, the chain's tails read only levels that halve, where the centres
/// are whole numbers: a half holds them up to 2048, an `f32` past it.
pub(super) const HALF_CENTRE: u32 = 2048;

/// The precision of field `f`'s arithmetic on the M3 Max: a half for the half floats and the
/// fields it carries through a half, `f32` for the rest.
pub(super) fn precision(f: &Field, pf_half: bool) -> Precision {
    match f.rule {
        Rule::Float16 | Rule::XrColour | Rule::XrAlpha | Rule::XrSrgb => Precision::F16,
        Rule::Grid { .. } if pf_half => Precision::F16,
        _ => Precision::F32,
    }
}

/// The value of half-float bits `h`.
pub(super) fn half_value(h: u32) -> f64 {
    let (e, m) = ((h >> 10) & 0x1f, h & 0x3ff);
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    if e == 31 {
        return if m == 0 {
            sign * f64::INFINITY
        } else {
            f64::NAN
        };
    }
    let v = if e == 0 {
        f64::from(m) * pow2(-24)
    } else {
        f64::from(m | 0x400) * pow2(e as i32 - 25)
    };
    sign * v
}

/// The value of an unsigned float's code of `mb` mantissa bits.
fn ufloat_value(c: u32, mb: u32) -> f64 {
    let (e, m) = (c >> mb, c & ((1 << mb) - 1));
    if e == 31 {
        return if m == 0 { f64::INFINITY } else { f64::NAN };
    }
    if e == 0 {
        f64::from(m) * pow2(-14 - mb as i32)
    } else {
        f64::from(m | (1 << mb)) * pow2(e as i32 - 15 - mb as i32)
    }
}

/// A sign-extended code of `width` bits, clamped below at `-den` (a snorm's most negative code
/// reads as the one above it).
fn signed_code(c: u32, width: u32, den: i64) -> i64 {
    let s = 64 - width;
    ((i64::from(c) << s) >> s).max(-den)
}

/// The value the M3 Max reads for a stored code `c` of field `f` (and, for `RGB9E5Float`'s
/// channels, the texel's word `word`).
pub(super) fn decode(f: &Field, c: u32, word: u32, pf_half: bool) -> f64 {
    let p = precision(f, pf_half);
    match f.rule {
        Rule::Grid { bits, signed, .. } => {
            let d = den(bits, signed);
            let n = if signed {
                signed_code(c, f.width, i64::from(d))
            } else {
                i64::from(c)
            };
            ratio(n, u64::from(d), p)
        }
        Rule::Fixed16 { signed } => {
            let d = den(16, signed);
            let n = if signed {
                signed_code(c, 16, i64::from(d))
            } else {
                i64::from(c)
            };
            ratio(n, u64::from(d), p)
        }
        Rule::Srgb8 => ratio(i64::from(SRGB_DECODE_STEPS[c as usize]), 4095, p),
        Rule::XrColour | Rule::XrAlpha => ratio(i64::from(c) - 384, 510, p),
        Rule::XrSrgb => {
            let o = i64::from(c) - 384;
            let s = i64::from(XR_DECODE[o.unsigned_abs().min(639) as usize]);
            ratio(if o < 0 { -s } else { s }, 8192, p)
        }
        Rule::Float32 => f64::from(f32::from_bits(c)),
        Rule::Float16 => half_value(c),
        Rule::Ufloat { mantissa } => ufloat_value(c, mantissa),
        Rule::Rgb9e5 { .. } => f64::from(c) * pow2((word >> 27) as i32 - 24),
        Rule::Constant(_) => 0.0,
    }
}

/// `x` rounded to the nearest integer, ties to even (exact here).
fn rne(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// The code the M3 Max stores for value `v` of field `f` (not `RGB9E5Float`'s, made by
/// [`rgb9e5`]).
pub(super) fn encode(f: &Field, v: f64) -> u32 {
    let mask = |c: i64| {
        (c as u32)
            & if f.width == 32 {
                u32::MAX
            } else {
                (1 << f.width) - 1
            }
    };
    match f.rule {
        Rule::Grid { bits, signed, .. } => {
            let d = i64::from(den(bits, signed));
            mask(rne(v * d as f64).clamp(if signed { -d } else { 0 }, d))
        }
        Rule::Fixed16 { signed } => {
            let d = i64::from(den(16, signed));
            mask(rne(v * d as f64).clamp(if signed { -d } else { 0 }, d))
        }
        Rule::Srgb8 => mask(search(&SRGB_ENCODE_FIRST, rne(v * 4095.0).clamp(0, 4095))),
        Rule::XrColour => mask((rne(v * 510.0) + 384).clamp(0, 1023)),
        Rule::XrAlpha => mask(rne(v.clamp(0.0, 1.0) * 510.0) + 384),
        Rule::XrSrgb => {
            let n = search(&XR_ENCODE, rne(v.abs() * 8192.0).min(16384));
            mask(if v >= 0.0 {
                384 + n.min(639)
            } else {
                384 - n.min(384)
            })
        }
        Rule::Float32 => {
            if v.is_nan() {
                0x7fc0_0000
            } else if v != 0.0 && v.abs() < pow2(-126) {
                if v < 0.0 {
                    0x8000_0000
                } else {
                    0
                }
            } else {
                (v as f32).to_bits()
            }
        }
        Rule::Float16 => half_bits(v),
        Rule::Ufloat { mantissa: mb } => {
            if v.is_nan() {
                (31 << mb) | (1 << (mb - 1))
            } else if v == f64::INFINITY {
                31 << mb
            } else if v <= 0.0 {
                0
            } else {
                // Toward zero to mb + 1 bits, subnormals below 2^-14, the largest finite above.
                let e = exponent(v);
                let m = (v / pow2(e.max(-14) - mb as i32)).floor() as u32;
                if e > 15 {
                    (31 << mb) - 1
                } else if e < -14 {
                    m
                } else {
                    (((e + 15) as u32) << mb) | (m & ((1 << mb) - 1))
                }
            }
        }
        Rule::Rgb9e5 { .. } | Rule::Constant(_) => 0,
    }
}

/// The half-float bits of a value that is a half float (zero, an infinity or a NaN included).
pub(super) fn half_bits(v: f64) -> u32 {
    if v.is_nan() {
        return 0x7e00;
    }
    let sign = if v < 0.0 { 0x8000 } else { 0 };
    let a = v.abs();
    if a == f64::INFINITY {
        return sign | 0x7c00;
    }
    if a == 0.0 {
        return 0;
    }
    let e = exponent(a);
    if e < -14 {
        sign | (a / pow2(-24)) as u32
    } else {
        sign | (((e + 15) as u32) << 10) | (((a / pow2(e - 10)) as u32) & 0x3ff)
    }
}

/// `RGB9E5Float`'s word from its three channels' values: each negative read as 0, the shared
/// exponent of the largest (at least -16 before the bias of 16, at most 31), each mantissa
/// floored, at most 511.
pub(super) fn rgb9e5(v: [f64; 3]) -> u32 {
    let v = v.map(|x| x.max(0.0));
    let mx = v[0].max(v[1]).max(v[2]);
    if mx <= 0.0 {
        return 0;
    }
    let es = (exponent(mx).max(-16) + 16).min(31);
    let mut word = (es as u32) << 27;
    for (c, x) in v.into_iter().enumerate() {
        let m = (x / pow2(es - 24)).floor().min(511.0) as u32;
        word |= m << (9 * c);
    }
    word
}

/// The M3 Max's tail level of `small` texels per layer from the level above (`big`), `layers`
/// layers of `layout` texels in `above`, reading the level above's kept values `kept` where
/// given: the level's bytes and the values it keeps.
pub(super) fn level(
    layout: &Layout,
    above: &[u8],
    kept: Option<&[Kept]>,
    big: [u32; 3],
    small: [u32; 3],
    layers: u32,
    pf_half: bool,
) -> (Vec<u8>, Vec<Kept>) {
    let bytes = layout.bytes as usize;
    let per_above = (big[0] * big[1]) as usize;
    let mut out = Vec::new();
    let mut keep = Vec::new();
    for layer in 0..layers as usize {
        let base = layer * per_above;
        for y in 0..small[1] {
            let (y0, ty) = axis(y, big[1], small[1]);
            let y1 = (y0 + 1).min(big[1] - 1);
            for x in 0..small[0] {
                let (x0, tx) = axis(x, big[0], small[0]);
                let x1 = (x0 + 1).min(big[0] - 1);
                let at = |xx: u32, yy: u32| base + (yy * big[0] + xx) as usize;
                let read = |i: usize, c: usize, f: &Field| -> f64 {
                    match kept {
                        Some(k) => k[i][c],
                        None => {
                            let texel = &above[i * bytes..(i + 1) * bytes];
                            let word = bits_of(texel, 0, 32.min(8 * layout.bytes));
                            decode(f, bits_of(texel, f.offset, f.width), word, pf_half)
                        }
                    }
                };
                let mut word = 0u128;
                let mut values = [0.0f64; 4];
                for (c, f) in layout.fields.iter().enumerate() {
                    if let Rule::Constant(v) = f.rule {
                        word |= u128::from(v) << f.offset;
                        continue;
                    }
                    let p = precision(f, pf_half);
                    // An axis of one texel reads it alone: no mix along it, so an infinity
                    // stays one.
                    let row = |yy: u32| {
                        let a = read(at(x0, yy), c, f);
                        if big[0] == 1 {
                            a
                        } else {
                            mix(a, read(at(x1, yy), c, f), tx, p)
                        }
                    };
                    let r0 = row(y0);
                    let v = if big[1] == 1 {
                        r0
                    } else {
                        mix(r0, row(y1), ty, p)
                    };
                    values[c] = v;
                    if !matches!(f.rule, Rule::Rgb9e5 { .. }) {
                        word |= u128::from(encode(f, v)) << f.offset;
                    }
                }
                if layout.rgb9e5() {
                    word = u128::from(rgb9e5([values[0], values[1], values[2]]));
                }
                out.extend_from_slice(&word.to_le_bytes()[..bytes]);
                keep.push(values);
            }
        }
    }
    (out, keep)
}
