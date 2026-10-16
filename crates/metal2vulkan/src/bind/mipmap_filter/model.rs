//! Metal's filter on the CPU, rule by rule ([`super::MipmapFilterChain::expected_levels`]): the
//! statement the kernel is checked against, written plainly from the module docs.

use super::layout::{
    axis_tap, den, ties, widen, Field, Layout, Rule, XR_ALPHA_FLOOR, XR_COLOUR_FLOOR, XR_SRGB_FLOOR,
};
use super::reciprocal::Reciprocal;
use super::tail::{self, Kept, Precision};
use crate::lower::arith::srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::extended_srgb::{
    SRGB_DECODE_STEPS as XR_DECODE, SRGB_ENCODE_FIRST as XR_ENCODE,
};

/// One tap of a field: its raw bits (the field's, and for a float the exponent's), and its weight
/// `n` (of `2^24`).
#[derive(Clone, Copy, Debug)]
struct Tap {
    bits: u32,
    exp: u32,
    n: u32,
}

/// The eight taps of destination texel `(x, y, z)` of `small` from `big`, from the coordinates of
/// `reciprocal`'s GPU: each source texel's index within its layer and its weight
/// ([`tap_weight`]).
pub(super) fn taps(
    at: [u32; 3],
    big: [u32; 3],
    small: [u32; 3],
    reciprocal: Reciprocal,
) -> [(u32, u32); 8] {
    let axes: [(u32, u32); 3] =
        std::array::from_fn(|a| axis_tap(at[a], big[a], small[a], reciprocal));
    std::array::from_fn(|k| {
        let mut pos = [0u32; 3];
        let mut w = [0u32; 3];
        for a in 0..3 {
            let (x0, q) = axes[a];
            let second = (k >> a) & 1 == 1;
            pos[a] = if second { (x0 + 1).min(big[a] - 1) } else { x0 };
            w[a] = if second { q } else { 256 - q };
        }
        let n = tap_weight(w[0] * w[2], axes[1].1, (k >> 1) & 1 == 1);
        ((pos[2] * big[1] + pos[1]) * big[0] + pos[0], n)
    })
}

/// A tap's weight (of `2^24`) from its weights along x and z, `nxz` (of `2^16`), and the y
/// split `qy` (of 256), on the second row along y where `far`: Metal splits `nxz` along y last,
/// the far row's part rounded half down to `2^-16` and the near row's the rest (the module docs'
/// "The taps"). A 2D level's depth weight is 256, so its weights are exact.
pub(super) fn tap_weight(nxz: u32, qy: u32, far: bool) -> u32 {
    let far_part = ((nxz * qy + 127) >> 8) << 8;
    if far {
        far_part
    } else {
        (nxz << 8) - far_part
    }
}

/// The `width` bits at bit `offset` of a texel's little-endian bytes.
pub(super) fn bits_of(texel: &[u8], offset: u32, width: u32) -> u32 {
    let mut v = 0u128;
    for (i, b) in texel.iter().enumerate() {
        v |= u128::from(*b) << (8 * i);
    }
    ((v >> offset) & ((1u128 << width) - 1)) as u32
}

/// `v` sign-extended from `width` bits.
fn extend(v: u32, width: u32) -> i64 {
    let s = 64 - width;
    (i64::from(v) << s) >> s
}

/// A normalized integer, an extended-range code, or an sRGB byte: the exact sum of the decoded
/// codes `v` times their weights.
fn sum(taps: &[Tap], v: impl Fn(u32) -> i64) -> i64 {
    taps.iter().map(|t| v(t.bits) * i64::from(t.n)).sum()
}

/// A grid's sum (code units of `2^24`): each term floored to `2^(width - 24)` code, `(v n) >>
/// width << width`, as Metal holds a product of a `width`-bit code to 24 bits below the code's
/// unit (the 16-bit fields' `per_term` the same at width 16).
fn floored(taps: &[Tap], v: impl Fn(u32) -> i64, width: u32) -> i64 {
    taps.iter()
        .map(|t| ((v(t.bits) * i64::from(t.n)) >> width) << width)
        .sum()
}

/// Each term floored to 1/256 code (`(v n) >> 16`), the sum rounded half up (`(T + 128) >> 8`).
fn per_term(taps: &[Tap], v: impl Fn(u32) -> i64) -> i64 {
    let t: i64 = taps
        .iter()
        .map(|t| (v(t.bits) * i64::from(t.n)) >> 16)
        .sum();
    (t + 128) >> 8
}

/// The grid rounding of the sum `s` (code units of `2^24`) of codes `m` times a code's on `k`
/// fraction bits: half up to the grid, `Q`, then the code nearest `Q / (m 2^k)` with the tie
/// table's rule at an exact half (a widened field's value `Q / (den m 2^k)` as an `f32`, stored as
/// its nearest code; `m` 1 but for a widened field).
fn grid(rule: Rule, s: i64, k: u32, m: i64, lo: i64, hi: i64) -> i64 {
    let q = (s + (1 << (23 - k))) >> (24 - k);
    let unit = m << k;
    let mut code = q.div_euclid(unit);
    let fraction = q.rem_euclid(unit);
    let half = unit / 2;
    let t = ties(rule).expect("a grid rule has ties");
    if fraction > half || (fraction == half && t.up(code as i32)) {
        code += 1;
    }
    code.clamp(lo, hi)
}

/// The largest index whose first entry is at most `i`.
pub(super) fn search(first: &[u16], i: i64) -> i64 {
    first.iter().rposition(|&f| i64::from(f) <= i).unwrap_or(0) as i64
}

/// A float value `(-1)^s M 2^z` (`M` an integer), or a special.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    Finite { negative: bool, m: u64, z: i32 },
    Inf { negative: bool },
    Nan,
}

/// The decode of a float field: f32 (subnormals read as 0), f16, unsigned float, or one of
/// RGB9E5's mantissas.
fn value(rule: Rule, t: &Tap) -> Value {
    let (sign, mantissa_bits, hidden_bias, exp_special) = match rule {
        Rule::Float32 => ((t.bits >> 31) == 1, 23, 150, Some(255)),
        Rule::Float16 => ((t.bits >> 15) == 1, 10, 25, Some(31)),
        Rule::Ufloat { mantissa } => (false, mantissa, 15 + mantissa, Some(31)),
        Rule::Rgb9e5 { .. } => {
            return Value::Finite {
                negative: false,
                m: u64::from(t.bits),
                z: t.exp as i32 - 24,
            }
        }
        _ => unreachable!("not a float rule"),
    };
    let e = t.exp;
    let m = t.bits & ((1 << mantissa_bits) - 1);
    if Some(e) == exp_special {
        return if m == 0 {
            Value::Inf { negative: sign }
        } else {
            Value::Nan
        };
    }
    if e == 0 {
        let m = if rule == Rule::Float32 { 0 } else { m };
        return Value::Finite {
            negative: sign,
            m: u64::from(m),
            z: 1 - hidden_bias as i32,
        };
    }
    Value::Finite {
        negative: sign,
        m: u64::from(m | (1 << mantissa_bits)),
        z: e as i32 - hidden_bias as i32,
    }
}

fn msb(v: u64) -> i32 {
    63 - v.leading_zeros() as i32
}

/// Metal's float accumulator over the finite taps, by depth slice (each four taps in turn: a 2D
/// level's one, a 3D level's two): each slice's terms `M n 2^z` truncated toward zero to its
/// quantum `2^(e_s - fraction - guard)`, `e_s` its largest term's exponent estimate, and summed;
/// then each slice's sum truncated toward zero to the quantum `2^sc` of the largest estimate of
/// all, `sc = e - fraction - guard`, and the slices summed: `(S, sc)`, or `None` with no live term.
fn accumulate(terms: &[(bool, u64, i32, u32)], fraction: i32, guard: i32) -> Option<(i64, i32)> {
    let slices: Vec<(i64, i32)> = terms
        .chunks(4)
        .filter_map(|slice| {
            let live: Vec<_> = slice
                .iter()
                .filter(|(_, m, _, n)| *m != 0 && *n != 0)
                .collect();
            let e = live
                .iter()
                .map(|(_, m, z, n)| msb(*m) + z + msb(u64::from(*n)) - 24)
                .max()?;
            let sc = e - fraction - guard;
            let mut s = 0i64;
            for (negative, m, z, n) in live {
                let sh = z - 24 - sc;
                let p = m * u64::from(*n);
                let mag = if sh >= 0 {
                    p << sh
                } else {
                    p.checked_shr((-sh) as u32).unwrap_or(0)
                } as i64;
                s += if *negative { -mag } else { mag };
            }
            Some((s, sc))
        })
        .collect();
    let sc = slices.iter().map(|&(_, sc)| sc).max()?;
    let s = slices
        .iter()
        .map(|&(s, at)| {
            let mag = s.unsigned_abs().checked_shr((sc - at) as u32).unwrap_or(0) as i64;
            if s < 0 {
                -mag
            } else {
                mag
            }
        })
        .sum();
    Some((s, sc))
}

/// `s 2^sc` rounded half away from zero to `fraction + 1` significant bits, `emin` the least
/// normal exponent (subnormals on its quantum): `(negative, r, ulp)`, the value `r 2^ulp`.
fn round(s: i64, sc: i32, fraction: i32, emin: i32) -> Option<(bool, u64, i32)> {
    if s == 0 {
        return None;
    }
    let negative = s < 0;
    let mag = s.unsigned_abs();
    let x = msb(mag) + sc;
    let mut ulp = x.max(emin) - fraction;
    let drop = ulp - sc;
    let mut r = if drop > 0 {
        let r = mag.checked_shr(drop as u32).unwrap_or(0);
        let half = mag.checked_shr(drop as u32 - 1).unwrap_or(0) & 1;
        r + half
    } else {
        mag << (-drop)
    };
    if r == 2 << fraction {
        r >>= 1;
        ulp += 1;
    }
    (r != 0).then_some((negative, r, ulp))
}

/// The f16 bits of a rounded value (`round` at 10 fraction bits and `emin` -14).
fn half_bits(v: Option<(bool, u64, i32)>) -> u32 {
    let Some((negative, r, ulp)) = v else {
        return 0;
    };
    let sign = u32::from(negative) << 15;
    let x = msb(r) + ulp;
    if x > 15 {
        sign | 0x7c00
    } else if x < -14 {
        sign | r as u32
    } else {
        sign | (((x + 15) as u32) << 10) | (r as u32 & 0x3ff)
    }
}

/// Specials among the live taps: NaN, or both infinities, is `Some(None)`; one infinity
/// `Some(Some(negative))`.
fn special(values: &[Value], taps: &[Tap]) -> Option<Option<bool>> {
    let mut nan = false;
    let (mut pos, mut neg) = (false, false);
    for (v, t) in values.iter().zip(taps) {
        if t.n == 0 {
            continue;
        }
        match v {
            Value::Nan => nan = true,
            Value::Inf { negative: true } => neg = true,
            Value::Inf { negative: false } => pos = true,
            Value::Finite { .. } => {}
        }
    }
    if nan || (pos && neg) {
        Some(None)
    } else if pos || neg {
        Some(Some(neg))
    } else {
        None
    }
}

/// Each tap's term `(negative, M, z, n)`, in tap order; an infinity's or a NaN's (of weight 0
/// where [`special`] found none) a zero term.
fn terms(values: &[Value], taps: &[Tap]) -> Vec<(bool, u64, i32, u32)> {
    values
        .iter()
        .zip(taps)
        .map(|(v, t)| match v {
            Value::Finite { negative, m, z } => (*negative, *m, *z, t.n),
            _ => (false, 0, 0, 0),
        })
        .collect()
}

/// The half-float answer of a field of the half accumulator (f16, unsigned float, RGB9E5).
fn half(rule: Rule, taps: &[Tap]) -> u32 {
    let values: Vec<Value> = taps.iter().map(|t| value(rule, t)).collect();
    match special(&values, taps) {
        Some(None) => 0x7e00,
        Some(Some(negative)) => (u32::from(negative) << 15) | 0x7c00,
        None => half_bits(
            accumulate(&terms(&values, taps), 10, 7).and_then(|(s, sc)| round(s, sc, 10, -14)),
        ),
    }
}

/// The f32 answer.
fn float32(taps: &[Tap]) -> u32 {
    let values: Vec<Value> = taps.iter().map(|t| value(Rule::Float32, t)).collect();
    match special(&values, taps) {
        Some(None) => return 0x7fc0_0000,
        Some(Some(negative)) => return (u32::from(negative) << 31) | 0x7f80_0000,
        None => {}
    }
    let Some((s, sc)) = accumulate(&terms(&values, taps), 23, 15) else {
        return 0;
    };
    // Below the least normal before rounding: +0.
    if s == 0 || msb(s.unsigned_abs()) + sc < -126 {
        return 0;
    }
    let Some((negative, r, ulp)) = round(s, sc, 23, -126) else {
        return 0;
    };
    let sign = u32::from(negative) << 31;
    let x = msb(r) + ulp;
    if x > 127 {
        sign | 0x7f80_0000
    } else {
        sign | (((x + 127) as u32) << 23) | (r as u32 & 0x7f_ffff)
    }
}

/// `a` rounded half away from zero to 11 significant bits (`a >= 0`).
fn half_away_11(a: i64) -> i64 {
    if a < 2048 {
        return a;
    }
    let drop = msb(a as u64) - 10;
    (((a >> (drop - 1)) + 1) >> 1) << drop
}

/// Field `f`'s answer from its taps, `carried` where the Apple M3 Max carries the field through a
/// half float (its own level of `ABGR4Unorm`, `A1BGR5Unorm` or an extended-range format): the
/// code (for an `RGB9E5Float` mantissa, the channel's half bits) and the value the M3 Max keeps
/// of it for a later level of its tail (the module docs).
fn field(f: &Field, taps: &[Tap], carried: bool) -> (u32, f64) {
    let code = |signed: bool, lo: i64, hi: i64| {
        move |b: u32| {
            let v = if signed {
                extend(b, f.width)
            } else {
                i64::from(b)
            };
            v.clamp(lo, hi)
        }
    };
    let mask = |v: i64| {
        (v as u32)
            & if f.width == 32 {
                u32::MAX
            } else {
                (1 << f.width) - 1
            }
    };
    // The grid sum `Q` (code units of `2^-k`) of the exact sum `s`.
    let q_of = |s: i64, k: u32| (s + (1 << (23 - k))) >> (24 - k);
    // The M3 Max's half of `n / d`, the code it stores of it, and the half.
    let through_half = |n: i64, d: u64, clamp: bool| {
        let v = tail::ratio(n, d, Precision::F16);
        let v = if clamp { v.clamp(0.0, 1.0) } else { v };
        (tail::encode(f, v), v)
    };
    match f.rule {
        Rule::Grid { bits, signed, wide } => {
            let d = i64::from(den(bits, signed));
            let lo = if signed { -d } else { 0 };
            // Each code as the code of `wide` bits its bits replicate to, summed on that code's
            // grid.
            let m = i64::from(widen(bits, signed, wide));
            let k = 12 - wide;
            let decoded = code(signed, lo, d);
            let s = floored(taps, |b| decoded(b) * m, wide);
            let unit = ((d * m) << k) as u64;
            if carried {
                through_half(q_of(s, k), unit, false)
            } else {
                let kept = tail::ratio(q_of(s, k), unit, Precision::F32);
                (mask(grid(f.rule, s, k, m, lo, d)), kept)
            }
        }
        Rule::XrColour => {
            let s = floored(taps, code(false, 0, 1023), XR_COLOUR_FLOOR);
            if carried {
                through_half(q_of(s, 4) - 384 * 16, 510 * 16, false)
            } else {
                (mask(grid(f.rule, s, 4, 1, 0, 1023)), 0.0)
            }
        }
        Rule::XrAlpha => {
            let s = floored(taps, code(false, 384, 894), XR_ALPHA_FLOOR);
            if carried {
                through_half(q_of(s, 3) - 384 * 8, 510 * 8, true)
            } else {
                (mask(grid(f.rule, s, 3, 1, 384, 894)), 0.0)
            }
        }
        Rule::Srgb8 => {
            let s = sum(taps, |b| i64::from(SRGB_DECODE_STEPS[b as usize]));
            let i = ((s + (1 << 23)) >> 24).clamp(0, 4095);
            let kept = tail::ratio(i, 4095, Precision::F32);
            (mask(search(&SRGB_ENCODE_FIRST, i)), kept)
        }
        Rule::Fixed16 { signed } => {
            let d = i64::from(den(16, signed));
            let lo = if signed { -d } else { 0 };
            let c = per_term(taps, code(signed, lo, d)).clamp(lo, d);
            (mask(c), tail::ratio(c, d as u64, Precision::F32))
        }
        Rule::XrSrgb => {
            let s = floored(
                taps,
                |b| {
                    let o = i64::from(b) - 384;
                    let d = i64::from(XR_DECODE[o.unsigned_abs().min(639) as usize]);
                    if o < 0 {
                        -d
                    } else {
                        d
                    }
                },
                XR_SRGB_FLOOR,
            );
            let q = (s + (1 << 23)) >> 24;
            // The M3 Max rounds the sum's magnitude to a half float, half away from zero.
            let i = if carried {
                half_away_11(q.abs())
            } else {
                q.abs()
            };
            let m = search(&XR_ENCODE, i);
            let c = mask(if q >= 0 {
                384 + m.min(639)
            } else {
                384 - m.min(384)
            });
            let v = i as f64 / 8192.0;
            (c, if q < 0 { -v } else { v })
        }
        Rule::Float32 => {
            let c = float32(taps);
            (c, f64::from(f32::from_bits(c)))
        }
        Rule::Float16 | Rule::Rgb9e5 { .. } => {
            let h = half(f.rule, taps);
            (h, tail::half_value(h))
        }
        Rule::Ufloat { mantissa } => {
            let h = half(f.rule, taps);
            (h >> (10 - mantissa), tail::half_value(h))
        }
        Rule::Constant(v) => (mask(i64::from(v)), 0.0),
    }
}

/// `RGB9E5Float`'s word from its three channels' half-float answers: the shared exponent of
/// the largest (at least -16 before the bias of 16, at most 31), each mantissa floored, at most
/// 511.
fn rgb9e5(h: [u32; 3]) -> u32 {
    let mx = h.into_iter().max().unwrap_or(0);
    if mx == 0 {
        return 0;
    }
    let dec = |h: u32| {
        let (e, m) = (h >> 10, h & 0x3ff);
        if e == 0 {
            (u64::from(m), -24)
        } else {
            (u64::from(m | 0x400), e as i32 - 25)
        }
    };
    let (m, z) = dec(mx);
    let es = ((msb(m) + z).max(-16) + 16).min(31);
    let mut word = (es as u32) << 27;
    for (c, h) in h.into_iter().enumerate() {
        let (m, z) = dec(h);
        let sh = z - es + 24;
        let v = if sh >= 0 {
            m << sh
        } else {
            m.checked_shr((-sh) as u32).unwrap_or(0)
        };
        word |= (v.min(511) as u32) << (9 * c);
    }
    word
}

/// Metal's level of `small` texels per layer from the level above (`big`), `layers` layers of
/// `layout` texels in `above`, its coordinates from `reciprocal`'s GPU, the fields the M3 Max
/// carries through a half converted where `half`: the bytes of every layer of the level, and
/// the values each texel keeps.
pub(super) fn level(
    layout: &Layout,
    above: &[u8],
    big: [u32; 3],
    small: [u32; 3],
    layers: u32,
    reciprocal: Reciprocal,
    half: bool,
) -> (Vec<u8>, Vec<Kept>) {
    let bytes = layout.bytes as usize;
    let per_above = (big[0] * big[1] * big[2]) as usize;
    let mut out = Vec::new();
    let mut keep = Vec::new();
    for layer in 0..layers as usize {
        let base = layer * per_above;
        for z in 0..small[2] {
            for y in 0..small[1] {
                for x in 0..small[0] {
                    let taps = taps([x, y, z], big, small, reciprocal);
                    let mut word = 0u128;
                    let mut halves = [0u32; 3];
                    let mut values = [0.0f64; 4];
                    for (c, f) in layout.fields.iter().enumerate() {
                        let t: Vec<Tap> = taps
                            .iter()
                            .map(|&(i, n)| {
                                let at = (base + i as usize) * bytes;
                                let texel = &above[at..at + bytes];
                                let exp = match f.rule {
                                    Rule::Float32 => bits_of(texel, f.offset + 23, 8),
                                    Rule::Float16 => bits_of(texel, f.offset + 10, 5),
                                    Rule::Ufloat { mantissa } => {
                                        bits_of(texel, f.offset + mantissa, 5)
                                    }
                                    Rule::Rgb9e5 { .. } => bits_of(texel, 27, 5),
                                    _ => 0,
                                };
                                Tap {
                                    bits: bits_of(texel, f.offset, f.width),
                                    exp,
                                    n,
                                }
                            })
                            .collect();
                        let (v, kept) = field(f, &t, half && layout.carries_half(f));
                        values[c] = kept;
                        match f.rule {
                            Rule::Rgb9e5 { channel } => halves[channel as usize] = v,
                            _ => word |= u128::from(v) << f.offset,
                        }
                    }
                    if layout.rgb9e5() {
                        word = u128::from(rgb9e5(halves));
                    }
                    out.extend_from_slice(&word.to_le_bytes()[..bytes]);
                    keep.push(values);
                }
            }
        }
    }
    (out, keep)
}
