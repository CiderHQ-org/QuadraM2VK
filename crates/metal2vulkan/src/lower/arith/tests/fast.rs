//! Metal's fast division and the fast functions composed with it, on the interpreter against
//! Metal's recorded answers (Apple M3 Max, probe case 08bc6611, compiled with the default fast
//! math): the fast reciprocal's flushes, and `fast::fmod` as `fma(-trunc(x / y), y, x)` of the fast
//! quotient on every fused multiply-add route.

use super::*;
use crate::lower::arith::divide::{Reciprocal, Rough};
use crate::lower::arith::fma::{self, Fma, Route};

/// `(y, 1 / y)`: every divisor of case 08bc6611 that is an `f32` denormal, a zero, at least
/// `2^126`, an infinity, or a NaN, with Metal's fast reciprocal of it.
const RECIPROCALS: [(u32, u32); 27] = [
    (0x00000000, 0x7F800000),
    (0x00000001, 0x7F800000),
    (0x00032804, 0x7F800000),
    (0x00088345, 0x7F800000),
    (0x00130911, 0x7F800000),
    (0x00137376, 0x7F800000),
    (0x001A8C56, 0x7F800000),
    (0x00221974, 0x7F800000),
    (0x00314B0F, 0x7F800000),
    (0x003EE65F, 0x7F800000),
    (0x0053C4EF, 0x7F800000),
    (0x007F47D9, 0x7F800000),
    (0x7F7FFFFF, 0x00000000),
    (0x7F800000, 0x00000000),
    (0x7FC00000, 0x7FC00000),
    (0x80000000, 0xFF800000),
    (0x8004B787, 0xFF800000),
    (0x8010F4B2, 0xFF800000),
    (0x801ED4BD, 0xFF800000),
    (0x802822A3, 0xFF800000),
    (0x802B599C, 0xFF800000),
    (0x803255FF, 0xFF800000),
    (0x80400000, 0xFF800000),
    (0x804188FC, 0xFF800000),
    (0x804230F6, 0xFF800000),
    (0x8056D96A, 0xFF800000),
    (0xFF800000, 0x80000000),
];

/// `(x, y, fast::fmod(x, y))`: the lanes of case 08bc6611 whose fast reciprocal of `y` is the
/// correctly rounded `1 / y` (917 of 1,024; these are its special lanes and 44 others drawn at
/// random), so the quotient the interpreter divides is Metal's.
const FMOD: [(u32, u32, u32); 108] = [
    (0xBFFE18C9, 0xC15FD2B5, 0xBFFE18C9),
    (0x3E57C427, 0x400109AA, 0x3E57C427),
    (0x3FFEC76F, 0xC11ADCA2, 0x3FFEC76F),
    (0xBF7A8F38, 0x40D82500, 0xBF7A8F38),
    (0x3FDACF22, 0xBF9E5534, 0x3EF1E7B8),
    (0x40370A25, 0x41351B59, 0x40370A25),
    (0x40333E0F, 0xC070368B, 0x40333E0F),
    (0x3F87699E, 0xBDCA9F43, 0x3D8C6142),
    (0x400F6B18, 0x404C13A5, 0x400F6B18),
    (0xC044DE3B, 0xC1067FAD, 0xC044DE3B),
    (0xBFC1C9F0, 0xC1596163, 0xBFC1C9F0),
    (0x3F455453, 0x40E5F119, 0x3F455453),
    (0xC004006C, 0xC1174EE9, 0xC004006C),
    (0xC033AD6C, 0xC1581FB8, 0xC033AD6C),
    (0xBF9346C4, 0xC0DA6A55, 0xBF9346C4),
    (0xBEABFA9E, 0x411845D4, 0xBEABFA9E),
    (0xAC7CF4DE, 0x3FC6FCE4, 0xAC7CF4DE),
    (0xAD71D04C, 0x401803AA, 0xAD71D04C),
    (0x3D24224C, 0x401C03CB, 0x3D24224C),
    (0x313F63AA, 0x405E6F2F, 0x313F63AA),
    (0xC3E60FEB, 0x40BE90BB, 0xBFC9CA04),
    (0x4808DD1C, 0xC0388B55, 0x3FBE1A92),
    (0xCB225FBF, 0x3E2F7E58, 0xBD97D9C0),
    (0x4AFD7F51, 0x4093C524, 0x3F760C00),
    (0x48354100, 0xC0B62F00, 0x408ED800),
    (0x5A900799, 0xBF634297, 0x4ECA3738),
    (0x5D4E041C, 0xC05F3547, 0xD0752700),
    (0x404AE76D, 0x3F9536FC, 0x3F56C1C4),
    (0x7CED6A0D, 0x412F29DB, 0x6E499AC0),
    (0x084A6E20, 0x4167E3DE, 0x084A6E20),
    (0x5785B164, 0xBE805703, 0xCA344400),
    (0x735934AF, 0xC10AA805, 0x66CD7F78),
    (0x584741F4, 0xBF6C5E08, 0xCC3E24C8),
    (0x7D0F380B, 0xC0FCCC48, 0x707E50E0),
    (0x47D98FB0, 0x411B6F4C, 0x403C7820),
    (0xC1C1650E, 0xB78404E2, 0xB47FF500),
    (0x42919FFE, 0x35DF6EE2, 0x365423D0),
    (0xC15B3515, 0x804230F6, 0x7FC00000),
    (0xC24E4367, 0xE310DC76, 0xC24E4367),
    (0x41A18EC6, 0xD1497DA4, 0x41A18EC6),
    (0xC285F86E, 0x61DE79B2, 0xC285F86E),
    (0x42831361, 0x804188FC, 0x7FC00000),
    (0x3F800020, 0xC10FA238, 0x3F800020),
    (0x3F7FFFF8, 0x417D4D92, 0x3F7FFFF8),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
];

/// Metal's fast reciprocal (`math::fast_reciprocal`) at every flushing divisor of the case: the
/// infinity of `y`'s sign for a zero or an `f32` denormal, the zero of its sign at or past
/// `2^126` (where the quotient is an `f32` denormal) and for an infinity, and NaN for a NaN.
#[test]
fn fast_reciprocal_flushes_as_metal() {
    let f = float(32);
    let cases: Vec<Case> = RECIPROCALS
        .iter()
        .map(|(y, _)| vec![vec![u64::from(*y)]])
        .collect();
    for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
        check_accept(
            &format!("fast reciprocal f32 against Metal, {reciprocal:?}"),
            &[f],
            f,
            &|fb, a| math::fast_reciprocal(fb, f, a[0], Contraction::Allowed, reciprocal),
            &cases,
            &|c| {
                let y = arg(c, 0) as u32;
                RECIPROCALS
                    .iter()
                    .find(|(k, _)| *k == y)
                    .map(|(_, r)| vec![u64::from(*r)])
            },
            &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
        );
        // 2^126 itself has a normal reciprocal, kept.
        let edge = vec![vec![vec![u64::from(2f32.powi(126).to_bits())]]];
        check_accept(
            &format!("fast reciprocal of 2^126, {reciprocal:?}"),
            &[f],
            f,
            &|fb, a| math::fast_reciprocal(fb, f, a[0], Contraction::Allowed, reciprocal),
            &edge,
            &|_| Some(vec![u64::from(2f32.powi(-126).to_bits())]),
            &|g, w| g == w,
        );
    }
}

/// Divisors for [`a_rounded_fast_reciprocal_is_correctly_rounded`]: the normal range's ends, the
/// range `(2^125, 2^126]` whose estimate is halved, all-ones significands, `3` (RADV's
/// `v_rcp_f32` answered `0x3eaaaaaa`, and `6 / 3` was `0x3fffffff`, RC3 case f8494e2f),
/// reciprocals near a rounding midpoint, the flushes, and random normal divisors.
fn reciprocal_divisors() -> Vec<u64> {
    use super::divide::bits;
    let mut rng = Rng(0x2790_8220_0000_0001);
    let mut out: Vec<u64> = [
        bits(0, 127, 0),
        bits(0, 128, 0x40_0000),
        bits(1, 128, 0x40_0000),
        bits(0, 127, 0x7F_FFFF),
        bits(1, 127, 0x7F_FFFF),
        bits(0, 127, 1),
        bits(0, 1, 0),
        bits(1, 1, 0x7F_FFFF),
        bits(0, 2, 0),
        bits(0, 252, 0),
        bits(0, 252, 1),
        bits(1, 252, 0x40_0000),
        bits(0, 252, 0x7F_FFFF),
        bits(1, 252, 0x7F_FFFF),
        bits(0, 253, 0),
        bits(1, 253, 0),
        bits(0, 253, 1),
        bits(0, 254, 0x7F_FFFF),
        bits(0, 255, 0),
        bits(1, 255, 0),
        bits(0, 255, 0x40_0000),
        bits(0, 0, 0),
        bits(1, 0, 0),
        bits(0, 0, 0x40_0000),
        bits(1, 0, 0x7F_FFFF),
    ]
    .to_vec();
    // `1 / y` within 2^-10 of a rounding midpoint: 2^47 mod Y near Y / 2.
    let mut near = 0;
    while near < 64 {
        let m = rng.next() & 0x7F_FFFF;
        let big_y = m | 0x80_0000;
        let r = (1u64 << 48) % (2 * big_y);
        if r.abs_diff(big_y) * 1024 >= 2 * big_y {
            continue;
        }
        near += 1;
        out.push(bits(rng.next() & 1, 1 + rng.next() % 253, m));
    }
    for _ in 0..256 {
        let m = if rng.next() & 7 == 0 {
            0x7F_FFFF
        } else {
            rng.next() & 0x7F_FFFF
        };
        out.push(bits(rng.next() & 1, 1 + rng.next() % 253, m));
    }
    out
}

/// Metal's fast reciprocal of `f32` bits `y` where it is `RN(1 / y)`, with its flushes: the
/// infinity of `y`'s sign for a zero or an `f32` denormal, the zero of its sign past `2^126`.
fn rounded_reciprocal_of(y: u64) -> u64 {
    let v = f32v(y);
    if y & 0x7F80_0000 == 0 {
        return (y & 0x8000_0000) | 0x7F80_0000;
    }
    if v.is_nan() {
        return 0x7FC0_0000;
    }
    if v.abs() > 2f32.powi(126) {
        return y & 0x8000_0000;
    }
    // The host's `f64` reciprocal of an `f32` is never within an `f64` ulp of an `f32` rounding
    // midpoint without being one, which it is not, so rounding it to `f32` rounds once.
    u64::from(((1.0 / f64::from(v)) as f32).to_bits())
}

/// The device's quotient one ulp low everywhere, as RADV's `v_rcp_f32` answered `1 / 3`.
fn one_ulp_low(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Value> {
    let u = super::super::narrow::u32_form(f);
    let (ut, ft) = (ops::ty(fb, u)?, ops::ty(fb, f)?);
    let q = fb.binary(BinOp::FDiv, x, y)?;
    let qb = fb.convert(ConvOp::Bitcast, q, ut)?;
    let one = ops::splat(fb, u, 1)?;
    let low = fb.binary(BinOp::ISub, qb, one)?;
    fb.convert(ConvOp::Bitcast, low, ft)
}

/// On profiles whose `float` fused multiply-add is `OpFmaKHR`, the fast reciprocal is `RN(1 / y)`
/// from any device estimate within 2.5 ulp: [`reciprocal_divisors`] on one and three lanes, with
/// the estimate exact, moved by up to two ulp either way, and one ulp low everywhere. The flushes
/// are the device reciprocal's. With the estimate one ulp low, the fast `6 / 3` is `2` (Metal's
/// answer), where the device's reciprocal gives `0x3fffffff`.
#[test]
fn a_rounded_fast_reciprocal_is_correctly_rounded() {
    let ys = reciprocal_divisors();
    let exact: Rough<'_> = &|fb, x, y| fb.binary(BinOp::FDiv, x, y);
    for lanes in [1u8, 3] {
        let f = vec(float(32), lanes);
        let n = usize::from(lanes);
        let cases: Vec<Case> = (0..ys.len())
            .step_by(n)
            .map(|i| vec![(0..n).map(|l| ys[(i + l) % ys.len()]).collect()])
            .collect();
        let moved = move |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| {
            super::divide::perturbed(fb, f, x, y)
        };
        let low = move |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| one_ulp_low(fb, f, x, y);
        for (name, rough) in [
            ("exact", exact),
            ("moved", &moved as Rough<'_>),
            ("low", &low as Rough<'_>),
        ] {
            check_accept(
                &format!("rounded fast reciprocal x{lanes} {name}"),
                &[f],
                f,
                &|fb, a| math::fast_reciprocal_with(fb, f, a[0], Reciprocal::Rounded, rough),
                &cases,
                &|c| Some(c[0].iter().map(|y| rounded_reciprocal_of(*y)).collect()),
                &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
            );
        }
    }
    let f = float(32);
    let low = move |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| one_ulp_low(fb, f, x, y);
    let six_by_three = vec![vec![
        vec![u64::from(6f32.to_bits())],
        vec![u64::from(3f32.to_bits())],
    ]];
    for (reciprocal, want) in [
        (Reciprocal::Rounded, 0x4000_0000u64),
        (Reciprocal::Device, 0x3FFF_FFFF),
    ] {
        check_accept(
            &format!("6 / 3 from a low estimate, {reciprocal:?}"),
            &[f, f],
            f,
            &|fb, a| {
                let r = math::fast_reciprocal_with(fb, f, a[1], reciprocal, &low)?;
                fb.binary(BinOp::FMul, a[0], r)
            },
            &six_by_three,
            &|_| Some(vec![want]),
            &|g, w| g == w,
        );
    }
}

/// Metal's fast quotient of `f32` bits `x` and `y` where its reciprocal is `RN(1 / y)`
/// ([`rounded_reciprocal_of`]): a denormal `x` read as a zero of its sign, a product whose
/// 24-bit rounding (with an unbounded exponent) is below `2^-126` a zero of its sign, and a NaN
/// `0x7FC00000`.
fn metal_fast_quotient(x: u64, y: u64) -> u64 {
    let r = f64::from(f32v(rounded_reciprocal_of(y)));
    let x = if x & 0x7F80_0000 == 0 {
        f64::from(f32v(x & 0x8000_0000))
    } else {
        f64::from(f32v(x))
    };
    // Exact: the two significands' product has 48 bits, and its exponent is an `f64`'s.
    let p = x * r;
    if p.is_nan() {
        return 0x7FC0_0000;
    }
    // Below `2^-126` the 24-bit rounding is decided `2^64` higher, where it is a normal `f32`.
    let scaled = (p * 2f64.powi(64)) as f32;
    if p.abs() < 2f64.powi(-126) && scaled.abs() < 2f32.powi(-62) {
        return u64::from(f32v(0).copysign(p as f32).to_bits());
    }
    u64::from((p as f32).to_bits())
}

/// `(x, y)`: fast divisions at Metal's NaNs and denormals. A NaN divisor (either sign), a NaN
/// numerator, `0 / 0`, `inf / inf`, and `inf · 0` (`inf / 2^127`) answer `0x7FC00000`; a denormal
/// numerator is a zero of its sign (`0 · inf = NaN` over a denormal divisor, A1 case f9a12252's
/// lanes); a quotient below `2^-126` is a zero of its sign, including `(2^24 - 1) · 2^-150`, which
/// the denormal grid rounds up to `2^-126`; a zero quotient of a numerator too large to scale by
/// `2^24` (`2^127 / 2^127`, whose reciprocal flushes) keeps its sign; and quotients at `2^-126` and
/// of normals are the product's.
const SPECIAL_DIVISIONS: [(u32, u32); 34] = [
    (0x3F80_0000, 0x7FC0_0000),
    (0x7F80_0000, 0x7FC0_0000),
    (0x3F80_0000, 0xFFC0_0000),
    (0x7FC0_0001, 0x4000_0000),
    (0xFFC0_0000, 0x4000_0000),
    (0x0000_0000, 0x0000_0000),
    (0x8000_0000, 0x0000_0000),
    (0x7F80_0000, 0x7F80_0000),
    (0x7F80_0000, 0x7F00_0000),
    (0xFF80_0000, 0x7F7F_FFFF),
    (0x0040_0000, 0x3F80_0000),
    (0x8040_0000, 0x4000_0000),
    (0x0000_0001, 0xBF80_0000),
    (0x807F_FFFF, 0x3F00_0000),
    (0x0040_0000, 0x0040_0000),
    (0x0000_0001, 0x0000_0001),
    (0x8000_0001, 0x0040_0000),
    (0x0D80_0000, 0x4E80_0000),
    (0x8D80_0000, 0x4E80_0000),
    (0x0D80_0000, 0xCE80_0000),
    (0x00C0_0000, 0x4000_0000),
    (0x80C0_0000, 0x4000_0000),
    (0x00FF_FFFF, 0x4000_0000),
    (0x80FF_FFFF, 0x4000_0000),
    (0x0100_0000, 0x4000_0000),
    (0x8100_0000, 0x4000_0000),
    (0x00FF_FFFF, 0x3F00_0000),
    (0x40C0_0000, 0x4040_0000),
    (0x3F80_0000, 0x4040_0000),
    (0x42AA_3A21, 0x44C8_0075),
    (0x7F7F_FFFF, 0x3F00_0000),
    (0x3F80_0000, 0x0080_0000),
    (0x7F00_0000, 0x7F00_0000),
    (0xFF00_0000, 0x7F00_0000),
];

/// The fast division's test vectors in lanes of `lanes`: [`SPECIAL_DIVISIONS`] and random normal
/// operands whose quotients reach the denormals.
fn fast_division_cases(lanes: u8) -> Vec<Case> {
    let mut rng = Rng(0x0864_0000_D279_0001);
    let mut pairs: Vec<(u64, u64)> = SPECIAL_DIVISIONS
        .iter()
        .map(|(x, y)| (u64::from(*x), u64::from(*y)))
        .collect();
    for _ in 0..96 {
        let x = super::divide::bits(rng.next() & 1, 1 + rng.next() % 40, rng.next() & 0x7F_FFFF);
        let y = super::divide::bits(
            rng.next() & 1,
            100 + rng.next() % 54,
            rng.next() & 0x7F_FFFF,
        );
        pairs.push((x, y));
    }
    let n = usize::from(lanes);
    (0..pairs.len())
        .step_by(n)
        .map(|i| {
            let lane = |l: usize| pairs[(i + l) % pairs.len()];
            vec![
                (0..n).map(|l| lane(l).0).collect(),
                (0..n).map(|l| lane(l).1).collect(),
            ]
        })
        .collect()
}

/// Metal's fast division (`math::fast_divide`) answers its NaNs and denormals on every device, by
/// either reciprocal, on one and four lanes: [`SPECIAL_DIVISIONS`] and random normal operands whose
/// quotients reach the denormals, against [`metal_fast_quotient`], bit for bit. The interpreter's
/// division is correctly rounded, its arithmetic keeps denormals, and its NaNs keep their payloads,
/// so no answer here comes from a device that happens to flush.
#[test]
fn a_fast_division_answers_metal_at_nans_and_denormals() {
    for lanes in [1u8, 4] {
        let f = vec(float(32), lanes);
        let cases = fast_division_cases(lanes);
        for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
            check_accept(
                &format!("fast division at NaNs and denormals x{lanes}, {reciprocal:?}"),
                &[f, f],
                f,
                &|fb, a| math::fast_divide(fb, f, a[0], a[1], Contraction::Allowed, reciprocal),
                &cases,
                &|c| {
                    Some(
                        c[0].iter()
                            .zip(&c[1])
                            .map(|(x, y)| metal_fast_quotient(*x, *y))
                            .collect(),
                    )
                },
                &|g, w| g == w,
            );
        }
    }
}

/// An `arcp` division's word estimate ([`ArithStrategy::FastDivide`], a call) with its routine's
/// ([`math::FastDivision`], charged once per module), and `fast::tanh`'s and `fast::fmod`'s with
/// the same routine, bound what they emit by either reciprocal, on scalars and vectors, and the
/// routine with one call is within the last third of its estimate. The module's growth over an
/// identity kernel is charged in full, its constants, types, and the `GLSL.std.450` import
/// included.
#[test]
fn a_fast_division_fits_its_word_estimate() {
    use crate::lower::strategy::FamilyStrategy;
    let module_words = |ins: &[Form], f: Form, defs: Defs<'_>, body: BodyCalling<'_>| {
        let m = kernel_defs(ins, f, defs, body, false, false).expect("kernel");
        crate::spirv::serialize::serialize(m)
            .expect("words")
            .words
            .len()
    };
    let mut over = Vec::new();
    for f in [float(32), vec(float(32), 2), vec(float(32), 4)] {
        let none: Defs<'_> = &|_| Ok(Vec::new());
        let bare = module_words(&[f, f], f, none, &|_, x, _| Ok(x[0]));
        for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
            for contract in [Contraction::Allowed, Contraction::Forbidden] {
                let routine = math::FastDivision::new(f, reciprocal, contract);
                let defs: Defs<'_> = &|b| Ok(vec![math::define_fast_division(b, routine)?]);
                let used = module_words(&[f, f], f, defs, &|fb, x, funcs| {
                    math::call_fast_division(fb, funcs[0], x[0], x[1])
                });
                let bound = super::super::ArithStrategy::FastDivide { form: f, routine }.words()
                    as usize
                    + routine.words() as usize;
                if used - bare > bound || (used - bare) * 3 < bound * 2 {
                    over.push(format!(
                        "{reciprocal:?} {contract:?} x{}: {} against {bound}",
                        f.lanes,
                        used - bare
                    ));
                }
            }
            let routine = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
            let defs: Defs<'_> = &|b| Ok(vec![math::define_fast_division(b, routine)?]);
            let tanh = FloatFn::FastTanh { reciprocal };
            let used = module_words(&[f, f], f, defs, &|fb, x, funcs| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, funcs[0], p, q)
                };
                math::fast_tanh(fb, tanh, f, x[0], &Lib::instructions(), &divide)
            });
            let bound = (tanh.words(f) + routine.words()) as usize;
            if used - bare > bound {
                over.push(format!(
                    "{tanh:?} x{}: {} over {bound}",
                    f.lanes,
                    used - bare
                ));
            }
        }
        for route in [Route::Khr, Route::Wide, Route::Soft] {
            let fma = Fma {
                form: f,
                route,
                tiny: None,
            };
            let reciprocal = Reciprocal::of(route);
            let routine = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
            let defs: Defs<'_> = &|b| {
                let mut funcs = vec![math::define_fast_division(b, routine)?];
                if let Some(r) = fma.routine() {
                    funcs.push(fma::define(b, r)?);
                }
                Ok(funcs)
            };
            let used = module_words(&[f, f], f, defs, &|fb, x, funcs| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, funcs[0], p, q)
                };
                let call = |fb: &mut FunctionBuilder<'_>, args: &[Value]| {
                    fma::emit(fb, fma, args, funcs.get(1).copied())
                };
                math::fast_fmod(fb, x[0], x[1], &divide, &call)
            });
            let bound = (super::super::ArithStrategy::FastFmod { fma, reciprocal }.words()
                + routine.words()
                + fma
                    .routine()
                    .map_or(0, |r| r.words(crate::spirv::builder::QuantizeForm::Integer)))
                as usize;
            if used - bare > bound {
                over.push(format!(
                    "fast fmod {route:?} x{}: {} over {bound}",
                    f.lanes,
                    used - bare
                ));
            }
        }
    }
    assert!(over.is_empty(), "{over:?}");
}

/// The fast division's routine answers as the division expanded in place, bit for bit: an `arcp`
/// division by a call of [`math::define_fast_division`] against `math::fast_divide` inline, on the
/// fast division's test vectors, by either reciprocal and either contraction, on one and four
/// lanes; and `fast::tanh`'s quotient and `fast::fmod` (every `float` route of its fused
/// multiply-add) with the routine against the same with the division inline, on their test vectors.
#[test]
fn a_fast_division_routine_answers_as_the_division_inline() {
    let compare = |what: &str,
                   ins: &[Form],
                   f: Form,
                   inline: TypedModule,
                   called: TypedModule,
                   cases: &[Case]| {
        let a = run_module(&format!("{what}, inline"), inline, ins, f, cases);
        let b = run_module(&format!("{what}, called"), called, ins, f, cases);
        let differ: Vec<String> = cases
            .iter()
            .zip(a.iter().zip(&b))
            .filter(|(_, (a, b))| a != b)
            .take(12)
            .map(|(c, (a, b))| format!("{c:x?}: inline {a:x?}, called {b:x?}"))
            .collect();
        assert!(differ.is_empty(), "{what}: {}", differ.join("\n"));
    };
    for lanes in [1u8, 4] {
        let f = vec(float(32), lanes);
        let cases = fast_division_cases(lanes);
        for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
            for contract in [Contraction::Allowed, Contraction::Forbidden] {
                let routine = math::FastDivision::new(f, reciprocal, contract);
                let inline = kernel(&[f, f], f, &|fb, x| {
                    math::fast_divide(fb, f, x[0], x[1], contract, reciprocal)
                })
                .expect("inline");
                let called = kernel_defs(
                    &[f, f],
                    f,
                    &|b| Ok(vec![math::define_fast_division(b, routine)?]),
                    &|fb, x, funcs| math::call_fast_division(fb, funcs[0], x[0], x[1]),
                    false,
                    false,
                )
                .expect("called");
                let what = format!("fast division x{lanes} {reciprocal:?} {contract:?}");
                compare(&what, &[f, f], f, inline, called, &cases);
            }
        }
    }
    // `fast::tanh`'s quotient, `(t - 1) / (t + 1)` of `t = exp(2x)` (the interpreter does not
    // model `Exp2`): zeros, small and large `t`, the reciprocal's flush past `2^126`, infinities,
    // and NaNs.
    let f = float(32);
    let tanh_cases: Vec<Case> = [
        0.0f32,
        -0.0,
        1e-40,
        3.9e-22,
        0.25,
        1.0,
        1.5,
        7.0,
        1e20,
        8.5e37,
        1.7e38,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ]
    .iter()
    .map(|t| vec![vec![u64::from(t.to_bits())]])
    .collect();
    for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
        let routine = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
        let inline = kernel(&[f], f, &|fb, t| {
            let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                math::fast_divide(fb, f, p, q, Contraction::Forbidden, reciprocal)
            };
            math::tanh_of_exp(fb, f, &divide, t[0])
        })
        .expect("inline");
        let called = kernel_defs(
            &[f],
            f,
            &|b| Ok(vec![math::define_fast_division(b, routine)?]),
            &|fb, t, funcs| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, funcs[0], p, q)
                };
                math::tanh_of_exp(fb, f, &divide, t[0])
            },
            false,
            false,
        )
        .expect("called");
        compare(
            &format!("fast tanh's quotient {reciprocal:?}"),
            &[f],
            f,
            inline,
            called,
            &tanh_cases,
        );
    }
    let fmod_cases: Vec<Case> = FMOD
        .iter()
        .map(|(x, y, _)| vec![vec![u64::from(*x)], vec![u64::from(*y)]])
        .collect();
    for route in [Route::Khr, Route::Wide, Route::Soft] {
        let fma = Fma {
            form: f,
            route,
            tiny: None,
        };
        let reciprocal = Reciprocal::of(route);
        let routine = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
        let fmas = |b: &mut ModuleBuilder| -> BResult<Vec<FuncId>> {
            Ok(fma
                .routine()
                .map(|r| fma::define(b, r))
                .transpose()?
                .into_iter()
                .collect())
        };
        let inline = kernel_defs(
            &[f, f],
            f,
            &fmas,
            &|fb, x, funcs| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::fast_divide(fb, f, p, q, Contraction::Forbidden, reciprocal)
                };
                let call = |fb: &mut FunctionBuilder<'_>, args: &[Value]| {
                    fma::emit(fb, fma, args, funcs.first().copied())
                };
                math::fast_fmod(fb, x[0], x[1], &divide, &call)
            },
            false,
            false,
        )
        .expect("inline");
        let called = kernel_defs(
            &[f, f],
            f,
            &|b| {
                let mut funcs = fmas(b)?;
                funcs.push(math::define_fast_division(b, routine)?);
                Ok(funcs)
            },
            &|fb, x, funcs| {
                let division = funcs.last().copied().expect("the division routine");
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, division, p, q)
                };
                let fma_id = fma.routine().and(funcs.first().copied());
                let call =
                    |fb: &mut FunctionBuilder<'_>, args: &[Value]| fma::emit(fb, fma, args, fma_id);
                math::fast_fmod(fb, x[0], x[1], &divide, &call)
            },
            false,
            false,
        )
        .expect("called");
        compare(
            &format!("fast fmod {route:?}"),
            &[f, f],
            f,
            inline,
            called,
            &fmod_cases,
        );
    }
}

/// `fast::fmod` (`math::fast_fmod`) on every `float` route of its fused multiply-add, at the
/// lanes of [`FMOD`], bit for bit: Metal's answer is the expression on its own quotient, and the
/// interpreter's quotient there is Metal's. `OpFmaKHR` is the device's (the interpreter's IEEE
/// `fma`, which keeps denormals), so on that route a lane with an `f32` denormal operand or a
/// zero or denormal answer is not asserted.
#[test]
fn fast_fmod_answers_as_metal_on_every_route() {
    let f = float(32);
    let cases: Vec<Case> = FMOD
        .iter()
        .map(|(x, y, _)| vec![vec![u64::from(*x)], vec![u64::from(*y)]])
        .collect();
    for route in [Route::Khr, Route::Wide, Route::Soft] {
        let fma = Fma {
            form: f,
            route,
            tiny: None,
        };
        let routine = fma.routine();
        let division = math::FastDivision::new(f, Reciprocal::of(route), Contraction::Forbidden);
        let m = kernel_defs(
            &[f, f],
            f,
            &|b| {
                let mut funcs = vec![math::define_fast_division(b, division)?];
                if let Some(r) = routine {
                    funcs.push(fma::define(b, r)?);
                }
                Ok(funcs)
            },
            &|fb, x, funcs| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, funcs[0], p, q)
                };
                let call = |fb: &mut FunctionBuilder<'_>, args: &[Value]| {
                    fma::emit(fb, fma, args, funcs.get(1).copied())
                };
                math::fast_fmod(fb, x[0], x[1], &divide, &call)
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("fast fmod {route:?}: build: {e:?}"));
        check_module(
            &format!("fast fmod {route:?} against Metal"),
            m,
            &[f, f],
            f,
            &cases,
            &|c| {
                let (x, y) = (arg(c, 0) as u32, arg(c, 1) as u32);
                let want = FMOD
                    .iter()
                    .find(|(a, b, _)| *a == x && *b == y)
                    .map(|(_, _, w)| *w)?;
                // `OpFmaKHR` keeps denormal operands and results (the device's denormal mode),
                // so on that route a lane with an `f32` denormal operand or a zero or denormal
                // answer is the device's.
                let denormal = |b: u32| b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0;
                if route == Route::Khr && (denormal(x) || denormal(y) || want & 0x7F80_0000 == 0) {
                    return None;
                }
                Some(vec![u64::from(want)])
            },
            &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
        );
    }
}

/// `(x, rsqrt(x), fast::sqrt(x))`: Metal's fast `rsqrt` and `sqrt` at the special lanes of case
/// 08bc6611 and 40 others drawn at random.
const SQRT: [(u32, u32, u32); 52] = [
    (0x00000000, 0x7F800000, 0x00000000),
    (0x00000001, 0x7F800000, 0x00000000),
    (0x03EB9990, 0x5D3CB198, 0x21ADA837),
    (0x0AD36389, 0x59C734EB, 0x25247E12),
    (0x10BF38EE, 0x56D1729C, 0x281C7316),
    (0x2813F70D, 0x4B285D4D, 0x33C2A01C),
    (0x30125D7E, 0x47294838, 0x37C19205),
    (0x3252A670, 0x460D1B6F, 0x38E8387D),
    (0x382DB84C, 0x431B6233, 0x3BD2E272),
    (0x392CDDDE, 0x429BC440, 0x3C525DB5),
    (0x3CB5ECBD, 0x40D6BBAE, 0x3E189946),
    (0x3CD3FD00, 0x40C6ECC3, 0x3E24B9BD),
    (0x3CE1B503, 0x40C0C8DC, 0x3E29F8DA),
    (0x3CFEE21B, 0x40B56A5C, 0x3E349FC2),
    (0x3EB47B35, 0x3FD79713, 0x3F17FDFC),
    (0x3F800000, 0x3F800000, 0x3F800000),
    (0x3F800040, 0x3F7FFFC0, 0x3F800020),
    (0x3FA97BAA, 0x3F5E79AD, 0x3F9349CB),
    (0x3FB5E18D, 0x3F56C249, 0x3F989495),
    (0x3FC90FDB, 0x3F4C422A, 0x3FA06C9A),
    (0x3FCFB1F9, 0x3F48F861, 0x3FA30C8E),
    (0x3FFFC4B1, 0x3F3519EF, 0x3FB4EFFA),
    (0x4020E4ED, 0x3F217543, 0x3FCAF358),
    (0x403F7694, 0x3F14023B, 0x3FDD6471),
    (0x40490FDB, 0x3F106EBA, 0x3FE2DFC4),
    (0x41A18EC6, 0x3E63DDE9, 0x408FCDA5),
    (0x42B06B8F, 0x3DDA0EB3, 0x411645B9),
    (0x46331409, 0x3C190A90, 0x42D61CBA),
    (0x4A1C15F4, 0x3A23ED1A, 0x44C7E515),
    (0x527736DD, 0x3602412F, 0x48FB919D),
    (0x56A91948, 0x33DEBA5C, 0x4B131F06),
    (0x71A27292, 0x26633DEC, 0x589032E3),
    (0x7F7FFFFF, 0x1F800000, 0x5F7FFFFF),
    (0x7F800000, 0x00000000, 0x7F800000),
    (0x7FC00000, 0x7FC00000, 0x7FC00000),
    (0x80000000, 0xFF800000, 0x80000000),
    (0x80400000, 0xFF800000, 0x80000000),
    (0xBDFF9977, 0x7FC00000, 0x7FC00000),
    (0xBF20930B, 0x7FC00000, 0x7FC00000),
    (0xBF46AC52, 0x7FC00000, 0x7FC00000),
    (0xBF6EF8AD, 0x7FC00000, 0x7FC00000),
    (0xBF800000, 0x7FC00000, 0x7FC00000),
    (0xBF9B0F44, 0x7FC00000, 0x7FC00000),
    (0xBF9E0E31, 0x7FC00000, 0x7FC00000),
    (0xBFF56F61, 0x7FC00000, 0x7FC00000),
    (0xC017738C, 0x7FC00000, 0x7FC00000),
    (0xC0269500, 0x7FC00000, 0x7FC00000),
    (0xC22E7F33, 0x7FC00000, 0x7FC00000),
    (0xC36B16DC, 0x7FC00000, 0x7FC00000),
    (0xC48E08E1, 0x7FC00000, 0x7FC00000),
    (0xC59BB47B, 0x7FC00000, 0x7FC00000),
    (0xFF800000, 0x7FC00000, 0x7FC00000),
];

/// `fast::sqrt` (`math::fast_sqrt_of`) from Metal's own `rsqrt` is Metal's answer bit for bit:
/// `x · rsqrt(x)`, with the zero of `x`'s sign for a zero or an `f32` denormal and `+inf` for
/// `+inf`, where the product is NaN.
#[test]
fn fast_sqrt_answers_as_metal_from_its_rsqrt() {
    let f = float(32);
    let cases: Vec<Case> = SQRT
        .iter()
        .map(|(x, r, _)| vec![vec![u64::from(*x)], vec![u64::from(*r)]])
        .collect();
    check_accept(
        "fast sqrt f32 against Metal",
        &[f, f],
        f,
        &|fb, a| math::fast_sqrt_of(fb, f, a[0], a[1]),
        &cases,
        &|c| {
            let (x, r) = (arg(c, 0) as u32, arg(c, 1) as u32);
            SQRT.iter()
                .find(|(a, b, _)| *a == x && *b == r)
                .map(|(_, _, w)| vec![u64::from(*w)])
        },
        &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
    );
}

/// The fast functions Metal composes are emitted as its compositions, read from the module:
/// `fast::exp` is one `Exp2` of a product with `float(log2 e)`, `fast::log` one `Log2` times
/// `float(ln 2)`, `fast::powr` and `fast::pow` one `Exp2` of a product with one `Log2`, and
/// `fast::sqrt` one `InverseSqrt`; none reads `Exp`, `Log`, `Pow`, or `Sqrt`.
#[test]
fn fast_functions_are_metals_compositions() {
    const OP_CONSTANT: u32 = 43;
    const OP_EXT_INST: u32 = 12;
    const OP_FMUL: u32 = 133;
    // GLSL.std.450: Pow, Exp, Log, Exp2, Log2, Sqrt, InverseSqrt.
    const GLSL: [u32; 7] = [26, 27, 28, 29, 30, 31, 32];
    let f = float(32);
    let read = |op: FloatFn| {
        let m = kernel(&vec![f; op.arity()], f, &|fb, a| {
            math::float_fn(
                fb,
                op,
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                a,
                &Lib::instructions(),
            )
        })
        .expect("kernel");
        let words = crate::spirv::serialize::serialize(m).expect("words").words;
        let (mut ext, mut fmul, mut consts) = ([0u32; 7], 0, Vec::new());
        let mut i = 5;
        while i < words.len() {
            let (n, code) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
            if code == OP_EXT_INST {
                if let Some(k) = GLSL.iter().position(|g| words.get(i + 4) == Some(g)) {
                    ext[k] += 1;
                }
            }
            if code == OP_FMUL {
                fmul += 1;
            }
            if code == OP_CONSTANT && n == 4 {
                consts.push(words[i + 3]);
            }
            i += n.max(1);
        }
        (ext, fmul, consts)
    };
    let log2e = (std::f64::consts::LOG2_E as f32).to_bits();
    let ln2 = (std::f64::consts::LN_2 as f32).to_bits();
    assert_eq!((log2e, ln2), (0x3FB8_AA3B, 0x3F31_7218));
    let (ext, fmul, consts) = read(FloatFn::FastExp);
    assert_eq!((ext, fmul), ([0, 0, 0, 1, 0, 0, 0], 1));
    assert!(consts.contains(&log2e));
    let (ext, fmul, consts) = read(FloatFn::FastLog);
    assert_eq!((ext, fmul), ([0, 0, 0, 0, 1, 0, 0], 1));
    assert!(consts.contains(&ln2));
    let (ext, fmul, _) = read(FloatFn::FastPowr);
    assert_eq!((ext, fmul), ([0, 0, 0, 1, 1, 0, 0], 1));
    let (ext, _, _) = read(FloatFn::FastPow);
    assert_eq!(ext, [0, 0, 0, 1, 1, 0, 0]);
    let (ext, _, _) = read(FloatFn::Pow);
    assert_eq!(ext, [1, 0, 0, 0, 0, 0, 0]);
    let (ext, _, _) = read(FloatFn::FastSqrt);
    assert_eq!(ext, [0, 0, 0, 0, 0, 0, 1]);
}
