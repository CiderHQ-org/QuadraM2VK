//! A fragment function's float color output rounded to the code Metal stores
//! (`color_write::round`), on the interpreter, against the rules probes measured on an Apple M3 Max
//! and an M4 (`runtime_lane::color_writes` holds the same rules).
//!
//! Each case writes one value to every lane of a four-lane output whose lanes take different
//! rules, so every rule runs beside the others. The reference is the value each rule writes:
//! `c` times the `f32` nearest `1 / m` for Metal's unorm or snorm code `c`, the linear value of
//! Metal's sRGB byte, and the small float Metal stores, widened. A further check asserts that a device
//! which rounds the `f32` product to nearest (every Vulkan driver measured, ties aside) stores
//! each such value as its code.

use super::*;
use crate::lower::arith::color_write;
use crate::lower::arith::srgb::{self, ColorTable};
use crate::pipeline::lower_inputs::ChannelWrite;

/// Metal's code of `x` in a unorm channel of maximum `m`: NaN 0, clamped to `[0, 1]`, the exact
/// product rounded to nearest (its only tie, at 0.5, to even).
fn unorm_code(x: f32, m: u32) -> u32 {
    let x = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) };
    exact_round(f64::from(x), m)
}

/// The exact product `x * m` (`x` an `f32` in `[0, 1]`, `m < 2^16`) rounded to nearest even:
/// the product of a 24-bit significand and a 16-bit integer is exact in `f64`.
fn exact_round(x: f64, m: u32) -> u32 {
    (x * f64::from(m)).round_ties_even() as u32
}

/// Metal's code of `x` in a snorm channel of maximum `m` (two's complement as `i32`).
fn snorm_code(x: f32, m: u32, blended: bool) -> i32 {
    if x.is_nan() {
        return if blended { -(m as i32) } else { 0 };
    }
    let c = exact_round(f64::from(x.abs().min(1.0)), m) as i32;
    if x < 0.0 {
        -c
    } else {
        c
    }
}

/// The sRGB byte Metal stores for `x`: its twelve-bit index (the exact product with 4095), and
/// the byte the MSL transfer with the exponent `0.41666` gives that index, which reproduces all
/// 4096 of Metal's (`srgb_conversions_answer_with_metals_tables`).
fn srgb_byte(x: f32) -> u32 {
    let i = unorm_code(x, 4095);
    let c = f64::from(i) / 4095.0;
    let v = if c < 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(0.41666) - 0.055
    };
    (255.0 * v).round_ties_even() as u32
}

/// The `f32` nearest the linear value of sRGB byte `b`.
fn srgb_linear(b: u32) -> f32 {
    let c = f64::from(b) / 255.0;
    let v = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    v as f32
}

/// Metal's code of `x` in a float channel of a five-bit exponent (bias 15) and `man` fraction
/// bits (a sign bit above where `signed`): rounded toward zero, a finite value past the largest
/// to the largest, an infinity kept, a NaN the quiet NaN, a negative value 0 where unsigned.
/// Computed in `f64` on the value's grid, apart from the emission's bit operations.
fn small_float_code(x: f32, man: u32, signed: bool) -> u32 {
    let infinity = 0x1F << man;
    if x.is_nan() {
        return infinity | (1 << (man - 1));
    }
    if x.is_sign_negative() && !signed {
        return 0;
    }
    let sign = if x.is_sign_negative() {
        1 << (man + 5)
    } else {
        0
    };
    let a = f64::from(x.abs());
    if a.is_infinite() {
        return sign | infinity;
    }
    if a >= (2.0 - 2f64.powi(-(man as i32))) * 32768.0 {
        return sign | (infinity - 1);
    }
    let man = man as i32;
    if a < 2f64.powi(-14) {
        return sign | (a / 2f64.powi(-14 - man)).floor() as u32;
    }
    let e = ((x.abs().to_bits() >> 23) & 0xFF) as i32 - 127;
    let k = (a / 2f64.powi(e - man)).floor() as u32;
    sign | (((e + 15) as u32) << man) | (k - (1 << man))
}

/// Metal's code of `x` in such a float channel rounded to nearest even (a render state's
/// half-float channel under `textureWriteRoundingMode` 2): [`small_float_code`] of `x` rounded to
/// nearest with ties to even on its grid, computed in `f64`, a finite value whose rounding reaches
/// 2^16 infinity.
fn small_float_nearest_code(x: f32, man: u32, signed: bool) -> u32 {
    if x.is_nan() || x.is_infinite() || (x.is_sign_negative() && !signed) {
        return small_float_code(x, man, signed);
    }
    let a = f64::from(x.abs());
    let e = if a < 2f64.powi(-14) {
        -14
    } else {
        ((x.abs().to_bits() >> 23) & 0xFF) as i32 - 127
    };
    let q = 2f64.powi(e - man as i32);
    let r = (a / q).round_ties_even() * q;
    let r = if r >= 65536.0 { f64::INFINITY } else { r };
    let v = (if x.is_sign_negative() { -r } else { r }) as f32;
    small_float_code(v, man, signed)
}

/// `x` rounded to a half to nearest even, widened (a NaN stays a NaN).
fn half_nearest(x: f32) -> f32 {
    let c = small_float_nearest_code(x, 10, true);
    if x.is_nan() {
        x
    } else {
        small_float_value(c, 10, true) as f32
    }
}

/// The value of code `c` of a float channel as [`small_float_code`] describes it (not a NaN).
fn small_float_value(c: u32, man: u32, signed: bool) -> f64 {
    let negative = signed && (c >> (man + 5)) & 1 == 1;
    let e = (c >> man) & 0x1F;
    let f = f64::from(c & ((1 << man) - 1));
    let man = man as i32;
    let a = match e {
        0 => f * 2f64.powi(-14 - man),
        0x1F => f64::INFINITY,
        _ => (1.0 + f * 2f64.powi(-man)) * 2f64.powi(e as i32 - 15),
    };
    if negative {
        -a
    } else {
        a
    }
}

/// The value written for code `c` of maximum `m`: `c` times the `f32` nearest `1 / m`.
fn code_value(c: u32, m: u32) -> f32 {
    c as f32 * (1.0 / m as f32)
}

/// The value `rule` writes for `x`, as `f32` bits.
fn written(rule: Option<ChannelWrite>, x: f32) -> u64 {
    let bits = match rule {
        None | Some(ChannelWrite::Device | ChannelWrite::BlendedNearestEven { .. }) => x.to_bits(),
        Some(ChannelWrite::Unorm { bits }) => {
            let m = (1u32 << bits) - 1;
            code_value(unorm_code(x, m), m).to_bits()
        }
        Some(ChannelWrite::Snorm { bits, blended }) => {
            let m = (1u32 << (bits - 1)) - 1;
            let c = snorm_code(x, m, blended);
            let q = code_value(c.unsigned_abs(), m);
            if c < 0 || (c == 0 && x < 0.0) {
                (-q).to_bits()
            } else {
                q.to_bits()
            }
        }
        Some(ChannelWrite::Srgb) => srgb_linear(srgb_byte(x)).to_bits(),
        // A selected channel writes by the rule its code selects: no reference of its own.
        Some(ChannelWrite::Selected) => unreachable!("a selected channel's rule is its code's"),
        Some(ChannelWrite::HalfUnorm { bits }) => {
            let m = (1u32 << bits) - 1;
            code_value(unorm_code(half_nearest(x), m), m).to_bits()
        }
        Some(
            rule @ (ChannelWrite::FloatTowardZero { mantissa, signed }
            | ChannelWrite::FloatNearestEven { mantissa, signed }),
        ) => {
            let man = u32::from(mantissa);
            let c = match rule {
                ChannelWrite::FloatNearestEven { .. } => small_float_nearest_code(x, man, signed),
                _ => small_float_code(x, man, signed),
            };
            if (c >> man) & 0x1F == 0x1F && c & ((1 << man) - 1) != 0 {
                0x7FC0_0000
            } else {
                (small_float_value(c, man, signed) as f32).to_bits()
            }
        }
    };
    u64::from(bits)
}

/// The values beside each code and each tie of a channel of maximum `m` (every `step`-th),
/// `n` `f32` steps either side, over `[lo, 1]`.
fn around_codes(m: u32, signed: bool, step: usize, n: u32) -> Vec<f32> {
    let lo = if signed { -(m as i64) } else { 0 };
    let mut out = Vec::new();
    for k in (lo..=i64::from(m)).step_by(step) {
        for c in [k as f64, k as f64 + 0.5] {
            let x = (c / f64::from(m)) as f32;
            for d in 0..=n {
                out.push(f32::from_bits(x.to_bits().wrapping_add(d)));
                out.push(f32::from_bits(x.to_bits().wrapping_sub(d)));
            }
        }
    }
    out
}

fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.0,
        -2.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-10,
        -1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xFFC0_0000),
        f32::from_bits(0x7F80_0001),
        65504.0,
        65519.0,
        65520.0,
        -65520.0,
        1e10,
        -1e10,
    ]
}

/// The half grid: every 7th finite half and the ties beside it, one `f32` step either side,
/// of either sign.
fn half_values() -> Vec<f32> {
    let mut out = Vec::new();
    for h in (0u16..0x7C00)
        .step_by(7)
        .chain([0x03FF, 0x0400, 0x7BFE, 0x7BFF])
    {
        let a = f16_to_f64(h);
        let b = if h == 0x7BFF {
            65536.0
        } else {
            f16_to_f64(h + 1)
        };
        let t = ((a + b) / 2.0) as f32;
        for v in [
            a as f32,
            t,
            f32::from_bits(t.to_bits() + 1),
            f32::from_bits(t.to_bits() - 1),
        ] {
            out.extend([v, -v]);
        }
    }
    out
}

/// An unsigned float grid of `man` fraction bits: every finite code's value and the ties beside
/// it, one `f32` step either side, and their negations.
fn small_float_values(man: u32) -> Vec<f32> {
    let mut out = Vec::new();
    let largest = (0x1F << man) - 1;
    for c in 0..=largest {
        let a = small_float_value(c, man, false);
        let b = if c == largest {
            65536.0
        } else {
            small_float_value(c + 1, man, false)
        };
        let t = ((a + b) / 2.0) as f32;
        for v in [
            a as f32,
            t,
            f32::from_bits(t.to_bits() + 1),
            f32::from_bits(t.to_bits() - 1),
        ] {
            out.extend([v, -v]);
        }
    }
    out
}

/// The sRGB grid: each twelve-bit index's tie with two `f32` steps either side, and each byte's
/// linear value.
fn srgb_values() -> Vec<f32> {
    let mut out = Vec::new();
    for i in 0..4095u32 {
        let t = ((f64::from(i) + 0.5) / 4095.0) as f32;
        for d in 0..=2 {
            out.push(f32::from_bits(t.to_bits() + d));
            out.push(f32::from_bits(t.to_bits() - d));
        }
    }
    out.extend((0..256).map(srgb_linear));
    out
}

fn run_rules(what: &str, rules: [Option<ChannelWrite>; 4], values: &[f32]) {
    run_rules_in(what, rules, values, ColorTable::Uniform);
}

/// [`run_rules`] with the color table in the buffer `class` names.
fn run_rules_in(what: &str, rules: [Option<ChannelWrite>; 4], values: &[f32], class: ColorTable) {
    let f = vec(float(32), 4);
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits()); 4]])
        .collect();
    check_accept(
        what,
        &[f],
        f,
        &|fb, x| {
            let table = srgb::declare_color_table(fb.module(), Stage::Kernel, class)?;
            color_write::round(fb, f, x[0], &rules, Some(table))
        },
        &cases,
        &|c| {
            let x = f32::from_bits(c[0][0] as u32);
            Some(rules.iter().map(|r| written(*r, x)).collect())
        },
        &|g, w| g == w,
    );
}

/// Every unorm and snorm code's value written lies within `2^-7` of the code once multiplied
/// by its maximum, exactly and as an `f32` product, so a device that rounds that product to
/// nearest stores the code, whatever it does at ties, and so does Metal; and the linear value of
/// every sRGB byte is that byte's in Metal's encode.
/// A render state's half-float channel under `textureWriteRoundingMode` 2 rounds to nearest even
/// (`color_writes::HALF_NEAREST`): the consumer's spot values on an M3 Max and our probe's
/// on an M4 (1 + 3 * 2^-12 to `0x3C01`, the tie 1 + 3 * 2^-11 to `0x3C02`, 65519 to `0x7BFF`, from
/// 65520 infinity), and on the interpreter every 7th half and the ties beside it, the unsigned
/// 11- and 10-bit grids under the same rule, and the specials: each lane of an output beside
/// lanes of the other rules, and every lane of a vector by one rule's instructions on the whole
/// vector (`color_write::small_float`, a storage write's), both rules.
#[test]
fn nearest_even_halves_round_to_mode_twos_codes() {
    let code = |x: f32| small_float_nearest_code(x, 10, true);
    assert_eq!(code(1.0 + 3.0 * 2f32.powi(-12)), 0x3C01);
    assert_eq!(code(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
    assert_eq!(code(65519.0), 0x7BFF);
    assert_eq!(code(65520.0), 0x7C00);
    assert_eq!(code(-65520.0), 0xFC00);
    assert_eq!(code(f32::NAN), 0x7E00);
    let mut values = half_values();
    values.extend(small_float_values(6));
    values.extend(small_float_values(5));
    values.extend(specials());
    let nearest = |mantissa, signed| Some(ChannelWrite::FloatNearestEven { mantissa, signed });
    run_rules(
        "nearest-even half, half, 11-bit, 10-bit",
        [
            Some(crate::pipeline::color_writes::HALF_NEAREST),
            nearest(10, true),
            nearest(6, false),
            nearest(5, false),
        ],
        &values,
    );
    let f = vec(float(32), 4);
    for rule in [
        crate::pipeline::color_writes::HALF_NEAREST,
        crate::pipeline::color_writes::HALF,
        ChannelWrite::FloatNearestEven {
            mantissa: 6,
            signed: false,
        },
    ] {
        let cases: Vec<Case> = values
            .iter()
            .map(|x| vec![vec![u64::from(x.to_bits()); 4]])
            .collect();
        check_accept(
            &format!("{rule:?} on the whole vector"),
            &[f],
            f,
            &|fb, x| color_write::small_float(fb, f, x[0], rule),
            &cases,
            &|c| Some(vec![written(Some(rule), f32::from_bits(c[0][0] as u32)); 4]),
            &|g, w| g == w,
        );
    }
}

#[test]
fn written_values_store_their_codes() {
    for m in [1u32, 3, 15, 31, 63, 255, 1023, 65535, 127, 32767] {
        for c in 0..=m {
            let q = code_value(c, m);
            let exact = f64::from(q) * f64::from(m);
            assert!(
                (exact - f64::from(c)).abs() < 1.0 / 128.0,
                "code {c} of {m}: {exact}"
            );
            let product = f64::from(q * m as f32);
            assert!(
                (product - f64::from(c)).abs() < 1.0 / 128.0,
                "code {c} of {m}: {product}"
            );
            assert_eq!(
                exact_round(f64::from(q.min(1.0)), m),
                c,
                "Metal's code {c} of {m}"
            );
        }
    }
    for b in 0..256 {
        assert_eq!(srgb_byte(srgb_linear(b)), b, "sRGB byte {b}");
    }
}

#[test]
fn eight_bit_channels_round_to_metals_codes() {
    let mut values = around_codes(255, true, 1, 2);
    values.extend(around_codes(255, false, 1, 2));
    values.extend(specials());
    let snorm = |blended| Some(ChannelWrite::Snorm { bits: 8, blended });
    run_rules(
        "unorm8, snorm8, blended snorm8, device",
        [
            Some(ChannelWrite::Unorm { bits: 8 }),
            snorm(false),
            snorm(true),
            Some(ChannelWrite::Device),
        ],
        &values,
    );
}

#[test]
fn wide_and_packed_channels_round_to_metals_codes() {
    let mut values = around_codes(65535, false, 61, 2);
    values.extend(around_codes(32767, true, 37, 2));
    values.extend(around_codes(1023, false, 1, 2));
    values.extend(around_codes(3, false, 1, 2));
    values.extend(specials());
    run_rules(
        "unorm16, snorm16, unorm10, unorm2",
        [
            Some(ChannelWrite::Unorm { bits: 16 }),
            Some(ChannelWrite::Snorm {
                bits: 16,
                blended: false,
            }),
            Some(ChannelWrite::Unorm { bits: 10 }),
            Some(ChannelWrite::Unorm { bits: 2 }),
        ],
        &values,
    );
}

/// The packed formats' narrow channels (`A1BGR5Unorm`'s alpha, `ABGR4Unorm`'s, `B5G6R5Unorm`'s)
/// and the narrowest snorm channel: the exact product's rounding for every `k` the formats take.
#[test]
fn narrow_channels_round_to_metals_codes() {
    let mut values = Vec::new();
    for m in [1, 15, 31, 63] {
        values.extend(around_codes(m, false, 1, 3));
    }
    values.extend(around_codes(1, true, 1, 3));
    values.extend(specials());
    run_rules(
        "unorm1, unorm4, unorm5, unorm6, snorm2",
        [
            Some(ChannelWrite::Unorm { bits: 1 }),
            Some(ChannelWrite::Unorm { bits: 4 }),
            Some(ChannelWrite::Unorm { bits: 6 }),
            Some(ChannelWrite::Snorm {
                bits: 2,
                blended: false,
            }),
        ],
        &values,
    );
    run_rules(
        "unorm5",
        [Some(ChannelWrite::Unorm { bits: 5 }), None, None, None],
        &values,
    );
}

/// The packed 16-bit formats' channels (`ChannelWrite::HalfUnorm`) round the value to a half
/// first, then as unorm channels: our probe's spot values on an M3 Max (`0x3C842FFF`, a
/// half below 1/62, to 5-bit code 0; `0x3D463000` to 2; `0x3EFFFFFF`, whose half is 0.5, to 16
/// of 31 and 32 of 63; 0.5 to 1-bit code 0; `0x3D089000` to 4-bit code 0; `0x3C021000` to 6-bit
/// code 0), then on the interpreter the values beside every code and tie of 1, 4, 5 and 6 bits,
/// the halves either side of each tie, and the specials, beside lanes of the other widths.
#[test]
fn packed_channels_round_through_a_half() {
    let code = |x: u32, bits: u32| unorm_code(half_nearest(f32::from_bits(x)), (1 << bits) - 1);
    assert_eq!(code(0x3C84_2FFF, 5), 0);
    assert_eq!(code(0x3D46_3000, 5), 2);
    assert_eq!(code(0x3EFF_FFFF, 5), 16);
    assert_eq!(code(0x3EFF_FFFF, 6), 32);
    assert_eq!(code(0.5f32.to_bits(), 1), 0);
    assert_eq!(code(0x3D08_9000, 4), 0);
    assert_eq!(code(0x3C02_1000, 6), 0);
    let mut values = Vec::new();
    for m in [1u32, 15, 31, 63] {
        values.extend(around_codes(m, false, 1, 3));
        // The halves nearest each tie and either side of it, and the midpoints between them
        // (a half's step is `0x2000` in the `f32` bits there), three `f32` steps either side.
        for k in 0..m {
            let tie = ((f64::from(k) + 0.5) / f64::from(m)) as f32;
            let h = half_nearest(tie).to_bits();
            for at in (h - 0x2000..=h + 0x2000).step_by(0x1000) {
                values.extend((0..3).flat_map(|d| [at + d, at - d]).map(f32::from_bits));
            }
        }
    }
    values.extend(specials());
    let packed = |bits| Some(ChannelWrite::HalfUnorm { bits });
    run_rules(
        "half unorm 1, 4, 5, 6",
        [packed(1), packed(4), packed(5), packed(6)],
        &values,
    );
}

/// The sRGB byte's estimate ([`srgb::color_linear`]) lies between `b + 1/3` and `b + 5/3` for
/// Metal's byte `b` of every encode index, evaluated in `f32` step by step and with every step
/// fused, so `floor` of it, clamped to `[1, 255]`, less one where the index is below that byte's
/// first index, is `b`.
#[test]
fn srgb_estimates_lie_a_byte_above_metals() {
    let first = |b: u32| u32::from(srgb::SRGB_ENCODE_FIRST[b as usize]);
    for fused in [false, true] {
        let step = |a: f32, b: f32, c: f32| if fused { a.mul_add(b, c) } else { a * b + c };
        for i in 0..4096u32 {
            let metal = (0..256).rev().find(|&b| first(b) <= i).unwrap_or(0);
            let x = i as f32;
            let e = if i < 13 {
                step(x, srgb::ENCODE_LINEAR, 1.0)
            } else {
                let t = (x * (1.0f32 / 4095.0)).sqrt();
                let [lead, rest @ ..] = srgb::ENCODE_ESTIMATE;
                rest.iter().fold(lead, |e, &a| step(e, t, a))
            };
            let above = f64::from(e) - f64::from(metal);
            assert!(
                (0.333..=1.667).contains(&above),
                "index {i} (fused {fused}): estimate {e}, Metal's byte {metal}"
            );
            let b0 = (e as u32).clamp(1, 255);
            let b = if i < first(b0) { b0 - 1 } else { b0 };
            assert_eq!(b, metal, "index {i} (fused {fused})");
        }
    }
}

/// An sRGB channel writes Metal's byte's linear value from the color table in either buffer, beside
/// an 8-bit alpha and a half.
#[test]
fn srgb_and_half_channels_take_metals_values() {
    let mut values = srgb_values();
    values.extend(half_values());
    values.extend(specials());
    for class in [ColorTable::Uniform, ColorTable::Storage] {
        run_rules_in(
            &format!("sRGB, sRGB alpha, half, none ({class:?})"),
            [
                Some(ChannelWrite::Srgb),
                Some(ChannelWrite::Unorm { bits: 8 }),
                Some(ChannelWrite::FloatTowardZero {
                    mantissa: 10,
                    signed: true,
                }),
                None,
            ],
            &values,
            class,
        );
    }
}

/// The color table's bytes are 256 little-endian `uvec4` entries, entry `b` holding byte `b`'s
/// first encode index, its linear value's bits and byte `b - 1`'s, so the write's choice between
/// the two linear values by the index against the first index is Metal's encode.
#[test]
fn the_color_table_holds_each_bytes_first_index_and_linear_values() {
    let table = srgb::color_write_table();
    assert_eq!(table.len(), 256 * 16);
    let word = |b: usize, k: usize| {
        let at = b * 16 + k * 4;
        let bytes: [u8; 4] = table[at..at + 4].try_into().expect("word");
        u32::from_le_bytes(bytes)
    };
    for b in 0..256usize {
        assert_eq!(
            word(b, 0),
            u32::from(srgb::SRGB_ENCODE_FIRST[b]),
            "byte {b}"
        );
        assert_eq!(word(b, 1), srgb_linear(b as u32).to_bits(), "byte {b}");
        let prev = if b == 0 {
            0
        } else {
            srgb_linear(b as u32 - 1).to_bits()
        };
        assert_eq!(word(b, 2), prev, "byte {b}");
        assert_eq!(word(b, 3), 0, "byte {b}");
    }
}

/// `RG11B10Float`'s unsigned 11- and 10-bit floats, beside a half.
#[test]
fn packed_float_channels_round_toward_zero() {
    let mut values = small_float_values(6);
    values.extend(small_float_values(5));
    values.extend(half_values());
    values.extend(specials());
    let f = |mantissa, signed| Some(ChannelWrite::FloatTowardZero { mantissa, signed });
    run_rules(
        "float11, float11, float10, half",
        [f(6, false), f(6, false), f(5, false), f(10, true)],
        &values,
    );
}

/// The reference rounding agrees with the half one Metal's probes gave (`0x7E00` for NaN,
/// `0x7BFF` past 65504) and with `RG11B10Float`'s spot answers.
#[test]
fn small_float_codes_give_metals_spot_answers() {
    assert_eq!(small_float_code(f32::NAN, 10, true), 0x7E00);
    assert_eq!(small_float_code(65520.0, 10, true), 0x7BFF);
    assert_eq!(small_float_code(-1.0, 10, true), 0xBC00);
    assert_eq!(small_float_code(8.940_697e-8, 10, true), 1);
    assert_eq!(small_float_code(f32::NAN, 6, false), 0x7E0);
    assert_eq!(small_float_code(f32::NAN, 5, false), 0x3F0);
    assert_eq!(small_float_code(f32::NEG_INFINITY, 6, false), 0);
    assert_eq!(small_float_code(-1.0, 5, false), 0);
    assert_eq!(small_float_code(f32::INFINITY, 6, false), 0x7C0);
    assert_eq!(small_float_code(1e10, 6, false), 0x7BF);
    assert_eq!(small_float_code(1e10, 5, false), 0x3DF);
    assert_eq!(small_float_code(1.0, 6, false), 15 << 6);
    assert_eq!(small_float_code(1.0 + 1.0 / 64.0 - 1e-7, 6, false), 15 << 6);
    for h in (0u16..0x7C00).step_by(5) {
        let x = f16_to_f64(h) as f32;
        assert_eq!(small_float_code(x, 10, true), u32::from(h), "{h:#06x}");
        assert_eq!(
            small_float_code(-x, 10, true),
            u32::from(h | 0x8000),
            "-{h:#06x}"
        );
    }
}

/// A scalar output and a half output (held widened) round as their lanes do.
#[test]
fn scalar_and_half_outputs_round() {
    let values: Vec<f32> = around_codes(255, false, 1, 1)
        .into_iter()
        .chain(specials())
        .collect();
    let s = float(32);
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits())]])
        .collect();
    let rules = [Some(ChannelWrite::Unorm { bits: 8 }), None, None, None];
    check_accept(
        "scalar unorm8",
        &[s],
        s,
        &|fb, x| color_write::round(fb, s, x[0], &rules, None),
        &cases,
        &|c| Some(vec![written(rules[0], f32::from_bits(c[0][0] as u32))]),
        &|g, w| g == w,
    );
    let h = vec(float(16), 2);
    let halves: Vec<u64> = (0u64..0x10000).step_by(3).collect();
    let cases: Vec<Case> = halves.iter().map(|b| vec![vec![*b, *b]]).collect();
    let rules = [
        Some(ChannelWrite::Srgb),
        Some(ChannelWrite::Snorm {
            bits: 8,
            blended: false,
        }),
        None,
        None,
    ];
    let out = vec(float(32), 2);
    check_accept(
        "half sRGB and snorm8",
        &[h],
        out,
        &|fb, x| {
            let table = srgb::declare_color_table(fb.module(), Stage::Kernel, ColorTable::Uniform)?;
            color_write::round(fb, h, x[0], &rules, Some(table))
        },
        &cases,
        &|c| {
            let x = f16_to_f64(c[0][0] as u16) as f32;
            Some(vec![written(rules[0], x), written(rules[1], x)])
        },
        &|g, w| g == w,
    );
}

/// A lane written as computed with its NaN chosen (`Lane::NanAs`) passes every other value through
/// unchanged, a zero's sign, the infinities and the denormals included, and writes every NaN of
/// either sign and any payload as the chosen bits; it runs beside a rounded lane and one written as
/// computed.
#[test]
fn nan_only_lanes_choose_the_nan_alone() {
    use color_write::Lane;
    let f = vec(float(32), 4);
    let minus_one = (-1.0f32).to_bits();
    let lanes = [
        Some(Lane::NanAs(0x7fc0_0000)),
        Some(Lane::NanAs(minus_one)),
        Some(Lane::Rounded(ChannelWrite::Unorm { bits: 16 })),
        None,
    ];
    let nans = [
        0x7fc0_0000u32,
        0xffc0_0000,
        0x7f80_0001,
        0xffff_ffff,
        0x7fff_ffff,
    ];
    let values: Vec<f32> = specials()
        .into_iter()
        .chain(nans.iter().map(|&b| f32::from_bits(b)))
        .chain(around_codes(65535, false, 4099, 1))
        .collect();
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits()); 4]])
        .collect();
    check_accept(
        "NaN-only lanes",
        &[f],
        f,
        &|fb, x| color_write::write(fb, f, x[0], &lanes, (None, None)),
        &cases,
        &|c| {
            let x = f32::from_bits(c[0][0] as u32);
            let nan_as = |bits: u32| u64::from(if x.is_nan() { bits } else { x.to_bits() });
            Some(vec![
                nan_as(0x7fc0_0000),
                nan_as(minus_one),
                written(Some(ChannelWrite::Unorm { bits: 16 }), x),
                u64::from(x.to_bits()),
            ])
        },
        &|g, w| g == w,
    );
    let rounded = color_write::words(&[Some(ChannelWrite::Unorm { bits: 16 }); 4]);
    let chosen = color_write::write_words(&[Some(Lane::NanAs(0x7fc0_0000)); 4]);
    assert!(chosen * 8 < rounded, "{chosen} of {rounded}");
}

/// A lane whose rule the draw selects (`Lane::Selected`) writes what the rule its code in the
/// rule word selects writes: the value as computed for no rule (0) and the device's conversion
/// (1), Metal's 8-bit unorm code (2), and a half-float channel's lane (3) as `half` says (rounded
/// toward zero, as computed, or its NaN chosen). Each lane reads its own code (bits `4i` to
/// `4i + 3`): over four words every lane takes every code, beside lanes of the others, across
/// the unorm grid, the half grid (every third of [`half_values`]) and the specials.
#[test]
fn selected_lanes_write_the_rule_their_code_selects() {
    use crate::pipeline::color_writes::{HALF, HALF_CODE, UNORM8_CODE};
    use color_write::{HalfLane, Lane};
    let f = vec(float(32), 4);
    let mut values = around_codes(255, false, 1, 1);
    values.extend(half_values().into_iter().step_by(3));
    values.extend(specials());
    let words = [0x3210u32, 0x2103, 0x1032, 0x0321];
    let cases: Vec<Case> = values
        .iter()
        .flat_map(|x| {
            words
                .iter()
                .map(|&w| vec![vec![u64::from(x.to_bits()); 4], vec![u64::from(w)]])
        })
        .collect();
    for half in [
        HalfLane::Rounded,
        HalfLane::AsComputed,
        HalfLane::NanAs(0x7fc0_0000),
    ] {
        let lanes = [Some(Lane::Selected { half }); 4];
        check_accept(
            &format!("selected lanes, half {half:?}"),
            &[f, int(32)],
            f,
            &|fb, x| color_write::write(fb, f, x[0], &lanes, (None, Some(x[1]))),
            &cases,
            &|c| {
                let x = f32::from_bits(c[0][0] as u32);
                let word = c[1][0] as u32;
                let lane = |i: u32| match (word >> (4 * i)) & 0xF {
                    UNORM8_CODE => written(Some(ChannelWrite::Unorm { bits: 8 }), x),
                    HALF_CODE => match half {
                        HalfLane::Rounded => written(Some(HALF), x),
                        HalfLane::AsComputed => u64::from(x.to_bits()),
                        HalfLane::NanAs(bits) => {
                            u64::from(if x.is_nan() { bits } else { x.to_bits() })
                        }
                    },
                    _ => u64::from(x.to_bits()),
                };
                Some((0..4).map(lane).collect())
            },
            &|g, w| g == w,
        );
    }
}

/// `color_write::half_bits` writes each half's own bits: every `half` widened (every NaN payload of
/// either sign, the denormals, both zeros and infinities), each value but a NaN through the
/// conversion to 16 bits and each NaN as `nan_codes`' code; a NaN no half widens to (its top ten
/// fraction bits zero) is the quiet NaN of its sign; and `HALF_BITS_WORDS` bounds the emission.
#[test]
fn half_bits_are_each_halfs_bits() {
    use crate::spirv::ir::{ConvOp, Type};
    use crate::spirv::serialize::serialize;
    // `half_bits`' lanes as their codes, widened to 32-bit integers.
    let codes = |fb: &mut FunctionBuilder<'_>, f: Form, x: Value| -> BResult<Value> {
        let h = color_write::half_bits(fb, f, x)?;
        let lanes = |fb: &mut FunctionBuilder<'_>, width: u8| -> BResult<_> {
            let elem = fb.module().ty(Type::Int {
                width,
                signed: false,
            })?;
            fb.module().ty(Type::Vector { elem, len: f.lanes })
        };
        let u16_ = lanes(fb, 16)?;
        let u32_ = lanes(fb, 32)?;
        let narrow = fb.convert(ConvOp::Bitcast, h, u16_)?;
        fb.convert(ConvOp::UConvert, narrow, u32_)
    };
    let h = vec(float(16), 4);
    let out = vec(int(32), 4);
    let cases: Vec<Case> = (0u64..0x10000)
        .step_by(4)
        .map(|b| vec![vec![b, b + 1, b + 2, b + 3]])
        .collect();
    check_accept(
        "every half",
        &[h],
        out,
        &|fb, x| codes(fb, h, x[0]),
        &cases,
        &|c| Some(c[0].clone()),
        &|g, w| g == w,
    );
    let f = vec(float(32), 4);
    let low_nans = [0x7f80_0001u32, 0xff80_0001, 0x7f80_1fff, 0xffc0_0000];
    let cases = vec![vec![low_nans.iter().map(|&b| u64::from(b)).collect()]];
    check_accept(
        "NaNs without a half's payload",
        &[f],
        out,
        &|fb, x| codes(fb, f, x[0]),
        &cases,
        &|_| Some(vec![0x7e00, 0xfe00, 0x7e00, 0xfe00]),
        &|g, w| g == w,
    );
    // The routine's words, with its codes' widening (two conversions) charged to it too.
    let size = |with: bool| -> usize {
        let m = kernel(&[f], out, &|fb, x| match with {
            true => codes(fb, f, x[0]),
            false => {
                let u = crate::lower::arith::ops::ty(fb, out)?;
                fb.convert(ConvOp::Bitcast, x[0], u)
            }
        })
        .expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let words = size(true) - size(false);
    assert!(
        words as u64 <= color_write::HALF_BITS_WORDS,
        "half_bits is {words} words, bound {}",
        color_write::HALF_BITS_WORDS
    );
}

/// A word attachment's encode (`stage_io::word_color::encode`) stores Metal's word for every
/// packed 16-bit layout: each lane rounded to `half`, then its code rounded to even
/// (`WordAttachment::output_word`, which the M3 Max's draws and clears matched on every texel),
/// over the values beside each code and tie of the 1-, 4-, 5- and 6-bit channels, the half grid
/// and the specials, each lane of a case a different value, and a three-lane output's alpha 1;
/// and `RGB9E5Float`'s word of a `float` output, its mantissas truncated, over those and values
/// across its range.
#[test]
fn word_attachments_store_metals_words() {
    use crate::metal::format::WordAttachment;
    use crate::metal::PixelFormat;
    let mut values = Vec::new();
    for m in [1, 15, 31, 63] {
        values.extend(around_codes(m, false, 1, 2));
    }
    values.extend(half_values().into_iter().step_by(5));
    values.extend(specials());
    // `RGB9E5Float`'s range: past its largest, its smallest step, and between.
    values.extend([
        70000.0,
        65408.0,
        65409.0,
        1e-7,
        2f32.powi(-24),
        3e-5,
        1000.3,
        2.5,
        0.375,
    ]);
    let n = values.len();
    let quad = |i: usize| [0, 1, 2, 3].map(|l| values[(i + l * 7919) % n]);
    for pf in [
        PixelFormat::B5G6R5Unorm,
        PixelFormat::A1BGR5Unorm,
        PixelFormat::ABGR4Unorm,
        PixelFormat::BGR5A1Unorm,
        PixelFormat::RGB9E5Float,
    ] {
        let word = WordAttachment::layout(pf).expect("a word layout");
        for lanes in [4u8, 3] {
            let f = vec(float(32), lanes);
            let cases: Vec<Case> = (0..n)
                .map(|i| {
                    let q = quad(i);
                    vec![q[..usize::from(lanes)]
                        .iter()
                        .map(|x| u64::from(x.to_bits()))
                        .collect()]
                })
                .collect();
            check_accept(
                &format!("{pf:?} word of {lanes} lanes"),
                &[f],
                int(32),
                &|fb, x| crate::lower::stage_io::word_color::encode(fb, word, (f, x[0]), None),
                &cases,
                &|c| {
                    let mut rgba = [0.0, 0.0, 0.0, 1.0];
                    for (out, b) in rgba.iter_mut().zip(&c[0]) {
                        *out = f32::from_bits(*b as u32);
                    }
                    Some(vec![u64::from(word.output_word(rgba))])
                },
                &|g, w| g == w,
            );
        }
    }
}
