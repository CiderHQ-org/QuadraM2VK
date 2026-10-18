//! The formats the filter computes, field by field, and the words a level's dispatch reads: the
//! taps of each destination texel and each field's rule ([`super::MipmapFilterChain`]).

use super::reciprocal::Reciprocal;
use crate::metal::PixelFormat;

/// How Metal computes one field of a texel from its taps (the module docs give each rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// A normalized integer of `bits` bits, snorm where `signed`, filtered as the code of `wide`
    /// bits its bits replicate to (`bits` but for `RGB10A2Unorm`'s and `BGR10A2Unorm`'s 2-bit
    /// alpha, filtered as a 10-bit code, and `ABGR4Unorm`'s fields as 8-bit codes and
    /// `A1BGR5Unorm`'s as 10-bit ones): the weighted sum of the wide codes, each term floored to
    /// `2^(wide - 24)` code, rounded half up to the wide code's 12-bit grid, then the code of
    /// `bits` bits nearest it, through `f32` at a half.
    Grid { bits: u32, signed: bool, wide: u32 },
    /// An 8-bit sRGB colour channel, filtered in Metal's twelve-bit linear steps.
    Srgb8,
    /// A 16-bit unorm or snorm (`signed`) code: each term floored to 1/256 code, the sum rounded
    /// half up.
    Fixed16 { signed: bool },
    /// An extended-range colour code: each term floored to `2^-13` code ([`XR_COLOUR_FLOOR`]),
    /// the sum rounded half up to 1/16 code, then to a code through `f32`.
    XrColour,
    /// An extended-range alpha code, clamped to `[384, 894]` first, each term floored to
    /// `2^-15` code ([`XR_ALPHA_FLOOR`]), on a 1/8 grid.
    XrAlpha,
    /// An extended-range sRGB colour code, filtered in Metal's `2^-13` linear steps.
    XrSrgb,
    /// A 32-bit float.
    Float32,
    /// A 16-bit float.
    Float16,
    /// An unsigned float of 5 exponent bits and `mantissa` mantissa bits (`RG11B10Float`).
    Ufloat { mantissa: u32 },
    /// One of `RGB9E5Float`'s three mantissas, `channel` of them, the shared exponent in bits
    /// 27 to 31: the field holds the channel's half-float answer until the texel's word is made.
    Rgb9e5 { channel: u32 },
    /// Bits Metal writes this constant into (`BGR10_XR`'s two padding bits: 3).
    Constant(u32),
}

/// The shift an extended-range colour's term, a code times its weight of `2^24`, is floored by
/// before the sum: to `2^-13` code, as if the filter held a signed 11-bit value `c - 384` times its
/// 16-bit weight to 24 bits (in 2D the same floor, `384 n` being a multiple of `2^15`).
pub(super) const XR_COLOUR_FLOOR: u32 = 11;

/// The shift an extended-range alpha's term is floored by: to `2^-15` code, as if its 9-bit value
/// `c - 384` (`[0, 510]`) times its weight were held to 24 bits.
pub(super) const XR_ALPHA_FLOOR: u32 = 9;

/// The shift an extended-range sRGB colour's term, its signed linear step (of `2^-13`) times its
/// weight of `2^24`, is floored by before the sum: to `2^-9` step (every one of
/// 167,136 values of five chains of the M3 Max's under macOS 27.0.1, each level from Metal's level
/// above).
pub(super) const XR_SRGB_FLOOR: u32 = 15;

/// One field of a texel: `width` bits at bit `offset` of the texel's little-endian bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Field {
    pub(super) offset: u32,
    pub(super) width: u32,
    pub(super) rule: Rule,
}

/// A format the filter computes: its texel's bytes and fields, in the bytes of the Vulkan
/// format the runtime holds it in (the red/blue twins swap only fields of one rule).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) bytes: u32,
    pub(super) fields: Vec<Field>,
    /// Whether the Apple M3 Max carries the format's values through a half float
    /// (`ABGR4Unorm`, `A1BGR5Unorm` and the extended-range formats: the module docs).
    pub(super) half: bool,
}

impl Layout {
    /// Whether the M3 Max carries field `f` through a half float.
    pub(super) fn carries_half(&self, f: &Field) -> bool {
        self.half && !matches!(f.rule, Rule::Constant(_))
    }

    /// Whether the texel's word is `RGB9E5Float`'s, made from its three fields' answers.
    pub(super) fn rgb9e5(&self) -> bool {
        self.fields
            .iter()
            .any(|f| matches!(f.rule, Rule::Rgb9e5 { .. }))
    }
}

fn at(offset: u32, width: u32, rule: Rule) -> Field {
    Field {
        offset,
        width,
        rule,
    }
}

/// `n` fields of `width` bits and one rule, from bit 0 up.
fn uniform(n: u32, width: u32, rule: Rule) -> Vec<Field> {
    (0..n).map(|i| at(i * width, width, rule)).collect()
}

/// The layout of `pf` where the filter computes Metal's mipmaps of it: every mipmappable colour
/// format but `B5G6R5Unorm` and `BGR5A1Unorm`, whose 5-bit fields miss the rule at a few grid
/// ties, and the two-plane extended-range formats.
pub(super) fn layout(pf: PixelFormat) -> Option<Layout> {
    use PixelFormat as P;
    let g8 = Rule::Grid {
        bits: 8,
        signed: false,
        wide: 8,
    };
    let s8 = Rule::Grid {
        bits: 8,
        signed: true,
        wide: 8,
    };
    // A code of `bits` bits filtered as the code of `wide` bits its bits replicate to.
    let widened = |bits, wide| Rule::Grid {
        bits,
        signed: false,
        wide,
    };
    let g = |bits| widened(bits, bits);
    let (bytes, fields) = match pf {
        P::R8Unorm | P::A8Unorm => (1, uniform(1, 8, g8)),
        P::RG8Unorm => (2, uniform(2, 8, g8)),
        P::RGBA8Unorm | P::BGRA8Unorm | P::ARGB8Unorm => (4, uniform(4, 8, g8)),
        P::R8Snorm => (1, uniform(1, 8, s8)),
        P::RG8Snorm => (2, uniform(2, 8, s8)),
        P::RGBA8Snorm => (4, uniform(4, 8, s8)),
        P::R8UnormSRGB => (1, uniform(1, 8, Rule::Srgb8)),
        P::RG8UnormSRGB => (2, uniform(2, 8, Rule::Srgb8)),
        P::RGBA8UnormSRGB | P::BGRA8UnormSRGB => {
            let mut f = uniform(3, 8, Rule::Srgb8);
            f.push(at(24, 8, g8));
            (4, f)
        }
        // The 2-bit alpha filtered as the 10-bit code its bits replicate to.
        P::RGB10A2Unorm | P::BGR10A2Unorm => {
            let mut f = uniform(3, 10, g(10));
            let alpha = Rule::Grid {
                bits: 2,
                signed: false,
                wide: 10,
            };
            f.push(at(30, 2, alpha));
            (4, f)
        }
        // Alpha in bit 0, then three 5-bit fields, in both its Vulkan formats; each filtered as a
        // 10-bit code.
        P::A1BGR5Unorm => (
            2,
            vec![
                at(0, 1, widened(1, 10)),
                at(1, 5, widened(5, 10)),
                at(6, 5, widened(5, 10)),
                at(11, 5, widened(5, 10)),
            ],
        ),
        // Each 4-bit field filtered as an 8-bit code.
        P::ABGR4Unorm => (2, uniform(4, 4, widened(4, 8))),
        P::R16Unorm => (2, uniform(1, 16, Rule::Fixed16 { signed: false })),
        P::RG16Unorm => (4, uniform(2, 16, Rule::Fixed16 { signed: false })),
        P::RGBA16Unorm => (8, uniform(4, 16, Rule::Fixed16 { signed: false })),
        P::R16Snorm => (2, uniform(1, 16, Rule::Fixed16 { signed: true })),
        P::RG16Snorm => (4, uniform(2, 16, Rule::Fixed16 { signed: true })),
        P::RGBA16Snorm => (8, uniform(4, 16, Rule::Fixed16 { signed: true })),
        P::R16Float => (2, uniform(1, 16, Rule::Float16)),
        P::RG16Float => (4, uniform(2, 16, Rule::Float16)),
        P::RGBA16Float => (8, uniform(4, 16, Rule::Float16)),
        P::R32Float => (4, uniform(1, 32, Rule::Float32)),
        P::RG32Float => (8, uniform(2, 32, Rule::Float32)),
        P::RGBA32Float => (16, uniform(4, 32, Rule::Float32)),
        P::RG11B10Float => (
            4,
            vec![
                at(0, 11, Rule::Ufloat { mantissa: 6 }),
                at(11, 11, Rule::Ufloat { mantissa: 6 }),
                at(22, 10, Rule::Ufloat { mantissa: 5 }),
            ],
        ),
        P::RGB9E5Float => (
            4,
            (0..3)
                .map(|channel| at(9 * channel, 9, Rule::Rgb9e5 { channel }))
                .collect(),
        ),
        // Blue, green and red codes from bit 0, and the padding Metal sets.
        P::BGR10XR | P::BGR10XRSRGB => {
            let rule = if pf == P::BGR10XR {
                Rule::XrColour
            } else {
                Rule::XrSrgb
            };
            let mut f = uniform(3, 10, rule);
            f.push(at(30, 2, Rule::Constant(3)));
            (4, f)
        }
        // Four 16-bit lanes, blue first, each code in the lane's top ten bits (its low six 0).
        P::BGRA10XR | P::BGRA10XRSRGB => {
            let rule = if pf == P::BGRA10XR {
                Rule::XrColour
            } else {
                Rule::XrSrgb
            };
            (
                8,
                vec![
                    at(6, 10, rule),
                    at(22, 10, rule),
                    at(38, 10, rule),
                    at(54, 10, Rule::XrAlpha),
                ],
            )
        }
        _ => return None,
    };
    let half = matches!(
        pf,
        P::ABGR4Unorm
            | P::A1BGR5Unorm
            | P::BGR10XR
            | P::BGR10XRSRGB
            | P::BGRA10XR
            | P::BGRA10XRSRGB
    );
    Some(Layout {
        bytes,
        fields,
        half,
    })
}

/// The two taps along one axis of destination texel `i` of `small` texels from `big`: the first
/// source texel `x0` and the weight `q` (of 256) of the next, `min(x0 + 1, big - 1)`, the first
/// weighing `256 - q`. Metal computes the centre's coordinate in `f32`, `u = (i + 0.5) * (1 /
/// small)` and `c = u * big - 0.5`, each operation rounded to nearest even from the GPU's own
/// reciprocal of `small`, and rounds the fraction of `c` half up to 8 bits (173,505 taps of widths,
/// heights and depths from 257 to 16383 on an M4: every level of R32Float ramps up to 16384 on an
/// M3 Max). An axis of one texel that stays one is one tap.
pub(super) fn axis_tap(i: u32, big: u32, small: u32, reciprocal: Reciprocal) -> (u32, u32) {
    if big == small {
        return (0, 0);
    }
    // Every operand is exact in f32 (below 2^24), and each operation is IEEE's, rounded to
    // nearest even, as on the GPU; the reciprocal is the GPU's own.
    let r = reciprocal.of(small);
    let u = (i as f32 + 0.5) * r;
    let c = u * big as f32 - 0.5;
    let x0 = c.floor();
    // `c - x0` is exact, and so is its product with 256 plus a half in f64.
    let q = (f64::from(c - x0) * 256.0 + 0.5).floor() as u32;
    let x0 = x0 as u32;
    if q == 256 {
        (x0 + 1, 0)
    } else {
        (x0, q)
    }
}

/// `p / q` (`q > 0`, `p != 0`) rounded to `f32` to nearest even, as `(negative, m, e)`: the value
/// `m * 2^e`, `2^23 <= m < 2^24`. The quotients here are far from the subnormals.
fn f32_ratio(p: i64, q: u64) -> (bool, u128, i32) {
    let negative = p < 0;
    let a = u128::from(p.unsigned_abs());
    let b = u128::from(q);
    let bits = |v: u128| 128 - v.leading_zeros() as i32;
    let mut e = bits(a) - bits(b) - 24;
    loop {
        let (num, den) = if e <= 0 { (a << -e, b) } else { (a, b << e) };
        let m = num / den;
        if m >= 1 << 24 {
            e += 1;
            continue;
        }
        if m < 1 << 23 {
            e -= 1;
            continue;
        }
        let rem = num - m * den;
        let up = 2 * rem > den || (2 * rem == den && m & 1 == 1);
        let m = m + u128::from(up);
        return if m == 1 << 24 {
            (negative, 1 << 23, e + 1)
        } else {
            (negative, m, e)
        };
    }
}

/// `lhs = ±m * k * 2^e + c` against `rhs * 1` (both scaled to integers): whether a value `x +
/// 1/2` that Metal rounds through `f32` lands above the half (`Greater`), on it, or below.
fn compare(negative: bool, m: u128, e: i32, k: i128, c: i128, rhs: i128) -> std::cmp::Ordering {
    let m = m as i128 * k;
    let m = if negative { -m } else { m };
    if e >= 0 { (m << e) + c } else { m + (c << -e) }.cmp(&if e >= 0 { rhs } else { rhs << -e })
}

/// Whether code `x + 1/2`, an exact half between two codes of a normalized format of `den`
/// steps, rounds up: Metal takes the half's `f32` value `RNE((x + 1/2) / den)` and rounds its
/// product with `den` to nearest even.
pub(super) fn grid_tie_up(x: i32, den: u32) -> bool {
    let (negative, m, e) = f32_ratio(2 * i64::from(x) + 1, 2 * u64::from(den));
    // t * den * 2 against 2x + 1.
    match compare(
        negative,
        m,
        e + 1,
        i128::from(den),
        0,
        2 * i128::from(x) + 1,
    ) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => x & 1 == 1,
        std::cmp::Ordering::Less => false,
    }
}

/// Whether extended-range code `x + 1/2` rounds up: Metal takes the value `RNE((x + 1/2 - 384) /
/// 510)` and rounds its code `v * 510 + 384` to nearest even.
pub(super) fn xr_tie_up(x: i32) -> bool {
    let (negative, m, e) = f32_ratio(2 * i64::from(x) + 1 - 768, 1020);
    // v * 1020 + 768 against 2x + 1.
    match compare(negative, m, e, 1020, 768, 2 * i128::from(x) + 1) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => x & 1 == 1,
        std::cmp::Ordering::Less => false,
    }
}

/// A field's tie table: the codes from `base` on, `len` of them, one bit each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ties {
    /// A normalized format of `den` steps (signed: codes `-den` to `den - 1`).
    Grid { den: u32, signed: bool },
    /// The extended-range codes 0 to 1022.
    Xr,
}

impl Ties {
    /// The first code and the number of codes.
    pub(super) fn span(self) -> (i32, u32) {
        match self {
            Ties::Grid { den, signed: true } => (-(den as i32), 2 * den),
            Ties::Grid { den, signed: false } => (0, den),
            Ties::Xr => (0, 1023),
        }
    }

    /// Whether code `x + 1/2` rounds up.
    pub(super) fn up(self, x: i32) -> bool {
        match self {
            Ties::Grid { den, .. } => grid_tie_up(x, den),
            Ties::Xr => xr_tie_up(x),
        }
    }

    /// The table's words: bit `j % 32` of word `j / 32` for code `base + j`.
    pub(super) fn words(self) -> Vec<u32> {
        let (base, len) = self.span();
        let mut words = vec![0u32; len.div_ceil(32) as usize];
        for j in 0..len {
            if self.up(base + j as i32) {
                words[(j / 32) as usize] |= 1 << (j % 32);
            }
        }
        words
    }
}

/// The denominator of a normalized integer of `bits` bits: its largest code.
pub(super) fn den(bits: u32, signed: bool) -> u32 {
    if signed {
        (1 << (bits - 1)) - 1
    } else {
        (1 << bits) - 1
    }
}

/// What a code of `bits` bits is multiplied by to be the code of `wide` bits its bits replicate
/// to, `(2^wide - 1) / (2^bits - 1)` (an unsigned field, `wide` a multiple of `bits`; 1 where
/// `wide` is `bits`).
pub(super) fn widen(bits: u32, signed: bool, wide: u32) -> u32 {
    let (d, w) = (den(bits, signed), den(wide, signed));
    assert!(
        w % d == 0 && (wide == bits || !signed),
        "an unsigned code replicated"
    );
    w / d
}

/// The tie table a rule rounds through, if any (a widened field's: its own code's).
pub(super) fn ties(rule: Rule) -> Option<Ties> {
    match rule {
        Rule::Grid { bits, signed, .. } => Some(Ties::Grid {
            den: den(bits, signed),
            signed,
        }),
        Rule::XrColour | Rule::XrAlpha => Some(Ties::Xr),
        _ => None,
    }
}

// ---- The words a dispatch reads ([`super::MipmapFilterModule::PARAMS_BINDING`]) ----

/// Words of the level's header.
pub(super) const HEADER_WORDS: u32 = 11;

/// Words of one field's record, which the kernel is built with (`kernel::Spec`).
pub(super) const RECORD_WORDS: u32 = 41;
/// Words a texel keeps for the next level (one per field, at most four; `RGB9E5Float`'s three
/// channels).
pub(super) const KEPT_WORDS: u32 = 4;

/// Header words, in order: the destination texels of the level (every layer), the invocations
/// of a dispatch row, the level's width, height and depth, the level above's, and where the x,
/// y and z tap tables start (words). The format's tie tables follow the header, at the words
/// [`tie_tables`] gives, then the tap tables: a sampler level's `x0 | q << 16`, a tail level's
/// `x0 | w << 16` (`w` of 2048).
pub(super) mod header {
    pub const TEXELS: u32 = 0;
    pub const ROW: u32 = 1;
    pub const EXTENT: u32 = 2;
    pub const ABOVE: u32 = 5;
    pub const TAPS: u32 = 8;
}

/// The tie tables of `layout`'s fields, from word [`HEADER_WORDS`] of a level's words: the
/// tables' words, and the word each field's starts at (0 for a field without one; fields of one
/// table share it). The same for every level of the format, so the kernel is built with them.
pub(super) fn tie_tables(layout: &Layout) -> Option<(Vec<u32>, Vec<u32>)> {
    let mut words = Vec::new();
    let mut tables: Vec<(Ties, u32)> = Vec::new();
    let mut at = Vec::with_capacity(layout.fields.len());
    for f in &layout.fields {
        at.push(match ties(f.rule) {
            Some(t) => match tables.iter().find(|(u, _)| *u == t) {
                Some((_, start)) => *start,
                None => {
                    let start = HEADER_WORDS.checked_add(u32::try_from(words.len()).ok()?)?;
                    words.extend(t.words());
                    tables.push((t, start));
                    start
                }
            },
            None => 0,
        });
    }
    Some((words, at))
}

/// A field's record words: its rule, as the kernel is built with it (`kernel::Spec`).
pub(super) mod record {
    /// The field's mode ([`super::mode`]).
    pub const MODE: u32 = 0;
    /// Its bit offset in the texel, and its mask (`2^width - 1`).
    pub const OFFSET: u32 = 1;
    pub const MASK: u32 = 2;
    /// An integer field: 32 minus its width where it is signed (a sign extension's shift),
    /// else 0; then the clamp of its codes before the sum.
    pub const EXTEND: u32 = 3;
    pub const IN_LO: u32 = 4;
    pub const IN_HI: u32 = 5;
    /// An integer field's decode ([`super::decode`]); the shift `PER_TERM` each term (a code
    /// times its weight of `2^24`) is floored by first, `(v n) >> PER_TERM` (16: to 1/256 code; a
    /// grid's: its code's width; an extended-range code's: 11 or 9; 0: the sum exact); and the
    /// rounding of the sum of the floored terms, `(S + ADD) >> SHIFT`.
    pub const DECODE: u32 = 6;
    pub const PER_TERM: u32 = 7;
    pub const ADD: u32 = 8;
    pub const SHIFT: u32 = 9;
    /// The encode ([`super::encode`]); a grid's fraction bits, their mask, and the half.
    pub const ENCODE: u32 = 10;
    pub const K: u32 = 11;
    pub const FRACTION: u32 = 12;
    pub const HALF: u32 = 13;
    /// The clamp of the result (a widened field's wide code's: times [`WIDEN`]), the tie table's
    /// word offset, first code and length.
    pub const LO: u32 = 14;
    pub const HI: u32 = 15;
    pub const TIE_AT: u32 = 16;
    pub const TIE_BASE: u32 = 17;
    pub const TIE_LEN: u32 = 18;
    /// A constant field's value.
    pub const CONSTANT: u32 = 19;
    /// A float field: its sign bit's mask, the exponent's bit offset in the texel and mask, the
    /// mantissa's mask and hidden bit, the exponent bias plus the mantissa bits, the exponent a
    /// subnormal takes (1, or 0 for `RGB9E5Float`), 1 where subnormal inputs read as zero, the
    /// exponent of infinities and NaNs (`u32::MAX` for none).
    pub const SIGN: u32 = 20;
    pub const EXP_OFFSET: u32 = 21;
    pub const EXP_MASK: u32 = 22;
    pub const MANTISSA: u32 = 23;
    pub const HIDDEN: u32 = 24;
    pub const BIAS: u32 = 25;
    pub const SUBNORMAL_EXP: u32 = 26;
    pub const FLUSH_IN: u32 = 27;
    pub const SPECIAL: u32 = 28;
    /// The accumulator and result: the result's fraction bits plus Metal's guard bits, in
    /// the low and high halves; the least normal exponent; 1 where a sum below it reads as +0.
    pub const PRECISION: u32 = 29;
    pub const EMIN: u32 = 30;
    pub const FLUSH_OUT: u32 = 31;
    /// The field's rule ([`super::rule`]); the denominator of its normalized codes (an
    /// extended-range code's 510, an sRGB step's 4095); 1 where the M3 Max carries it through a
    /// half; the significand bits and least normal exponent of the M3 Max's arithmetic for it;
    /// the first level's kept value `(Q - KEPT_BIAS) / KEPT_UNIT` of the grid sum `Q`; an
    /// unsigned float's mantissa bits.
    pub const RULE: u32 = 32;
    pub const DEN: u32 = 33;
    pub const CARRIED: u32 = 34;
    pub const BITS: u32 = 35;
    pub const LEAST: u32 = 36;
    pub const KEPT_UNIT: u32 = 37;
    pub const KEPT_BIAS: u32 = 38;
    pub const MB: u32 = 39;
    /// A normalized field filtered as a wider code: what each code is multiplied by
    /// ([`super::widen`]; 1 for the rest), its grid sum the wide code's, rounded to the nearest
    /// code of the field.
    pub const WIDEN: u32 = 40;
}

/// A field's rule, for the tail and the kept values.
pub(super) mod rule {
    pub const GRID: u32 = 1;
    pub const SRGB8: u32 = 2;
    pub const FIXED16: u32 = 3;
    pub const XR_COLOUR: u32 = 4;
    pub const XR_ALPHA: u32 = 5;
    pub const XR_SRGB: u32 = 6;
    pub const FLOAT32: u32 = 7;
    pub const FLOAT16: u32 = 8;
    pub const UFLOAT: u32 = 9;
    pub const RGB9E5: u32 = 10;
    pub const CONSTANT: u32 = 11;
}

/// A field's mode.
pub(super) mod mode {
    pub const INTEGER: u32 = 1;
    pub const FLOAT32: u32 = 2;
    pub const HALF: u32 = 3;
    pub const CONSTANT: u32 = 4;
}

/// An integer field's decode.
pub(super) mod decode {
    pub const CODE: u32 = 0;
    pub const SRGB: u32 = 1;
    pub const XR_SRGB: u32 = 2;
}

/// An integer field's encode; a half field's (in [`super::record::ENCODE`] too).
pub(super) mod encode {
    pub const GRID: u32 = 0;
    pub const SRGB: u32 = 1;
    pub const CLAMP: u32 = 2;
    pub const XR_SRGB: u32 = 3;
    /// A half field: the half-float bits, or shifted right by `K` (`RG11B10Float`), or kept
    /// for the texel's `RGB9E5Float` word.
    pub const HALF_BITS: u32 = 0;
    pub const HALF_SHIFTED: u32 = 1;
    pub const HALF_RGB9E5: u32 = 2;
}

/// The record of field `f` of `layout`, its tie table at word `tie_at` where it has one.
pub(super) fn record(layout: &Layout, f: &Field, tie_at: u32) -> [u32; RECORD_WORDS as usize] {
    use record as r;
    let mut w = [0u32; RECORD_WORDS as usize];
    w[r::OFFSET as usize] = f.offset;
    w[r::MASK as usize] = if f.width == 32 {
        u32::MAX
    } else {
        (1 << f.width) - 1
    };
    let integer = |w: &mut [u32; RECORD_WORDS as usize],
                   signed: bool,
                   lo: i32,
                   hi: i32,
                   decode: u32,
                   per_term: u32,
                   encode: u32| {
        w[r::MODE as usize] = mode::INTEGER;
        w[r::EXTEND as usize] = if signed { 32 - f.width } else { 0 };
        w[r::IN_LO as usize] = lo as u32;
        w[r::IN_HI as usize] = hi as u32;
        w[r::DECODE as usize] = decode;
        w[r::PER_TERM as usize] = per_term;
        w[r::ENCODE as usize] = encode;
        w[r::LO as usize] = lo as u32;
        w[r::HI as usize] = hi as u32;
        if per_term == 16 {
            w[r::ADD as usize] = 128;
            w[r::SHIFT as usize] = 8;
        }
    };
    // The sum's rounding half up to `k` fraction bits of terms floored by `per_term`.
    let grid = |w: &mut [u32; RECORD_WORDS as usize], k: u32| {
        let floor = w[r::PER_TERM as usize];
        w[r::ADD as usize] = 1 << (23 - k - floor);
        w[r::SHIFT as usize] = 24 - k - floor;
        w[r::K as usize] = k;
        w[r::FRACTION as usize] = (1 << k) - 1;
        w[r::HALF as usize] = 1 << (k - 1);
        if let Some(t) = ties(f.rule) {
            let (base, len) = t.span();
            w[r::TIE_AT as usize] = tie_at;
            w[r::TIE_BASE as usize] = base as u32;
            w[r::TIE_LEN as usize] = len;
        }
    };
    let float = |w: &mut [u32; RECORD_WORDS as usize],
                 m: u32,
                 sign: u32,
                 exp: (u32, u32),
                 mantissa: u32,
                 hidden: u32,
                 bias: u32,
                 subnormal: u32,
                 special: u32| {
        w[r::MODE as usize] = m;
        w[r::SIGN as usize] = sign;
        w[r::EXP_OFFSET as usize] = exp.0;
        w[r::EXP_MASK as usize] = exp.1;
        w[r::MANTISSA as usize] = mantissa;
        w[r::HIDDEN as usize] = hidden;
        w[r::BIAS as usize] = bias;
        w[r::SUBNORMAL_EXP as usize] = subnormal;
        w[r::SPECIAL as usize] = special;
    };
    match f.rule {
        Rule::Grid { bits, signed, wide } => {
            let d = den(bits, signed) as i32;
            let lo = if signed { -d } else { 0 };
            integer(&mut w, signed, lo, d, decode::CODE, wide, encode::GRID);
            grid(&mut w, 12 - wide);
            w[r::WIDEN as usize] = widen(bits, signed, wide);
        }
        Rule::XrColour => {
            integer(
                &mut w,
                false,
                0,
                1023,
                decode::CODE,
                XR_COLOUR_FLOOR,
                encode::GRID,
            );
            grid(&mut w, 4);
        }
        Rule::XrAlpha => {
            integer(
                &mut w,
                false,
                384,
                894,
                decode::CODE,
                XR_ALPHA_FLOOR,
                encode::GRID,
            );
            grid(&mut w, 3);
        }
        Rule::Srgb8 => {
            integer(&mut w, false, 0, 4095, decode::SRGB, 0, encode::SRGB);
            w[r::ADD as usize] = 1 << 23;
            w[r::SHIFT as usize] = 24;
        }
        Rule::Fixed16 { signed } => {
            let d = den(16, signed) as i32;
            let lo = if signed { -d } else { 0 };
            integer(&mut w, signed, lo, d, decode::CODE, 16, encode::CLAMP);
        }
        Rule::XrSrgb => {
            integer(
                &mut w,
                false,
                0,
                1023,
                decode::XR_SRGB,
                XR_SRGB_FLOOR,
                encode::XR_SRGB,
            );
            w[r::ADD as usize] = 1 << (23 - XR_SRGB_FLOOR);
            w[r::SHIFT as usize] = 24 - XR_SRGB_FLOOR;
        }
        Rule::Float32 => {
            float(
                &mut w,
                mode::FLOAT32,
                1 << 31,
                (f.offset + 23, 0xff),
                0x7f_ffff,
                1 << 23,
                150,
                1,
                0xff,
            );
            w[r::FLUSH_IN as usize] = 1;
            w[r::PRECISION as usize] = 23 | (15 << 16);
            w[r::EMIN as usize] = -126i32 as u32;
            w[r::FLUSH_OUT as usize] = 1;
        }
        Rule::Float16 => {
            float(
                &mut w,
                mode::HALF,
                1 << 15,
                (f.offset + 10, 0x1f),
                0x3ff,
                1 << 10,
                25,
                1,
                0x1f,
            );
            w[r::ENCODE as usize] = encode::HALF_BITS;
        }
        Rule::Ufloat { mantissa } => {
            float(
                &mut w,
                mode::HALF,
                0,
                (f.offset + mantissa, 0x1f),
                (1 << mantissa) - 1,
                1 << mantissa,
                15 + mantissa,
                1,
                0x1f,
            );
            w[r::ENCODE as usize] = encode::HALF_SHIFTED;
            w[r::K as usize] = 10 - mantissa;
        }
        Rule::Rgb9e5 { .. } => {
            float(&mut w, mode::HALF, 0, (27, 0x1f), 0x1ff, 0, 24, 0, u32::MAX);
            w[r::ENCODE as usize] = encode::HALF_RGB9E5;
        }
        Rule::Constant(v) => {
            w[r::MODE as usize] = mode::CONSTANT;
            w[r::CONSTANT as usize] = v;
        }
    }
    if w[r::MODE as usize] == mode::HALF {
        w[r::PRECISION as usize] = 10 | (7 << 16);
        w[r::EMIN as usize] = -14i32 as u32;
    }
    let (id, den_, unit, bias) = match f.rule {
        Rule::Grid { bits, signed, wide } => {
            let d = den(bits, signed);
            (rule::GRID, d, den(wide, signed) << (12 - wide), 0)
        }
        Rule::Srgb8 => (rule::SRGB8, 4095, 4095, 0),
        Rule::Fixed16 { signed } => {
            let d = den(16, signed);
            (rule::FIXED16, d, d, 0)
        }
        Rule::XrColour => (rule::XR_COLOUR, 510, 510 * 16, 384 * 16),
        Rule::XrAlpha => (rule::XR_ALPHA, 510, 510 * 8, 384 * 8),
        Rule::XrSrgb => (rule::XR_SRGB, 8192, 8192, 0),
        Rule::Float32 => (rule::FLOAT32, 0, 0, 0),
        Rule::Float16 => (rule::FLOAT16, 0, 0, 0),
        Rule::Ufloat { mantissa } => {
            w[r::MB as usize] = mantissa;
            (rule::UFLOAT, 0, 0, 0)
        }
        Rule::Rgb9e5 { .. } => (rule::RGB9E5, 0, 0, 0),
        Rule::Constant(_) => (rule::CONSTANT, 0, 0, 0),
    };
    w[r::RULE as usize] = id;
    w[r::DEN as usize] = den_;
    w[r::KEPT_UNIT as usize] = unit;
    w[r::KEPT_BIAS as usize] = bias;
    w[r::CARRIED as usize] = u32::from(layout.carries_half(f));
    let (bits, least) = match super::tail::precision(f, layout.half) {
        super::tail::Precision::F32 => (24, -126i32),
        super::tail::Precision::F16 => (11, -14),
    };
    w[r::BITS as usize] = bits;
    w[r::LEAST as usize] = least as u32;
    w
}
