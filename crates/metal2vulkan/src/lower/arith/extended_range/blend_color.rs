//! The blend color as Metal's blend into an extended-range attachment reads it
//! (`super::blend`), which the binder writes into a draw's push bytes (`PushField::BlendColor`),
//! and the source ranges the blend clamps to. Nothing here builds SPIR-V: the runtime reaches it.

/// The range Metal clamps a blend's `half` source color to before blending into an attachment
/// of a linear form, `BGR10_XR_sRGB` or the two-plane `_sRGB` form: the linear values of codes 0
/// and 1022, `-384 / 510` and `638 / 510`, rounded to `half`, nearest even (every
/// one of the probe's 8,192 sources a NaN or beyond the range lands there, for each format):
/// `-0.7529296875` and `1.2509765625`. A linear form's blend clamps the blend color's color
/// channels to it too ([`constant_words`]).
pub(crate) const SOURCE_RANGE: (f32, f32) = (-1542.0 / 2048.0, 1281.0 / 1024.0);

/// [`SOURCE_RANGE`] for `BGRA10_XR_sRGB`: the tile values of codes 0 and 1022
/// (`super::tile_value`, the table's steps 384 and 638 over 8192 rounded to `half`, ties away from
/// zero): `-0.52734375` and `1.6689453125`. A blend into it clamps the blend color's color channels
/// to it too (the lane table without fast math: every configuration reading the blend color into
/// `BGRA10_XR_sRGB`).
pub(crate) const SRGB_SOURCE_RANGE: (f32, f32) = (-1080.0 / 2048.0, 1709.0 / 1024.0);

/// The smallest range holding [`SOURCE_RANGE`] and [`SRGB_SOURCE_RANGE`], which the binder clamps
/// the blend color's color channels to ([`constant_words`]): `-0.7529296875` and `1.6689453125`.
pub(crate) const COLOR_HULL: (f32, f32) = (SOURCE_RANGE.0, SRGB_SOURCE_RANGE.1);

/// The least `f32` a color channel of the blend color reads, under `BlendColor` into
/// `BGRA10_XR_sRGB`, as [`SRGB_DIRECT_TOP`] rather than its `half`: `1.6689399480819702`
/// (`0x3FD5_9FD3`, 45 `f32` steps below [`SRGB_SOURCE_RANGE`]'s top). Below it the channel reads
/// its `half` clamped to the range, as its one minus does at every value (plain
/// Metal on an Apple M4 under macOS 27.0.1: every `f32` from `1.6689000129699707` to
/// `1.6689451932907104`, each read under `BlendColor` and under `OneMinusBlendColor` without fast
/// math; our probe about both ends of the range, the top's `half`s, `2.0`, `1e30`, `65504`,
/// the infinities and a NaN, which reads the low end).
pub(crate) const SRGB_DIRECT_BOUND: f32 = f32::from_bits(0x3FD5_9FD3);

/// What `BlendColor` reads into `BGRA10_XR_sRGB` for a color channel at or past
/// [`SRGB_DIRECT_BOUND`]: `1.66796875`, the `half` below [`SRGB_SOURCE_RANGE`]'s top.
pub(crate) const SRGB_DIRECT_TOP: f32 = 1708.0 / 1024.0;

/// The first row's color channel at or past [`SRGB_DIRECT_BOUND`]: `1.669921875`, the `half` above
/// [`COLOR_HULL`]'s top, which every form's clamp brings back to its range's top and
/// `BGRA10_XR_sRGB`'s `BlendColor` reads as [`SRGB_DIRECT_TOP`] (`super::blend::blend`).
pub(crate) const PAST_DIRECT_BOUND: f32 = 1710.0 / 1024.0;

/// `x` rounded to `half`, nearest even, as an `f32` (`|x|` below 2).
fn half_even(x: f64) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let ulp = 2f64.powi(e.max(-14) - 10);
    ((x / ulp).round_ties_even() * ulp) as f32
}

/// The two `uvec4`s of `PushField::BlendColor` for the encoder's blend color `c`
/// (`setBlendColorRed:green:blue:alpha:`), each lane an `f32`'s bits. The first: the color
/// clamped to [`COLOR_HULL`] (a NaN to its low end) and alpha to `[0, 1]` (a NaN to 0), each
/// rounded to `half`, nearest even, which a blend clamps again to its form's source range
/// (`super::blend::blend`): the ranges' ends are `half`s, so that is the color clamped to the
/// form's range and then rounded, as Metal reads `BlendColor` and `OneMinusBlendColor`; a color
/// channel at or past [`SRGB_DIRECT_BOUND`] holds [`PAST_DIRECT_BOUND`] instead, which every form
/// clamps to its top but `BGRA10_XR_sRGB`'s `BlendColor` reads as [`SRGB_DIRECT_TOP`]. The second:
/// one minus each, `h(1 - x)` of the color clamped to [`SOURCE_RANGE`] and `h(1 - h(a))` of alpha,
/// as Metal reads `OneMinusBlendColor` and `OneMinusBlendAlpha` under fast math (a NaN of either
/// sign, the infinities and values beyond `half`'s range), which only a blend into a form of that
/// range reads (no blend into `BGRA10_XR_sRGB` reading the blend color is built under fast math,
/// and without it the blend rounds one minus the first row's value itself). `1 - x` is exact in
/// `f64` but for `|x|` below `2^-29`, where its rounding stays clear of a `half` midpoint, so each
/// lane rounds once.
pub fn constant_words(c: [f32; 4]) -> [u32; 8] {
    let clamp = |x: f32, (lo, hi): (f32, f32)| match x.is_nan() {
        true => f64::from(lo),
        false => f64::from(x).clamp(f64::from(lo), f64::from(hi)),
    };
    // A color channel at or past `BGRA10_XR_sRGB`'s direct bound is marked past the hull's top.
    let color = |x: f32| match x >= SRGB_DIRECT_BOUND {
        true => f64::from(PAST_DIRECT_BOUND),
        false => clamp(x, COLOR_HULL),
    };
    let [r, g, b, a] = c;
    let wide = [color(r), color(g), color(b), clamp(a, (0.0, 1.0))];
    let h: [f32; 4] = wide.map(half_even);
    let minus = [
        1.0 - clamp(r, SOURCE_RANGE),
        1.0 - clamp(g, SOURCE_RANGE),
        1.0 - clamp(b, SOURCE_RANGE),
        1.0 - f64::from(h[3]),
    ];
    let [r, g, b, a] = h.map(f32::to_bits);
    let [mr, mg, mb, ma] = minus.map(|m| half_even(m).to_bits());
    [r, g, b, a, mr, mg, mb, ma]
}
