//! IEEE-754 binary16 encoding of a host `f32`.
//!
//! Two layers mint `half` constants — the native emitter, for AIR constants typed `half`, and the
//! passes layer, for the 0/1/scale operands its intrinsic lowerings synthesize. Both were
//! encoding the value themselves, and the two encoders did not agree: the passes copy rounded
//! half-away-from-zero and combined the rounded significand into the encoding with `|` instead of
//! `+`, so a value whose rounding carries out of the significand kept the exponent it started
//! with. `1.999755859375` encoded as `1.0` rather than `2.0`. Only `0.0`, `1.0` and one tanh scale
//! reached it, all of which round without a carry, so nothing observed the difference — the next
//! constant would have.
//!
//! One encoder, exercised over every representable `half`.

/// The IEEE-754 binary16 bit pattern nearest to `value`, rounding ties to even.
///
/// Out-of-range magnitudes become the signed infinity, matching what a `float` -> `half`
/// conversion does on the device. A NaN keeps its sign and the high bits of its payload, and is
/// forced non-zero so it cannot silently become an infinity.
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let significand = bits & 0x007f_ffff;

    if exponent == 0xff {
        if significand == 0 {
            return sign | 0x7c00;
        }
        let payload = (significand >> 13) as u16;
        return sign | 0x7c00 | payload | u16::from(payload == 0);
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        // Subnormal: shift the implicit leading one back in and round at the binade the encoding
        // can express. A carry out of the significand lands on the smallest normal, which is the
        // correct next value up, so the same `|` that is wrong for a normal is right here.
        let mantissa = significand | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        return sign | round_shift_right_ties_even(mantissa, shift) as u16;
    }

    // `+`, not `|`: rounding can carry out of the ten significand bits, and the carry belongs in
    // the exponent. Clamping at `0x7c00` turns a carry out of the top exponent into infinity.
    let rounded = round_shift_right_ties_even(significand, 13);
    let encoded = ((half_exponent as u32) << 10) + rounded;
    sign | encoded.min(0x7c00) as u16
}

/// The exact `f32` value of a binary16 bit pattern.
///
/// Every finite `half` is exactly representable as an `f32`, so this is a widening with no
/// rounding of its own; the infinities and NaNs widen to their `f32` counterparts.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let negative = bits & 0x8000 != 0;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let significand = (bits & 0x03ff) as u32;
    let magnitude = if exponent == 0 {
        // Subnormal: significand * 2^-24, exact in `f32`.
        (significand as f32) * (2.0f32).powi(-24)
    } else if exponent == 0x1f {
        // The all-ones exponent is not a binade. Widening it arithmetically lands on 65536, which
        // is a plausible-looking number and the wrong one; and 65536 encodes back to this same
        // pattern, so a round-trip test cannot tell the difference.
        if significand == 0 {
            f32::INFINITY
        } else {
            f32::from_bits(0x7f80_0000 | (significand << 13))
        }
    } else {
        f32::from_bits(((exponent + 112) << 23) | (significand << 13))
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// `value >> shift`, rounded to nearest with ties going to the even result.
fn round_shift_right_ties_even(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    truncated + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

#[cfg(test)]
mod tests {
    use super::{f16_bits_to_f32, f32_to_f16_bits};

    /// Every finite `half`, widened and encoded again, must come back unchanged.
    ///
    /// Widening is exact, so this alone only proves the encoding of values that need no rounding.
    /// [`every_boundary_between_two_halves_rounds_correctly`] covers the rounding.
    #[test]
    fn every_representable_half_round_trips() {
        for bits in 0u16..=u16::MAX {
            if (bits >> 10) & 0x1f == 0x1f {
                continue; // infinities and NaNs are covered separately
            }
            let widened = f16_bits_to_f32(bits);
            assert_eq!(
                f32_to_f16_bits(widened),
                bits,
                "0x{bits:04x} widened to {widened} did not encode back"
            );
        }
    }

    /// The rounding decision at every boundary in the format: for each pair of adjacent finite
    /// `half` values, the exact midpoint must go to whichever of the two has an even significand,
    /// and the two neighbouring `f32` values on either side of it must go to their own side.
    ///
    /// A midpoint between two adjacent halves is exactly representable in `f32` -- it needs one
    /// bit more than a `half` and `f32` has thirteen more -- so no rounding of the test's own
    /// sneaks in. This is what a round trip of representable values cannot see: it never asks the
    /// encoder to round at all, which is where both the tie rule and the carry out of the
    /// significand live.
    #[test]
    fn every_boundary_between_two_halves_rounds_correctly() {
        // Positive finite halves, stopping one short of the largest so `bits + 1` stays finite.
        for bits in 0u16..0x7bff {
            let (low, high) = (f16_bits_to_f32(bits), f16_bits_to_f32(bits + 1));
            let midpoint = (low + high) * 0.5;
            assert!(
                low < midpoint && midpoint < high,
                "midpoint of 0x{bits:04x} and its successor is not exactly between them"
            );

            let even = if bits & 1 == 0 { bits } else { bits + 1 };
            assert_eq!(
                f32_to_f16_bits(midpoint),
                even,
                "the tie at {midpoint} between 0x{bits:04x} and 0x{:04x} must go to the even one",
                bits + 1
            );

            let below = f32::from_bits(midpoint.to_bits() - 1);
            let above = f32::from_bits(midpoint.to_bits() + 1);
            assert_eq!(
                f32_to_f16_bits(below),
                bits,
                "{below} is nearest 0x{bits:04x}"
            );
            assert_eq!(
                f32_to_f16_bits(above),
                bits + 1,
                "{above} is nearest 0x{:04x}",
                bits + 1
            );

            // The sign is carried, not computed: the negative of each probe encodes to the same
            // pattern with the sign bit set.
            for probe in [midpoint, below, above] {
                assert_eq!(
                    f32_to_f16_bits(-probe),
                    f32_to_f16_bits(probe) | 0x8000,
                    "the encoding of {probe} and its negation must differ only in the sign"
                );
            }
        }
    }

    /// A value whose rounding carries out of the ten significand bits moves to the next exponent.
    /// Combining the rounded significand with `|` instead of `+` loses exactly this case, and only
    /// when the target exponent is odd — which is why it survived the constants in use.
    #[test]
    fn rounding_that_carries_reaches_the_next_exponent() {
        assert_eq!(f32_to_f16_bits(1.999_755_9), 0x4000); // -> 2.0, exponent 15 is odd
        assert_eq!(f32_to_f16_bits(3.999_511_7), 0x4400); // -> 4.0, exponent 16 is even
        assert_eq!(f32_to_f16_bits(-1.999_755_9), 0xc000);
        // The largest `half` is 65504; anything that rounds past it is the infinity.
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::MAX), 0x7c00);
    }

    /// Ties go to the even significand, in both the normal and the subnormal range.
    #[test]
    fn ties_round_to_even() {
        // A `half` ulp at 1.0 is `0x2000` of `f32` significand. 1.5 ulp ties between 0x3c01 and
        // 0x3c02 and resolves to the even 0x3c02; 0.5 ulp ties between 0x3c00 and 0x3c01 and
        // resolves to the even 0x3c00. Three-quarters of an ulp is not a tie and rounds up.
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x3f80_3000)), 0x3c02);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x3f80_1000)), 0x3c00);
        assert_eq!(f32_to_f16_bits(f32::from_bits(0x3f80_1800)), 0x3c01);
        // Half of the smallest subnormal is a tie with zero, and zero is even.
        assert_eq!(f32_to_f16_bits((2.0f32).powi(-25)), 0x0000);
        // One and a half of it ties between 0x0001 and 0x0002, resolving to the even 0x0002.
        assert_eq!(f32_to_f16_bits((2.0f32).powi(-25) * 3.0), 0x0002);
    }

    /// Signed zero, the signed infinities, and a NaN that must not become one.
    #[test]
    fn zeros_infinities_and_nans_keep_their_identity() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        // A NaN whose payload lives entirely below the bits `half` keeps still encodes as a NaN,
        // not as the infinity that a bare truncation of the payload would produce.
        let low_payload_nan = f32::from_bits(0x7f80_0001);
        assert_ne!(f32_to_f16_bits(low_payload_nan) & 0x03ff, 0);
        assert_eq!(f32_to_f16_bits(low_payload_nan) & 0x7c00, 0x7c00);
        // Underflow below the smallest subnormal is the signed zero, not a denormal-looking value.
        assert_eq!(f32_to_f16_bits((2.0f32).powi(-30)), 0x0000);
        assert_eq!(f32_to_f16_bits(-(2.0f32).powi(-30)), 0x8000);
    }
}
