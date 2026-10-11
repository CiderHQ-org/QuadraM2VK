//! Float exactness: binary16 conversions round trip and round to nearest-even, and the `f64`
//! route for 32-bit `+ - * /` and `sqrt` gives the correctly rounded results of native `f32`.

use super::super::float;

#[test]
fn every_half_round_trips() {
    for h in 0..=u16::MAX {
        let x = float::f16_to_f64(h);
        let back = float::f64_to_f16(x);
        if x.is_nan() {
            assert_eq!(back & 0x7c00, 0x7c00, "{h:#06x}");
            assert_ne!(back & 0x3ff, 0, "{h:#06x}");
        } else {
            assert_eq!(back, h, "{h:#06x}");
        }
    }
}

#[test]
fn halves_round_to_nearest_even() {
    // Halfway between 1.0 (0x3c00) and the next half (0x3c01): ties to the even 0x3c00.
    let ulp = 2f64.powi(-10);
    assert_eq!(float::f64_to_f16(1.0 + ulp / 2.0), 0x3c00);
    assert_eq!(float::f64_to_f16(1.0 + 1.5 * ulp), 0x3c02);
    assert_eq!(float::f64_to_f16(1.0 + ulp / 2.0 + 1e-12), 0x3c01);
    // Largest finite, the overflow tie, and subnormals.
    assert_eq!(float::f64_to_f16(65504.0), 0x7bff);
    assert_eq!(float::f64_to_f16(65519.99), 0x7bff);
    assert_eq!(float::f64_to_f16(65520.0), 0x7c00);
    assert_eq!(float::f64_to_f16(2f64.powi(-24)), 0x0001);
    assert_eq!(float::f64_to_f16(2f64.powi(-25)), 0x0000);
    assert_eq!(float::f64_to_f16(1.5 * 2f64.powi(-24)), 0x0002);
    assert_eq!(float::f64_to_f16(-0.0), 0x8000);
    // The largest subnormal rounds up into the smallest normal.
    assert_eq!(float::f64_to_f16(2f64.powi(-14) - 2f64.powi(-26)), 0x0400);
}

/// A seeded xorshift generator (no dependency).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A random f32 bit pattern, biased toward interesting exponents.
    fn f32_bits(&mut self) -> u64 {
        let r = self.next();
        let special = [
            0u32,
            0x8000_0000,
            0x7f7f_ffff,
            0x0000_0001,
            0x0080_0000,
            0x3f80_0000,
        ];
        if r.is_multiple_of(16) {
            u64::from(special[(r >> 8) as usize % special.len()])
        } else {
            u64::from(self.next() as u32)
        }
    }
}

#[test]
fn f32_operations_match_native() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..200_000 {
        let (a, b) = (rng.f32_bits(), rng.f32_bits());
        let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
        let cases = [
            (float::add(32, a, b).expect("add"), x + y),
            (float::sub(32, a, b).expect("sub"), x - y),
            (float::mul(32, a, b).expect("mul"), x * y),
            (float::bin(32, a, b, |p, q| p / q).expect("div"), x / y),
            (
                float::bin(32, a, a, |p, _| p.sqrt()).expect("sqrt"),
                x.sqrt(),
            ),
            (float::fma(32, a, b, a).expect("fma"), x.mul_add(y, x)),
        ];
        for (got, want) in cases {
            let got = f32::from_bits(got as u32);
            if want.is_nan() {
                assert!(got.is_nan(), "{x:e} {y:e}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{x:e} {y:e}");
            }
        }
    }
}
