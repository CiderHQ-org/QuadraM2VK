//! The extended-range bicubic table: Metal's bicubic samples of each of Apple's extended-range
//! formats through the owned fixture `xr_bicubic`'s kernels, taken on an Apple M3 Max under macOS
//! 26.7 and committed as `validation/fixtures/public/engine/xr_bicubic.tsv` ([`table`]).
//!
//! The probe filled a texture of each raw format at each of [`SIZES`] with random codes
//! ([`textures`], the probe's generator, so both sides hold the same texels; a two-plane texture's
//! alpha bytes written by a kernel's `float` write of `b / 255`, then the color words by
//! `replaceRegion`), and sampled it through each kernel at [`COORDINATES`] normalized
//! coordinates: each texel's centre, then random ones in `[-0.15, 1.15)` on each axis. The
//! runtime kernels' sampler has `Cubic` minification and magnification, clamps to edge on `s` and
//! `t`, and takes normalized coordinates (`MTLSamplerDescriptor`'s other defaults). The product's
//! unit test runs the table through the CPU interpreter, and validation's lane on a device.

use std::collections::BTreeMap;

/// The texture sizes the probe sampled, in its order: width, height.
pub const SIZES: [Size; 5] = [(16, 16), (7, 5), (3, 3), (2, 6), (1, 1)];

/// The coordinates the probe sampled each texture at, its texel centres first.
pub const COORDINATES: u32 = 2304;

/// The raw formats the probe sampled: 550 `RGB10A8_2P_XR10`, 551 its `_sRGB` twin, 552
/// `BGRA10_XR`, 553 its twin, 554 `BGR10_XR`, 555 its twin.
pub const RAW_FORMATS: std::ops::RangeInclusive<u32> = 550..=555;

/// The fixture's kernels: through the runtime sampler (`float` and `half` results), and through
/// constexpr bicubic samplers (clamp to edge with `float` and `half` results; repeat; mirrored
/// repeat; clamp to zero; the opaque-white and opaque-black borders; pixel coordinates; a constant
/// offset `(1, -2)` under clamp to edge and under repeat).
pub const KERNELS: [&str; 12] = [
    "xrb_runtime",
    "xrb_runtime_half",
    "xrb_edge",
    "xrb_edge_half",
    "xrb_repeat",
    "xrb_mirror",
    "xrb_zero",
    "xrb_white",
    "xrb_black",
    "xrb_pixel",
    "xrb_edge_offset",
    "xrb_repeat_offset",
];

/// A texture's width and height.
pub type Size = (u32, u32);

/// The table's samples by the raw format, the size and the kernel ([`table`]).
pub type Table = BTreeMap<(u32, Size, String), Vec<Sample>>;

/// The textures' codes by the raw format and the size ([`textures`]).
pub type Textures = BTreeMap<(u32, Size), Vec<[u32; 4]>>;

/// One of Metal's samples: its coordinate, the result's lanes as bits (`half` lanes for a `_half`
/// kernel), and the lanes Metal rounds to the other side of a tie the exact sum does not.
pub struct Sample {
    pub at: [f32; 2],
    pub lanes: [u32; 4],
    pub ties: Vec<usize>,
}

/// The committed table, one sample per line: the raw format, the texture's width and height, the
/// kernel, the coordinate's `u` and `v` as `f32` bits, the result's lanes as bits, and the lanes at
/// a tie (`-` for none). Of the 2,304 samples the probe took of each texture through each kernel,
/// it holds the middle texel's centre, four random coordinates, the first sample with a color lane
/// at a clamp bound and the first with a lane at a tie. By the raw format, the size and the kernel.
pub fn table() -> Table {
    let path = crate::engine_dir().join("xr_bicubic.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = BTreeMap::<_, Vec<Sample>>::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split(' ').collect();
        let [raw, w, h, kernel, u, v, lanes, ties] = f[..] else {
            panic!("xr_bicubic.tsv: {line}");
        };
        let number = |s: &str| s.parse::<u32>().unwrap_or_else(|e| panic!("{line}: {e}"));
        let bits = |s: &str| u32::from_str_radix(s, 16).unwrap_or_else(|e| panic!("{line}: {e}"));
        let lanes: Vec<u32> = lanes.split(',').map(bits).collect();
        let ties = match ties {
            "-" => Vec::new(),
            t => t.split(',').map(|c| number(c) as usize).collect(),
        };
        out.entry((number(raw), (number(w), number(h)), kernel.to_string()))
            .or_default()
            .push(Sample {
                at: [f32::from_bits(bits(u)), f32::from_bits(bits(v))],
                lanes: lanes
                    .try_into()
                    .unwrap_or_else(|_| panic!("{line}: four lanes")),
                ties,
            });
    }
    out
}

/// The codes of each texture the probe sampled (red, green, blue, alpha; row by row), by raw
/// format and size: splitmix64 from the state `0x9E3779B97F4A7C15`, size by size in [`SIZES`]'
/// order and within a size raw format by raw format, each texel's red, green and blue codes the top
/// ten bits of a draw each, its alpha the top eight (a two-plane alpha byte) or ten (`BGRA10_XR`)
/// bits of the next (`BGR10_XR` draws none, its alpha 0 here), then two draws for each random
/// coordinate.
pub fn textures() -> Textures {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = BTreeMap::new();
    for size in SIZES {
        for raw in RAW_FORMATS {
            let texels = size.0 * size.1;
            let codes = (0..texels)
                .map(|_| {
                    let [r, g, b] = [(); 3].map(|_| (next() >> 54) as u32);
                    let a = match raw {
                        550 | 551 => (next() >> 56) as u32,
                        552 | 553 => (next() >> 54) as u32,
                        _ => 0,
                    };
                    [r, g, b, a]
                })
                .collect();
            out.insert((raw, size), codes);
            for _ in texels..COORDINATES {
                next();
                next();
            }
        }
    }
    out
}

/// The grid of lane `lane` (red to alpha) of a sample of raw format `raw`: `16 · 510` steps a
/// unit of a color lane, `8192` of an `_sRGB` twin's, and `8 · 510` of alpha.
pub fn grid(raw: u32, lane: usize) -> f64 {
    match (lane, raw % 2 == 1) {
        (3, _) => 4080.0,
        (_, true) => 8192.0,
        (_, false) => 8160.0,
    }
}

/// Whether the lane `got` is one step of `grid` from Metal's `want` (`half` lanes where `half`):
/// apart, and no further apart than a step and the result's rounding.
pub fn one_step(got: u32, want: u32, grid: f64, half: bool) -> bool {
    let value = |bits: u32| match half {
        true => half_value(bits as u16),
        false => f64::from(f32::from_bits(bits)),
    };
    let (g, w) = (value(got), value(want));
    let exponent = g.abs().max(w.abs()).log2().floor();
    let ulp = match half {
        true => 2f64.powf(exponent.max(-14.0) - 10.0),
        false => 2f64.powf(exponent.max(-126.0) - 23.0),
    };
    got != want && (g - w).abs() <= 1.0 / grid + ulp
}

/// The finite `half` of `bits`.
fn half_value(bits: u16) -> f64 {
    let magnitude = match (bits >> 10) & 0x1f {
        0 => f64::from(bits & 0x3ff) * 2f64.powi(-24),
        e => (1.0 + f64::from(bits & 0x3ff) / 1024.0) * 2f64.powi(i32::from(e) - 15),
    };
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}
