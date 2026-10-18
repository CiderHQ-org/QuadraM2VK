//! Metal's linear filter of the two-plane YCbCr formats, on the CPU interpreter against Metal's own
//! samples and its rule, bit for bit.
//!
//! A format's view holds each channel of the converted color as Metal reads it, a step `k / D`
//! (alpha 1) on its field's grid: `2048` (`PackedTexels::YCbCr2048`), `8192`
//! (`PackedTexels::YCbCr8192`), or `8192` signed within `[-2, 2]`
//! (`PackedTexels::YCbCrExtended8192`). Metal's 2D linear sample sums the steps on the grid under
//! the weights `w = wx · wy` in units of `2^-16` (an axis whose two taps address one texel weighs
//! `(256, 0)`), each term truncated: `T = Σ ⌊k · w / 2^7⌋`, then the step `⌊(T + 2^8) / 2^9⌋`,
//! and a `half` result the step's value rounded with ties away from zero (an Apple M3 Max on macOS
//! 26.7 and an Apple M4 on macOS 27.0.1, byte-identical; our probe over the family; and our probe's
//! samples past a texture's edges). The class is the normalized family with `EXACT`
//! (signed for the signed grid) and the field. On MoltenVK, whose sampler is Metal's for every
//! other format, the slot's YCbCr variant (`LoweringInputs::ycbcr_textures`) takes the same
//! emulation.

use super::*;
use crate::lower::select::LowerInputs;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{compile_with, typed_module_with};

/// A format of each YCbCr field: the format, its grid `D`, and its steps' bounds.
#[derive(Clone, Copy, Debug)]
struct Family {
    pf: PixelFormat,
    d: i64,
    lo: i64,
    hi: i64,
}

/// `YCBCR8_420_2P`, on the grid `2048`.
const G2048: Family = Family {
    pf: PixelFormat::YCBCR84202P,
    d: 2048,
    lo: 0,
    hi: 2048,
};

/// `YCBCR12_420_2P`, on the grid `8192`.
const G8192: Family = Family {
    pf: PixelFormat::YCBCR124202P,
    d: 8192,
    lo: 0,
    hi: 8192,
};

/// `YCBCR10_420_2P_XR`, on the signed grid `8192` within `[-2, 2]`.
const X8192: Family = Family {
    pf: PixelFormat::YCBCR104202PXR,
    d: 8192,
    lo: -16384,
    hi: 16384,
};

/// The format's class.
fn class_of(f: Family) -> u32 {
    texel_format(f.pf).0
}

/// Lowering inputs with texture slot 0 in its YCbCr variant, or none.
fn inputs(variant: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    if variant {
        i.ycbcr_textures.insert(TEXTURE);
    }
    i
}

/// Metal's step of a 2D linear sample of a YCbCr format: the taps' steps `k` (`(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`) under the fractions `a` and `b` in 1/256 steps, a merged
/// axis's 0; a signed term and sum floored.
fn metal_step(taps: [i64; 4], a: i64, b: i64) -> i64 {
    let w = [(256 - a) * (256 - b), a * (256 - b), (256 - a) * b, a * b];
    let t: i64 = taps.iter().zip(w).map(|(k, w)| (k * w) >> 7).sum();
    (t + 256) >> 9
}

/// The `half` bits of `k / d` (`d` a power of two, `1 <= |k| / d`'s binade above `2^-14`, or `k
/// = 0`), rounded with ties away from zero.
fn half_away(k: i64, d: i64) -> u16 {
    if k == 0 {
        return 0;
    }
    let sign = if k < 0 { 0x8000 } else { 0 };
    let m = k.unsigned_abs();
    let lead = i64::from(63 - m.leading_zeros());
    let e = lead - i64::from(d.trailing_zeros());
    assert!(e >= -14, "{k} / {d} is below the normal halves");
    // The significand's 11 bits: the magnitude rounded half up there, a carry raising `e`.
    let shift = lead - 10;
    let q = match shift > 0 {
        true => (m + (1 << (shift - 1))) >> shift,
        false => m << -shift,
    };
    let (q, e) = if q == 2048 { (1024, e + 1) } else { (q, e) };
    sign | ((e + 15) as u16) << 10 | (q as u16 - 1024)
}

/// The result lanes of steps `steps` on grid `d` (alpha 1): `k / d` as `f32` bits (exact), or as
/// `half` bits rounded with ties away from zero when `half`.
fn result_bits(steps: [i64; 3], half: bool, d: i64) -> [u32; 4] {
    let lanes = [steps[0], steps[1], steps[2], d];
    lanes.map(|k| match half {
        true => u32::from(half_away(k, d)),
        false => (k as f32 / d as f32).to_bits(),
    })
}

/// The device's texel of steps `steps` on grid `d` (alpha 1), as the view holds it: each `k / d`
/// exactly.
fn texel(steps: [i64; 3], d: i64) -> [u32; 4] {
    [steps[0], steps[1], steps[2], d].map(|k| (k as f32 / d as f32).to_bits())
}

/// One of Metal's samples: the fractions `kx` and `ky` in 1/256 steps, the four taps' steps
/// (red, green, and blue; `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, `(i + 1, j + 1)`), Metal's steps,
/// and Metal's `half` result's bits where the probe read it.
type Row = (i64, i64, [[i64; 3]; 4], [i64; 3], Option<[u16; 3]>);

/// Ten of our probes' samples of `YCBCR8_420_2P` away from the edges (64 x 64
/// textures, clamp to edge, exact 1/256 fractions), chosen so that each of the rule's neighbours
/// misses some: the exact sum rounded once, and the terms truncated by `2^4`, `2^6`, or `2^8`
/// (the first two rows all four).
const INTERIOR: [Row; 10] = [
    (
        199,
        180,
        [
            [208, 1950, 1471],
            [0, 1149, 593],
            [0, 1048, 609],
            [1165, 2048, 1816],
        ],
        [651, 1677, 1322],
        None,
    ),
    (
        126,
        167,
        [
            [1247, 994, 1537],
            [0, 244, 0],
            [238, 111, 446],
            [26, 721, 157],
        ],
        [307, 485, 470],
        None,
    ),
    (
        207,
        135,
        [
            [314, 100, 630],
            [406, 890, 836],
            [403, 2016, 867],
            [0, 1008, 0],
        ],
        [224, 982, 464],
        None,
    ),
    (
        50,
        74,
        [
            [208, 1950, 1471],
            [0, 1149, 593],
            [0, 1048, 609],
            [1165, 2048, 1816],
        ],
        [185, 1634, 1168],
        None,
    ),
    (
        107,
        37,
        [
            [314, 100, 630],
            [406, 890, 836],
            [403, 2016, 867],
            [0, 1008, 0],
        ],
        [335, 598, 686],
        None,
    ),
    (
        19,
        209,
        [[1313, 115, 0], [850, 1053, 143], [763, 0, 852], [0, 0, 538]],
        [811, 34, 678],
        None,
    ),
    (
        234,
        186,
        [[1169; 3], [973; 3], [1599; 3], [393; 3]],
        [631; 3],
        None,
    ),
    (
        45,
        205,
        [
            [1247, 994, 1537],
            [0, 244, 0],
            [238, 111, 446],
            [26, 721, 157],
        ],
        [365, 347, 569],
        None,
    ),
    (
        46,
        167,
        [[252; 3], [636; 3], [1496; 3], [1496; 3]],
        [1088; 3],
        None,
    ),
    (
        11,
        117,
        [[252; 3], [636; 3], [1496; 3], [1496; 3]],
        [830; 3],
        None,
    ),
];

/// Four of our probe's interior samples on the grid `8192` (`YCBCR12_420_2P`'s video
/// range, two; `YCBCR10_420_2P_sRGB`'s, two), each a `half` tie that rounding to nearest even
/// misses, and each missed by the terms truncated by `2^6` or by `2^8`.
const FINE: [Row; 4] = [
    (
        179,
        1,
        [
            [0, 2347, 5566],
            [5646, 3359, 7657],
            [0, 2403, 3884],
            [1218, 0, 3742],
        ],
        [3936, 3045, 7015],
        Some([0x37b0, 0x35f3, 0x3ada]),
    ),
    (
        155,
        200,
        [
            [0, 1580, 0],
            [5414, 7117, 4813],
            [923, 4740, 3276],
            [5919, 8191, 7290],
        ],
        [3801, 6415, 5096],
        Some([0x376d, 0x3a44, 0x38fa]),
    ),
    (
        227,
        91,
        [
            [272, 0, 0],
            [1366, 6118, 1560],
            [698, 147, 0],
            [3883, 5660, 184],
        ],
        [2053, 5286, 950],
        Some([0x3403, 0x392a, 0x2f6c]),
    ),
    (
        202,
        168,
        [
            [1329, 7168, 929],
            [8192, 6439, 1506],
            [0, 3286, 0],
            [3795, 4112, 111],
        ],
        [4284, 4850, 533],
        Some([0x382f, 0x38bd, 0x2c2a]),
    ),
];

/// Six of our probe's interior samples on the signed grid `8192` (`YCBCR12_420_2P_XR`'s,
/// three; `YCBCR10_420_2P_XR`'s, three): negative steps, steps past `8192`, `half` ties, and
/// each of the rule's neighbours missing some.
const EXTENDED: [Row; 6] = [
    (
        145,
        243,
        [
            [-3312, 501, 4446],
            [-2852, 2437, 6496],
            [359, 3997, 6571],
            [-546, 3427, 6304],
        ],
        [-301, 3569, 6378],
        Some([0xa8b4, 0x36f9, 0x3a3b]),
    ),
    (
        127,
        158,
        [
            [3214, 5328, 4412],
            [6673, 3136, 3042],
            [3383, 7503, 5832],
            [3286, 2551, 1666],
        ],
        [3946, 4738, 3753],
        Some([0x37b5, 0x38a1, 0x3755]),
    ),
    (
        203,
        69,
        [
            [-4637, 2482, 1205],
            [-3569, 2421, -664],
            [-1307, 5280, 2263],
            [3023, 8928, 3462],
        ],
        [-2195, 3980, 664],
        Some([0xb44a, 0x37c6, 0x2d30]),
    ),
    (
        71,
        49,
        [
            [2134, 6376, 4112],
            [2666, 5585, 8194],
            [3919, 2879, 1990],
            [1118, 39, 676],
        ],
        [2446, 5378, 4552],
        Some([0x34c7, 0x3941, 0x3872]),
    ),
    (
        93,
        132,
        [
            [3796, 5019, 11274],
            [1131, 1004, 8300],
            [5669, 6740, 13647],
            [1172, 684, 10343],
        ],
        [3450, 4065, 11355],
        Some([0x36bd, 0x37f1, 0x3d8b]),
    ),
    (
        31,
        144,
        [
            [-36, 521, 1296],
            [794, 823, 351],
            [-444, 146, 347],
            [1186, 3088, 905],
        ],
        [-111, 526, 750],
        Some([0xa2f0, 0x2c1c, 0x2ddc]),
    ),
];

/// Two of our probe's samples of `YCBCR8_420_2P` (an Apple M4; a 40 x 24 texture, clamp to
/// edge) whose `y` taps clamp onto one row, at the top and the bottom edge: the unmerged
/// fraction (8 and 216 here) answers one step low on red chroma. Each row's taps are `(i, j)`
/// and `(i + 1, j)`.
const EDGES: [(i64, i64, [[i64; 3]; 2], [i64; 3]); 2] = [
    (
        72,
        8,
        [[373, 2048, 957], [239, 2048, 333]],
        [335, 2048, 782],
    ),
    (
        184,
        216,
        [[1216, 497, 649], [1539, 2048, 1849]],
        [1448, 1612, 1512],
    ),
];

/// A compiled kernel sampling a YCbCr texture through a constexpr linear clamp-to-edge sampler or
/// a runtime sampler of that state, with a `half` or `float` result: its plan, module, and the
/// runtime sampler's word.
struct Kernel {
    plan: Plan,
    m: TypedModule,
    word: Option<u32>,
    a: std::sync::Arc<crate::library::FunctionArtifact>,
}

/// The kernel of `half` and `runtime` (module docs) on lavapipe, or on MoltenVK under the slot's
/// YCbCr variant when `metal`; `None` without `llvm-as`.
fn kernel_for(half: bool, runtime: bool, metal: bool) -> Option<Kernel> {
    let p = if metal { m3() } else { lavapipe() };
    let sampler = (!runtime).then(|| linear(1, false, 0));
    let name = format!("ycbcr half={half} runtime={runtime} metal={metal}");
    let bc = assemble(
        &name,
        &kernel_of(sampler, Form::Plain { offset: (0, 0) }, !half),
    )?;
    let a = compile_with(&bc, "k", &p, &inputs(metal)).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert!(
        a.strategies.entries.contains_key("sampler.linear.quad"),
        "{name}"
    );
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(metal)).expect("typed module");
    let word = runtime.then(|| {
        runtime_word(
            SamplerAddressMode::ClampToEdge,
            false,
            SamplerBorderColor::TransparentBlack,
        )
    });
    Some(Kernel { plan, m, word, a })
}

/// `k`'s results at normalized `coords` over `image`, bound in `f`'s format.
fn sample(
    k: &Kernel,
    image: &Image,
    coords: &[(f32, f32)],
    half: bool,
    f: Family,
) -> Vec<[u32; 4]> {
    let bytes: Vec<u8> = coords
        .iter()
        .flat_map(|(u, v)| [u.to_le_bytes(), v.to_le_bytes()])
        .flatten()
        .collect();
    let size = if half { 8 } else { 16 };
    let n = coords.len();
    let res = resources(&k.plan, &[bytes, vec![0xab; size * n]], class_of(f));
    let res = with_word(res, &k.a, k.word);
    run(&k.m, &res, image, n as u32, half)
}

/// Each result, sampler, and device (lavapipe, and MoltenVK under the variant).
fn combinations() -> Vec<(bool, bool, bool)> {
    let flags = [false, true];
    flags
        .iter()
        .flat_map(|&h| flags.iter().flat_map(move |&r| flags.map(|m| (h, r, m))))
        .collect()
}

/// `rows`' samples of `f`'s format through `k`, away from the edges: row `r`'s taps at `x` `2r`
/// and `2r + 1`, rows 0 and 1; each Metal's steps (and, where read, its `half` bits).
fn check_rows(k: &Kernel, half: bool, f: Family, rows: &[Row], name: &str) {
    let n = rows.len() as u32;
    let mut image = Image::new(combined_slot(&k.m), [2 * n, 2, 1], 1, 1, 1);
    for (r, (_, _, taps, _, _)) in (0u32..).zip(rows) {
        let sites = [
            [2 * r, 0, 0],
            [2 * r + 1, 0, 0],
            [2 * r, 1, 0],
            [2 * r + 1, 1, 0],
        ];
        for (at, t) in sites.into_iter().zip(taps) {
            let i = image.index(0, 0, 0, at).expect("texel");
            image.texels[i] = texel(*t, f.d);
        }
    }
    let coords: Vec<(f32, f32)> = (0u32..)
        .zip(rows)
        .map(|(r, (a, b, _, _, _))| {
            let u = (512 * i64::from(r) + 128 + a) as f32 / (512 * n) as f32;
            (u, (128 + b) as f32 / 512.0)
        })
        .collect();
    let got = sample(k, &image, &coords, half, f);
    for ((a, b, taps, want, metal_half), g) in rows.iter().zip(got) {
        assert_eq!(
            g,
            result_bits(*want, half, f.d),
            "{name} {f:?} k {a} {b} taps {taps:?}: {g:x?}"
        );
        if let (true, Some(h)) = (half, metal_half) {
            assert_eq!(g[..3], h.map(u32::from), "{name} {f:?} Metal's half bits");
        }
    }
}

/// Metal's samples (module docs) through a constexpr sampler and a runtime one, with `float` and
/// `half` results, on lavapipe and on MoltenVK under the variant, are Metal's bytes: on the grid
/// `2048` (each step exact in `half`; our probe's `half` samples of the format are its
/// `float` ones), `8192`, and the signed `8192`; and the rule answers each of them.
#[test]
fn ycbcr_linear_samples_are_metals_bytes() {
    for (f, rows) in [
        (G2048, &INTERIOR[..]),
        (G8192, &FINE[..]),
        (X8192, &EXTENDED[..]),
    ] {
        for (a, b, taps, want, metal_half) in rows {
            let steps = std::array::from_fn(|c| metal_step(taps.map(|t| t[c]), *a, *b));
            assert_eq!(steps, *want, "the rule at {a} {b} on {f:?}");
            if let Some(h) = metal_half {
                let away = std::array::from_fn(|c| half_away(want[c], f.d));
                assert_eq!(away, *h, "ties away at {a} {b} on {f:?}");
            }
        }
    }
    for (a, _, taps, want) in EDGES {
        let [lo, hi] = taps;
        let steps = std::array::from_fn(|c| metal_step([lo[c], hi[c], lo[c], hi[c]], a, 0));
        assert_eq!(steps, want, "the rule at {a}, merged");
    }
    for (half, runtime, metal) in combinations() {
        let Some(k) = kernel_for(half, runtime, metal) else {
            return;
        };
        let name = format!("half={half} runtime={runtime} metal={metal}");
        check_rows(&k, half, G2048, &INTERIOR, &name);
        check_rows(&k, half, G8192, &FINE, &name);
        check_rows(&k, half, X8192, &EXTENDED, &name);
        // Past the edges: a texture one row tall, whose `y` taps clamp onto that row.
        let n = EDGES.len() as u32;
        let mut image = Image::new(combined_slot(&k.m), [2 * n, 1, 1], 1, 1, 1);
        for (r, (_, _, taps, _)) in (0u32..).zip(EDGES) {
            for (x, t) in [2 * r, 2 * r + 1].into_iter().zip(taps) {
                let i = image.index(0, 0, 0, [x, 0, 0]).expect("texel");
                image.texels[i] = texel(t, G2048.d);
            }
        }
        let coords: Vec<(f32, f32)> = (0u32..)
            .zip(EDGES)
            .map(|(r, (a, b, _, _))| {
                let u = (512 * i64::from(r) + 128 + a) as f32 / (512 * n) as f32;
                (u, (128 + b) as f32 / 256.0)
            })
            .collect();
        let got = sample(&k, &image, &coords, half, G2048);
        for ((a, b, taps, want), g) in EDGES.into_iter().zip(got) {
            assert_eq!(
                g,
                result_bits(want, half, G2048.d),
                "{name} edge k {a} {b} taps {taps:?}: {g:x?}"
            );
        }
    }
}

/// The emulation is the rule (module docs) at random steps and positions over an 8 x 4 texture
/// of each field's format, a quarter of them past an edge on one axis or both (each axis's
/// positions within one and a half texels of the texture), through both samplers, both results,
/// and both devices.
#[test]
fn ycbcr_linear_samples_take_metals_rule() {
    const W: i64 = 8;
    const H: i64 = 4;
    let mut rng = Lcg(0x0059_3593);
    // Footprint positions `q = 256 (x - 1/2)`, exact: `x = (q + 128) / 256` texels.
    let positions: Vec<(i64, i64)> = (0..96)
        .map(|_| {
            let mut q = |n: i64| (rng.next() % (256 * n as u64 + 768)) as i64 - 384;
            (q(W), q(H))
        })
        .collect();
    let edges = positions
        .iter()
        .filter(|&&(qx, qy)| {
            let out = |q: i64, n: i64| q < 0 || q >= 256 * (n - 1);
            out(qx, W) || out(qy, H)
        })
        .count();
    assert!(edges >= positions.len() / 4, "{edges} past an edge");
    let coords: Vec<(f32, f32)> = positions
        .iter()
        .map(|&(qx, qy)| {
            let n = |q: i64, n: i64| (q + 128) as f32 / (256 * n) as f32;
            (n(qx, W), n(qy, H))
        })
        .collect();
    let clamped = |i: i64, n: i64| i.clamp(0, n - 1);
    let mut rng = Lcg(0x5935_5a62_d593);
    let families = [G2048, G8192, X8192].map(|f| {
        let span = (f.hi - f.lo + 1) as u64;
        let mut step = || f.lo + (rng.next() % span) as i64;
        let steps: Vec<[i64; 3]> = (0..W * H).map(|_| [step(), step(), step()]).collect();
        let want: Vec<[i64; 3]> = positions
            .iter()
            .map(|&(qx, qy)| {
                let (i, j, a, b) = (qx >> 8, qy >> 8, qx & 255, qy & 255);
                let (i0, i1) = (clamped(i, W), clamped(i + 1, W));
                let (j0, j1) = (clamped(j, H), clamped(j + 1, H));
                let a = if i0 == i1 { 0 } else { a };
                let b = if j0 == j1 { 0 } else { b };
                let at = |x: i64, y: i64| steps[(y * W + x) as usize];
                let taps = [at(i0, j0), at(i1, j0), at(i0, j1), at(i1, j1)];
                std::array::from_fn(|c| metal_step(taps.map(|t| t[c]), a, b))
            })
            .collect();
        (f, steps, want)
    });
    for (half, runtime, metal) in combinations() {
        let Some(k) = kernel_for(half, runtime, metal) else {
            return;
        };
        for (f, steps, want) in &families {
            let mut image = Image::new(combined_slot(&k.m), [W as u32, H as u32, 1], 1, 1, 1);
            for y in 0..H {
                for x in 0..W {
                    let i = image
                        .index(0, 0, 0, [x as u32, y as u32, 0])
                        .expect("texel");
                    image.texels[i] = texel(steps[(y * W + x) as usize], f.d);
                }
            }
            let got = sample(&k, &image, &coords, half, *f);
            for ((p, w), g) in positions.iter().zip(want).zip(got) {
                assert_eq!(
                    g,
                    result_bits(*w, half, f.d),
                    "half={half} runtime={runtime} metal={metal} {f:?} q {p:?}: {g:x?}"
                );
            }
        }
    }
}

/// On MoltenVK a filtered sample records `RuntimeFact::YCbCrTexture`, and outside the variant the
/// device filters it, Metal's sampler, unnoted; under the variant a sample the emulation reaches
/// takes it, noted where its device arm runs (a class of no measured family), and one it does
/// not reach (a linear mip filter) is the device's, noted. Every one of these `half` samples
/// requires `VkProperty::MetalSampler` there, whose texel's rounding to `half` is Metal's only
/// through Metal's own texture unit. A nearest sample records nothing. On lavapipe the variant
/// changes nothing.
#[test]
fn moltenvk_emulates_a_ycbcr_slot_under_its_variant() {
    let plain = Form::Plain { offset: (0, 0) };
    // (name, sampler, emulated under the variant, filtered)
    let cases = [
        ("linear", Some(linear(1, false, 0)), true, true),
        ("runtime", None, true, true),
        ("mip", Some(sampler_words(1, 2, 1, false, 0)), false, true),
        (
            "nearest",
            Some(sampler_words(0, 0, 1, false, 0)),
            false,
            false,
        ),
    ];
    let emulated = |a: &crate::library::FunctionArtifact| {
        a.strategies.entries.contains_key("sampler.linear.quad")
    };
    let metal_sampler = |a: &crate::library::FunctionArtifact| {
        a.requirements
            .properties
            .contains(&VkProperty::MetalSampler)
    };
    let fact = RuntimeFact::YCbCrTexture { slot: TEXTURE };
    for (name, sampler, reached, filtered) in cases {
        let Some(bc) = assemble(name, &kernel(sampler, plain)) else {
            return;
        };
        for (stem, p) in [("moltenvk-m3", m3()), ("lavapipe", lavapipe())] {
            let metal = stem == "moltenvk-m3";
            for variant in [false, true] {
                let what = format!("{name} on {stem}, variant {variant}");
                let a = compile_with(&bc, "k", &p, &inputs(variant))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(variant))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(plan.facts.contains(&fact), metal && filtered, "{what}");
                let takes = !metal || variant;
                assert_eq!(emulated(&a), reached && takes, "{what}");
                assert_eq!(
                    a.precision.contains(&PrecisionNote::HalfSampleFilter),
                    filtered && takes,
                    "{what}"
                );
                assert_eq!(metal_sampler(&a), metal, "{what}");
            }
        }
    }
}

/// Under native sampling (the default), a filtered sample of a texture slot records
/// `RuntimeFact::YCbCrTexture` only under `YCbCrFilterPolicy::Exact`, whose variant emulates
/// Metal's filter of a two-plane YCbCr format bound there, on lavapipe and on MoltenVK. Under the
/// default, `YCbCrFilterPolicy::Device`, it records none, so a draw binding such a texture makes no
/// variant, and the device filters the slot whatever the inputs name: on lavapipe the module records
/// `DeviceSampling` (its note of the device's filter) and no filter note; on MoltenVK, whose
/// sampler is Metal's own, the filter note, which holds only where the format bound at the slot is
/// two-plane YCbCr (`SpecKey::BoundTexelFormat`, the runtime's `ycbcr-filter-not-exact`), and no
/// `DeviceSampling`.
#[test]
fn a_device_takes_the_ycbcr_variant_only_under_the_exact_filter() {
    use crate::device::YCbCrFilterPolicy;
    use crate::metal::texel_format::{PackedTexels, TexelFormat};
    use crate::pipeline::sym::SpecKey;
    let Some(bc) = assemble(
        "ycbcr policy",
        &kernel(Some(linear(1, false, 0)), Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let fact = RuntimeFact::YCbCrTexture { slot: TEXTURE };
    let ycbcr_fields: Vec<u32> = [
        PackedTexels::YCbCr2048,
        PackedTexels::YCbCr8192,
        PackedTexels::YCbCrExtended8192,
    ]
    .map(|p| p.class().0)
    .into();
    for stem in ["lavapipe", "moltenvk-m3"] {
        for ycbcr in [YCbCrFilterPolicy::Device, YCbCrFilterPolicy::Exact] {
            let policy = DevicePolicy {
                ycbcr_filter: ycbcr,
                ..DevicePolicy::UNTRUSTED
            };
            let p = profile(stem, policy).expect("profile");
            let native = stem == "lavapipe";
            let takes = ycbcr == YCbCrFilterPolicy::Exact;
            assert_eq!(p.fields().ycbcr_variant, takes, "{stem} {ycbcr:?}");
            for variant in [false, true] {
                let what = format!("{stem} {ycbcr:?}, variant {variant}");
                let a = compile_with(&bc, "k", &p, &inputs(variant))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(variant))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(plan.facts.contains(&fact), takes, "{what}");
                assert_eq!(
                    a.strategies.entries.contains_key("sampler.linear.quad"),
                    takes && variant,
                    "{what}"
                );
                // Metal's own sampler without the variant: the filter note, under a two-plane
                // YCbCr format bound at the slot alone.
                let bound = !native && !takes;
                assert_eq!(
                    a.precision.contains(&PrecisionNote::HalfSampleFilter),
                    (takes && variant) || bound,
                    "{what}"
                );
                if bound {
                    let s = a
                        .specialized
                        .iter()
                        .find(|s| s.note == PrecisionNote::HalfSampleFilter)
                        .unwrap_or_else(|| panic!("{what}: the filter note's clauses"));
                    let mut tested: Vec<u32> = s
                        .when
                        .iter()
                        .map(|c| match c.all[..] {
                            [t] if matches!(t.key, SpecKey::BoundTexelFormat(_))
                                && t.mask == TexelFormat::PACKED_MASK
                                && t.equal =>
                            {
                                t.bits
                            }
                            _ => panic!("{what}: {c:?}"),
                        })
                        .collect();
                    tested.sort_unstable();
                    assert_eq!(tested, ycbcr_fields, "{what}");
                }
                // The device's sample, wherever the emulation is not taken: noted on lavapipe.
                if !(takes && variant) {
                    assert_eq!(
                        a.precision.contains(&PrecisionNote::DeviceSampling),
                        native,
                        "{what}"
                    );
                }
            }
        }
    }
}

/// A YCbCr texture has one level, which Metal's mip filter leaves as it is: through a runtime
/// sampler whose word states a nearest or a linear mip filter, the emulation answers each field's
/// samples as through none, on lavapipe and on MoltenVK under the variant; and the filter note
/// holds under such a word for a format of another class (`RGBA8Unorm`, whose textures may have
/// levels) and not for `YCBCR8_420_2P`, the one the binder binds yet.
#[test]
fn a_runtime_mip_filter_takes_metals_filter_of_one_level() {
    use crate::bind::SamplerDescriptor;
    use crate::metal::{SamplerMinMagFilter, SamplerMipFilter};
    for (half, metal) in [(false, false), (true, false), (false, true), (true, true)] {
        for mip in [1u32, 2] {
            let Some(k) = kernel_for(half, true, metal) else {
                return;
            };
            let k = Kernel {
                word: k.word.map(|w| w | mip << SamplerWord::MIP_FILTER_SHIFT),
                ..k
            };
            let name = format!("half={half} metal={metal} mip={mip}");
            check_rows(&k, half, G2048, &INTERIOR, &name);
            check_rows(&k, half, G8192, &FINE, &name);
            check_rows(&k, half, X8192, &EXTENDED, &name);
        }
    }
    let Some(bc) = assemble(
        "notes runtime mip",
        &kernel(None, Form::Plain { offset: (0, 0) }),
    ) else {
        return;
    };
    let lvp = device_of("lavapipe");
    let runtime = notes_pipeline(&lvp, &bc);
    for mip in [SamplerMipFilter::Nearest, SamplerMipFilter::Linear] {
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        sampler.mip_filter = mip;
        for (pf, noted) in [(PixelFormat::RGBA8Unorm, true), (G2048.pf, false)] {
            let plane = (crate::metal::TextureType::Type2D, pf);
            let notes = notes_bound(&runtime, &lvp, plane, Some(sampler));
            assert_eq!(
                notes.contains(&PrecisionNote::HalfSampleFilter),
                noted,
                "{mip:?} {pf:?}: {notes:?}"
            );
        }
    }
}

/// On MoltenVK the slot's YCbCr variant emulates an `EXACT` class alone, through the exact
/// routine: every other class, a two-plane format's unmeasured one (its class made with a swizzle,
/// or none) and the float families no two-plane format has, takes the device's sample (Metal's own
/// filter there), which the interpreter does not model, through a constexpr sampler and a runtime
/// one, with `float` and `half` results. On lavapipe the float families keep their emulation. An
/// `EXACT` class's samples stay Metal's (`ycbcr_linear_samples_are_metals_bytes`).
#[test]
fn the_ycbcr_variant_emulates_only_an_exact_class() {
    use crate::metal::texel_format::TexelFormat;
    use crate::spirv::interp::{InterpError, Unmodelled};
    let unmeasured = [0, TexelFormat(class_of(G2048)).swizzled().0];
    let float = [
        texel_format(PixelFormat::RGBA32Float).0,
        texel_format(PixelFormat::RGBA16Float).0,
    ];
    for class in unmeasured {
        assert_eq!(class & BicubicTexels::EXACT, 0, "{class:#x}");
    }
    for (half, runtime, metal) in combinations() {
        let Some(k) = kernel_for(half, runtime, metal) else {
            return;
        };
        let name = format!("half={half} runtime={runtime} metal={metal}");
        let mut image = Image::new(combined_slot(&k.m), [4, 4, 1], 1, 1, 1);
        for t in image.texels.iter_mut() {
            *t = texel([512, 1024, 1536], G2048.d);
        }
        let coords = [(0.3f32, 0.6f32), (0.55, 0.45)];
        let bytes: Vec<u8> = coords
            .iter()
            .flat_map(|(u, v)| [u.to_le_bytes(), v.to_le_bytes()])
            .flatten()
            .collect();
        let size = if half { 8 } else { 16 };
        let classes = unmeasured.iter().map(|c| (*c, false));
        let classes = classes.chain(float.iter().map(|c| (*c, !metal)));
        let classes = classes.chain([(class_of(G2048), true)]);
        for (class, emulated) in classes {
            let init = [bytes.clone(), vec![0xab; size * coords.len()]];
            let res = with_word(resources(&k.plan, &init, class), &k.a, k.word);
            let cfg = Config {
                workgroups: [coords.len() as u32, 1, 1],
                ..Config::default()
            };
            let r = run_with_images(&k.m, &cfg, res, vec![image.clone()]);
            let device = matches!(
                r,
                Err(InterpError::Unmodelled(Unmodelled::Opcode(
                    crate::spv::Op::ImageSampleExplicitLod | crate::spv::Op::ImageSampleImplicitLod
                )))
            );
            assert_eq!(
                (r.is_ok(), device),
                (emulated, !emulated),
                "{name} class {class:#x}: {:?}",
                r.err()
            );
        }
    }
}
