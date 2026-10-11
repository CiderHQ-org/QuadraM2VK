//! Device tests of a constant depth bias on a `Depth16Unorm` depth on the device the cfg selects,
//! under the validation layer.
//!
//! Metal probes (EVs RRS-probe-7-m3, RRS-probe-7-m4 and RRS-probe-7b-m3; an M3 Max and an M4 alike,
//! each case plain and under Metal's API validation) drew a full-screen triangle at each depth
//! below into a `Depth16Unorm` texture cleared to 1, its depth test `Always` and written, under
//! each constant bias, and read the stored codes: probe 7 at 16 depths (0, near the ends, at codes
//! and between them) under 27 factors (0, ±0.25 to ±8, ±100, ±1000 and ±30000), probe 7b at 39
//! depths around the midpoints of codes 100 and 40000 under 7 (0, ±0.5, ±1 and ±2). Every code is
//! Metal's rule ([`metal_code`]): a finite nonzero constant `c` biases by `c + sign(c)` units of
//! 1/65535, as `float`, added to the depth rounded toward zero, and the sum clamped to `[0, 1]` and
//! stored rounded to nearest.
//!
//! The runtime records Metal's bias where the device pins Vulkan's unit
//! (`metal::state::depth_bias`). The device then rounds the sum and converts it its own way.
//! RADV (Mesa 26.2.4, the Radeon 780M) stored every code Metal's, and MoltenVK (1.4.1, an M4),
//! which hands Metal its constant, too (`DriverFact::UnormDepthBiasIsMetals` and
//! `Depth16UnormStoresMetals`). ANV (Mesa 26.2.4) stored every unbiased code Metal's
//! (`Depth16UnormStoresMetals`), but rounded some biased sums near a code's midpoint to nearest.
//! lavapipe (Mesa 26.0.8 and 26.2.4) scales every depth by 65535 as a `float` and rounds a tie
//! to even (0.9 as 58982). NVIDIA's (615.71.09) converts every depth its own way, storing the
//! code below Metal's for a depth up to 0.55 of a code above a midpoint (0.5 as 32767). Such a "near tie", within [`NEAR_TIE`] of a midpoint, stores the
//! code next to Metal's, which the runtime announces as `unorm-depth-bias` and
//! `unorm-depth-rounding` where the driver lacks the fact.
//!
//! Each case draws into one pixel of a pass (its scissor rectangle), a pass per probe.

use super::buffer_with;
use super::degrade_notices::{assert_made_once, forced, shared_in_system_memory, unclamped_bias};
use super::render_state::{quiet, validated, Probe, PASS, WHITE};
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{Degradation, DriverFact, ScissorRect};

/// The side of each probe's pass: a pixel per case.
const SIDE: u64 = 32;

const PROBE_7_FACTORS: [f32; 27] = [
    -30000.0, -1000.0, -100.0, -8.0, -7.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0,
    0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0, 8.0, 100.0, 1000.0, 30000.0,
];
const PROBE_7: [(u32, [u16; 27]); 16] = [
    (
        0x00000000,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 8, 9, 101, 1001,
            30001,
        ],
    ),
    (
        0x37800080,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 9, 10, 102, 1002,
            30002,
        ],
    ),
    (
        0x38000080,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 10, 11, 103, 1003,
            30003,
        ],
    ),
    (
        0x3dcccccd,
        [
            0, 5552, 6452, 6544, 6545, 6548, 6549, 6550, 6550, 6551, 6551, 6552, 6552, 6554, 6555,
            6555, 6556, 6556, 6557, 6557, 6558, 6559, 6561, 6562, 6655, 7554, 36554,
        ],
    ),
    (
        0x3e800000,
        [
            0, 15383, 16283, 16375, 16376, 16379, 16380, 16380, 16381, 16381, 16382, 16382, 16382,
            16384, 16385, 16385, 16386, 16386, 16387, 16387, 16388, 16389, 16392, 16393, 16485,
            17385, 46385,
        ],
    ),
    (
        0x3e800080,
        [
            0, 15383, 16283, 16375, 16376, 16379, 16380, 16380, 16381, 16381, 16382, 16382, 16383,
            16384, 16385, 16385, 16386, 16386, 16387, 16387, 16388, 16389, 16392, 16393, 16485,
            17385, 46385,
        ],
    ),
    (
        0x3e99999a,
        [
            0, 18660, 19560, 19651, 19652, 19655, 19656, 19657, 19657, 19658, 19658, 19659, 19659,
            19661, 19662, 19662, 19663, 19663, 19664, 19664, 19665, 19666, 19669, 19670, 19761,
            20661, 49662,
        ],
    ),
    (
        0x3effff00,
        [
            2766, 31766, 32666, 32758, 32759, 32762, 32763, 32763, 32764, 32764, 32765, 32765,
            32766, 32767, 32768, 32768, 32769, 32769, 32770, 32770, 32771, 32772, 32775, 32776,
            32868, 33768, 62768,
        ],
    ),
    (
        0x3f000000,
        [
            2767, 31766, 32666, 32758, 32759, 32762, 32763, 32764, 32764, 32765, 32765, 32766,
            32766, 32768, 32769, 32769, 32769, 32770, 32770, 32771, 32771, 32772, 32775, 32776,
            32868, 33768, 62768,
        ],
    ),
    (
        0x3f000080,
        [
            2767, 31767, 32667, 32759, 32760, 32763, 32764, 32764, 32765, 32765, 32766, 32766,
            32767, 32768, 32769, 32769, 32770, 32770, 32771, 32771, 32772, 32773, 32776, 32777,
            32869, 33769, 62769,
        ],
    ),
    (
        0x3f3fffc0,
        [
            19150, 48150, 49050, 49142, 49143, 49146, 49147, 49147, 49148, 49148, 49149, 49149,
            49150, 49151, 49152, 49152, 49153, 49153, 49154, 49154, 49155, 49156, 49159, 49160,
            49252, 50152, 65535,
        ],
    ),
    (
        0x3f400000,
        [
            19150, 48150, 49050, 49142, 49143, 49146, 49147, 49148, 49148, 49149, 49149, 49150,
            49150, 49151, 49152, 49153, 49153, 49154, 49154, 49155, 49155, 49156, 49159, 49160,
            49252, 50152, 65535,
        ],
    ),
    (
        0x3f666666,
        [
            28980, 57980, 58880, 58972, 58973, 58976, 58977, 58978, 58978, 58979, 58979, 58980,
            58980, 58981, 58983, 58983, 58983, 58984, 58984, 58985, 58985, 58986, 58989, 58990,
            59082, 59982, 65535,
        ],
    ),
    (
        0x3f7d70a4,
        [
            34879, 63879, 64779, 64871, 64872, 64875, 64876, 64876, 64877, 64877, 64878, 64878,
            64878, 64880, 64881, 64881, 64882, 64882, 64883, 64883, 64884, 64885, 64888, 64889,
            64981, 65535, 65535,
        ],
    ),
    (
        0x3f7fff00,
        [
            35533, 64533, 65433, 65525, 65526, 65529, 65530, 65530, 65531, 65531, 65532, 65532,
            65533, 65534, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
            65535, 65535, 65535,
        ],
    ),
    (
        0x3f800000,
        [
            35534, 64534, 65434, 65526, 65527, 65530, 65531, 65531, 65532, 65532, 65533, 65533,
            65534, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535,
            65535, 65535, 65535,
        ],
    ),
];
const PROBE_7B_FACTORS: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
const PROBE_7B: [(u32, [u16; 7]); 39] = [
    (0x3ac90046, [97, 98, 99, 100, 102, 102, 103]),
    (0x3ac900bc, [97, 98, 99, 100, 102, 102, 103]),
    (0x3ac900c9, [97, 98, 99, 100, 102, 102, 103]),
    (0x3ac900ca, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac900cd, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac900d6, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac900e3, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac900fd, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac90118, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac90132, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac9014c, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac90166, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac9019b, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac901cf, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac90252, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac902d5, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac903db, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac904e2, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac906ee, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac908fa, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac90d13, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac9112b, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac91a63, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac920c9, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac933fc, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac940c9, [98, 99, 99, 101, 102, 103, 104]),
    (0x3ac96730, [98, 99, 99, 101, 102, 103, 104]),
    (
        0x3f1c411c,
        [39997, 39998, 39999, 40000, 40002, 40002, 40003],
    ),
    (
        0x3f1c411d,
        [39997, 39998, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c411e,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c411f,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c4120,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c4122,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c4124,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c4129,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c412c,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c4136,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c413c,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
    (
        0x3f1c414f,
        [39998, 39999, 39999, 40001, 40002, 40003, 40004],
    ),
];

/// The depth `z` biased by the constant `c` (`z` where `c` is 0 or not finite): Metal's bias,
/// `c + sign(c)` units of 1/65535 as `float`, added to `z` rounded toward zero (`toward_zero`,
/// Metal's) or to nearest, clamped to `[0, 1]`, and scaled by 65535 exactly.
fn scaled(z: f32, c: f32, toward_zero: bool) -> f64 {
    let biased = if c.is_finite() && c != 0.0 {
        let bias = ((f64::from(c) + f64::from(c.signum())) / 65535.0) as f32;
        // Two `float`s sum exactly in `f64` here (their exponents lie within 29 of each other).
        let exact = f64::from(z) + f64::from(bias);
        let nearest = exact as f32;
        if toward_zero && f64::from(nearest).abs() > exact.abs() {
            f32::from_bits(nearest.to_bits() - 1)
        } else {
            nearest
        }
    } else {
        z
    };
    f64::from(biased.clamp(0.0, 1.0)) * 65535.0
}

/// Metal's stored code: the scaled sum ([`scaled`], rounded toward zero) rounded to nearest, a
/// tie up (Metal stored 0x8000 for 0.5).
fn metal_code(z: f32, c: f32) -> u16 {
    (scaled(z, c, true) + 0.5).floor() as u16
}

/// How close to a code's midpoint a device's conversion may round its own way: NVIDIA's rounded
/// down every depth up to 0.55 of a code above a midpoint, and Metal's way from 0.5625 (the
/// module docs).
const NEAR_TIE: f64 = 1.0 / 16.0;

/// The two codes beside the midpoint the scaled sum of `z` and `c` lies within [`NEAR_TIE`] of,
/// its sum rounded either way: a near tie, which a device may store as either (the module docs).
/// `None` for every other depth.
fn near_tie(z: f32, c: f32) -> Option<[u16; 2]> {
    [true, false].into_iter().find_map(|toward_zero| {
        let scaled = scaled(z, c, toward_zero);
        let below = (scaled - 0.5).round();
        let near = (scaled - (below + 0.5)).abs() <= NEAR_TIE && (0.0..65535.0).contains(&below);
        near.then(|| [below as u16, below as u16 + 1])
    })
}

/// The codes `cases` store on the device: each case `(depth, constant)` drawn into its own pixel
/// of one pass, a full-screen triangle at the depth under the constant bias.
fn stored_codes(probe: &Probe, cases: &[(f32, f32)]) -> Vec<u16> {
    assert!(cases.len() as u64 <= SIDE * SIDE);
    let d16 = PixelFormat::Depth16Unorm;
    let pipeline = probe.pipeline(WHITE.depth(d16));
    let state = probe.state(CompareFunction::Always, true, None);
    // Each case's `[z, slope]`, which the fixture's vertex function reads at its offset.
    let bytes: Vec<u8> = cases
        .iter()
        .flat_map(|&(z, _)| [z.to_le_bytes(), 0f32.to_le_bytes()].concat())
        .collect();
    let depths = buffer_with(&probe.device, &bytes).unwrap();
    let stored = probe.run(PASS.size(SIDE).depth(d16, 1.0), |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_depth_stencil_state(&state).unwrap();
        for (at, &(_, constant)) in (0u64..).zip(cases) {
            e.set_scissor_rect(ScissorRect::new(at % SIDE, at / SIDE, 1, 1))
                .unwrap();
            e.set_depth_bias(constant, 0.0, 0.0).unwrap();
            e.set_vertex_buffer(Some(&depths), at * 8, 0).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        }
    });
    stored
        .depth
        .unwrap()
        .chunks(2)
        .take(cases.len())
        .map(|t| u16::from_le_bytes([t[0], t[1]]))
        .collect()
}

/// The cases of a probe's grid, `(depth, constant, Metal's code)`.
fn grid<const N: usize>(factors: &[f32; N], rows: &[(u32, [u16; N])]) -> Vec<(f32, f32, u16)> {
    rows.iter()
        .flat_map(|&(z, codes)| {
            factors
                .iter()
                .zip(codes)
                .map(move |(&c, code)| (f32::from_bits(z), c, code))
        })
        .collect()
}

/// The kinds `unorm_depth_bias` takes: a constant bias on a 16-bit unorm depth where the driver
/// is not measured to store Metal's codes.
pub(super) const UNORM_DEPTH_BIAS: [Degradation; 1] = [Degradation::UnormDepthBias];

/// The kinds `unorm_depth_bias` takes: a 16-bit unorm depth tested or written where the driver is
/// not measured to convert a depth to Metal's code.
pub(super) const UNORM_DEPTH_ROUNDING: [Degradation; 1] = [Degradation::UnormDepthRounding];

/// Every code probes 7 and 7b measured is Metal's rule's ([`metal_code`]). An unbiased row (a
/// constant of 0) stores Metal's code where the driver is measured to convert a depth so
/// (`DriverFact::Depth16UnormStoresMetals`), a biased one where it is measured to bias so
/// (`DriverFact::UnormDepthBiasIsMetals`); elsewhere a row stores Metal's code or, at a near tie
/// ([`near_tie`]), the other code beside the midpoint, printed, and the draws announce the
/// fact's kind once (`unorm-depth-rounding`, `unorm-depth-bias`).
#[test]
fn a_constant_bias_on_a_16_bit_unorm_depth_stores_metals_codes() {
    let probes = [
        ("probe 7", grid(&PROBE_7_FACTORS, &PROBE_7)),
        ("probe 7b", grid(&PROBE_7B_FACTORS, &PROBE_7B)),
    ];
    let unruled: Vec<_> = probes
        .iter()
        .flat_map(|(_, cases)| cases)
        .filter(|&&(z, c, code)| metal_code(z, c) != code)
        .collect();
    assert!(unruled.is_empty(), "codes off Metal's rule: {unruled:?}");
    let (messages, _cache, probe) = validated();
    let device = probe.device.name();
    let facts = &probe.device.properties().driver_facts;
    let biases = facts.contains(&DriverFact::UnormDepthBiasIsMetals);
    let stores = facts.contains(&DriverFact::Depth16UnormStoresMetals);
    let mut differ = Vec::new();
    for (name, cases) in &probes {
        let pairs: Vec<(f32, f32)> = cases.iter().map(|&(z, c, _)| (z, c)).collect();
        let stored = stored_codes(&probe, &pairs);
        let mut near = Vec::new();
        for (&(z, c, metal), stored) in cases.iter().zip(stored) {
            if stored == metal {
                continue;
            }
            let row = format!("{:#010x} by {c}: {stored}, Metal {metal}", z.to_bits());
            let measured = if c == 0.0 { stores } else { biases };
            if !measured && near_tie(z, c).is_some_and(|codes| codes.contains(&stored)) {
                near.push(row);
            } else {
                differ.push(format!("{name} {row}"));
            }
        }
        eprintln!(
            "UNORM-DEPTH-BIAS {device} {name}: {} cases, {} near ties off Metal's: {near:?}",
            cases.len(),
            near.len()
        );
    }
    let mut kinds = Vec::new();
    let mut quiet_kinds = Vec::new();
    for (measured, kind) in [(biases, UNORM_DEPTH_BIAS), (stores, UNORM_DEPTH_ROUNDING)] {
        match measured {
            true => quiet_kinds.extend(kind),
            false => kinds.extend(kind),
        }
    }
    // The draws under constant biases move their depth (`biased-depth-past-unit-range` where the
    // device stores it unclamped), from a `Shared` buffer, through the white pipeline.
    let mut first = forced(&[
        (
            Degradation::BiasedDepthPastUnitRange,
            unclamped_bias(&probe.device),
        ),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&probe.device),
        ),
    ]);
    first.extend(probe.made(WHITE.depth(PixelFormat::Depth16Unorm)));
    let announced = assert_made_once(&messages, &kinds, &first);
    let unexpected: Vec<_> = quiet_kinds
        .iter()
        .filter(|kind| announced.contains_key(kind))
        .collect();
    assert!(
        unexpected.is_empty(),
        "announced {unexpected:?} on a driver measured to store Metal's codes"
    );
    quiet(probe, &messages);
    assert!(differ.is_empty(), "codes not Metal's: {differ:#?}");
}
