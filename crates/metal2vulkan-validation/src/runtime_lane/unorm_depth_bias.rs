//! Device tests of a constant depth bias on a `Depth16Unorm` depth on the device the cfg selects,
//! under the validation layer.
//!
//! Metal probes (an M3 Max and an M4 alike, each case plain and under Metal's API validation) drew
//! a full-screen triangle at each depth below into a `Depth16Unorm` texture cleared to 1, its depth
//! test `Always` and written, under each constant bias, and read the stored codes: probe 7 at 16
//! depths (0, near the ends, at codes and between them) under 27 factors (0, ±0.25 to ±8, ±100,
//! ±1000 and ±30000), probe 7b at 39 depths around the midpoints of codes 100 and 40000 under 7 (0,
//! ±0.5, ±1 and ±2). Every code is Metal's rule ([`metal_code`]): wherever the constant `c`
//! (finite) or the slope factor is nonzero, the constant biases by `c + 1` units of 1/65535 for `c`
//! of 0 or more and `c - 1` below, as `float`, added to the depth rounded toward zero, and the sum
//! clamped to `[0, 1]` and stored rounded to nearest. Our probe `depth16.swift` on the M3 Max under
//! macOS 27.0.1 drew both grids again (every code as before) and 24000 random depths and constants
//! (every code the rule's), and found that a slope alone brings the one unit: [`SLOPE_ALONE`], flat
//! triangles (so the slope adds nothing itself) under a constant of 0.
//!
//! Where the profile takes `DriverDeviation::MetalDepth16Stores`, the device converts an unbiased
//! depth, and where it also takes `DriverDeviation::MetalDepth16BiasedStores` a biased one: the
//! runtime records Metal's bias where the device pins Vulkan's unit (`metal::state::depth_bias`),
//! and the device rounds the sum and converts it its own way. RADV (Mesa 26.2.4, the Radeon 780M)
//! stored every code Metal's, and MoltenVK (1.4.1, an M4), which hands Metal its constant, too
//! (both deviations). ANV (Mesa 26.2.4) stored every unbiased code Metal's (the first), but
//! rounded some biased sums near a code's midpoint to nearest, storing the code next to Metal's,
//! so a biased draw there runs the variant whose fragment stage adds the bias
//! (`VariantKey::depth16_biased`).
//!
//! Elsewhere the fragment stage writes Metal's codes itself (`LoweringInputs::depth16_codes`),
//! its bias added in the shader, and every code is Metal's (`shader-unorm-depth`; a sloped bias
//! takes its slope from the quad's depths, `unorm-depth-slope`). Without it lavapipe (Mesa 26.0.8
//! and 26.2.4) scaled every depth by 65535 as a `float` and rounded a tie to even (0.9 as 58982),
//! and NVIDIA's (615.71.09) stored the code below Metal's for a depth up to 0.55 of a code above
//! a midpoint (0.5 as 32767). A biased draw whose fragment stage cannot add the bias (one that
//! declares early fragment tests, here) keeps the device's conversion, which the runtime
//! announces as `unorm-depth-bias` where the profile lacks the second deviation.
//!
//! The render pass's depth decides what a draw stores, whatever the pipeline names: the cases
//! drawn by a pipeline that names no depth store the same codes.
//!
//! Each case draws into one pixel of a pass (its scissor rectangle), a pass per probe.

use super::buffer_with;
use super::degrade_notices::{
    assert_made_once, forced, made_render_kinds, shared_in_system_memory, unclamped_bias,
};
use super::render_state::{quiet, validated, Pipe, Probe, PASS, WHITE};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::{
    CompareFunction, DepthClipMode, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Degradation, LoadAction, PipelineOptions, RenderPassDescriptor, ResourceOptions, ScissorRect,
    StoreAction, TextureDescriptor, Viewport,
};

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

/// Flat triangles at a depth (its bits) under a slope factor alone (a constant of 0), and the
/// code Metal stored: the M3 Max under macOS 27.0.1 (our probe `depth16.swift`), the depth one
/// unit up before the rounding ([`metal_code`]).
const SLOPE_ALONE: [(u32, f32, u16); 28] = [
    (0x3ea474a5, 0.5, 21051),
    (0x3f2da090, 0.5, 44449),
    (0x3e85acd5, 0.5, 17111),
    (0x3f12f307, 0.5, 37619),
    (0x3f225df6, 1.0, 41566),
    (0x3f283eb5, 1.0, 43071),
    (0x3f0ff56e, 1.0, 36854),
    (0x3f1ec515, 1.0, 40645),
    (0x3f1040c8, 2.0, 36929),
    (0x3efe92f1, 2.0, 32586),
    (0x3ea76d9f, 2.0, 21431),
    (0x3f1f9cf1, 2.0, 40861),
    (0x3ea5c404, -1.0, 21219),
    (0x3f110eb8, -1.0, 37135),
    (0x3e8edc5e, -1.0, 18287),
    (0x3f3549b8, -1.0, 46410),
    (0x3ef3da1c, 100.0, 31214),
    (0x3f27c8a5, 100.0, 42953),
    (0x3ebc23a2, 100.0, 24082),
    (0x3ed43cd2, 100.0, 27167),
    (0x3e91cded, -100.0, 18664),
    (0x3efb372e, -100.0, 32156),
    (0x3f1bb82e, -100.0, 39865),
    (0x3f316b5a, -100.0, 45420),
    (0x3f14a090, 1000.0, 38049),
    (0x3ee65c02, 1000.0, 29487),
    (0x3f12edfd, 1000.0, 37614),
    (0x3ebc19cf, 1000.0, 24078),
];

/// The depth `z` of a flat triangle biased by the constant `c` and the slope factor `s` (`z`
/// where both are 0, or `c` is not finite): Metal's bias, `c + 1` units of 1/65535 as `float` for
/// `c` of 0 or more and `c - 1` below, added to `z` rounded toward zero, clamped to `[0, 1]`, and
/// scaled by 65535 exactly.
fn scaled(z: f32, c: f32, s: f32) -> f64 {
    let biased = if c.is_finite() && (c != 0.0 || s != 0.0) {
        let unit = if c >= 0.0 { 1.0 } else { -1.0 };
        let bias = ((f64::from(c) + unit) / 65535.0) as f32;
        // Two `float`s sum exactly in `f64` here (their exponents lie within 29 of each other).
        let exact = f64::from(z) + f64::from(bias);
        let nearest = exact as f32;
        if f64::from(nearest).abs() > exact.abs() {
            f32::from_bits(nearest.to_bits() - 1)
        } else {
            nearest
        }
    } else {
        z
    };
    f64::from(biased.clamp(0.0, 1.0)) * 65535.0
}

/// Metal's stored code: the scaled sum ([`scaled`]) rounded to nearest, a tie up (Metal stored 0x8000 for 0.5).
fn metal_code(z: f32, c: f32, s: f32) -> u16 {
    (scaled(z, c, s) + 0.5).floor() as u16
}

/// The codes `cases` store on the device: each case `(depth, constant, slope factor)` drawn into
/// its own pixel of one pass of a `Depth16Unorm` depth, a flat full-screen triangle at the depth
/// under the bias, by `pipe`.
fn stored_codes(probe: &Probe, pipe: Pipe, cases: &[(f32, f32, f32)]) -> Vec<u16> {
    assert!(cases.len() as u64 <= SIDE * SIDE);
    let d16 = PixelFormat::Depth16Unorm;
    let pipeline = probe.pipeline(pipe);
    let state = probe.state(CompareFunction::Always, true, None);
    // Each case's `[z, slope]`, which the fixture's vertex function reads at its offset.
    let bytes: Vec<u8> = cases
        .iter()
        .flat_map(|&(z, _, _)| [z.to_le_bytes(), 0f32.to_le_bytes()].concat())
        .collect();
    let depths = buffer_with(&probe.device, &bytes).unwrap();
    let stored = probe.run(PASS.size(SIDE).depth(d16, 1.0), |_, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_depth_stencil_state(&state).unwrap();
        for (at, &(_, constant, slope)) in (0u64..).zip(cases) {
            e.set_scissor_rect(ScissorRect::new(at % SIDE, at / SIDE, 1, 1))
                .unwrap();
            e.set_depth_bias(constant, slope, 0.0).unwrap();
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

/// The cases of a probe's grid, `(depth, constant, slope factor, Metal's code)`.
fn grid<const N: usize>(factors: &[f32; N], rows: &[(u32, [u16; N])]) -> Vec<(f32, f32, f32, u16)> {
    rows.iter()
        .flat_map(|&(z, codes)| {
            factors
                .iter()
                .zip(codes)
                .map(move |(&c, code)| (f32::from_bits(z), c, 0.0, code))
        })
        .collect()
}

/// The kinds `unorm_depth_bias` takes: a constant bias on a 16-bit unorm depth the device
/// converts where the profile lacks `DriverDeviation::MetalDepth16BiasedStores`, past a fragment
/// stage that cannot add it (one that declares early fragment tests).
pub(super) const UNORM_DEPTH_BIAS: [Degradation; 1] = [Degradation::UnormDepthBias];

/// The kinds `unorm_depth_bias` takes: a 16-bit unorm depth the device converts its own way where
/// the profile lacks `DriverDeviation::MetalDepth16Stores`, here a depth clamped to a viewport
/// range's end inside `(0, 1)` past a fragment stage that writes Metal's codes.
pub(super) const UNORM_DEPTH_ROUNDING: [Degradation; 1] = [Degradation::UnormDepthRounding];

/// The kinds `unorm_depth_bias` takes where the fragment stage writes Metal's 16-bit codes: a
/// depth tested or written so, and a slope it takes from the quad's depths.
pub(super) const SHADER_UNORM_DEPTH: [Degradation; 2] =
    [Degradation::ShaderUnormDepth, Degradation::UnormDepthSlope];

/// Every code probes 7 and 7b and [`SLOPE_ALONE`] measured is Metal's rule's ([`metal_code`]),
/// and every row stores Metal's code. The device converts an unbiased row (a constant and a slope
/// factor of 0) where the profile takes `DriverDeviation::MetalDepth16Stores`, and a biased one
/// where it also takes `DriverDeviation::MetalDepth16BiasedStores`; elsewhere the fragment stage
/// writes the row's code, and the draws announce `shader-unorm-depth` and, for the sloped rows,
/// `unorm-depth-slope` once. No draw announces `unorm-depth-bias`. Either way a pipeline that
/// names no depth stores the same codes: the pass's depth decides them. Where the fragment stage
/// writes a biased row's code, every row whose bias has a known side
/// (`metal::state::depth16_bias_side`) runs the variant that declares it, counted
/// (`WorkCounts::depth16_sided_draws`), and stores Metal's code all the same.
#[test]
fn a_constant_bias_on_a_16_bit_unorm_depth_stores_metals_codes() {
    let slope_alone: Vec<_> = SLOPE_ALONE
        .iter()
        .map(|&(z, s, code)| (f32::from_bits(z), 0.0, s, code))
        .collect();
    let probes = [
        ("probe 7", grid(&PROBE_7_FACTORS, &PROBE_7)),
        ("probe 7b", grid(&PROBE_7B_FACTORS, &PROBE_7B)),
        ("a slope alone", slope_alone),
    ];
    let unruled: Vec<_> = probes
        .iter()
        .flat_map(|(_, cases)| cases)
        .filter(|&&(z, c, s, code)| metal_code(z, c, s) != code)
        .collect();
    assert!(unruled.is_empty(), "codes off Metal's rule: {unruled:?}");
    let (messages, _cache, probe) = validated();
    let device = probe.device.name();
    let profile = probe.device.compile_device().profile();
    let stores = profile.takes_driver_deviation(DriverDeviation::MetalDepth16Stores);
    let biased_stores = profile.takes_driver_deviation(DriverDeviation::MetalDepth16BiasedStores);
    // The fragment stage writes Metal's codes, its bias included, wherever the device does not:
    // every row's where it converts no depth so, a biased row's where it converts only unbiased
    // ones so.
    let shader = !stores || !biased_stores;
    let d16 = WHITE.depth(PixelFormat::Depth16Unorm);
    let sided_before = probe.device.work_counts().depth16_sided_draws;
    let mut sided_rows = 0u64;
    let mut differ = Vec::new();
    for (name, cases) in &probes {
        let drawn: Vec<(f32, f32, f32)> = cases.iter().map(|&(z, c, s, _)| (z, c, s)).collect();
        sided_rows += drawn
            .iter()
            .filter(|&&(_, c, s)| {
                metal2vulkan::metal::state::depth16_bias_side(c, s, 0.0).is_some()
            })
            .count() as u64;
        let stored = stored_codes(&probe, d16, &drawn);
        let unnamed = stored_codes(&probe, WHITE, &drawn);
        assert_eq!(
            unnamed, stored,
            "{name} on {device}: a pipeline that names no depth stores other codes"
        );
        let off: Vec<String> = cases
            .iter()
            .zip(stored)
            .filter(|&(&(.., metal), stored)| stored != metal)
            .map(|(&(z, c, s, metal), stored)| {
                format!(
                    "{name} {:#010x} by {c} and slope {s}: {stored}, Metal {metal}",
                    z.to_bits()
                )
            })
            .collect();
        eprintln!(
            "UNORM-DEPTH-BIAS {device} {name}: {} cases, {} off Metal's",
            cases.len(),
            off.len()
        );
        differ.extend(off);
    }
    // Each row is drawn twice: by the pipeline that names the depth and by the one that names
    // none.
    let sided = probe.device.work_counts().depth16_sided_draws - sided_before;
    eprintln!("UNORM-DEPTH-SIDED {device}: {sided} draws declared their bias's side");
    assert_eq!(
        sided,
        if shader { 2 * sided_rows } else { 0 },
        "{device}: draws of a known side through the variant that declares it"
    );
    let mut kinds = Vec::new();
    let mut quiet_kinds = Vec::new();
    for (taken, list) in [
        (false, &UNORM_DEPTH_BIAS[..]),
        (false, &UNORM_DEPTH_ROUNDING[..]),
        (shader, &SHADER_UNORM_DEPTH[..]),
    ] {
        match taken {
            true => kinds.extend(list),
            false => quiet_kinds.extend(list),
        }
    }
    // The draws under constant biases move their depth (`biased-depth-past-unit-range` where the
    // device stores it unclamped; a fragment stage that adds the bias clamps it to `[0, 1]`),
    // from a `Shared` buffer, through the white pipelines.
    let mut first = forced(&[
        (
            Degradation::BiasedDepthPastUnitRange,
            unclamped_bias(&probe.device) && !shader,
        ),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(&probe.device),
        ),
    ]);
    first.extend(probe.made(d16));
    first.extend(probe.made(WHITE));
    let announced = assert_made_once(&messages, &kinds, &first);
    let unexpected: Vec<_> = quiet_kinds
        .iter()
        .filter(|kind| announced.contains_key(kind))
        .collect();
    assert!(
        unexpected.is_empty(),
        "announced {unexpected:?} where Metal's codes are stored"
    );
    quiet(probe, &messages);
    assert!(differ.is_empty(), "codes not Metal's: {differ:#?}");
}

/// A draw in depth clamp mode through a viewport depth range whose near end lies inside `(0, 1)`
/// clamps a depth below it to that end, which the device converts to its code its own way past a
/// fragment stage that writes Metal's codes: such a draw announces `unorm-depth-rounding` once
/// there (and `shader-unorm-depth`), and neither where the profile takes
/// `DriverDeviation::MetalDepth16Stores`. Skipped, printed, on a device without `depthClamp`.
#[test]
fn a_depth_clamped_to_a_range_end_inside_the_unit_range_is_the_device_s_conversion() {
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    let profile = probe.device.compile_device().profile();
    if !profile.enabled().features.contains(&VkFeature::DepthClamp) {
        eprintln!("UNORM-DEPTH-CLAMP {device}: skipped, the device has no depthClamp");
        return;
    }
    let shader = !profile.takes_driver_deviation(DriverDeviation::MetalDepth16Stores);
    let d16 = PixelFormat::Depth16Unorm;
    let pipe = WHITE.depth(d16);
    let stored = probe.run(PASS.depth(d16, 1.0), |p, e, h| {
        e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, None))
            .unwrap();
        e.set_viewport(Viewport::new(0.0, 0.0, 4.0, 4.0, 0.25, 0.75))
            .unwrap();
        e.set_depth_clip_mode(DepthClipMode::Clamp).unwrap();
        // A normalized depth of -0.5: 0 in the window, clamped to 0.25.
        p.draw(e, h, pipe, -0.5, 0.0);
    });
    let codes: Vec<u16> = stored
        .depth
        .unwrap_or_default()
        .chunks(2)
        .map(|t| u16::from_le_bytes([t[0], t[1]]))
        .collect();
    eprintln!("UNORM-DEPTH-CLAMP {device}: stored {codes:?}");
    let kinds: Vec<Degradation> = match shader {
        true => vec![
            Degradation::ShaderUnormDepth,
            Degradation::UnormDepthRounding,
        ],
        false => Vec::new(),
    };
    let mut first = forced(&[(
        Degradation::SharedInSystemMemory,
        shared_in_system_memory(&probe.device),
    )]);
    first.extend(probe.made(pipe));
    let announced = assert_made_once(&messages, &kinds, &first);
    assert!(
        shader || !announced.contains_key(&Degradation::UnormDepthRounding),
        "announced unorm-depth-rounding on {device}, whose profile takes the deviation"
    );
    quiet(probe, &messages);
}

/// A draw biasing a `Depth16Unorm` depth through a fragment function that declares early
/// fragment tests (the owned fixture `cfb_depth_never`'s `cfb_dn_red_early`, a full-screen
/// triangle at depth 0.5) keeps the device's conversion, as no fragment stage that tests first
/// can write the depth: it announces `unorm-depth-bias` once where the profile lacks
/// `DriverDeviation::MetalDepth16BiasedStores`, and `unorm-depth-rounding` where it lacks
/// `DriverDeviation::MetalDepth16Stores`. The stored code is printed.
#[test]
fn a_bias_a_fragment_stage_testing_first_cannot_add_is_the_device_s() {
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    let profile = probe.device.compile_device().profile();
    let stores = profile.takes_driver_deviation(DriverDeviation::MetalDepth16Stores);
    let biased_stores = profile.takes_driver_deviation(DriverDeviation::MetalDepth16BiasedStores);
    let d16 = PixelFormat::Depth16Unorm;
    let bitcode = llvm_assemble(&engine_dir().join("cfb_depth_never.ll")).unwrap();
    let library = probe.device.new_library_with_air(&bitcode).unwrap();
    let function = |name| library.new_function_with_name(name).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        function("cfb_dn_full").function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        function("cfb_dn_red_early").function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.depth_format = Some(d16);
    let pipeline = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let stored = probe.run(PASS.depth(d16, 1.0), |p, e, _| {
        e.set_render_pipeline_state(&pipeline).unwrap();
        e.set_depth_stencil_state(&p.state(CompareFunction::Always, true, None))
            .unwrap();
        e.set_depth_bias(2.0, 0.0, 0.0).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    });
    let code = stored.depth.unwrap()[..2].to_vec();
    eprintln!(
        "UNORM-DEPTH-EARLY {device}: stored {}, Metal {}",
        u16::from_le_bytes([code[0], code[1]]),
        metal_code(0.5, 2.0, 0.0)
    );
    let kinds: Vec<Degradation> = [
        (Degradation::UnormDepthBias, !biased_stores),
        (Degradation::UnormDepthRounding, !stores),
    ]
    .into_iter()
    .filter_map(|(kind, taken)| taken.then_some(kind))
    .collect();
    let mut first = forced(&[(
        Degradation::BiasedDepthPastUnitRange,
        unclamped_bias(&probe.device),
    )]);
    first.extend(made_render_kinds(&probe.device, &descriptor));
    assert_made_once(&messages, &kinds, &first);
    quiet(probe, &messages);
}

/// The side of the cost test's pass: about a 1080p frame's pixels.
const COST_SIDE: u64 = 1440;

/// The full-screen layers the cost test draws front to back.
const LAYERS: u16 = 16;

/// Command buffers per form of the cost test, of which the median GPU time is printed.
const TIMED: usize = 15;

/// What a fragment stage that writes Metal's 16-bit codes costs against the device's own
/// conversion: [`LAYERS`] full-screen layers drawn front to back (the nearest first, `Less`,
/// written) into a [`COST_SIDE`]² pass, by the white pipeline into a `Depth16Unorm` depth (whose
/// fragment stage writes the codes where the profile lacks `DriverDeviation::MetalDepth16Stores`,
/// or for the biased draws `DriverDeviation::MetalDepth16BiasedStores`, so every layer runs it
/// before the test, but where it declares the bias's side, which a constant of 2 raises:
/// `WorkCounts::depth16_sided_draws`) and into a `Depth32Float` one (which the device tests before
/// the stage runs), unbiased and under a constant bias of 2; each the median GPU time of its
/// command buffers after one untimed, printed with the form. Nothing about the times is asserted.
#[test]
fn the_shader_unorm_depth_costs_against_a_float_depth() {
    let (messages, _cache, probe) = validated();
    let device = probe.device.name().to_owned();
    let profile = probe.device.compile_device().profile();
    let form = |deviation| match profile.takes_driver_deviation(deviation) {
        true => "the device's conversion",
        false => "the fragment stage's codes",
    };
    let target = |pf: PixelFormat| {
        let mut d = TextureDescriptor::texture_2d(pf, COST_SIDE, COST_SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        probe.device.new_texture(&d).unwrap()
    };
    let color = target(PixelFormat::RGBA8Unorm);
    let state = probe.state(CompareFunction::Less, true, None);
    let mut lines = Vec::new();
    for depth_format in [PixelFormat::Depth16Unorm, PixelFormat::Depth32Float] {
        let depth = target(depth_format);
        let pipe = WHITE.depth(depth_format);
        for constant in [0.0f32, 2.0] {
            let sided_before = probe.device.work_counts().depth16_sided_draws;
            let mut times = Vec::with_capacity(TIMED);
            for run in 0..=TIMED {
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.color_attachments[0];
                attachment.texture = Some(color.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                pass.depth_attachment.texture = Some(depth.clone());
                pass.depth_attachment.load_action = LoadAction::Clear;
                pass.depth_attachment.clear_depth = 1.0;
                pass.depth_attachment.store_action = StoreAction::Store;
                let command_buffer = probe.queue.command_buffer().unwrap();
                let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
                let mut held = Vec::new();
                encoder.set_depth_stencil_state(&state).unwrap();
                encoder.set_depth_bias(constant, 0.0, 0.0).unwrap();
                for layer in 0..LAYERS {
                    let z = 0.1 + f32::from(layer) * 0.05;
                    probe.draw(&mut encoder, &mut held, pipe, z, 0.0);
                }
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer
                    .wait_until_completed_for(super::WAIT)
                    .unwrap();
                assert_eq!(command_buffer.error(), None);
                if run > 0 {
                    let start = command_buffer.gpu_start_time().unwrap();
                    let end = command_buffer.gpu_end_time().unwrap();
                    times.push((end - start) * 1e6);
                }
            }
            times.sort_by(f64::total_cmp);
            let deviation = match constant == 0.0 {
                true => DriverDeviation::MetalDepth16Stores,
                false => DriverDeviation::MetalDepth16BiasedStores,
            };
            // A stage that declares its bias's side (`WorkCounts::depth16_sided_draws`) lets the
            // device test the depth before it runs.
            let sided = probe.device.work_counts().depth16_sided_draws > sided_before;
            let through = match depth_format {
                PixelFormat::Depth16Unorm if sided => {
                    "the fragment stage's codes, its side declared"
                }
                PixelFormat::Depth16Unorm => form(deviation),
                _ => "the device's test",
            };
            lines.push(format!(
                "{depth_format:?} biased by {constant} ({through}): {:.0} us",
                times[TIMED / 2]
            ));
        }
    }
    eprintln!(
        "UNORM-DEPTH-COST {device}: {COST_SIDE}x{COST_SIDE}, {LAYERS} layers front to back, \
         median GPU time of {TIMED}: {}",
        lines.join(", ")
    );
    quiet(probe, &messages);
}
