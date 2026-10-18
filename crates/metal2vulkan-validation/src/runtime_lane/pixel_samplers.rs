//! Samples through pixel-coordinate samplers (`PrecisionNote::PixelSamplerScaling`; the consumer's
//! ask of 2026-10-09) on the device the cfg selects, under the validation layer, against plain
//! Metal's answers.
//!
//! Metal's answers (our own probe, an Apple M4 under macOS 27.0.1, two processes,
//! identical): a 5x3 `RGBA8Unorm` and a 5x3 `RGBA16Float` texture, sampled at level 0 on a grid of
//! 80 x 10 coordinates in and outside the texture (each texel edge and centre, one ulp either
//! side, quarter and 1/512 steps; [`grid`]) through a runtime sampler state of
//! `normalizedCoordinates` NO at a sampler slot (`slot`) and in an argument buffer beside its
//! texture (`arg`), and through a constexpr `coord::pixel` sampler (`const`); nearest and linear
//! filters; clamp to edge, to zero, to an opaque white border, and (runtime states) mirror clamp
//! to edge. Each configuration's `float4` results hash ([`METAL`]); a configuration names its
//! format, source, filter and address mode. Metal answered each `slot` and `arg` configuration
//! exactly as the `norm` control: an argument buffer's normalized state of the same filter and
//! address mode sampled at the coordinate divided by the size. A runtime state's clamp to zero
//! reads its border color, here opaque white, as clamp to the border does, and a constexpr one
//! transparent black.
//!
//! Each configuration here runs the same grid through the runtime: the hash of its results must
//! be Metal's, and each pixel configuration's results must equal the device's own `norm` control's
//! cell by cell, which names the cells that differ. On a device with `VkProperty::MetalSampler`
//! none announces `pixel-sampler-scaling`, and an argument buffer's pixel sampler announces
//! `argument-pixel-sampler-scaling` only on a device without it, at the first dispatch while a
//! pixel state with a linear filter lives
//! ([`the_argument_buffer_note_waits_for_a_filtering_pixel_state`]).

use super::argument_linear_samples::kernel;
use super::degrade_notices::counts;
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::extended_range_targets::half_bits;
use super::{validated_device, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::device::VkProperty;
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Degradation, Device, Region, SamplerState, SamplerStateOptions, Texture,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// The textures' size: neither side a power of two.
const W: u64 = 5;
const H: u64 = 3;

/// Plain Metal's answers: each configuration's results hashed (the module docs).
const METAL: [(&str, &str); 60] = [
    (
        "rgba8_slot_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_arg_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_norm_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_const_nearest_edge",
        "250024876cde0c2f67dccea354f400d28c94b6cd9634aae0b38c6354f35070b7",
    ),
    (
        "rgba8_slot_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_arg_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_norm_nearest_zero",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_const_nearest_zero",
        "4ce73456509dc569e5a67d724ecd5f2bea5dda4d8b7279b5f8e4a7351998c5be",
    ),
    (
        "rgba8_slot_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_arg_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_norm_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_const_nearest_border",
        "a0a4cd79e2609250f0131240bdb85e230771e5052b8ae7ed2f9af30c4a901a72",
    ),
    (
        "rgba8_slot_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_arg_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_norm_nearest_mirror",
        "27c2ee7d7782b153e1e9daa8f2e37dba6ad073e4dc1f08b6350634a0b66fdf32",
    ),
    (
        "rgba8_slot_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_arg_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_norm_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_const_linear_edge",
        "4c832fb335efa8b6f389e8a2b8a6721efff8bef2d0d933df0670336ee3841051",
    ),
    (
        "rgba8_slot_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_arg_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_norm_linear_zero",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_const_linear_zero",
        "8c8ebb90b930331d3790933171428b8edff1c8296f8bb344a6c06362dc382c83",
    ),
    (
        "rgba8_slot_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_arg_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_norm_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_const_linear_border",
        "8640b83f43b188201afa3e7ed424318caa473f82511150f87921d8ac871eb157",
    ),
    (
        "rgba8_slot_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba8_arg_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba8_norm_linear_mirror",
        "26940521965a1456adaba62f9dc004a9f27967fd355737d6efdca7c46209b80b",
    ),
    (
        "rgba16f_slot_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_arg_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_norm_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_const_nearest_edge",
        "e28b6390ff83ff95e1c5d00dae0fcf2f7b51ad9c20c68e88223e25cfc18700bd",
    ),
    (
        "rgba16f_slot_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_arg_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_norm_nearest_zero",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_const_nearest_zero",
        "7b239607fcf4299166fc92e5300ea888e7224f8a7ed32d7e5265cb78cc62657e",
    ),
    (
        "rgba16f_slot_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_arg_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_norm_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_const_nearest_border",
        "3288187d8ade9557ef4b727bf31750abd8136d15c98b6d29b17ccf6423c0d497",
    ),
    (
        "rgba16f_slot_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_arg_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_norm_nearest_mirror",
        "fb1a84e8f744e0bd92aab91af807fe789fdf105b5ae8bd526fb427ef0d04409d",
    ),
    (
        "rgba16f_slot_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_arg_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_norm_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_const_linear_edge",
        "7a6a5f64bc4531517ba0d4d4912c2d687ee0779777af9376c02b5e09f4af2421",
    ),
    (
        "rgba16f_slot_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_arg_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_norm_linear_zero",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_const_linear_zero",
        "dbb87cfd8c4ab3c54dede294e5ebc42ad0a2fd84166cc1b46a33e3a3b61e1777",
    ),
    (
        "rgba16f_slot_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_arg_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_norm_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_const_linear_border",
        "a33152229f2a9e0a2cedd5edd958a262c214bd5b4a682110837f3523a6fe5deb",
    ),
    (
        "rgba16f_slot_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
    (
        "rgba16f_arg_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
    (
        "rgba16f_norm_linear_mirror",
        "24724e535ae68eaccd7460b0c8948ce18d3118bb92395b690d7c8918dc521e06",
    ),
];

/// The probe's coordinates, `x` fastest: per integer `i` from -1 to 6 the texel edge `i` and one
/// ulp either side (`2^-24` at 0), `i + 1/512`, `i + 0.25`, the centre `i + 0.5` and one ulp
/// either side, `i + 0.75` and `i + 255/256`; `y` from a list of ten across and past the 3 rows.
fn grid() -> Vec<[f32; 2]> {
    let mut xs = Vec::new();
    for i in -1..=(W as i32 + 1) {
        let (b, h) = (i as f32, i as f32 + 0.5);
        let (lo, hi) = match i {
            0 => (-(2f32.powi(-24)), 2f32.powi(-24)),
            _ => (b.next_down(), b.next_up()),
        };
        xs.extend([
            lo,
            b,
            hi,
            b + 2f32.powi(-9),
            b + 0.25,
            h.next_down(),
            h,
            h.next_up(),
            b + 0.75,
            b + 255.0 / 256.0,
        ]);
    }
    let ys = [
        -0.5,
        0.0,
        0.5,
        1f32.next_down(),
        1.0,
        1.25,
        1.5f32.next_up(),
        2.5,
        3.0,
        3.25,
    ];
    ys.iter()
        .flat_map(|&y| xs.iter().map(move |&x| [x, y]))
        .collect()
}

/// The texture of `format`, its texel `(x, y)` channel `c` the probe's.
fn texture(device: &Device, format: PixelFormat) -> Texture {
    let texture = device
        .new_texture(&TextureDescriptor::texture_2d(format, W, H, false))
        .unwrap();
    let mut bytes = Vec::new();
    for y in 0..H {
        for x in 0..W {
            for c in 0..4 {
                match format {
                    PixelFormat::RGBA8Unorm => {
                        bytes.push(((x * 53 + y * 97 + c * 29 + 11) % 256) as u8)
                    }
                    _ => {
                        let k = ((x * 7 + y * 13 + c * 5 + 3) % 29) as i32 - 14;
                        bytes.extend(half_bits(k as f32 / 4.0).to_le_bytes());
                    }
                }
            }
        }
    }
    let row = W * if format == PixelFormat::RGBA8Unorm {
        4
    } else {
        8
    };
    texture
        .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, row, 0)
        .unwrap();
    texture
}

/// An address mode: the probe's name, the runtime's mode, and a constexpr sampler's code and
/// border color (none for mirror clamp to edge, which a constexpr sampler cannot state).
type Address = (&'static str, SamplerAddressMode, Option<(u64, u64)>);

/// The address modes sampled.
const ADDRESSES: [Address; 4] = [
    ("edge", SamplerAddressMode::ClampToEdge, Some((1, 0))),
    ("zero", SamplerAddressMode::ClampToZero, Some((0, 0))),
    (
        "border",
        SamplerAddressMode::ClampToBorderColor,
        Some((0, 2)),
    ),
    ("mirror", SamplerAddressMode::MirrorClampToEdge, None),
];

/// A runtime sampler state of `filter` and `address` on every axis, an opaque white border, and
/// argument-buffer support, of pixel coordinates unless `normalized`.
fn sampler(
    device: &Device,
    filter: SamplerMinMagFilter,
    address: SamplerAddressMode,
    normalized: bool,
) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    d.s_address_mode = address;
    d.t_address_mode = address;
    d.r_address_mode = address;
    d.border_color = SamplerBorderColor::OpaqueWhite;
    d.normalized_coordinates = normalized;
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    device.new_sampler_state_with_options(&d, options).unwrap()
}

/// A constexpr `coord::pixel` sampler's words: address `code` on every axis, `linear` filters or
/// nearest, no mip filter, no comparison, the frontend's LOD clamps, border color `border`.
fn pixel_words(code: u64, linear: bool, border: u64) -> u64 {
    let filters = if linear { 1 << 9 | 1 << 11 } else { 0 };
    (code | code << 3 | code << 6) | filters | 1 << 15 | 8 << 16 | 0x7bff << 40 | border << 56
}

/// The kernels sampling a slot's texture through runtime sampler slot 0, and an argument
/// buffer's texture through its sampler.
struct Kernels {
    slot: ComputePipelineState,
    argument: ComputePipelineState,
}

/// A configuration's results beside Metal's.
struct Outcome {
    /// The configuration's name (the module docs).
    name: String,
    /// Its source: `slot`, `arg`, `norm` or `const`.
    source: &'static str,
    /// Whether its results hash as Metal's.
    metal: bool,
    /// Its hash, and its first cells that differ from the device's `norm` control (none for the
    /// control itself).
    differs: Vec<String>,
}

/// Every configuration's results on `device` beside Metal's.
fn compare(device: &Device) -> Vec<Outcome> {
    let make = |heap: bool, through: Through| {
        pipeline_of(
            device,
            &llvm_assemble_text(&kernel(heap, through, false, false)).unwrap(),
            "k",
        )
    };
    let kernels = Kernels {
        slot: make(false, Through::Runtime),
        argument: make(true, Through::Argument),
    };
    let metal: BTreeMap<&str, &str> = METAL.into_iter().collect();
    let cells = grid();
    let jobs = |scale: bool| -> Vec<[f32; 4]> {
        cells
            .iter()
            .map(|&[x, y]| match scale {
                true => [x / W as f32, y / H as f32, 0.0, 0.0],
                false => [x, y, 0.0, 0.0],
            })
            .collect()
    };
    let (pixel, normalized) = (jobs(false), jobs(true));
    let n = cells.len();
    let run = |texture: &Texture,
               held: Held<'_>,
               sampler: Option<&SamplerState>,
               state: &ComputePipelineState,
               jobs: &[[f32; 4]]| {
        let mut out = run_held(device, (texture, held, sampler), state, jobs, 1);
        out.truncate(4 * n);
        out
    };
    let mut outcomes = Vec::new();
    for (format, fname) in [
        (PixelFormat::RGBA8Unorm, "rgba8"),
        (PixelFormat::RGBA16Float, "rgba16f"),
    ] {
        let texture = texture(device, format);
        for (filter, lname) in [
            (SamplerMinMagFilter::Nearest, "nearest"),
            (SamplerMinMagFilter::Linear, "linear"),
        ] {
            for &(aname, address, constexpr) in &ADDRESSES {
                let pixel_state = sampler(device, filter, address, false);
                let norm_state = sampler(device, filter, address, true);
                let norm = run(
                    &texture,
                    Held::Argument(Some(&norm_state)),
                    None,
                    &kernels.argument,
                    &normalized,
                );
                let mut results = vec![
                    ("norm", norm.clone()),
                    (
                        "slot",
                        run(
                            &texture,
                            Held::Slot,
                            Some(&pixel_state),
                            &kernels.slot,
                            &pixel,
                        ),
                    ),
                    (
                        "arg",
                        run(
                            &texture,
                            Held::Argument(Some(&pixel_state)),
                            None,
                            &kernels.argument,
                            &pixel,
                        ),
                    ),
                ];
                if let Some((code, border)) = constexpr {
                    let words = pixel_words(code, filter == SamplerMinMagFilter::Linear, border);
                    let state = make(false, Through::Constexpr(words));
                    results.push(("const", run(&texture, Held::Slot, None, &state, &pixel)));
                }
                for (source, out) in results {
                    let name = format!("{fname}_{source}_{lname}_{aname}");
                    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let hash = format!("{:x}", Sha256::digest(&bytes));
                    let metal = metal.get(name.as_str()) == Some(&hash.as_str());
                    // The cells that differ from the `norm` control, which Metal answered alike
                    // but where a constexpr sampler clamps to zero.
                    let mut cells_wrong = Vec::new();
                    for (j, &[x, y]) in cells.iter().enumerate() {
                        let (got, control) = (&out[4 * j..4 * j + 4], &norm[4 * j..4 * j + 4]);
                        if got
                            .iter()
                            .zip(control)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                            && cells_wrong.len() < 6
                        {
                            cells_wrong.push(format!(
                                "({x:e}, {y:e}): {got:?}, the normalized control {control:?}"
                            ));
                        }
                    }
                    if !metal {
                        cells_wrong.insert(0, format!("hash {hash}"));
                    }
                    outcomes.push(Outcome {
                        name,
                        source,
                        metal,
                        differs: cells_wrong,
                    });
                }
            }
        }
    }
    outcomes
}

/// Every configuration answers Metal's results (the module docs), but where the device announced
/// `float-sample-filter` (no `VkProperty::MetalSampler`): there an argument buffer's texture
/// sampled with a `float` result keeps the device's conversion of its texels (an `RGBA8Unorm`
/// code `k` read as other than `k / 255` rounded once, lavapipe), so a `norm` configuration may
/// differ from Metal's, and an `arg` one only as its `norm` control does, cell by cell: its pixel
/// coordinates still read Metal's texels.
#[test]
fn pixel_samplers_answer_metal_s_texels() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (outcomes, metal_sampler) = {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("pixel samplers on {}", device.name());
        (compare(&device), metal_sampler(&device))
    };
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    let announced: BTreeSet<Degradation> = counts(&messages.take_notices()).into_keys().collect();
    eprintln!("announced: {announced:?}");
    let converted = announced.contains(&Degradation::FloatSampleFilter);
    // A device with Metal's own sampler reads each configuration's texels exactly; elsewhere a
    // linear pixel sample the emulation does not reach divides by the size (`OpFDiv`, Vulkan's
    // bound) and filters on the device, which keeps the note.
    if metal_sampler {
        assert!(!announced.contains(&Degradation::PixelSamplerScaling));
    }
    assert_eq!(
        announced.contains(&Degradation::ArgumentPixelSamplerScaling),
        !metal_sampler,
        "the argument buffer's pixel note where the device lacks Metal's sampler"
    );
    assert_eq!(outcomes.len(), METAL.len());
    let mut allowed = Vec::new();
    let mut wrong = BTreeMap::new();
    for o in outcomes {
        if o.metal {
            continue;
        }
        let pixel_exact = match o.source {
            "norm" => true,
            "arg" => o.differs.len() == 1,
            _ => false,
        };
        if converted && pixel_exact {
            allowed.push(o.name);
        } else {
            wrong.insert(o.name, o.differs);
        }
    }
    eprintln!("the device's conversion under float-sample-filter: {allowed:?}");
    assert!(
        wrong.is_empty(),
        "{} of {} configurations are not Metal's: {wrong:#?}",
        wrong.len(),
        METAL.len()
    );
}

/// Whether `device` samples through Metal's own sampler (`VkProperty::MetalSampler`).
fn metal_sampler(device: &Device) -> bool {
    device
        .compile_device()
        .profile()
        .enabled()
        .properties
        .contains(&VkProperty::MetalSampler)
}

/// `argument-pixel-sampler-scaling` fires at a dispatch, never at a state's making: an argument
/// buffer's linear sample, dispatched while the device holds a normalized linear state and a
/// nearest pixel state, announces nothing; making a pixel state of linear filters announces
/// nothing; the next dispatch, through the normalized state still, announces it on a device without
/// `VkProperty::MetalSampler`, where the kernel records the note, and nowhere else.
#[test]
fn the_argument_buffer_note_waits_for_a_filtering_pixel_state() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let kind = Degradation::ArgumentPixelSamplerScaling;
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let argument = pipeline_of(
            &device,
            &llvm_assemble_text(&kernel(true, Through::Argument, false, false)).unwrap(),
            "k",
        );
        let texture = texture(&device, PixelFormat::RGBA8Unorm);
        let jobs: Vec<[f32; 4]> = grid()
            .iter()
            .take(64)
            .map(|&[x, y]| [x / W as f32, y / H as f32, 0.0, 0.0])
            .collect();
        let normalized = sampler(
            &device,
            SamplerMinMagFilter::Linear,
            SamplerAddressMode::ClampToEdge,
            true,
        );
        let dispatch = |state: &SamplerState| {
            run_held(
                &device,
                (&texture, Held::Argument(Some(state)), None),
                &argument,
                &jobs,
                1,
            );
        };
        let nearest = sampler(
            &device,
            SamplerMinMagFilter::Nearest,
            SamplerAddressMode::ClampToEdge,
            false,
        );
        dispatch(&normalized);
        dispatch(&nearest);
        let before = counts(&messages.take_notices());
        assert!(!before.contains_key(&kind), "{before:?}");
        let linear = sampler(
            &device,
            SamplerMinMagFilter::Linear,
            SamplerAddressMode::ClampToEdge,
            false,
        );
        let made = counts(&messages.take_notices());
        assert!(!made.contains_key(&kind), "{made:?}");
        dispatch(&normalized);
        let after = counts(&messages.take_notices());
        assert_eq!(
            after.get(&kind).copied(),
            (!metal_sampler(&device)).then_some(1),
            "{after:?}"
        );
        drop(linear);
    }
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}
