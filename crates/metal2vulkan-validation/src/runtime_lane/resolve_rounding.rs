//! Metal's color resolve arithmetic over every input it measured, on the device the cfg selects,
//! under the validation layer.
//!
//! Metal probes (an Apple M3 Max and an M4, macOS 26.7, byte-identical) resolved every pair of
//! 8-bit samples and hundreds of thousands of 2- and 4-sample texels per format, and closed each
//! format's average as one rule (every case measured):
//!
//! - a unorm channel of `b` bits: `(sum + n / 2) / n` over the integer codes (8, 10, 16 and the
//!   2-bit alpha);
//! - a snorm channel: `floor(sum / n + 1 / 2)` over the codes, `-2^(b-1)` read as
//!   `-(2^(b-1) - 1)` (both are -1.0);
//! - an sRGB channel: each code's linear value as Metal's 12-bit table holds it, their integer
//!   average `(sum + n / 2) / n`, and Metal's encode of that 12-bit value
//!   (`bind::SRGB_RESOLVE_LINEAR`, `bind::SRGB_RESOLVE_THRESHOLDS`);
//! - a 32-bit float channel: the samples summed in order in `f32`, each input, partial sum and the
//!   result times `1 / n` a denormal flushed to the zero of its sign, and a NaN anywhere the
//!   canonical `0x7fc00000`.
//!
//! Each case draws a multisampled texture sample by sample (the owned fixture `rt6_samples`'s
//! fill, each sample's code as the float that encodes it exactly), reads every sample back
//! through an unsigned view (`candidate::runtime_samples`, the oracle's readback), resolves it in
//! a pass of no draws that loads it, and compares every channel of the resolve texture with the
//! rule applied to the samples read back: every pair of codes of an 8-bit channel at 2 samples,
//! every pair as `(a, a, b, b)` and random texels at 4, and random texels with the edge values of
//! a wider channel. The runtime resolves every one of these formats by Metal's rules on every
//! device (its exact resolve, or a driver measured to average as Metal does), so every channel must
//! match. Metal has no texture of more than 4 samples on Apple silicon, so a resolve of 8 is
//! compared with the same rules and printed as unmeasured, never as Metal's. A device without
//! `imageViewFormatReinterpretation` (MoltenVK) views no texel as an unsigned format of other
//! components, so the formats only such a view reads back are printed as unread there. The runs
//! before the runtime resolved so (EVs `RPL-avg2-*` at 2b41d71d) measured each driver's own
//! average, which `DriverFact::ColorResolveAverageIsMetals` records.

use super::presenting::{image_info, made_image, shared_among};
use super::resolves::{lane, quiet, Lane, FULL, WHITE};
use crate::candidate::runtime_samples::{encode_through, library, shape};
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::runtime_lane::buffer_with;
use metal2vulkan::bind::{SRGB_RESOLVE_LINEAR, SRGB_RESOLVE_THRESHOLDS};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, LoadAction, PipelineOptions, RenderPassDescriptor,
    ResourceOptions, StoreAction, Texture, TextureDescriptor, VkImageLayouts,
};

/// How a channel's bits encode its value.
#[derive(Clone, Copy, Debug)]
enum Channel {
    Unorm(u32),
    Snorm(u32),
    Srgb,
    Float,
}

impl Channel {
    fn bits(self) -> u32 {
        match self {
            Channel::Unorm(bits) | Channel::Snorm(bits) => bits,
            Channel::Srgb => 8,
            Channel::Float => 32,
        }
    }
}

/// A format the probes closed: its texel bytes and each channel's bit offset in the texel, its
/// component (0 to 3 for R, G, B, A) and its encoding.
struct Case {
    pf: PixelFormat,
    bytes: usize,
    channels: &'static [(u32, usize, Channel)],
}

const U8: Channel = Channel::Unorm(8);
const S8: Channel = Channel::Snorm(8);
const U16: Channel = Channel::Unorm(16);
const S16: Channel = Channel::Snorm(16);
const U10: Channel = Channel::Unorm(10);
const U2: Channel = Channel::Unorm(2);
const SRGB: Channel = Channel::Srgb;
const F32: Channel = Channel::Float;

const fn case(pf: PixelFormat, bytes: usize, channels: &'static [(u32, usize, Channel)]) -> Case {
    Case {
        pf,
        bytes,
        channels,
    }
}

const CASES: &[Case] = &[
    case(PixelFormat::R8Unorm, 1, &[(0, 0, U8)]),
    case(PixelFormat::RG8Unorm, 2, &[(0, 0, U8), (8, 1, U8)]),
    case(
        PixelFormat::RGBA8Unorm,
        4,
        &[(0, 0, U8), (8, 1, U8), (16, 2, U8), (24, 3, U8)],
    ),
    case(
        PixelFormat::BGRA8Unorm,
        4,
        &[(0, 2, U8), (8, 1, U8), (16, 0, U8), (24, 3, U8)],
    ),
    case(PixelFormat::R8UnormSRGB, 1, &[(0, 0, SRGB)]),
    case(PixelFormat::RG8UnormSRGB, 2, &[(0, 0, SRGB), (8, 1, SRGB)]),
    case(
        PixelFormat::RGBA8UnormSRGB,
        4,
        &[(0, 0, SRGB), (8, 1, SRGB), (16, 2, SRGB), (24, 3, U8)],
    ),
    case(
        PixelFormat::BGRA8UnormSRGB,
        4,
        &[(0, 2, SRGB), (8, 1, SRGB), (16, 0, SRGB), (24, 3, U8)],
    ),
    case(PixelFormat::R8Snorm, 1, &[(0, 0, S8)]),
    case(PixelFormat::RG8Snorm, 2, &[(0, 0, S8), (8, 1, S8)]),
    case(
        PixelFormat::RGBA8Snorm,
        4,
        &[(0, 0, S8), (8, 1, S8), (16, 2, S8), (24, 3, S8)],
    ),
    case(PixelFormat::R16Unorm, 2, &[(0, 0, U16)]),
    case(PixelFormat::RG16Unorm, 4, &[(0, 0, U16), (16, 1, U16)]),
    case(
        PixelFormat::RGBA16Unorm,
        8,
        &[(0, 0, U16), (16, 1, U16), (32, 2, U16), (48, 3, U16)],
    ),
    case(PixelFormat::R16Snorm, 2, &[(0, 0, S16)]),
    case(PixelFormat::RG16Snorm, 4, &[(0, 0, S16), (16, 1, S16)]),
    case(
        PixelFormat::RGBA16Snorm,
        8,
        &[(0, 0, S16), (16, 1, S16), (32, 2, S16), (48, 3, S16)],
    ),
    case(
        PixelFormat::RGB10A2Unorm,
        4,
        &[(0, 0, U10), (10, 1, U10), (20, 2, U10), (30, 3, U2)],
    ),
    case(
        PixelFormat::BGR10A2Unorm,
        4,
        &[(0, 2, U10), (10, 1, U10), (20, 0, U10), (30, 3, U2)],
    ),
    case(PixelFormat::R32Float, 4, &[(0, 0, F32)]),
    case(PixelFormat::RG32Float, 8, &[(0, 0, F32), (32, 1, F32)]),
    case(
        PixelFormat::RGBA32Float,
        16,
        &[(0, 0, F32), (32, 1, F32), (64, 2, F32), (96, 3, F32)],
    ),
];

/// How a case's samples are read back: the unsigned view, the fixture's kernel that packs a texel
/// of it into words, and the readback's texel size. A view of the texel's own components where the
/// fixture packs one, else the unsigned format of the texel's size, which needs
/// `imageViewFormatReinterpretation`.
fn readback_view(case: &Case) -> (PixelFormat, &'static str, ReadbackView) {
    use PixelFormat as P;
    match case.pf {
        P::R8Unorm | P::R8UnormSRGB | P::R8Snorm => {
            (P::R8Uint, "rt6_read_words1", ReadbackView::R8Uint)
        }
        P::RGBA8Unorm | P::RGBA8UnormSRGB | P::RGBA8Snorm | P::BGRA8Unorm | P::BGRA8UnormSRGB => {
            (P::RGBA8Uint, "rt6_read_bytes4", ReadbackView::R32Uint)
        }
        P::R16Unorm | P::R16Snorm => (P::R16Uint, "rt6_read_words1", ReadbackView::R16Uint),
        P::RG16Unorm | P::RG16Snorm => (P::RG16Uint, "rt6_read_halves2", ReadbackView::R32Uint),
        P::RGBA16Unorm | P::RGBA16Snorm => {
            (P::RGBA16Uint, "rt6_read_halves4", ReadbackView::Rg32Uint)
        }
        P::R32Float => (P::R32Uint, "rt6_read_words1", ReadbackView::R32Uint),
        P::RG32Float => (P::RG32Uint, "rt6_read_words2", ReadbackView::Rg32Uint),
        P::RGBA32Float => (P::RGBA32Uint, "rt6_read_words4", ReadbackView::Rgba32Uint),
        _ => match case.bytes {
            2 => (P::R16Uint, "rt6_read_words1", ReadbackView::R16Uint),
            _ => (P::R32Uint, "rt6_read_words1", ReadbackView::R32Uint),
        },
    }
}

/// The float a fragment writes for `code` of `channel`: the value that encodes it exactly (an
/// sRGB code's linear value, a snorm code below -1.0 as -1.0, a float's own bits).
fn encoding(channel: Channel, code: u64) -> f32 {
    match channel {
        Channel::Unorm(bits) => (code as f64 / ((1u64 << bits) - 1) as f64) as f32,
        Channel::Snorm(bits) => {
            let half = 1i64 << (bits - 1);
            let v = code as i64;
            let v = if v >= half { v - 2 * half } else { v };
            (v.max(-(half - 1)) as f64 / (half - 1) as f64) as f32
        }
        Channel::Srgb => {
            let c = code as f64 / 255.0;
            let linear = if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            };
            linear as f32
        }
        Channel::Float => f32::from_bits(code as u32),
    }
}

/// SplitMix64, the probes' generator.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Float bits that reach every rule: denormals, zeros, infinities, NaNs, the largest finite, and
/// pairs of close values whose sum is a denormal.
fn float_bits(r: &mut SplitMix) -> u64 {
    let pick = r.next();
    let sign = (pick >> 8) & 1;
    let mantissa = r.next() & 0x7f_ffff;
    let bits = match pick % 16 {
        0 => mantissa.max(1),               // a denormal
        1 => 0,                             // zero
        2 => 0x7f80_0000,                   // infinity
        3 => 0x7f80_0000 | mantissa.max(1), // a NaN
        4 => 0x7f7f_ffff - (r.next() % 4),  // near the largest finite
        5 => 0x0080_0000 + (r.next() % 64), // near the smallest normal
        6 => (1 << 23) | mantissa,          // the smallest normal binade
        _ => r.next() % 0x7f80_0000,        // finite
    };
    (sign << 31) | bits
}

/// The value of channel `channel` of sample `s` of pixel `(x, y)` among `n`, on an image `width`
/// wide: an 8-bit or narrower channel takes every pair `(x, y)` (2 samples), and `(a, a, b, b)`
/// over every pair in the image's first 256 rows at more; the rest is random, wider integer
/// channels drawing their edge values a quarter of the time.
fn value(
    channel: Channel,
    index: usize,
    (x, y): (u32, u32),
    (s, n): (u32, u32),
    r: &mut SplitMix,
) -> u64 {
    let bits = channel.bits();
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let spread = |v: u32| -> u64 {
        // A bijection of the codes per channel, so channels differ.
        (u64::from(v).wrapping_mul(2 * index as u64 + 1) + 53 * index as u64) & mask
    };
    if bits <= 8 && y < 256 {
        let pick = if n == 2 {
            [x, y][s as usize]
        } else if s < n / 2 {
            x
        } else {
            y
        };
        return spread(pick) & mask;
    }
    match channel {
        Channel::Float => float_bits(r),
        _ if bits > 8 && r.next().is_multiple_of(4) => {
            let half = 1u64 << (bits - 1);
            let edges = [0, 1, half - 1, half, half + 1, mask - 1, mask];
            edges[(r.next() % edges.len() as u64) as usize]
        }
        _ => r.next() & mask,
    }
}

/// Metal's resolve of one channel's `samples` (the module docs).
fn metal(channel: Channel, samples: &[u64]) -> u64 {
    let n = samples.len() as i64;
    match channel {
        Channel::Unorm(_) => {
            let sum: i64 = samples.iter().map(|&s| s as i64).sum();
            ((sum + n / 2) / n) as u64
        }
        Channel::Snorm(bits) => {
            let half = 1i64 << (bits - 1);
            let sum: i64 = samples
                .iter()
                .map(|&s| {
                    let v = s as i64;
                    let v = if v >= half { v - 2 * half } else { v };
                    v.max(-(half - 1))
                })
                .sum();
            let resolved = (2 * sum + n).div_euclid(2 * n);
            (resolved as u64) & ((1u64 << bits) - 1)
        }
        Channel::Srgb => {
            let sum: i64 = samples
                .iter()
                .map(|&s| i64::from(SRGB_RESOLVE_LINEAR[s as usize]))
                .sum();
            let linear = ((sum + n / 2) / n) as u16;
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| t <= linear) as u64
        }
        Channel::Float => {
            let flush = |v: f32| {
                if v.is_subnormal() {
                    0.0f32.copysign(v)
                } else {
                    v
                }
            };
            let mut sum = flush(f32::from_bits(samples[0] as u32));
            for &s in &samples[1..] {
                sum = flush(sum + flush(f32::from_bits(s as u32)));
            }
            let resolved = flush(sum * (1.0 / n as f32));
            if resolved.is_nan() {
                0x7fc0_0000
            } else {
                u64::from(resolved.to_bits())
            }
        }
    }
}

/// The bits of `texel` (little-endian) at `offset`, `bits` wide.
fn field(texel: &[u8], offset: u32, bits: u32) -> u64 {
    let mut word = 0u128;
    for (i, &b) in texel.iter().enumerate() {
        word |= u128::from(b) << (8 * i);
    }
    ((word >> offset) as u64)
        & if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
}

/// `texel`'s bytes with `value` at `offset`, `bits` wide.
fn set_field(texel: &mut [u8], offset: u32, bits: u32, value: u64) {
    for bit in 0..bits {
        let at = offset + bit;
        let byte = &mut texel[(at / 8) as usize];
        let mask = 1u8 << (at % 8);
        if (value >> bit) & 1 == 1 {
            *byte |= mask;
        } else {
            *byte &= !mask;
        }
    }
}

/// What one case's resolve gave against Metal's rule.
struct Outcome {
    channels: usize,
    differ: usize,
    /// Samples the fill did not store as drawn (a float the device changed on the way).
    changed: usize,
    examples: Vec<String>,
}

/// Draw a `width` by `height` texture of `case` at `n` samples, read its samples back, resolve
/// it, and compare.
fn resolve_case(lane: &Lane, case: &Case, n: u32) -> Result<Outcome, String> {
    let (width, height) = (256u32, if n == 2 { 256u32 } else { 512u32 });
    let mut r = SplitMix(0x5250_4c00 ^ (u64::from(n) << 32) ^ case.pf as u64);
    let pixels = (width * height) as usize;
    let texels = pixels * n as usize;
    // Each sample's codes, as drawn, and the float4 that draws them.
    let mut drawn = vec![0u8; texels * case.bytes];
    let mut floats = Vec::with_capacity(texels * 16);
    for y in 0..height {
        for x in 0..width {
            let mut sample_floats = vec![[0f32; 4]; n as usize];
            for (index, &(offset, component, channel)) in case.channels.iter().enumerate() {
                for (s, lanes) in sample_floats.iter_mut().enumerate() {
                    let code = value(channel, index, (x, y), (s as u32, n), &mut r);
                    let at = (((y * width + x) * n) as usize + s) * case.bytes;
                    set_field(
                        &mut drawn[at..at + case.bytes],
                        offset,
                        channel.bits(),
                        code,
                    );
                    lanes[component] = encoding(channel, code);
                }
            }
            for lanes in sample_floats {
                for lane in lanes {
                    floats.extend_from_slice(&lane.to_le_bytes());
                }
            }
        }
    }
    let device = &lane.device;
    let mut descriptor = TextureDescriptor::texture_2d(case.pf, width.into(), height.into(), false);
    descriptor.texture_type = TextureType::Type2DMultisample;
    descriptor.sample_count = n.into();
    descriptor.usage =
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = ResourceOptions::private();
    let msaa = device
        .new_texture(&descriptor)
        .map_err(|e| format!("new_texture: {e}"))?;
    let (view_pf, kernel, readback) = readback_view(case);
    let view = msaa
        .new_texture_view(view_pf)
        .map_err(|e| format!("unread: no {view_pf:?} view ({e})"))?;
    let library = library(device).map_err(|e| e.to_string())?;
    let function = |name: &str| {
        library
            .new_function_with_name(name)
            .map(|f| f.function_descriptor())
            .map_err(|e| format!("{name}: {e}"))
    };
    let mut pipeline =
        RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")?));
    pipeline.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")?));
    pipeline.color_attachments[0] = Some(ColorAttachmentDescriptor::new(case.pf));
    pipeline.raster_sample_count = n;
    let pipeline = device
        .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
        .map_err(|e| format!("the fill's pipeline: {e}"))?;
    let values = buffer_with(device, &floats).map_err(|e| e.to_string())?;
    let shape = shape(device, width, n).map_err(|e| e.to_string())?;
    let mut single = TextureDescriptor::texture_2d(case.pf, width.into(), height.into(), false);
    single.usage = TextureUsage::RENDER_TARGET;
    single.resource_options = ResourceOptions::private();
    let resolved: Texture = device
        .new_texture(&single)
        .map_err(|e| format!("new_texture: {e}"))?;
    let command_buffer = lane.queue.command_buffer().map_err(|e| e.to_string())?;
    let mut fill = RenderPassDescriptor::new();
    let target = &mut fill.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = LoadAction::DontCare;
    target.store_action = StoreAction::Store;
    let mut encoder = command_buffer
        .render_command_encoder(&fill)
        .map_err(|e| format!("the fill pass: {e}"))?;
    let draw = |e: metal2vulkan_runtime::Error| format!("the fill: {e}");
    encoder.set_render_pipeline_state(&pipeline).map_err(draw)?;
    encoder
        .set_fragment_buffer(Some(&values), 0, 0)
        .map_err(draw)?;
    encoder
        .set_fragment_buffer(Some(&shape), 0, 1)
        .map_err(draw)?;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .map_err(draw)?;
    encoder.end_encoding().map_err(draw)?;
    let samples = encode_through(
        device,
        &command_buffer,
        &view,
        kernel,
        SampleRead::Color(readback),
        [width, height],
        n,
    )
    .map_err(|e| format!("the readback: {e}"))?;
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = LoadAction::Load;
    target.store_action = StoreAction::MultisampleResolve;
    target.resolve_texture = Some(resolved.clone());
    let encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(|e| format!("the resolve pass: {e}"))?;
    encoder.end_encoding().map_err(|e| e.to_string())?;
    command_buffer.commit().map_err(|e| e.to_string())?;
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .map_err(|e| format!("{e:?}"))?;
    if let Some(error) = command_buffer.error() {
        return Err(format!("the command buffer failed: {error:?}"));
    }
    let stored = samples.bytes().map_err(|e| e.to_string())?;
    let got = lane.read(&resolved, (0, 0), case.bytes as u64, BlitOption::None);
    let mut outcome = Outcome {
        channels: 0,
        differ: 0,
        changed: 0,
        examples: Vec::new(),
    };
    for p in 0..pixels {
        let at = p * case.bytes;
        for &(offset, _, channel) in case.channels {
            outcome.channels += 1;
            let bits = channel.bits();
            let field_of = |texels: &[u8], s: usize| {
                let at = (p * n as usize + s) * case.bytes;
                field(&texels[at..at + case.bytes], offset, bits)
            };
            let samples: Vec<u64> = (0..n as usize).map(|s| field_of(&stored, s)).collect();
            outcome.changed += (0..n as usize)
                .filter(|&s| field_of(&drawn, s) != samples[s])
                .count();
            let device_value = field(&got[at..at + case.bytes], offset, bits);
            let metal_value = metal(channel, &samples);
            if device_value != metal_value {
                outcome.differ += 1;
                if outcome.examples.len() < 4 {
                    let shown: Vec<String> = samples.iter().map(|v| format!("{v:#x}")).collect();
                    outcome.examples.push(format!(
                        "[{}] at bit {offset}: {device_value:#x}, Metal's rule {metal_value:#x}",
                        shown.join(", ")
                    ));
                }
            }
        }
    }
    Ok(outcome)
}

/// Every closed format at every sample count the device takes, printed against Metal's rules
/// and held to them (the module docs).
#[test]
fn color_resolves_against_metals_rules() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let description = &lane.device.properties().description;
    eprintln!(
        "resolve rounding on {name}: driver {:?}, driverVersion {:#x}, vendor {:#x}",
        description.driver_id, description.driver_version, description.vendor_id
    );
    let mut differ = Vec::new();
    for n in [2u32, 4, 8] {
        let supported = lane
            .device
            .properties()
            .supports_texture_sample_count(n.into())
            .unwrap();
        if !supported {
            eprintln!("resolve rounding on {name}: no texture of {n} samples");
            continue;
        }
        for case in CASES {
            let pf = case.pf;
            match resolve_case(&lane, case, n) {
                Err(why) => eprintln!("resolve rounding {pf:?} x{n} on {name}: {why}"),
                Ok(outcome) => {
                    if outcome.differ != 0 {
                        differ.push(format!("{pf:?} x{n}: {} channels", outcome.differ));
                    }
                    let verdict = match (n, outcome.differ) {
                        (2 | 4, 0) => "Metal's".to_owned(),
                        (2 | 4, _) => "not Metal's".to_owned(),
                        (_, 0) => {
                            "unmeasured (no Metal answer at this count), Metal's rules".to_owned()
                        }
                        _ => "unmeasured (no Metal answer at this count), not Metal's rules"
                            .to_owned(),
                    };
                    eprintln!(
                        "resolve rounding {pf:?} x{n} on {name}: {verdict}, {} of {} channels \
                         differ ({} samples stored otherwise than drawn){}",
                        outcome.differ,
                        outcome.channels,
                        outcome.changed,
                        if outcome.examples.is_empty() {
                            String::new()
                        } else {
                            format!(": {}", outcome.examples.join("; "))
                        }
                    );
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "resolves not by Metal's rules on {name}: {differ:#?}"
    );
    quiet(lane, &messages);
}

/// The side of [`a_maps_form_pass_resolves_bgra8_by_metals_rule`]'s targets.
const MAPS_SIDE: u32 = 64;

/// One pass of `pf` at 4 samples beside Maps' depth form (a 4x `Depth32Float_Stencil8` of
/// `RenderTarget` usage alone, cleared to 0, its test `GreaterEqual` with writes, which the fill at
/// depth 0 and the half at 0.5 pass): `random`, a
/// fullscreen fill whose every sample takes its own random codes ([`value`]), else a clear to
/// `(0.2, 0.4, 0.6, 1)`; then half the target (`rt6_resolve_half`, its diagonal's pixels half
/// covered) drawn per sample by `rt6_resolve_samples` (red `s / 3`, green 1 on sample 3, blue
/// 0.5). The color is stored and resolved in the pass (`StoreAndMultisampleResolve`); its samples
/// are read back after it. Each channel of the resolve is compared with Metal's rule ([`metal`])
/// over the samples read back; the count of differing channels, and of channels on half-covered
/// pixels whose samples differ, are returned.
fn maps_form_pass(
    lane: &Lane,
    case: &Case,
    random: bool,
    run: u64,
) -> Result<(usize, usize), String> {
    let (n, side) = (4u32, MAPS_SIDE);
    let device = &lane.device;
    let pixels = (side * side) as usize;
    let mut r = SplitMix(0x4d41_5053 ^ (run << 40) ^ case.pf as u64);
    let mut floats = Vec::with_capacity(pixels * n as usize * 16);
    for y in 0..side {
        for x in 0..side {
            for s in 0..n {
                let mut lanes = [0f32; 4];
                for (index, &(_, component, channel)) in case.channels.iter().enumerate() {
                    let code = value(channel, index, (x, y + 256), (s, n), &mut r);
                    lanes[component] = encoding(channel, code);
                }
                for lane in lanes {
                    floats.extend_from_slice(&lane.to_le_bytes());
                }
            }
        }
    }
    let texture = |pf, samples: u32, usage| {
        let mut descriptor = TextureDescriptor::texture_2d(pf, side.into(), side.into(), false);
        if samples > 1 {
            descriptor.texture_type = TextureType::Type2DMultisample;
            descriptor.sample_count = samples.into();
        }
        descriptor.usage = usage;
        descriptor.resource_options = ResourceOptions::private();
        device
            .new_texture(&descriptor)
            .map_err(|e| format!("new_texture {pf:?} x{samples}: {e}"))
    };
    let msaa = texture(
        case.pf,
        n,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
    )?;
    let depth = texture(
        PixelFormat::Depth32FloatStencil8,
        n,
        TextureUsage::RENDER_TARGET,
    )?;
    let resolved = texture(case.pf, 1, TextureUsage::RENDER_TARGET)?;
    let (view_pf, kernel, readback) = readback_view(case);
    let view = msaa
        .new_texture_view(view_pf)
        .map_err(|e| format!("unread: no {view_pf:?} view ({e})"))?;
    let samples_library = library(device).map_err(|e| e.to_string())?;
    let ds = Some(PixelFormat::Depth32FloatStencil8);
    let fill = {
        let function = |name: &str| {
            samples_library
                .new_function_with_name(name)
                .map(|f| f.function_descriptor())
                .map_err(|e| format!("{name}: {e}"))
        };
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(function("rt6_fill_vertex")?));
        descriptor.fragment = Some(StageSource::Function(function("rt6_fill_rgba32float")?));
        descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(case.pf));
        descriptor.depth_format = ds;
        descriptor.stencil_format = ds;
        descriptor.raster_sample_count = n;
        device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .map_err(|e| format!("the fill's pipeline: {e}"))?
    };
    let half = lane.pipeline(
        ("rt6_resolve_half", "rt6_resolve_samples"),
        &[case.pf],
        (ds, ds),
        4,
    );
    let mut tested = DepthStencilDescriptor::default();
    tested.depth_compare_function = CompareFunction::GreaterEqual;
    tested.depth_write_enabled = true;
    let tested = device
        .new_depth_stencil_state(&tested)
        .map_err(|e| e.to_string())?;
    let values = buffer_with(device, &floats).map_err(|e| e.to_string())?;
    let shape = shape(device, side, n).map_err(|e| e.to_string())?;
    let mut pass = RenderPassDescriptor::new();
    let target = &mut pass.color_attachments[0];
    target.texture = Some(msaa.clone());
    target.load_action = if random {
        LoadAction::DontCare
    } else {
        LoadAction::Clear
    };
    target.clear_color = ClearColor::new(0.2, 0.4, 0.6, 1.0);
    target.store_action = StoreAction::StoreAndMultisampleResolve;
    target.resolve_texture = Some(resolved.clone());
    pass.depth_attachment.texture = Some(depth.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.clear_depth = 0.0;
    pass.depth_attachment.store_action = StoreAction::DontCare;
    pass.stencil_attachment.texture = Some(depth.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::DontCare;
    let command_buffer = lane.queue.command_buffer().map_err(|e| e.to_string())?;
    let mut encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(|e| format!("the pass: {e}"))?;
    let draw = |e: metal2vulkan_runtime::Error| format!("the draws: {e}");
    encoder.set_depth_stencil_state(&tested).map_err(draw)?;
    if random {
        encoder.set_render_pipeline_state(&fill).map_err(draw)?;
        encoder
            .set_fragment_buffer(Some(&values), 0, 0)
            .map_err(draw)?;
        encoder
            .set_fragment_buffer(Some(&shape), 0, 1)
            .map_err(draw)?;
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .map_err(draw)?;
    }
    encoder.set_render_pipeline_state(&half).map_err(draw)?;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .map_err(draw)?;
    encoder.end_encoding().map_err(draw)?;
    let samples = encode_through(
        device,
        &command_buffer,
        &view,
        kernel,
        SampleRead::Color(readback),
        [side, side],
        n,
    )
    .map_err(|e| format!("the readback: {e}"))?;
    command_buffer.commit().map_err(|e| e.to_string())?;
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .map_err(|e| format!("{e:?}"))?;
    if let Some(error) = command_buffer.error() {
        return Err(format!("the command buffer failed: {error:?}"));
    }
    let stored = samples.bytes().map_err(|e| e.to_string())?;
    let got = lane.read(&resolved, (0, 0), case.bytes as u64, BlitOption::None);
    let (mut differ, mut mixed) = (0, 0);
    for p in 0..pixels {
        let at = p * case.bytes;
        for &(offset, _, channel) in case.channels {
            let bits = channel.bits();
            let samples: Vec<u64> = (0..n as usize)
                .map(|s| {
                    let at = (p * n as usize + s) * case.bytes;
                    field(&stored[at..at + case.bytes], offset, bits)
                })
                .collect();
            mixed += usize::from(samples.iter().any(|&v| v != samples[0]));
            if field(&got[at..at + case.bytes], offset, bits) != metal(channel, &samples) {
                differ += 1;
            }
        }
    }
    Ok((differ, mixed))
}

/// Maps' color formats at 4 samples (`BGRA8Unorm_sRGB`, `BGRA8Unorm`), each in a pass beside its
/// depth form, resolved in the pass: on a cleared and on a randomly filled target, three runs
/// each, every channel of the resolve is the rule over the samples read back (half-covered diagonal
/// pixels and per-sample colors among them), and no pass announces `device-rounded-resolve`. The
/// coordinator's ask of 2026-10-09 for the consumer, who relays that kind only for the half-float
/// formats measured on the M3.
#[test]
fn a_maps_form_pass_resolves_bgra8_by_metals_rule() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    messages.take_notices();
    for pf in [PixelFormat::BGRA8UnormSRGB, PixelFormat::BGRA8Unorm] {
        let case = CASES.iter().find(|c| c.pf == pf).expect("a closed format");
        for random in [false, true] {
            for run in 0..3 {
                let what = format!(
                    "{pf:?} x4 beside Maps' depth form, {} target, run {run}, on {name}",
                    if random { "a random" } else { "a cleared" }
                );
                match maps_form_pass(&lane, case, random, run) {
                    Err(why) if why.starts_with("unread") => {
                        eprintln!("Maps' form resolve {what}: {why}");
                    }
                    Err(why) => panic!("{what}: {why}"),
                    Ok((differ, mixed)) => {
                        eprintln!(
                            "Maps' form resolve {what}: {differ} channels differ from Metal's \
                             rule ({mixed} channels over samples that differ)"
                        );
                        assert_eq!(differ, 0, "{what}");
                        assert!(mixed > 0, "{what}: no pixel's samples differ");
                    }
                }
            }
        }
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    eprintln!("Maps' form resolve on {name}: announced {notices:?}");
    assert_eq!(
        notices.get(&Degradation::DeviceRoundedResolve),
        None,
        "device-rounded-resolve on {name}"
    );
    quiet(lane, &messages);
}

/// The width and height of the cost test's target: a 1080p frame.
const FRAME: [u32; 2] = [1920, 1080];

/// Command buffers per variant of the cost test, of which the median GPU time is printed.
const TIMED: usize = 15;

/// The exact resolve's cost against the device's: a 1920x1080 `RGBA8Unorm` target of 4 samples,
/// cleared and drawn once white, then resolved into a single-sampled texture by the device in the
/// pass (a caller's image made without input attachment usage, which the runtime resolves by the
/// device's average), by the exact resolve after the pass (the runtime's own texture, on a device
/// that resolves so), or stored without a resolve; each the median GPU time of its command buffers,
/// printed. Nothing about the times is asserted.
#[test]
fn the_exact_resolve_costs_against_the_devices() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let pf = PixelFormat::RGBA8Unorm;
    let [width, height] = FRAME;
    let descriptor = |samples: u64| {
        let mut d = TextureDescriptor::texture_2d(pf, width.into(), height.into(), false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
        }
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::private();
        d
    };
    let own = lane.device.new_texture(&descriptor(4)).unwrap();
    let resolved = lane.device.new_texture(&descriptor(1)).unwrap();
    // SAFETY: used only while the device lives; the image is destroyed after its texture and
    // every command buffer that used it.
    let raw = unsafe { lane.device.raw() };
    // SAFETY: a physical device of the instance.
    let memory = unsafe {
        raw.instance
            .get_physical_device_memory_properties(raw.physical_device)
    };
    let usage = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        | ash::vk::ImageUsageFlags::TRANSFER_SRC
        | ash::vk::ImageUsageFlags::TRANSFER_DST;
    let info = image_info(ash::vk::Format::R8G8B8A8_UNORM, usage)
        .extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .samples(ash::vk::SampleCountFlags::TYPE_4);
    let info = shared_among(info, raw.queue_families);
    // SAFETY: the device is live.
    let (image, image_memory) = unsafe { made_image(raw.device, &memory, &info) };
    // SAFETY: the image is the device's, made with `info`, and nothing else uses it.
    let callers = unsafe {
        lane.device.new_texture_with_vk_image(
            &descriptor(4),
            image,
            &info,
            VkImageLayouts::new(
                ash::vk::ImageLayout::UNDEFINED.into(),
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL.into(),
            ),
        )
    }
    .unwrap();
    let pipeline = lane.pipeline((FULL, WHITE), &[pf], (None, None), 4);
    let time = |msaa: &Texture, resolve: bool| {
        let mut times = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let mut pass = RenderPassDescriptor::new();
            let target = &mut pass.color_attachments[0];
            target.texture = Some(msaa.clone());
            target.load_action = LoadAction::Clear;
            if resolve {
                target.store_action = StoreAction::MultisampleResolve;
                target.resolve_texture = Some(resolved.clone());
            } else {
                target.store_action = StoreAction::Store;
            }
            let command_buffer = lane.queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            let mut held = Vec::new();
            lane.draw(&mut encoder, &mut held, &pipeline, 0.5);
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
                .wait_until_completed_for(super::WAIT)
                .unwrap();
            let (start, end) = (
                command_buffer.gpu_start_time().unwrap(),
                command_buffer.gpu_end_time().unwrap(),
            );
            times.push((end - start) * 1e6);
        }
        times.sort_by(f64::total_cmp);
        times[TIMED / 2]
    };
    let device = time(&callers, true);
    let exact = time(&own, true);
    let stored = time(&own, false);
    eprintln!(
        "resolve cost on {name}: {width}x{height} RGBA8Unorm x4, median GPU time of {TIMED}: \
         the device's resolve {device:.0} us, the runtime's own texture's resolve {exact:.0} us, \
         stored without a resolve {stored:.0} us"
    );
    drop(callers);
    // SAFETY: nothing uses them: the texture is gone and every command buffer completed.
    unsafe {
        raw.device.destroy_image(image, None);
        raw.device.free_memory(image_memory, None);
    }
    quiet(lane, &messages);
}
