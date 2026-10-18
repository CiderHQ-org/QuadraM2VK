//! Swizzled extended-range and bfloat textures at sampled slots on the device the cfg selects,
//! under the validation layer.
//!
//! Metal applies a texture's swizzle to the texel it reads: each channel the source its swizzle
//! names, `Zero` 0 and `One` 1.0, border texels included, and a gather of a component gathers the
//! swizzle's source of it. An extended-range or bfloat texture's image holds raw codes, which the
//! function decodes; this lane asks whether the runtime's swizzled reads, samples and gathers of
//! such a texture are its unswizzled ones swizzled, as Metal's are (an Apple M4
//! under macOS 27.0.1: the same textures, swizzles, positions and samplers).
//!
//! The kernels are `fpv_swizzled_reads`'s: reads of every texel of an 8 x 4 texture and four past
//! it, samples through a nearest and a linear runtime sampler at clamp-to-border opaque white, and
//! gathers of each component through the nearest one, at 540 positions inside and around the
//! texture. Each format's texture holds the probe's pseudo-random bytes and is made with each of
//! five swizzles: the default, `(a, r, 1, 0)`, `(0, a, g, 1)`, `(b, g, r, a)` and `(g, g, g, g)`.
//! A swizzled result that is not the unswizzled one swizzled is printed on an
//! `EXT-SWIZZLE-DIFFERS` line; a dispatch the device refuses, typed, on an `EXT-SWIZZLE-REFUSED`
//! line, counted apart (a refusal, not an answer).
//!
//! The packed normalized formats `B5G6R5Unorm`, `A1BGR5Unorm`, `BGR5A1Unorm` and `ABGR4Unorm` take
//! the same kernels, textures of the same bytes and the same swizzles, on `PACKED-SWIZZLE` lines:
//! Metal converts each channel's code `c` of `b` bits to `c / (2^b - 1)`, which a swizzle in the
//! view's component mapping would hand the function on another channel's lane, where NVIDIA and ANV
//! read the code's bits replicated to 8 (up to 0.4/255 off). On a device
//! without Metal's own sampler the function converts the unswizzled texel and swizzles it.
//!
//! A packed texture's read of a texel inside it, nearest sample inside it and gather of a footprint
//! inside it are compared with Metal's answer, each channel's code `c / (2^b - 1)` correctly
//! rounded, unswizzled and swizzled; its linear sample and its accesses past the edges with the
//! unswizzled texture's on the same device (Metal's filter, emulated, and its border), swizzled. A
//! difference of the unswizzled texture from Metal's answer is printed on a
//! `PACKED-SWIZZLE-DIFFERS` line too (its swizzle `rgba`).

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::{BindError, Unsupported};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter, TextureSwizzle,
    TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    Error, PipelineOptions, Region, ResourceOptions, SamplerState, Size, Texture,
    TextureDescriptor, TextureSwizzleChannels,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's width and height.
const W: u64 = 8;
const H: u64 = 4;

/// The formats, each with its texel's bytes: the one-plane extended-range formats, the padded
/// 10-bit ones, their `_sRGB` twins, and the bfloat formats.
const FORMATS: [(PixelFormat, u64); 11] = [
    (PixelFormat::BGRA10XR, 8),
    (PixelFormat::BGRA10XRSRGB, 8),
    (PixelFormat::BGR10XR, 4),
    (PixelFormat::BGR10XRSRGB, 4),
    (PixelFormat::R10UnormX6, 2),
    (PixelFormat::R10UnormX6SRGB, 2),
    (PixelFormat::RG10UnormX12, 4),
    (PixelFormat::RG10UnormX12SRGB, 4),
    (PixelFormat::R16Bfloat, 2),
    (PixelFormat::RG16Bfloat, 4),
    (PixelFormat::RGBA16Bfloat, 8),
];

/// The packed normalized formats, each with its texel's bytes.
const PACKED: [(PixelFormat, u64); 4] = [
    (PixelFormat::B5G6R5Unorm, 2),
    (PixelFormat::A1BGR5Unorm, 2),
    (PixelFormat::BGR5A1Unorm, 2),
    (PixelFormat::ABGR4Unorm, 2),
];

/// Metal's texel `(x, y)` of a texture of the packed normalized `format` holding `bytes`: each
/// channel's code `c` of `b` bits reads `c / (2^b - 1)`, correctly rounded, and an alpha the format
/// lacks 1. The fields are the Vulkan formats' that hold them (`R5G6B5_UNORM_PACK16`,
/// `R5G5B5A1_UNORM_PACK16`, `A1R5G5B5_UNORM_PACK16`, `R4G4B4A4_UNORM_PACK16`). `None` for another
/// format, and a texel outside the texture.
fn packed_texel(format: PixelFormat, bytes: &[u8], (x, y): (i64, i64)) -> Option<[u32; 4]> {
    if !(0..W as i64).contains(&x) || !(0..H as i64).contains(&y) {
        return None;
    }
    // Each channel's (shift, bits), red first; `None` where the format lacks it.
    let fields: [Option<(u32, u32)>; 4] = match format {
        PixelFormat::B5G6R5Unorm => [Some((11, 5)), Some((5, 6)), Some((0, 5)), None],
        PixelFormat::A1BGR5Unorm => [Some((11, 5)), Some((6, 5)), Some((1, 5)), Some((0, 1))],
        PixelFormat::BGR5A1Unorm => [Some((10, 5)), Some((5, 5)), Some((0, 5)), Some((15, 1))],
        PixelFormat::ABGR4Unorm => [Some((12, 4)), Some((8, 4)), Some((4, 4)), Some((0, 4))],
        _ => return None,
    };
    let at = ((y as u64 * W + x as u64) * 2) as usize;
    let v = u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
    Some(fields.map(|f| match f {
        Some((shift, bits)) => {
            let max = (1u32 << bits) - 1;
            let c = (v >> shift) & max;
            // An odd denominator's quotient is never an `f32` tie, so rounding it through `f64`
            // is the correctly rounded quotient.
            ((f64::from(c) / f64::from(max)) as f32).to_bits()
        }
        None => 1.0f32.to_bits(),
    }))
}

/// Metal's result of `op` (as [`dispatch`] names it) at position `i` of an unswizzled texture of
/// the packed normalized `format` holding `bytes`, where it reads texels inside the texture
/// alone: a read of a texel, a nearest sample of a position, `floor(u n)`'s texel, and a gather's
/// footprint, the texels `i, i + 1` from `floor(u n - 1/2)` on each axis, in gather order
/// `(i, j + 1)`, `(i + 1, j + 1)`, `(i + 1, j)`, `(i, j)`. `None` for a linear sample, an access
/// past the edges, and another format.
fn metal_packed(format: PixelFormat, bytes: &[u8], op: &str, i: usize) -> Option<[u32; 4]> {
    let texel = |at| packed_texel(format, bytes, at);
    let position = |i: usize| {
        positions()
            .get(i)
            .map(|p| (p[0] as f64 * W as f64, p[1] as f64 * H as f64))
    };
    match op {
        "rf" => {
            let [x, y] = *reads().get(i)?;
            texel((i64::from(x), i64::from(y)))
        }
        "sn" => {
            let (x, y) = position(i)?;
            texel((x.floor() as i64, y.floor() as i64))
        }
        "sl" => None,
        gather => {
            let c: usize = gather.strip_prefix('g')?.parse().ok()?;
            let (x, y) = position(i)?;
            let (i0, j0) = ((x - 0.5).floor() as i64, (y - 0.5).floor() as i64);
            let taps = [(i0, j0 + 1), (i0 + 1, j0 + 1), (i0 + 1, j0), (i0, j0)];
            let mut out = [0; 4];
            for (o, at) in out.iter_mut().zip(taps) {
                *o = texel(at)?[c];
            }
            Some(out)
        }
    }
}

/// Metal's result of an operation of an unswizzled texture where the lane computes it
/// ([`metal_packed`]): the format, the texture's bytes, the operation and the position.
type MetalRule = fn(PixelFormat, &[u8], &str, usize) -> Option<[u32; 4]>;

/// No format's answer: the extended-range formats' lane compares with the unswizzled texture's.
fn no_rule(_: PixelFormat, _: &[u8], _: &str, _: usize) -> Option<[u32; 4]> {
    None
}

/// The swizzles, the default first.
fn swizzles() -> [(&'static str, TextureSwizzleChannels); 5] {
    use TextureSwizzle::{Alpha, Blue, Green, One, Red, Zero};
    let s = |red, green, blue, alpha| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("ar10", s(Alpha, Red, One, Zero)),
        ("0ag1", s(Zero, Alpha, Green, One)),
        ("bgra", s(Blue, Green, Red, Alpha)),
        ("gggg", s(Green, Green, Green, Green)),
    ]
}

/// The probe's bytes of a texture of the format of raw value `raw`, `bytes` a texel: a 64-bit
/// linear congruential state seeded by the raw value, each byte its top 8 bits.
fn texels(raw: u32, bytes: u64) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(raw);
    (0..W * H * bytes)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

/// The reads' texel coordinates: every texel, then four past the texture.
fn reads() -> Vec<[u32; 2]> {
    let (w, h) = (W as u32, H as u32);
    let mut at: Vec<[u32; 2]> = (0..w * h).map(|i| [i % w, i / w]).collect();
    at.extend([[w, 0], [0, h], [w + 3, 1], [2, h + 5]]);
    at
}

/// The samples' and gathers' normalized positions: three in each texel and in a texel's width
/// past each edge, rows outer.
fn positions() -> Vec<[f32; 2]> {
    let offsets = [0.25, 0.5, 0.875];
    let mut p = Vec::new();
    for ky in -1..=H as i64 {
        for oy in offsets {
            for kx in -1..=W as i64 {
                for ox in offsets {
                    p.push([
                        ((kx as f64 + ox) / W as f64) as f32,
                        ((ky as f64 + oy) / H as f64) as f32,
                    ]);
                }
            }
        }
    }
    p
}

/// The pipeline of `entry`, a kernel of the owned fixture.
fn state(device: &Device, entry: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("fpv_swizzled_reads.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A runtime sampler of `filter` at clamp-to-border opaque white.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = filter;
    descriptor.mag_filter = filter;
    descriptor.s_address_mode = SamplerAddressMode::ClampToBorderColor;
    descriptor.t_address_mode = SamplerAddressMode::ClampToBorderColor;
    descriptor.border_color = SamplerBorderColor::OpaqueWhite;
    device.new_sampler_state(&descriptor).unwrap()
}

/// What every dispatch shares.
struct Fixed {
    queue: CommandQueue,
    pipelines: BTreeMap<&'static str, ComputePipelineState>,
    nearest: SamplerState,
    linear: SamplerState,
    reads: Buffer,
    positions: Buffer,
}

/// The device's typed refusal in `error`, if it is one.
fn refusal(error: &Error) -> Option<&Unsupported> {
    match error {
        Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)) => Some(u),
        _ => None,
    }
}

fn encode(
    encoder: &mut ComputeCommandEncoder,
    (pipeline, texture, sampler): (&ComputePipelineState, &Texture, Option<&SamplerState>),
    (at, out, lanes): (&Buffer, &Buffer, u64),
) -> Result<(), Error> {
    encoder.set_compute_pipeline_state(pipeline)?;
    encoder.set_texture(Some(texture), 0)?;
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0)?;
    }
    encoder.set_buffer(Some(out), 0, 0)?;
    encoder.set_buffer(Some(at), 0, 1)?;
    encoder.dispatch_threads(Size::new(lanes, 1, 1), Size::new(lanes.min(64), 1, 1))
}

/// The results of the operation `op` (`rf` a read; `sn`, `sl` a nearest or linear sample; `g0`
/// to `g3` a gather of x, y, z or w) of `texture`, one `float4`'s bits a position.
fn dispatch(
    device: &Device,
    fixed: &Fixed,
    texture: &Texture,
    op: &str,
) -> Result<Vec<[u32; 4]>, Error> {
    let (entry, sampler, at, lanes) = match op {
        "rf" => ("fpv_swz_read", None, &fixed.reads, reads().len()),
        "sn" => (
            "fpv_swz_sample",
            Some(&fixed.nearest),
            &fixed.positions,
            positions().len(),
        ),
        "sl" => (
            "fpv_swz_sample",
            Some(&fixed.linear),
            &fixed.positions,
            positions().len(),
        ),
        "g0" => (
            "fpv_swz_gather_x",
            Some(&fixed.nearest),
            &fixed.positions,
            positions().len(),
        ),
        "g1" => (
            "fpv_swz_gather_y",
            Some(&fixed.nearest),
            &fixed.positions,
            positions().len(),
        ),
        "g2" => (
            "fpv_swz_gather_z",
            Some(&fixed.nearest),
            &fixed.positions,
            positions().len(),
        ),
        _ => (
            "fpv_swz_gather_w",
            Some(&fixed.nearest),
            &fixed.positions,
            positions().len(),
        ),
    };
    let lanes = lanes as u64;
    let out = buffer_with(device, &vec![0x5a; (lanes * 16) as usize]).unwrap();
    let command_buffer = fixed.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let pipeline = &fixed.pipelines[entry];
    let encoded = encode(
        &mut encoder,
        (pipeline, texture, sampler),
        (at, &out, lanes),
    );
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    encoded?;
    let bytes = read_buffer(&out, 0, lanes * 16).unwrap();
    Ok(bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|v| {
            let w = |i: usize| u32::from_le_bytes([v[i], v[i + 1], v[i + 2], v[i + 3]]);
            [w(0), w(4), w(8), w(12)]
        })
        .collect())
}

/// The value a channel whose swizzle source is `source` reads from the unswizzled `texel`.
fn pick(source: TextureSwizzle, texel: [u32; 4]) -> u32 {
    match source {
        TextureSwizzle::Zero => 0,
        TextureSwizzle::One => 1.0f32.to_bits(),
        TextureSwizzle::Red => texel[0],
        TextureSwizzle::Green => texel[1],
        TextureSwizzle::Blue => texel[2],
        TextureSwizzle::Alpha => texel[3],
        other => panic!("a swizzle this test does not know: {other:?}"),
    }
}

/// What `op` of a texture made with `swizzle` reads at position `i`, from the unswizzled
/// texture's results `plain` (by operation): a read's and a sample's channels each its source's;
/// a gather of component `c` the gather of its source's component, or 0 or 1 at every tap.
fn swizzled(
    op: &str,
    swizzle: TextureSwizzleChannels,
    plain: &BTreeMap<&str, Vec<[u32; 4]>>,
    i: usize,
) -> Option<[u32; 4]> {
    let sources = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
    match op.strip_prefix('g') {
        None => Some(sources.map(|s| pick(s, plain[op][i]))),
        Some(c) => {
            let source = sources[c.parse::<usize>().unwrap()];
            let gathered = match source {
                TextureSwizzle::Zero => return Some([0; 4]),
                TextureSwizzle::One => return Some([1.0f32.to_bits(); 4]),
                TextureSwizzle::Red => "g0",
                TextureSwizzle::Green => "g1",
                TextureSwizzle::Blue => "g2",
                _ => "g3",
            };
            plain.get(gathered).map(|g| g[i])
        }
    }
}

/// Each swizzled read, sample and gather of each extended-range and bfloat texture is the
/// unswizzled texture's swizzled, as Metal's are. A dispatch the device refuses, typed, is
/// printed and counted apart.
#[test]
fn swizzled_extended_textures_read_their_texels_swizzled() {
    swizzled_reads(&FORMATS, "EXT-SWIZZLE", no_rule);
}

/// Each swizzled read, sample and gather of each packed normalized texture is Metal's answer
/// swizzled (the unswizzled texture's, for a linear sample and past the edges), as Metal's are,
/// and the unswizzled texture's is Metal's. A dispatch the device refuses, typed, is printed and
/// counted apart.
#[test]
fn swizzled_packed_textures_read_their_texels_swizzled() {
    swizzled_reads(&PACKED, "PACKED-SWIZZLE", metal_packed);
}

/// Each swizzled read, sample and gather of a texture of each of `formats` against the
/// reference swizzled, printed on `label` lines: `metal`'s answer for the unswizzled texture where
/// it has one, which the unswizzled texture's results must equal too, else the unswizzled
/// texture's.
fn swizzled_reads(formats: &[(PixelFormat, u64)], label: &str, metal: MetalRule) {
    on_device(|device, _| {
        let reads: Vec<u8> = reads()
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let positions: Vec<u8> = positions()
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let entries = [
            "fpv_swz_read",
            "fpv_swz_sample",
            "fpv_swz_gather_x",
            "fpv_swz_gather_y",
            "fpv_swz_gather_z",
            "fpv_swz_gather_w",
        ];
        let fixed = Fixed {
            queue: device.new_command_queue().unwrap(),
            pipelines: entries.map(|e| (e, state(device, e))).into_iter().collect(),
            nearest: sampler(device, SamplerMinMagFilter::Nearest),
            linear: sampler(device, SamplerMinMagFilter::Linear),
            reads: buffer_with(device, &reads).unwrap(),
            positions: buffer_with(device, &positions).unwrap(),
        };
        let ops = ["rf", "sn", "sl", "g0", "g1", "g2", "g3"];
        let (mut checked, mut differ, mut refused) = (0usize, 0usize, 0usize);
        for &(format, bytes) in formats {
            let data = texels(format.as_raw(), bytes);
            let mut plain: BTreeMap<&str, Vec<[u32; 4]>> = BTreeMap::new();
            for (name, swizzle) in swizzles() {
                let mut descriptor = TextureDescriptor::texture_2d(format, W, H, false);
                descriptor.usage = TextureUsage::SHADER_READ;
                descriptor.resource_options = ResourceOptions::shared();
                descriptor.swizzle = swizzle;
                let texture = match device.new_texture(&descriptor) {
                    Ok(t) => t,
                    Err(error) => match refusal(&error) {
                        Some(u) => {
                            println!(
                                "{label}-REFUSED {}: {format:?} {name} texture: {u:?}",
                                device.name()
                            );
                            refused += 1;
                            continue;
                        }
                        None => panic!("{format:?} {name}: {error}"),
                    },
                };
                let row = W * bytes;
                texture
                    .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &data, row, 0)
                    .unwrap();
                for op in ops {
                    let got = match dispatch(device, &fixed, &texture, op) {
                        Ok(got) => got,
                        Err(error) => match refusal(&error) {
                            Some(u) => {
                                println!(
                                    "{label}-REFUSED {}: {format:?} {name} {op}: {:?} {:?}",
                                    device.name(),
                                    u.feature,
                                    u.needs
                                );
                                refused += 1;
                                continue;
                            }
                            None => panic!("{format:?} {name} {op}: {error}"),
                        },
                    };
                    if name == "rgba" {
                        // The reference: Metal's answer where `metal` has one, else the
                        // unswizzled texture's on this device.
                        let reference = (0..got.len())
                            .map(|i| metal(format, &data, op, i).unwrap_or(got[i]))
                            .collect();
                        plain.insert(op, reference);
                    }
                    if !plain.contains_key(op) {
                        continue;
                    }
                    let mut wrong = 0usize;
                    let mut first = None;
                    for (i, &g) in got.iter().enumerate() {
                        let Some(want) = swizzled(op, swizzle, &plain, i) else {
                            continue;
                        };
                        checked += 1;
                        if g != want {
                            wrong += 1;
                            first.get_or_insert((i, g, want));
                        }
                    }
                    if let Some((i, g, want)) = first {
                        differ += wrong;
                        println!(
                            "{label}-DIFFERS {}: {format:?} {name} {op}: {wrong} of {} \
                             results, first at {i}: {g:08x?}, the reference swizzled {want:08x?}",
                            device.name(),
                            got.len()
                        );
                    }
                }
            }
        }
        println!(
            "{label} {}: {checked} results checked, {differ} differ, {refused} refused",
            device.name()
        );
        assert!(checked > 0);
        assert_eq!(differ, 0, "swizzled results differ from Metal's rule");
    });
}
