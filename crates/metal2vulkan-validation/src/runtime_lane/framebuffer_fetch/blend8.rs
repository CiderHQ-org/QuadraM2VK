//! A fetch of a blended 8-bit unorm attachment on the device the cfg selects, under the validation
//! layer: the owned fixtures `fbl_fetch_blend8` (fast math) and `fbl_fetch_blend8_safe` (without),
//! in the shape of a CoreAnimation composite of the x86 macOS guest. Color 0 is a `SIDE`² private
//! target of each 8-bit unorm format (`BGRA8Unorm`, the guest's, `RGBA8Unorm`, `RG8Unorm`,
//! `R8Unorm`) blended one over one minus the source alpha for color and alpha, color 1 an
//! `RGBA16Float` target of mask none. Each function reads color 0 and returns, at pixel
//! `i = SIDE · y + x`, a loaded source chosen by the fetched alpha (`src[i + SIDE²]` where it is
//! zero, else `src[i]`): `*_loaded` a `half4`, `*_loaded_float` a `float4`; or `*_product` the
//! loaded `half4` times the fetched color. One draw of the fullscreen triangle.
//!
//! Metal's answer (EV FB8-sw1-m3: plain Metal on the M3 Max under macOS 27.0.1, our own MSL of
//! this shape, every `half` alpha in `[0, 1]` against every destination code of each channel, four
//! source draws per pair, each format, `half4` and `float4` outputs, fast and safe math, every
//! channel; EV FB8-fold1-m3: products and sums of the output, the fetched color among them, fold
//! nothing into the blend): the output clamped to `[0, 1]`, a NaN to 0; the destination's code `c`
//! read as `h(c / 255)` (`f(c / 255)` for a `float4` output); `h(fma(d, h(1 - a), s))` under fast
//! math and `h(s + h(d · h(1 - a)))` without (in `float` for a `float4` output, likewise); the
//! result's code by Metal's unorm write, the exact product with 255 rounded to nearest. The texels
//! are codes covering every value in each channel, a zero alpha at times; the sources random `half`s
//! or `float`s, mostly in `[0, 1]`, some out of range, infinite or NaN.
//!
//! Where the draw announces no `half-output-fast-math-blend`, `half-output-precision` or
//! `device-color-conversion`, every channel is Metal's. On a device that fetches through the pixel
//! interlock (RADV, ANV, NVIDIA, MoltenVK) the function computes Metal's blend itself, exactly,
//! announcing none of the three, which is asserted; a format whose storage the device lacks there
//! is refused by that capability, which is printed. Through input attachments (lavapipe) the device
//! blends; what it announces and how many channels are Metal's is printed. Each draw prints a
//! `FETCH-BLEND8` line naming its function and format.

use super::{form, library, through_inputs};
use crate::runtime_lane::buffer_with;
use crate::runtime_lane::degrade_notices::counts;
use crate::runtime_lane::resolves::{lane_with, quiet, Lane};
use crate::runtime_lane::WAIT;
use metal2vulkan::error::Needs;
use metal2vulkan::metal::{
    BlendFactor, BlendOperation, ColorWriteMask, PixelFormat, PrimitiveType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, PrecisionNote, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, Degradation, Error, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
};

/// The target's side, as the fixtures'.
const SIDE: u64 = 64;

/// The target's texels.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The notes under which a channel may be another than Metal's.
const NOTES: [PrecisionNote; 3] = [
    PrecisionNote::HalfOutputFastMathBlend,
    PrecisionNote::HalfOutputPrecision,
    PrecisionNote::DeviceColorConversion,
];

/// Each format, its bytes a texel, and the channel (red 0 to alpha 3) of each byte.
const FORMATS: [(PixelFormat, &[usize]); 4] = [
    (PixelFormat::BGRA8Unorm, &[2, 1, 0, 3]),
    (PixelFormat::RGBA8Unorm, &[0, 1, 2, 3]),
    (PixelFormat::RG8Unorm, &[0, 1]),
    (PixelFormat::R8Unorm, &[0]),
];

/// What a function returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Output {
    /// A loaded `half4`.
    Half,
    /// A loaded `float4`.
    Float,
    /// The loaded `half4` times the fetched color.
    Product,
}

/// A fixture's functions: its vertex function, each fragment function and its output, and
/// whether the module computes without fast math.
struct Fixture {
    stem: &'static str,
    vertex: &'static str,
    cases: [(&'static str, Output); 3],
    safe: bool,
}

const FIXTURES: [Fixture; 2] = [
    Fixture {
        stem: "fbl_fetch_blend8",
        vertex: "fbl8_vertex",
        cases: [
            ("fbl8_over_loaded", Output::Half),
            ("fbl8_over_loaded_float", Output::Float),
            ("fbl8_over_product", Output::Product),
        ],
        safe: false,
    },
    Fixture {
        stem: "fbl_fetch_blend8_safe",
        vertex: "fbl8_safe_vertex",
        cases: [
            ("fbl8_safe_loaded", Output::Half),
            ("fbl8_safe_loaded_float", Output::Float),
            ("fbl8_safe_product", Output::Product),
        ],
        safe: true,
    },
];

/// A deterministic stream of 64-bit words (xorshift).
struct Words(u64);

impl Words {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A source lane: one time in 16 out of range (`specials`: or infinite, or NaN), else in
    /// `[0, 1]`.
    fn source(&mut self, specials: bool) -> f64 {
        const SPECIALS: [f64; 8] = [
            -0.0,
            -1.0,
            2.0,
            f64::INFINITY,
            -f64::INFINITY,
            f64::NAN,
            1.5,
            -0.25,
        ];
        let w = self.next();
        let k = ((w >> 4) % 8) as usize;
        match w.is_multiple_of(16) {
            true if specials || SPECIALS[k].is_finite() => SPECIALS[k],
            _ => ((w >> 16) & 0xFF_FFFF) as f64 / f64::from(0xFF_FFFFu32),
        }
    }
}

/// `x` rounded to `half`, nearest even (a NaN `0x7e00`).
fn h(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let e = ((x.abs().to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let ulp = 2f64.powi(e.max(-14) - 10);
    let r = (x / ulp).round_ties_even() * ulp;
    if r.abs() > 65504.0 {
        return f64::INFINITY.copysign(x);
    }
    r
}

/// The bits of `half` `x` (a value [`h`] gives).
fn half_bits(x: f64) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a.is_infinite() {
        return sign | 0x7c00;
    }
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    if e < -14 {
        return sign | (a / 2f64.powi(-24)) as u16;
    }
    sign | (((e + 15) as u16) << 10) | ((a / 2f64.powi(e - 10)) as u16 - 1024)
}

/// `x` clamped to `[0, 1]`, a NaN to 0.
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Metal's code of source lane `s` (alpha `a`) over destination code `c`, in `float` where
/// `float`, without fast math where `safe` (the module docs).
fn metal_code(float: bool, safe: bool, (s, a): (f64, f64), c: u8) -> u8 {
    let (s, a) = (clamp01(s), clamp01(a));
    let v = match float {
        false => {
            let d = h(f64::from(c) / 255.0);
            let m = h(1.0 - a);
            match safe {
                false => h(d * m + s),
                true => h(s + h(d * m)),
            }
        }
        true => {
            let d = f32::from(c) / 255.0;
            let (s, a) = (s as f32, a as f32);
            let m = 1.0 - a;
            f64::from(match safe {
                false => d.mul_add(m, s),
                true => s + d * m,
            })
        }
    };
    (clamp01(v) * 255.0 + 0.5).floor() as u8
}

/// The texels' codes, red to alpha, and the sources (two a pixel), as the output holds them:
/// each channel's code runs through every value, alpha zero at times. A product's sources are
/// finite: under fast math a product of an infinity or a NaN is undefined to Metal's compiler.
fn data(output: Output) -> (Vec<[u8; 4]>, Vec<[f64; 4]>) {
    let mut words = Words(0x51ab_1e8b_17c0_de55);
    let texels = (0..TEXELS)
        .map(|i| std::array::from_fn(|c| ((i * (2 * c + 1) + 37 * c) % 256) as u8))
        .collect();
    let sources = (0..2 * TEXELS)
        .map(|_| {
            [0; 4].map(|_: u8| {
                let x = words.source(output != Output::Product);
                match output {
                    Output::Float => f64::from(x as f32),
                    Output::Half | Output::Product => h(x),
                }
            })
        })
        .collect();
    (texels, sources)
}

/// A private `SIDE`² texture of `pf` holding `bytes`.
fn filled(lane: &Lane, pf: PixelFormat, bytes: &[u8], bpp: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    let texture = lane.device.new_texture(&descriptor).unwrap();
    let source = buffer_with(&lane.device, bytes).unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        bpp * SIDE,
        bytes.len() as u64,
        Size::new(SIDE, SIDE, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    texture
}

/// Each draw's channels are Metal's where it announces none of [`NOTES`], and on a device that
/// fetches through the pixel interlock it announces none (the module docs).
#[test]
fn a_blended_eight_bit_unorm_fetch_stores_metals_blend() {
    let kinds = NOTES.map(Degradation::from_precision_note);
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = kinds.into_iter().collect();
    });
    let device = lane.device.name();
    let interlock = !through_inputs(&lane);
    let mut drawn = 0;
    for fixture in &FIXTURES {
        let library = library(&lane, fixture.stem);
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        for (pf, order) in FORMATS {
            for (entry, output) in fixture.cases {
                let what = format!("{entry} into {pf:?} on {device}");
                let mut descriptor = RenderPipelineDescriptor::new(function(fixture.vertex));
                descriptor.fragment = Some(function(entry));
                let mut blended = ColorAttachmentDescriptor::new(pf);
                blended.blending_enabled = true;
                blended.source_rgb_blend_factor = BlendFactor::One;
                blended.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                blended.rgb_blend_operation = BlendOperation::Add;
                blended.source_alpha_blend_factor = BlendFactor::One;
                blended.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                blended.alpha_blend_operation = BlendOperation::Add;
                let mut unwritten = ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float);
                unwritten.write_mask = ColorWriteMask::NONE;
                descriptor.color_attachments[0] = Some(blended);
                descriptor.color_attachments[1] = Some(unwritten);
                let state = match lane
                    .device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                {
                    Ok(state) => state,
                    // A format whose storage the device lacks is refused by that capability.
                    Err(Error::Unsupported(u)) if interlock && u.needs != Needs::NotImplemented => {
                        println!("FETCH-BLEND8 {entry} {pf:?} | {device} | refused: {u}");
                        continue;
                    }
                    Err(error) => panic!("{what}: {error}"),
                };
                let bpp = order.len();
                let (texels, sources) = data(output);
                let bytes: Vec<u8> = texels
                    .iter()
                    .flat_map(|t| order.iter().map(|&c| t[c]))
                    .collect();
                let target = filled(&lane, pf, &bytes, bpp as u64);
                let mut second =
                    TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, SIDE, SIDE, false);
                second.usage = TextureUsage::RENDER_TARGET;
                second.resource_options = ResourceOptions::private();
                let second = lane.device.new_texture(&second).unwrap();
                let src: Vec<u8> = match output {
                    Output::Float => sources
                        .iter()
                        .flat_map(|s| s.map(|v| v as f32))
                        .flat_map(f32::to_le_bytes)
                        .collect(),
                    Output::Half | Output::Product => sources
                        .iter()
                        .flat_map(|s| s.map(half_bits))
                        .flat_map(u16::to_le_bytes)
                        .collect(),
                };
                let src = buffer_with(&lane.device, &src).unwrap();
                let mut pass = RenderPassDescriptor::new();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(target.clone());
                color.load_action = LoadAction::Load;
                color.store_action = StoreAction::Store;
                let color = &mut pass.color_attachments[1];
                color.texture = Some(second);
                color.load_action = LoadAction::Clear;
                color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
                color.store_action = StoreAction::DontCare;
                let _ = messages.take_notices();
                lane.run(&pass, |_, encoder, _| {
                    encoder.set_render_pipeline_state(&state).unwrap();
                    encoder.set_fragment_buffer(Some(&src), 0, 0).unwrap();
                    encoder
                        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                        .unwrap();
                })
                .unwrap();
                let announced = counts(&messages.take_notices());
                let noted: Vec<_> = kinds
                    .iter()
                    .filter(|k| announced.contains_key(*k))
                    .collect();
                let read = lane.read(&target, (0, 0), bpp as u64, BlitOption::None);
                let has_alpha = order.contains(&3);
                let mut wrong = Vec::new();
                for i in 0..TEXELS {
                    let codes = texels[i];
                    let got = &read[bpp * i..bpp * i + bpp];
                    // The fetched color: each channel's code, a missing color channel 0, a
                    // missing alpha one.
                    let fetched: [f64; 4] = std::array::from_fn(|c| match order.contains(&c) {
                        true => f64::from(codes[c]) / 255.0,
                        false => f64::from(u8::from(c == 3)),
                    });
                    // A load chosen by the fetched alpha, or the first load times the fetched
                    // color.
                    let chosen = match has_alpha && codes[3] == 0 {
                        true => i + TEXELS,
                        false => i,
                    };
                    let s: [f64; 4] = match output {
                        Output::Product => {
                            std::array::from_fn(|c| h(sources[i][c] * h(fetched[c])))
                        }
                        _ => sources[chosen],
                    };
                    let float = output == Output::Float;
                    let want: Vec<u8> = order
                        .iter()
                        .map(|&c| metal_code(float, fixture.safe, (s[c], s[3]), codes[c]))
                        .collect();
                    if got != want.as_slice() {
                        wrong.push((i, got.to_vec(), want));
                    }
                }
                println!(
                    "FETCH-BLEND8 {entry} {pf:?} | {device} | through {} | Metal's {}/{} texels \
                     | announced {noted:?}",
                    form(&lane),
                    TEXELS - wrong.len(),
                    TEXELS,
                );
                if interlock {
                    assert!(noted.is_empty(), "{what}: announced {noted:?}");
                }
                if noted.is_empty() {
                    let shown = &wrong[..wrong.len().min(8)];
                    assert!(
                        wrong.is_empty(),
                        "{what}: {} of {TEXELS} texels differ from Metal's, first (index, read, \
                         Metal's) {shown:?}",
                        wrong.len()
                    );
                }
                drawn += 1;
            }
        }
    }
    assert!(drawn > 0, "no draw on {device}");
    let _ = messages.take_notices();
    quiet(lane, &messages);
}
