//! Border texels through swizzled textures on the device the cfg selects, under the validation
//! layer.
//!
//! Metal's answer, measured on an Apple M4 (eleven formats, six swizzles,
//! the three border colours and clamp-to-zero, constexpr and runtime samplers alike, 1056 border
//! values, every one of them this rule's): a border texel is the border colour in the format's
//! channels, a missing red, green or blue read as 0 and a missing alpha as 1, and then the
//! texture's swizzle, as for any texel. `A8Unorm`'s one channel takes the border colour's red,
//! so its opaque-black border reads alpha 0, as its transparent one does.
//!
//! The owned `a2_clamp_to_zero` kernel samples its float texture through constexpr
//! clamp-to-zero samplers (nearest and linear) and through the runtime sampler at
//! `[[sampler(0)]]`, here a state of each border colour, of clamp-to-zero with each border
//! colour, and of clamp-to-zero on one axis beside clamp-to-border on the other, one dispatch
//! each. Metal's clamp-to-zero reads the sampler's border colour as clamp-to-border does (an
//! Apple M3 Max and an M4), the mixed samplers' too. The float texture is each of `formats` made
//! with each of `swizzles`. Lanes sample inside, at a texel centre, and outside on both sides,
//! where nearest and linear read only border texels.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::{BindError, Unsupported};
use metal2vulkan::metal::{
    PixelFormat, SamplerAddressMode, SamplerBorderColor, TextureSwizzle, TextureUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    Error, PipelineOptions, Region, ResourceOptions, SamplerState, Size, Texture,
    TextureDescriptor, TextureSwizzleChannels,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Lanes: a texel centre inside, and points outside on the left and the right, at least one
/// texel past the edge, where linear filtering's footprint holds only border texels.
const COORDS: [[f32; 2]; 3] = [[0.375, 0.375], [-0.5, 0.375], [1.5, 0.375]];

/// A float texture format: which of red, green, blue and alpha it has, the channel of the border
/// colour each of them takes on Metal, the bytes of its one texel value, and what a shader reads
/// of each channel it has.
struct Format {
    format: PixelFormat,
    present: [bool; 4],
    border_source: [usize; 4],
    texel: &'static [u8],
    value: [f32; 4],
}

fn unorm(c: u8) -> f32 {
    f32::from(c) / 255.0
}

/// Each channel's border colour channel, its own.
const OWN: [usize; 4] = [0, 1, 2, 3];

fn formats() -> [Format; 8] {
    // Half-float bits of 0.25, 0.5, 0.75 and 0.125, little-endian.
    const R16: &[u8] = &[0x00, 0x34];
    const RG16: &[u8] = &[0x00, 0x34, 0x00, 0x38];
    const RGBA16: &[u8] = &[0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x30];
    let rgba8 = [unorm(0x40), unorm(0x80), unorm(0xc0), unorm(0x20)];
    let halves = [0.25, 0.5, 0.75, 0.125];
    [
        Format {
            format: PixelFormat::R8Unorm,
            present: [true, false, false, false],
            border_source: OWN,
            texel: &[0x40],
            value: rgba8,
        },
        Format {
            format: PixelFormat::RG8Unorm,
            present: [true, true, false, false],
            border_source: OWN,
            texel: &[0x40, 0x80],
            value: rgba8,
        },
        Format {
            format: PixelFormat::RGBA8Unorm,
            present: [true; 4],
            border_source: OWN,
            texel: &[0x40, 0x80, 0xc0, 0x20],
            value: rgba8,
        },
        Format {
            format: PixelFormat::BGRA8Unorm,
            present: [true; 4],
            border_source: OWN,
            texel: &[0xc0, 0x80, 0x40, 0x20],
            value: rgba8,
        },
        Format {
            format: PixelFormat::A8Unorm,
            present: [false, false, false, true],
            border_source: [0, 1, 2, 0],
            texel: &[0x20],
            value: rgba8,
        },
        Format {
            format: PixelFormat::R16Float,
            present: [true, false, false, false],
            border_source: OWN,
            texel: R16,
            value: halves,
        },
        Format {
            format: PixelFormat::RG16Float,
            present: [true, true, false, false],
            border_source: OWN,
            texel: RG16,
            value: halves,
        },
        Format {
            format: PixelFormat::RGBA16Float,
            present: [true; 4],
            border_source: OWN,
            texel: RGBA16,
            value: halves,
        },
    ]
}

/// The swizzles, by name: each output channel's source.
fn swizzles() -> [(&'static str, TextureSwizzleChannels); 6] {
    use TextureSwizzle::{Alpha as A, Blue as B, Green as G, One as O, Red as R, Zero as Z};
    [
        ("rgba", TextureSwizzleChannels::DEFAULT),
        ("rrr1", TextureSwizzleChannels::new(R, R, R, O)),
        ("argb", TextureSwizzleChannels::new(A, R, G, B)),
        ("01rg", TextureSwizzleChannels::new(Z, O, R, G)),
        ("bgra", TextureSwizzleChannels::new(B, G, R, A)),
        ("ga0r", TextureSwizzleChannels::new(G, A, Z, R)),
    ]
}

/// The runtime samplers: a name, the address mode of `s` and of `t` and `r`, and the border
/// colour. Lanes leave the texture along `s` alone. The mixed ones clamp to zero on one axis and
/// to the border on the other, which plain Metal makes and reads the border colour through on
/// every border axis.
const SAMPLERS: [(&str, [SamplerAddressMode; 2], SamplerBorderColor); 10] = {
    use SamplerAddressMode::{ClampToBorderColor as Border, ClampToZero as Zero};
    use SamplerBorderColor::{OpaqueBlack, OpaqueWhite, TransparentBlack};
    [
        ("transparent", [Border, Border], TransparentBlack),
        ("white", [Border, Border], OpaqueWhite),
        ("zero", [Zero, Zero], TransparentBlack),
        ("zero white", [Zero, Zero], OpaqueWhite),
        ("zero black", [Zero, Zero], OpaqueBlack),
        ("black", [Border, Border], OpaqueBlack),
        ("zero s, border t, white", [Zero, Border], OpaqueWhite),
        ("border s, zero t, white", [Border, Zero], OpaqueWhite),
        ("zero s, border t, black", [Zero, Border], OpaqueBlack),
        (
            "zero s, border t, transparent",
            [Zero, Border],
            TransparentBlack,
        ),
    ]
};

/// Metal's texel of `f` through `swizzle` whose channels are `rgba`'s (picked by
/// `f.border_source` when `border`), a missing red, green or blue 0 and a missing alpha 1.
fn metal_texel(
    f: &Format,
    rgba: [f32; 4],
    border: bool,
    swizzle: TextureSwizzleChannels,
) -> [f32; 4] {
    let mut v = [0.0, 0.0, 0.0, 1.0];
    for c in 0..4 {
        if f.present[c] {
            v[c] = rgba[if border { f.border_source[c] } else { c }];
        }
    }
    let pick = |s: TextureSwizzle| match s {
        TextureSwizzle::Zero => 0.0,
        TextureSwizzle::One => 1.0,
        TextureSwizzle::Red => v[0],
        TextureSwizzle::Green => v[1],
        TextureSwizzle::Blue => v[2],
        TextureSwizzle::Alpha => v[3],
        other => panic!("a swizzle this test does not know: {other:?}"),
    };
    [
        pick(swizzle.red),
        pick(swizzle.green),
        pick(swizzle.blue),
        pick(swizzle.alpha),
    ]
}

/// What `border` clamps to, before the format's channels: the sampler's border colour, under
/// clamp-to-zero as under clamp-to-border.
fn border_rgba(border: SamplerBorderColor) -> [f32; 4] {
    match border {
        SamplerBorderColor::TransparentBlack => [0.0; 4],
        SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
        SamplerBorderColor::OpaqueWhite => [1.0; 4],
        other => panic!("a border this test does not know: {other:?}"),
    }
}

fn state(device: &Device) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("a2_clamp_to_zero.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("a2_clamp_to_zero").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A 4x4 texture of `format`, every texel `texel`, made with `swizzle`.
fn texture(
    device: &Device,
    format: PixelFormat,
    texel: &[u8],
    swizzle: TextureSwizzleChannels,
) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, 4, 4, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    descriptor.swizzle = swizzle;
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = (0..16).flat_map(|_| texel.iter().copied()).collect();
    let row = 4 * texel.len() as u64;
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, &bytes, row, 0)
        .unwrap();
    texture
}

/// What every dispatch shares: the pipeline, the queue, the integer textures, the coordinates.
struct Fixed {
    state: ComputePipelineState,
    queue: CommandQueue,
    uint: Texture,
    int: Texture,
    coords: Buffer,
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
    fixed: &Fixed,
    float: &Texture,
    sampler: &SamplerState,
    out: &Buffer,
) -> Result<(), Error> {
    encoder.set_compute_pipeline_state(&fixed.state)?;
    encoder.set_texture(Some(float), 0)?;
    encoder.set_texture(Some(&fixed.uint), 1)?;
    encoder.set_texture(Some(&fixed.int), 2)?;
    encoder.set_sampler_state(Some(sampler), 0)?;
    encoder.set_buffer(Some(&fixed.coords), 0, 0)?;
    encoder.set_buffer(Some(out), 0, 1)?;
    let lanes = COORDS.len() as u64;
    encoder.dispatch_threads(Size::new(lanes, 1, 1), Size::new(lanes, 1, 1))
}

/// What one dispatch read, `float` bound with the runtime sampler of the address modes `[s, tr]`
/// (`s`'s, and `t`'s and `r`'s) and `border`:
/// per lane, five `uint4`s (constexpr nearest and linear clamp-to-zero, the runtime sampler, then
/// the uint and int textures).
fn dispatch(
    device: &Device,
    fixed: &Fixed,
    float: &Texture,
    [s, tr]: [SamplerAddressMode; 2],
    border: SamplerBorderColor,
) -> Result<Vec<[u32; 4]>, Error> {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.s_address_mode = s;
    descriptor.t_address_mode = tr;
    descriptor.r_address_mode = tr;
    descriptor.border_color = border;
    let sampler = device.new_sampler_state(&descriptor)?;
    let lanes = COORDS.len() as u64;
    let out = buffer_with(device, &vec![0xab; (lanes * 5 * 16) as usize]).unwrap();
    let command_buffer = fixed.queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    let encoded = encode(&mut encoder, fixed, float, &sampler, &out);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    encoded?;
    let bytes = read_buffer(&out, 0, lanes * 5 * 16).unwrap();
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

/// Whether `got` is `metal`: border texels bit for bit (each channel is 0 or 1), the inside texel
/// within one part in a million (a format conversion's rounding, which this test is not about).
fn same(got: [u32; 4], metal: [f32; 4], inside: bool) -> bool {
    (0..4).all(|c| {
        if inside {
            (f32::from_bits(got[c]) - metal[c]).abs() <= 1e-6
        } else {
            got[c] == metal[c].to_bits()
        }
    })
}

/// Every texel each sample reads through a swizzled texture, inside and at the border, is
/// Metal's: the border colour in the format's channels, then the swizzle. A combination the
/// device refuses, typed, is printed and counted apart: a refusal, not an answer.
#[test]
fn border_texels_through_swizzled_textures_are_metal_s() {
    on_device(|device, _| {
        let coords: Vec<u8> = COORDS
            .iter()
            .flatten()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let fixed = Fixed {
            state: state(device),
            queue: device.new_command_queue().unwrap(),
            uint: texture(
                device,
                PixelFormat::R32Uint,
                &1u32.to_le_bytes(),
                TextureSwizzleChannels::DEFAULT,
            ),
            int: texture(
                device,
                PixelFormat::R32Sint,
                &(-1i32).to_le_bytes(),
                TextureSwizzleChannels::DEFAULT,
            ),
            coords: buffer_with(device, &coords).unwrap(),
        };
        let mut wrong = Vec::new();
        let mut checked = 0usize;
        let mut refused = 0usize;
        for f in formats() {
            for (swizzle_name, swizzle) in swizzles() {
                let float = texture(device, f.format, f.texel, swizzle);
                for (sampler_name, address, border) in SAMPLERS {
                    let at = format!("{:?} {swizzle_name} {sampler_name}", f.format);
                    let read = match dispatch(device, &fixed, &float, address, border) {
                        Ok(read) => read,
                        Err(error) => match refusal(&error) {
                            Some(u) => {
                                println!("BORDER-REFUSED {}: {at}: {u:?}", device.name());
                                refused += 1;
                                continue;
                            }
                            None => panic!("{at}: {error}"),
                        },
                    };
                    let inside = metal_texel(&f, f.value, false, swizzle);
                    let zero = metal_texel(&f, [0.0; 4], true, swizzle);
                    let runtime = metal_texel(&f, border_rgba(border), true, swizzle);
                    for lane in 0..COORDS.len() {
                        let reads = [
                            ("constexpr zero nearest", zero),
                            ("constexpr zero linear", zero),
                            ("runtime sampler", runtime),
                        ];
                        for (k, (what, outside)) in reads.into_iter().enumerate() {
                            let metal = if lane == 0 { inside } else { outside };
                            let got = read[lane * 5 + k];
                            checked += 1;
                            if !same(got, metal, lane == 0) {
                                wrong.push(format!(
                                    "{at} lane {lane} {what}: {got:08x?}, Metal {:08x?}",
                                    metal.map(f32::to_bits)
                                ));
                            }
                        }
                    }
                }
            }
        }
        for line in &wrong {
            println!("BORDER-DIFFERS {}: {line}", device.name());
        }
        println!(
            "BORDER {}: {checked} texels checked, {} differ from Metal, {refused} dispatches \
             refused",
            device.name(),
            wrong.len()
        );
        assert!(wrong.is_empty(), "{} texels differ from Metal", wrong.len());
    });
}
