//! A kernel's storage reads and writes of the 8-bit sRGB formats `R8Unorm_sRGB`, `RG8Unorm_sRGB`,
//! `RGBA8Unorm_sRGB`, `RGBX8Unorm_sRGB` and `BGRA8Unorm_sRGB` on the device the cfg selects, under
//! the validation layer, against Metal's answers (an M4 under macOS 27.0.1, our own
//! probe `srgb-storage`; `R8Unorm_sRGB`, `RG8Unorm_sRGB` and `RGBA8Unorm_sRGB` again on an M3 Max
//! under macOS 27.0.1, our own probe `srgb8`: every texel of the same rule).
//!
//! Metal's answers, recorded there for the values below: a kernel's write (`access::write` and
//! `read_write`, `float4` and `half4`) stores each color channel as its twelve-bit encode (a NaN
//! 0, the value clamped to `[0, 1]`, the exact product with 4095 rounded to nearest even, the
//! largest byte whose first encode index is at or below it) and alpha as the exact product with
//! 255 rounded to nearest even, the X field of `RGBX8Unorm_sRGB` green's byte: every texel equal
//! to `pack_float_to_srgb_unorm4x8` of the same values. A read (`read_write`, `float4` and
//! `half4`) of code `c` answers its twelve-bit step `k / 4095` correctly rounded (alpha `c / 255`,
//! 1 for `RGBX8Unorm_sRGB`; a channel the format lacks 0, alpha 1). The tables are Metal's own: the first encode indices are the color
//! table's (`bind::color_write_table`), the steps `round(decode(c / 255) * 4095)` of the sRGB
//! transfer (every one at least `1.6e-4` of a step from a half step).
//!
//! The values are the probe's, in its order: the `f32`s around each twelve-bit index's half step
//! (three ulps either side), each step's value and its neighbours, each alpha code's half step
//! (three ulps either side), the specials, and its 81,920 random values; and every half. Channel
//! `c` of texel `t` takes value `4t + (c + p) % 4` in pass `p` of 4, so each value reaches each
//! channel. The kernels are the owned fixture `fpv_srgb_storage`'s.

use super::degrade_notices::{on_device, on_device_with};
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::StorageImageClass;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Region, ResourceOptions,
    Size, Texture, TextureDescriptor,
};

/// The texture's width; its height holds every value.
const WIDTH: u64 = 256;

/// Each format, whether the probe measured its `read_write` access, its texel's bytes, and the
/// byte of each channel (red, green, blue, alpha) in its texel, `None` for a channel it lacks.
const FORMATS: [(PixelFormat, bool, usize, [Option<usize>; 4]); 5] = [
    (
        PixelFormat::R8UnormSRGB,
        true,
        1,
        [Some(0), None, None, None],
    ),
    (
        PixelFormat::RG8UnormSRGB,
        true,
        2,
        [Some(0), Some(1), None, None],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        true,
        4,
        [Some(0), Some(1), Some(2), Some(3)],
    ),
    (
        PixelFormat::RGBX8UnormSRGB,
        true,
        4,
        [Some(0), Some(1), Some(2), Some(3)],
    ),
    (
        PixelFormat::BGRA8UnormSRGB,
        false,
        4,
        [Some(2), Some(1), Some(0), Some(3)],
    ),
];

/// The byte of each twelve-bit index: the largest whose first encode index (the color table's
/// first word of each entry, `bind::color_write_table`) is at or below it.
fn encode_bytes() -> Vec<u8> {
    let first: Vec<u32> = metal2vulkan::bind::color_write_table()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|e| u32::from_le_bytes([e[0], e[1], e[2], e[3]]))
        .collect();
    (0..4096u32)
        .map(|i| (1..256).rev().find(|&b| first[b] <= i).unwrap_or(0) as u8)
        .collect()
}

/// Metal's byte of channel value `v`, a color channel's (`color`, through `bytes`, the
/// [`encode_bytes`]) or alpha's: a NaN 0, the value clamped to `[0, 1]`, its product with 4095
/// (255), exact in `f64`, rounded to nearest even.
fn code(bytes: &[u8], v: f32, color: bool) -> u8 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    match color {
        true => bytes[(f64::from(v) * 4095.0).round_ties_even() as usize],
        false => (f64::from(v) * 255.0).round_ties_even() as u8,
    }
}

/// `k / d` (`0 < k <= d`, `d` odd, so no ties) correctly rounded to `bits` significant bits: its
/// significand `q` and exponent `e`, the value `q * 2^-e`.
fn ratio(k: u64, d: u64, bits: u32) -> (u128, i32) {
    let (k, d) = (u128::from(k), u128::from(d));
    let mut e = 0;
    while k << e < d << (bits - 1) {
        e += 1;
    }
    let n = k << e;
    (n / d + u128::from(2 * (n % d) > d), e)
}

/// `k / d` correctly rounded to `f32`, as bits.
fn ratio_f32(k: u64, d: u64) -> u32 {
    match k {
        0 => 0,
        _ => {
            let (q, e) = ratio(k, d, 24);
            ((q as f64 * 2f64.powi(-e)) as f32).to_bits()
        }
    }
}

/// `k / d` correctly rounded to `half`, as bits (every nonzero quotient here a normal `half`).
fn ratio_half(k: u64, d: u64) -> u16 {
    if k == 0 {
        return 0;
    }
    let (mut q, mut e) = ratio(k, d, 11);
    if q == 1 << 11 {
        q >>= 1;
        e -= 1;
    }
    (((10 - e + 15) as u16) << 10) | (q as u16 - (1 << 10))
}

/// The value of `half` bits `h`, exactly (a NaN for a NaN).
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * match e {
        0 => m * 2f32.powi(-24),
        31 if m == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024.0 + m) * 2f32.powi(e - 25),
    }
}

/// The probe's floats, in its order (see the module docs).
fn floats() -> Vec<f32> {
    let next = |f: f32, k: i32| f32::from_bits((f.to_bits() as i64 + i64::from(k)) as u32);
    let mut v = Vec::new();
    for i in 0..=4096 {
        let x = ((f64::from(i) - 0.5) / 4095.0) as f32;
        v.extend((-3..=3).map(|k| next(x, k)));
    }
    for i in 0..4096 {
        let x = (f64::from(i) / 4095.0) as f32;
        v.extend((-1..=1).map(|k| next(x, k)));
    }
    for j in 0..=256 {
        let x = ((f64::from(j) - 0.5) / 255.0) as f32;
        v.extend((-3..=3).map(|k| next(x, k)));
    }
    let specials: [u32; 21] = [
        0x7fc0_0000,
        0x7f80_0001,
        0xffc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x3f80_0000,
        0x3f80_0001,
        0x3f7f_ffff,
        0x4000_0000,
        0x7f7f_ffff,
        0xbf00_0000,
        0xbf80_0000,
        0x3f00_0000,
        0x2edb_e6ff,
        0x3b4d_2e1c,
        0x3d25_aee6,
    ];
    v.extend(specials.map(f32::from_bits));
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut lcg = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f64 / f64::from(1u32 << 24)) as f32
    };
    v.extend((0..65536).map(|_| lcg()));
    v.extend((0..16384).map(|_| lcg() * 2.0 - 0.5));
    while v.len() % 4 != 0 {
        v.push(0.0);
    }
    v
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("fpv_srgb_storage.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// A shared texture of `pf` (`size` bytes a texel), [`WIDTH`] texels wide and `height` high, that
/// kernels read and write, holding `bytes` where given.
fn texture(
    device: &Device,
    (pf, size): (PixelFormat, u64),
    height: u64,
    bytes: Option<&[u8]>,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, WIDTH, height, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&d).unwrap();
    if let Some(bytes) = bytes {
        texture
            .replace_region(
                Region::new_2d(0, 0, WIDTH, height),
                0,
                0,
                bytes,
                WIDTH * size,
                0,
            )
            .unwrap();
    }
    texture
}

/// `texture`'s bytes (`size` a texel), `height` rows.
fn bytes_of(texture: &Texture, size: u64, height: u64) -> Vec<u8> {
    let mut out = vec![0u8; (WIDTH * height * size) as usize];
    texture
        .get_bytes(
            &mut out,
            WIDTH * size,
            0,
            Region::new_2d(0, 0, WIDTH, height),
            0,
            0,
        )
        .unwrap();
    out
}

/// Run `state` over every texel of `texture` (`height` rows) with `data` at buffer 0, and return
/// buffer 0's bytes.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    height: u64,
    data: &[u8],
) -> Vec<u8> {
    let data_buffer = buffer_with(device, data).unwrap();
    let width = buffer_with(device, &(WIDTH as u32).to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&data_buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&width), 0, 1).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(WIDTH / 16, height, 1), Size::new(16, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    read_buffer(&data_buffer, 0, data.len() as u64).unwrap()
}

/// The values a write kernel takes, each its value and its bytes in the kernel's buffer: the
/// probe's floats, or every half's bits (`half`).
fn write_values(half: bool) -> Vec<(f32, Vec<u8>)> {
    match half {
        true => (0..=u16::MAX)
            .map(|h| (half_value(h), h.to_le_bytes().to_vec()))
            .collect(),
        false => floats()
            .into_iter()
            .map(|v| (v, v.to_bits().to_le_bytes().to_vec()))
            .collect(),
    }
}

/// Every kernel write of the probe's floats and of every half into each format: Metal's bytes,
/// each texel's (the X field of `RGBX8Unorm_sRGB` green's byte), over bytes of `0xA5`.
#[test]
fn srgb_storage_writes_store_metals_codes() {
    on_device(|device, _| writes_store_metals_codes(device));
}

/// [`srgb_storage_writes_store_metals_codes`] on the device made without its storage accesses
/// without a format (`shaderStorageImageReadWithoutFormat`, `shaderStorageImageWriteWithoutFormat`
/// and `VK_KHR_format_feature_flags2`, under which a format's own bits would grant them at the
/// runtime's Vulkan 1.2 instance), as ARL's and Pascal's profiles state none for
/// `B8G8R8A8_UNORM`: a `BGRA8Unorm_sRGB` slot is then an `R8G8B8A8_UNORM` view that encodes and
/// exchanges red and blue (`StorageImageClass::SrgbEncodeBgra`), and stores Metal's bytes as the
/// twin's view does.
#[test]
fn srgb_storage_writes_through_rgba8_store_metals_codes() {
    on_device_with(
        |options| {
            options.withheld_features.extend([
                VkFeature::ShaderStorageImageReadWithoutFormat,
                VkFeature::ShaderStorageImageWriteWithoutFormat,
            ]);
            options
                .withheld_extensions
                .insert(VkExtension::KhrFormatFeatureFlags2);
        },
        |device, _| {
            let profile = device.compile_device().profile();
            let choice = storage_image_choice(PixelFormat::BGRA8UnormSRGB, profile).unwrap();
            assert_eq!(choice.class, StorageImageClass::SrgbEncodeBgra);
            writes_store_metals_codes(device);
        },
    );
}

fn writes_store_metals_codes(device: &Device) {
    let bytes = encode_bytes();
    let name = device.name().to_owned();
    let mut differ = Vec::new();
    for (pf, read_write, texel, at) in FORMATS {
        let rgbx = pf == PixelFormat::RGBX8UnormSRGB;
        let kernels: &[(&str, bool)] = match read_write {
            true => &[
                ("fpv_srgb_write", false),
                ("fpv_srgb_update", false),
                ("fpv_srgb_write_half", true),
                ("fpv_srgb_update_half", true),
            ],
            false => &[("fpv_srgb_write", false), ("fpv_srgb_write_half", true)],
        };
        for &(kernel, half) in kernels {
            let state = state(device, kernel);
            let list = write_values(half);
            let texels = list.len() / 4;
            let height = (texels as u64).div_ceil(WIDTH);
            let cells = (WIDTH * height) as usize;
            let size = list[0].1.len();
            let (mut wrong, mut example) = (0, None);
            for pass in 0..4 {
                let entry = |t: usize, c: usize| &list[4 * t + (c + pass) % 4];
                let mut data = vec![0u8; cells * 4 * size];
                for t in 0..texels {
                    for c in 0..4 {
                        let at = (4 * t + c) * size;
                        data[at..at + size].copy_from_slice(&entry(t, c).1);
                    }
                }
                let fill = vec![0xA5u8; cells * texel];
                let target = texture(device, (pf, texel as u64), height, Some(&fill));
                run(device, &state, &target, height, &data);
                let got = bytes_of(&target, texel as u64, height);
                for t in 0..texels {
                    let mut want = [0u8; 4];
                    for (c, byte) in at.into_iter().enumerate() {
                        if let Some(byte) = byte {
                            want[byte] = code(&bytes, entry(t, c).0, c < 3);
                        }
                    }
                    if rgbx {
                        want[3] = want[1];
                    }
                    let got = &got[texel * t..texel * (t + 1)];
                    if got != &want[..texel] {
                        wrong += 1;
                        example.get_or_insert((
                            (0..4).map(|c| entry(t, c).0).collect::<Vec<_>>(),
                            got.to_vec(),
                            want,
                        ));
                    }
                }
            }
            let example = example.map_or(String::new(), |e| format!(", e.g. {e:?}"));
            eprintln!(
                "SRGB write {pf:?} by {kernel} on {name}: {wrong} of {} texels differ from \
                 Metal's{example}",
                4 * texels
            );
            if wrong > 0 {
                differ.push(format!("{pf:?} by {kernel}: {wrong} of {}", 4 * texels));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "sRGB storage writes not Metal's: {differ:#?}"
    );
}

/// Every `read_write` read of every code in every channel of each format the probe read so:
/// Metal's quotients, `float` and `half`.
#[test]
fn srgb_storage_reads_are_metals_table() {
    on_device(|device, _| {
        let name = device.name().to_owned();
        let mut differ = Vec::new();
        for (pf, read_write, texel, at) in FORMATS {
            if !read_write {
                continue;
            }
            let rgbx = pf == PixelFormat::RGBX8UnormSRGB;
            let codes = |b: u32| [b, 255 - b, (37 * b) & 255, (b + 128) & 255];
            let mut bytes = vec![0u8; 256 * texel];
            for b in 0..256u32 {
                for (c, code) in codes(b).into_iter().enumerate() {
                    if let Some(byte) = at[c] {
                        bytes[texel * b as usize + byte] = code as u8;
                    }
                }
            }
            let source = texture(device, (pf, texel as u64), 1, Some(&bytes));
            for (kernel, half) in [
                ("fpv_srgb_read_write", false),
                ("fpv_srgb_read_write_half", true),
            ] {
                let size = if half { 8 } else { 16 };
                let state = state(device, kernel);
                let got = run(device, &state, &source, 1, &vec![0u8; 256 * size]);
                let mut wrong = 0;
                for b in 0..256u32 {
                    for (c, code) in codes(b).into_iter().enumerate() {
                        // A channel the format lacks reads 0, alpha 1.
                        let (k, d) = match (c, rgbx, at[c]) {
                            (3, true, _) | (3, _, None) => (1, 1),
                            (_, _, None) => (0, 1),
                            (3, false, Some(_)) => (u64::from(code), 255),
                            _ => (u64::from(super::srgb_step(code)), 4095),
                        };
                        let lane = &got[(b as usize * 4 + c) * size / 4..];
                        let (read, want) = match half {
                            true => (
                                u32::from(u16::from_le_bytes([lane[0], lane[1]])),
                                u32::from(ratio_half(k, d)),
                            ),
                            false => (
                                u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]),
                                ratio_f32(k, d),
                            ),
                        };
                        wrong += usize::from(read != want);
                    }
                }
                eprintln!(
                    "SRGB read {pf:?} by {kernel} on {name}: {wrong} of 1024 lanes differ from \
                     Metal's"
                );
                if wrong > 0 {
                    differ.push(format!("{pf:?} by {kernel}: {wrong} of 1024"));
                }
            }
        }
        assert!(
            differ.is_empty(),
            "sRGB storage reads not Metal's: {differ:#?}"
        );
    });
}
