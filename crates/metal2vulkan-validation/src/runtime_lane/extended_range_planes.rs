//! A two-plane extended-range texture held as two single-plane textures on the device the cfg
//! selects, under the validation layer.
//!
//! A consumer holds Apple's `RGB10A8_2P_XR10` so: plane 0 viewed in `BGR10_XR` (or its sRGB twin)
//! and the alpha plane in `ARGB8Unorm`, whose first component is the texel's second byte. The
//! owned fixture `xr_planes` joins the two into a `BGRA10_XR` texture (or its sRGB twin) and splits
//! one back, one thread per texel, reading and writing `half`; plane 0 is written with alpha 1 and
//! the alpha view with the alpha in every component. Every input code of every channel occurs,
//! with the bits Metal does not read set.
//!
//! Metal's answer: the SHA-256 of each output texture's bytes, for every pairing of the formats,
//! as an Apple M3 Max (macOS 26.7) and an Apple M4 (macOS 27.0.1) wrote them from the same inputs,
//! plain and under Metal's validation layer (which keep the bytes). A differing output is printed
//! whole, on one `XR-PLANES-DIFFERS` line, to compare texel by texel with the bytes the probes
//! keep.

use super::degrade_notices::on_device;
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, CommandQueue, ComputePipelineState, Device, PipelineOptions, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Each texture's side: 32 x 32 texels, one per code.
const SIDE: u64 = 32;

/// The texels of a texture.
const COUNT: usize = (SIDE * SIDE) as usize;

/// The alpha view's format.
const ALPHA: PixelFormat = PixelFormat::ARGB8Unorm;

/// The SHA-256 of plane 0's, the alpha view's and the joined texture's input bytes (the
/// generator below, which the probes' Metal harness writes in Swift).
const INPUT_COLOR: &str = "76bb1c1c9bd72a95199eb8fa090b7efbaf536dc9621b35343e3123356d0dd1c5";
const INPUT_ALPHA: &str = "0add0982965c72b0d76c77c3d019a78d2e0cfbb61fb8952f9b75d1194d8496dd";
const INPUT_JOINED: &str = "9e268ae2fa3dfc66242184ae46a352fd93c03f942d83a6022084e74e402b467c";

/// Metal's outputs: the kernel, plane 0's format, the joined texture's format, the output
/// texture, and the SHA-256 of its bytes.
const METAL: [(&str, PixelFormat, PixelFormat, &str, &str); 12] = [
    (
        "xr_planes_join",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XR,
        "joined",
        "dc47fe6bbda17bc06a0cb9c4a242ec886f0019ca40322606303f33ee65fed3cd",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XR,
        "color",
        "765e13747416e861632a53cd463d73a29eb410a5b7f357a216e60980aad80dd4",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XR,
        "alpha",
        "e70f6b7127321a563d2194c38944c37b4b0d63f568fb8503533562a1bbd76651",
    ),
    (
        "xr_planes_join",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XRSRGB,
        "joined",
        "37d9ed39be88417ee375f9dfc47bf6aced3b3569df392e3f802c4f7113fa5607",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XRSRGB,
        "color",
        "8b7af345769a66b5240f704813785f7ec279176791b7b55b29b9f5322d390a4a",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XR,
        PixelFormat::BGRA10XRSRGB,
        "alpha",
        "e70f6b7127321a563d2194c38944c37b4b0d63f568fb8503533562a1bbd76651",
    ),
    (
        "xr_planes_join",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        "joined",
        "5d91000cdae931671196c8abf6c0ae1969c4bd32aced52a47d8591b079260c64",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        "color",
        "9a24bb3f6b9ea4de385dba2ca1c09ff94575ee30f71253985ed8e9d62c90cd62",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XR,
        "alpha",
        "e70f6b7127321a563d2194c38944c37b4b0d63f568fb8503533562a1bbd76651",
    ),
    (
        "xr_planes_join",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XRSRGB,
        "joined",
        "dc47fe6bbda17bc06a0cb9c4a242ec886f0019ca40322606303f33ee65fed3cd",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XRSRGB,
        "color",
        "765e13747416e861632a53cd463d73a29eb410a5b7f357a216e60980aad80dd4",
    ),
    (
        "xr_planes_split",
        PixelFormat::BGR10XRSRGB,
        PixelFormat::BGRA10XRSRGB,
        "alpha",
        "e70f6b7127321a563d2194c38944c37b4b0d63f568fb8503533562a1bbd76651",
    ),
];

/// Texel `n`'s codes: red, green, blue, alpha.
fn codes(n: usize) -> [u32; 4] {
    let n = n as u32;
    [(n * 7 + 3) % 1024, 1023 - n, n, (n * 13 + 5) % 1024]
}

/// Plane 0: `BGR10_XR` words, blue in bits 0 to 9, green 10 to 19, red 20 to 29, and bits 30 and
/// 31 set to `n % 4`.
fn color_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            let [r, g, b, _] = codes(n);
            ((n as u32 % 4) << 30 | r << 20 | g << 10 | b).to_le_bytes()
        })
        .collect()
}

/// The alpha view: four bytes per texel, the second (its first component) taking every value.
fn alpha_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            [
                (n * 11 + 7) as u8,
                (n * 37 + 11) as u8,
                (n * 13) as u8,
                (n * 17 + 3) as u8,
            ]
        })
        .collect()
}

/// The joined texture: `BGRA10_XR`'s four 16-bit fields (blue, green, red, alpha), each code in
/// bits 6 to 15 and the low six bits set.
fn joined_bytes() -> Vec<u8> {
    (0..COUNT)
        .flat_map(|n| {
            let [r, g, b, a] = codes(n);
            let junk = n as u32 % 64;
            [
                b << 6 | junk,
                g << 6 | (63 - junk),
                r << 6 | junk,
                a << 6 | 42,
            ]
            .into_iter()
            .flat_map(|f| (f as u16).to_le_bytes())
        })
        .collect()
}

/// The bytes of one texel of `format`.
fn texel_bytes(format: PixelFormat) -> u64 {
    match format {
        PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => 8,
        _ => 4,
    }
}

/// A `SIDE` x `SIDE` texture of `format` read and written by kernels, holding `bytes`.
fn texture(device: &Device, format: PixelFormat, bytes: &[u8]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {format:?} texture: {e}"));
    let row = SIDE * texel_bytes(format);
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, bytes, row, 0)
        .unwrap();
    texture
}

/// A texture of `format` holding bytes `0xab`, for a kernel to write.
fn blank(device: &Device, format: PixelFormat) -> Texture {
    texture(
        device,
        format,
        &vec![0xab; COUNT * texel_bytes(format) as usize],
    )
}

/// The bytes of `texture`, of `format`.
fn bytes_of(texture: &Texture, format: PixelFormat) -> Vec<u8> {
    let row = SIDE * texel_bytes(format);
    let mut bytes = vec![0; (row * SIDE) as usize];
    texture
        .get_bytes(&mut bytes, row, 0, Region::new_2d(0, 0, SIDE, SIDE), 0, 0)
        .unwrap();
    bytes
}

/// The fixture's kernel `name`.
fn state(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("xr_planes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs `state` over the textures bound at 0, 1 and 2, one thread per texel.
fn dispatch(queue: &CommandQueue, state: &ComputePipelineState, textures: [&Texture; 3]) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, texture) in textures.into_iter().enumerate() {
        encoder.set_texture(Some(texture), index as u32).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// The SHA-256 of `bytes`, in hex.
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Metal's digest of `kernel`'s output `texture` for plane 0 in `color` and the joined texture
/// in `joined`.
fn metal(kernel: &str, color: PixelFormat, joined: PixelFormat, texture: &str) -> &'static str {
    METAL
        .iter()
        .find(|m| m.0 == kernel && m.1 == color && m.2 == joined && m.3 == texture)
        .map(|m| m.4)
        .unwrap_or_else(|| panic!("no Metal digest of {kernel} {color:?} {joined:?} {texture}"))
}

/// The generator writes the bytes the probes' Metal harness wrote, by digest.
#[test]
fn the_inputs_are_the_harness_s() {
    assert_eq!(digest(&color_bytes()), INPUT_COLOR);
    assert_eq!(digest(&alpha_bytes()), INPUT_ALPHA);
    assert_eq!(digest(&joined_bytes()), INPUT_JOINED);
}

/// Every output texture of both kernels, for every pairing of plane 0's and the joined texture's
/// formats, holds Metal's bytes.
#[test]
fn two_extended_range_planes_join_and_split_as_metal_does() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let join = state(device, "xr_planes_join");
        let split = state(device, "xr_planes_split");
        let mut wrong = Vec::new();
        for color in [PixelFormat::BGR10XR, PixelFormat::BGR10XRSRGB] {
            for joined in [PixelFormat::BGRA10XR, PixelFormat::BGRA10XRSRGB] {
                let plane = texture(device, color, &color_bytes());
                let alpha = texture(device, ALPHA, &alpha_bytes());
                let target = blank(device, joined);
                dispatch(&queue, &join, [&plane, &alpha, &target]);
                let mut outputs = vec![("xr_planes_join", "joined", bytes_of(&target, joined))];
                let source = texture(device, joined, &joined_bytes());
                let plane = blank(device, color);
                let alpha = blank(device, ALPHA);
                dispatch(&queue, &split, [&source, &plane, &alpha]);
                outputs.push(("xr_planes_split", "color", bytes_of(&plane, color)));
                outputs.push(("xr_planes_split", "alpha", bytes_of(&alpha, ALPHA)));
                for (kernel, what, bytes) in outputs {
                    let got = digest(&bytes);
                    let want = metal(kernel, color, joined, what);
                    if got != want {
                        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
                        wrong.push(format!(
                            "{kernel} {color:?} {joined:?} {what}: {got}, Metal {want}; bytes \
                             {hex}"
                        ));
                    }
                }
            }
        }
        for line in &wrong {
            println!("XR-PLANES-DIFFERS {}: {line}", device.name());
        }
        assert!(
            wrong.is_empty(),
            "{} outputs differ from Metal",
            wrong.len()
        );
    });
}
