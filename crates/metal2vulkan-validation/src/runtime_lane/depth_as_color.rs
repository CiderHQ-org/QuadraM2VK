//! A device test of depth textures read through color texture types on the device the cfg selects,
//! under the validation layer. Metal reads a depth texel through `texture2d<float>` as `(d, d, d,
//! 1)`: a read, nearest, linear, gradient and implicit samples answer it, gathers of x, y and z
//! answer the four depths and gathers of w four ones (Apple M4, `Depth16Unorm`,
//! `Depth32Float` and `Depth32Float_Stencil8`). Vulkan converts a depth texel to `(d, 0, 0, 1)`
//! before the view's component mapping, which the sampled view's `(R, R, R, ONE)` replaces.
//!
//! Each of those formats the device makes holds 16 distinct depths, exact in 16-bit unorm, filled
//! by a blit, and is read through owned fixtures: `denormal_texels` (a kernel: nearest samples,
//! one with a wrapped texel offset, gathers of x, y and w through constexpr samplers, and a nearest
//! sample of layer 1 of a 2D array of the same format), `rrp_sw_read_f` (a kernel's `read` of
//! texel (0, 0)), and `tob_read` and `tob_sample` (fragment functions' `read`, and implicit
//! samples through a nearest and a linear runtime sampler at the centre of each pixel of a 4x4
//! `RGBA32Float` target, where the linear filter weighs one texel).

use super::depth_compares::pipeline_of;
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandQueue, ComputePipelineState, Device, LoadAction, Origin,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, Size, StoreAction, Texture, TextureDescriptor,
};

/// The textures' side.
const SIDE: u64 = 4;

/// The depths' 16-bit codes, row by row: distinct, the ends included, and texel (0, 0) not zero
/// (the kernel's `read` reads it).
const CODES: [u16; 16] = [
    12345, 0, 65535, 16384, 1, 65534, 32768, 32767, 54321, 4096, 61440, 777, 30000, 40000, 21845,
    49151,
];

/// The depth formats Metal's answer was measured for.
const FORMATS: [PixelFormat; 3] = [
    PixelFormat::Depth16Unorm,
    PixelFormat::Depth32Float,
    PixelFormat::Depth32FloatStencil8,
];

/// The code of texel `(x, y)` of `layer`: layer 0 holds [`CODES`], layer 1 them reversed.
fn code(layer: u64, x: u64, y: u64) -> u16 {
    let i = (SIDE * y + x) as usize;
    match layer {
        0 => CODES[i],
        _ => CODES[CODES.len() - 1 - i],
    }
}

/// The texel bytes of `layer` in `pf`, and the bytes a texel: a `Depth16Unorm` texel its code,
/// a 32-bit float depth `code / 65535` in `f32` (the depth plane alone of `Depth32Float_Stencil8`,
/// which the blit fills with `DepthFromDepthStencil`).
fn layer_bytes(pf: PixelFormat, layer: u64) -> (Vec<u8>, u64) {
    let codes = (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| code(layer, x, y)));
    match pf {
        PixelFormat::Depth16Unorm => (codes.flat_map(u16::to_le_bytes).collect(), 2),
        _ => (
            codes
                .flat_map(|c| ((f64::from(c) / 65535.0) as f32).to_le_bytes())
                .collect(),
            4,
        ),
    }
}

/// Whether `x` is the depth of `code` (a unorm conversion within 2^-20).
fn near(x: f32, code: u16) -> bool {
    (f64::from(x) - f64::from(code) / 65535.0).abs() <= 1.0 / f64::from(1u32 << 20)
}

/// Whether `got` is Metal's `(d, d, d, 1)` of `code`.
fn depth_texel(got: [f32; 4], code: u16) -> bool {
    near(got[0], code)
        && got[1].to_bits() == got[0].to_bits()
        && got[2].to_bits() == got[0].to_bits()
        && got[3] == 1.0
}

/// A private texture of `pf` that shaders read (a 2D array of two layers where `array`), each
/// layer filled by a blit; none where the device makes no such texture.
fn depth_texture(
    device: &Device,
    queue: &CommandQueue,
    pf: PixelFormat,
    array: bool,
) -> Option<Texture> {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let layers = if array {
        d.texture_type = TextureType::Type2DArray;
        d.array_length = 2;
        2
    } else {
        1
    };
    let texture = match device.new_texture(&d) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: no {pf:?} texture ({e:?})", device.name());
            return None;
        }
    };
    let option = match pf {
        PixelFormat::Depth32FloatStencil8 => BlitOption::DepthFromDepthStencil,
        _ => BlitOption::None,
    };
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let mut held = Vec::new();
    for layer in 0..layers {
        let (bytes, per) = layer_bytes(pf, layer);
        let source = buffer_with(device, &bytes).unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            SIDE * per,
            SIDE * SIDE * per,
            Size::new(SIDE, SIDE, 1),
            &texture,
            layer,
            0,
            Origin::new(0, 0, 0),
            option,
        )
        .unwrap();
        held.push(source);
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    Some(texture)
}

/// The `float4`s kernel `state` writes reading `textures` at texture slots 0 up, `jobs` at buffer
/// 0 (one `float4` each), `per_job` outputs each at buffer 1.
fn dispatch(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    textures: &[&Texture],
    jobs: &[[f32; 4]],
    per_job: usize,
) -> Vec<[f32; 4]> {
    let bytes: Vec<u8> = jobs
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let input = buffer_with(device, &bytes).unwrap();
    let length = (jobs.len().max(1) * per_job * 16) as u64;
    let output = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, texture) in textures.iter().enumerate() {
        encoder.set_texture(Some(*texture), index as u32).unwrap();
    }
    match jobs.is_empty() {
        true => encoder.set_buffer(Some(&output), 0, 0).unwrap(),
        false => {
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&output), 0, 1).unwrap();
        }
    }
    let n = jobs.len().max(1) as u64;
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(n, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    read_buffer(&output, 0, length)
        .unwrap()
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect()
}

/// The texels of a 4x4 `RGBA32Float` target that `pipeline` (`tob_rect` and a fragment function)
/// draws over the whole of, reading `texture` at fragment texture slot 0 (through `sampler` at
/// sampler slot 0, where given), row by row.
fn draw(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &RenderPipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<[f32; 4]> {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let rect: Vec<u8> = [-1.0f32, -1.0, 1.0, 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let rect = buffer_with(device, &rect).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&rect), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    if sampler.is_some() {
        encoder.set_fragment_sampler_state(sampler, 0).unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let mut bytes = vec![0u8; (16 * SIDE * SIDE) as usize];
    target
        .get_bytes(
            &mut bytes,
            16 * SIDE,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
        })
        .collect()
}

/// Every read, sample and gather of a depth texture through a color texture type answers
/// Metal's `(d, d, d, 1)`, its gathers the depths and of w ones (module docs).
#[test]
fn a_depth_texture_read_as_a_color_reads_metals_texel() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let queue = device.new_command_queue().unwrap();
        let ll = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        let texels = pipeline_of(&device, &ll("denormal_texels"), "denormal_texels");
        let read = pipeline_of(&device, &ll("rrp_swizzled_storage"), "rrp_sw_read_f");
        let library = device.new_library_with_air(&ll("tob_renders")).unwrap();
        let render = |fragment: &str| {
            let function = |name| library.new_function_with_name(name).unwrap();
            let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
                function("tob_rect").function_descriptor(),
            ));
            descriptor.fragment = Some(StageSource::Function(
                function(fragment).function_descriptor(),
            ));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let (tob_read, tob_sample) = (render("tob_read"), render("tob_sample"));
        let sampler = |filter| {
            let mut d = SamplerDescriptor::new();
            d.min_filter = filter;
            d.mag_filter = filter;
            device.new_sampler_state(&d).unwrap()
        };
        let (nearest, linear) = (
            sampler(SamplerMinMagFilter::Nearest),
            sampler(SamplerMinMagFilter::Linear),
        );
        let centres: Vec<(u64, u64)> = (0..SIDE)
            .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
            .collect();
        let corners: Vec<(u64, u64)> = (0..SIDE - 1)
            .flat_map(|j| (0..SIDE - 1).map(move |i| (i, j)))
            .collect();
        let at = |u: f64, v: f64| [u as f32, v as f32, 0.0, 0.0];
        let side = SIDE as f64;
        let mut compared = 0usize;
        let mut wrong = Vec::new();
        let mut made = 0usize;
        for pf in FORMATS {
            let (Some(plain), Some(array)) = (
                depth_texture(&device, &queue, pf, false),
                depth_texture(&device, &queue, pf, true),
            ) else {
                continue;
            };
            made += 1;
            let mut check = |what: String, got: [f32; 4], want: &dyn Fn([f32; 4]) -> bool| {
                compared += 1;
                if !want(got) {
                    wrong.push(format!("{pf:?} {what}: {got:?}"));
                }
            };
            let textures = [&plain, &plain, &array];
            // Samples at each texel's centre: `r.sample` (out 1), `rgba.sample` (4), the wrapped
            // sample offset by (1, -1) (6), and layer 1's sample (7).
            let jobs: Vec<[f32; 4]> = centres
                .iter()
                .map(|&(x, y)| at((x as f64 + 0.5) / side, (y as f64 + 0.5) / side))
                .collect();
            let out = dispatch(&device, &queue, &texels, &textures, &jobs, 8);
            for (j, &(x, y)) in centres.iter().enumerate() {
                let own = code(0, x, y);
                let offset = code(0, (x + 1) % SIDE, (y + SIDE - 1) % SIDE);
                let layer = code(1, x, y);
                for (k, c, what) in [
                    (1, own, "sample"),
                    (4, own, "sample of slot 1"),
                    (6, offset, "wrapped sample offset (1, -1)"),
                    (7, layer, "array layer 1 sample"),
                ] {
                    check(format!("{what} at ({x}, {y})"), out[8 * j + k], &|g| {
                        depth_texel(g, c)
                    });
                }
            }
            // Gathers at each inner corner, in gather order: x (out 0, and 5 through a repeating
            // sampler), y (2), w (3).
            let jobs: Vec<[f32; 4]> = corners
                .iter()
                .map(|&(i, j)| at((i + 1) as f64 / side, (j + 1) as f64 / side))
                .collect();
            let out = dispatch(&device, &queue, &texels, &textures, &jobs, 8);
            for (n, &(i, j)) in corners.iter().enumerate() {
                let taps = [
                    code(0, i, j + 1),
                    code(0, i + 1, j + 1),
                    code(0, i + 1, j),
                    code(0, i, j),
                ];
                let depths = |g: [f32; 4]| (0..4).all(|k| near(g[k], taps[k]));
                for (k, what) in [(0, "gather x"), (5, "gather x, repeating"), (2, "gather y")] {
                    check(
                        format!("{what} at corner ({i}, {j})"),
                        out[8 * n + k],
                        &depths,
                    );
                }
                check(
                    format!("gather w at corner ({i}, {j})"),
                    out[8 * n + 3],
                    &|g| g == [1.0; 4],
                );
            }
            // A kernel's read of texel (0, 0).
            let out = dispatch(&device, &queue, &read, &[&plain], &[], 1);
            check("kernel read of (0, 0)".into(), out[0], &|g| {
                depth_texel(g, code(0, 0, 0))
            });
            // Fragment functions' read and implicit samples at each pixel.
            for (what, pipeline, sampler) in [
                ("fragment read", &tob_read, None),
                ("fragment nearest sample", &tob_sample, Some(&nearest)),
                ("fragment linear sample", &tob_sample, Some(&linear)),
            ] {
                let out = draw(&device, &queue, pipeline, &plain, sampler);
                for &(x, y) in &centres {
                    let c = code(0, x, y);
                    check(
                        format!("{what} at ({x}, {y})"),
                        out[(SIDE * y + x) as usize],
                        &|g| depth_texel(g, c),
                    );
                }
            }
        }
        assert!(made > 0, "{name}: no depth texture made");
        eprintln!(
            "{name}: {compared} reads of {made} depth formats compared, {} not Metal's",
            wrong.len()
        );
        assert!(wrong.is_empty(), "{name}: {wrong:#?}");
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
