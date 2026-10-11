//! Each group of state a draw depends on, changed alone between two draws of one render encoder,
//! reaches the second draw: the guard of every reuse the draw path makes of what an earlier draw
//! of the encoder settled or recorded. On the device the cfg selects, under the validation layer,
//! on a device that keeps no execution record, as a consumer's keeps none.
//!
//! Each case encodes three command buffers into a 16x16 `RGBA8Unorm` target cleared to zero (and,
//! for the cases that test depth or stencil, a `Depth32Float_Stencil8` texture cleared to 0.5 and
//! 0), each draw covering the whole target unless its state keeps it from pixels:
//!
//! - **together**: one encoder takes the case's shared state, draws, takes the change, and draws
//!   again;
//! - **apart**: one encoder takes the shared state and draws; a second, loading what the first
//!   stored, takes the shared state and the change, and draws;
//! - **unchanged**: as apart, without the change.
//!
//! An encoder's first draw reuses nothing an earlier encoder settled, so apart's second draw is
//! the changed state's own. Together must store apart's texels (and visibility result) exactly,
//! and apart must differ from unchanged, so the change is one the second draw shows. The
//! pipelines blend additively, so the texels count the draws that rendered: a change of which
//! draws render (culling, winding, the depth and stencil tests, a viewport, a scissor) shows too.
//!
//! The groups: the pipeline state; the vertex stage's inline bytes, buffer and buffer offset; the
//! fragment stage's inline bytes, buffer, buffer offset, texture (one of another format too) and
//! sampler; the viewport and the scissor rectangle; the blend color; the cull mode, the winding
//! and the triangle fill mode; the depth-stencil state, the stencil reference, the depth bias and
//! the depth clip mode; the visibility result mode; and the vertex amplification count, where the
//! device draws amplified.

use super::executions::on_device_with;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    BlendFactor, CompareFunction, CullMode, DepthClipMode, PixelFormat, PrimitiveType,
    SamplerAddressMode, TextureUsage, TriangleFillMode, Winding,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, DepthStencilState, Device, DeviceOptions, Library,
    LoadAction, PipelineOptions, Region, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, ScissorRect, StoreAction, Texture,
    TextureDescriptor, Viewport, VisibilityResultMode,
};

/// The target's side, in pixels.
const SIDE: u64 = 16;

/// The colors the fragment stage adds: a quarter of red, of green, and white.
const RED: [f32; 4] = [0.25, 0.0, 0.0, 0.25];
const GREEN: [f32; 4] = [0.0, 0.25, 0.0, 0.25];
const WHITE: [f32; 4] = [1.0; 4];
/// `RED` and `GREEN` as `RGBA8Unorm` texels.
const RED_TEXEL: [u8; 4] = [64, 0, 0, 64];
const GREEN_TEXEL: [u8; 4] = [0, 64, 0, 64];

/// `tob_rect`'s rectangles: the whole target, its left half, and its right half.
const FULL: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];
const LEFT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];
const RIGHT: [f32; 4] = [0.0, -1.0, 1.0, 1.0];

/// The pass a case draws in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    /// The color target alone.
    Color,
    /// The color target and the depth and stencil texture.
    Depth,
    /// The color target and the visibility buffer.
    Visibility,
}

/// One state group's case (the module docs).
struct Case {
    name: &'static str,
    pass: Pass,
    /// The vertices each draw draws.
    vertices: u32,
    /// The state both draws take.
    shared: fn(&Lane, &mut RenderCommandEncoder),
    /// The change before the second draw.
    change: fn(&Lane, &mut RenderCommandEncoder),
}

/// What a case's command buffer stored: the target's texels and the visibility result.
#[derive(Debug, PartialEq, Eq)]
struct Stored {
    texels: Vec<u8>,
    visibility: u64,
}

/// The lane's device, queue, and what the cases draw with.
struct Lane {
    device: Device,
    queue: CommandQueue,
    /// `tob_rect` and `tob_solid` adding the fragment's color.
    solid: RenderPipelineState,
    /// `tob_rect` and `tob_sample` adding the sampled color.
    sample: RenderPipelineState,
    /// `tob_rect` and `tob_solid` adding the blend color times the fragment's color.
    constant: RenderPipelineState,
    /// `rt6_full` and `rt6_half` adding half, with the depth and stencil texture's formats.
    depth: RenderPipelineState,
    /// `solid` made for two vertex amplifications, where the device makes it.
    amplified: Option<RenderPipelineState>,
    /// The rectangles and colors as buffers: one each, and both in one buffer, 16 bytes apart.
    left: Buffer,
    right: Buffer,
    rects: Buffer,
    red: Buffer,
    green: Buffer,
    colors: Buffer,
    /// 4x4 `RED_TEXEL` and `GREEN_TEXEL` textures, a 4x4 `RGBA16Float` one of `GREEN`, and a 2x1
    /// one of `RED_TEXEL` then `GREEN_TEXEL`.
    red_texture: Texture,
    green_texture: Texture,
    half_texture: Texture,
    stripes: Texture,
    /// Nearest samplers that clamp to the edge, and that repeat.
    clamp: SamplerState,
    repeat: SamplerState,
    /// Depth-stencil states: a depth test that never passes, one that always does, one of equal
    /// depths, and one that passes where the stencil equals the reference.
    never: DepthStencilState,
    always: DepthStencilState,
    equal: DepthStencilState,
    stencil_equal: DepthStencilState,
    target: Texture,
    depth_stencil: Texture,
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let library = |name: &str| -> Library {
            let ll = engine_dir().join(name);
            device
                .new_library_with_air(&llvm_assemble(&ll).unwrap())
                .unwrap()
        };
        let (tob, rt6) = (library("tob_renders.ll"), library("rt6_render_state.ll"));
        let function = |library: &Library, name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let adding = |source: BlendFactor| {
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
            color.blending_enabled = true;
            color.source_rgb_blend_factor = source;
            color.source_alpha_blend_factor = source;
            color.destination_rgb_blend_factor = BlendFactor::One;
            color.destination_alpha_blend_factor = BlendFactor::One;
            color
        };
        let make = |descriptor: &RenderPipelineDescriptor| {
            device.new_render_pipeline_state(descriptor, &PipelineOptions::default())
        };
        let tob_pipeline = |fragment: &str, source: BlendFactor| {
            let mut descriptor = RenderPipelineDescriptor::new(function(&tob, "tob_rect"));
            descriptor.fragment = Some(function(&tob, fragment));
            descriptor.color_attachments[0] = Some(adding(source));
            descriptor
        };
        let mut depth = RenderPipelineDescriptor::new(function(&rt6, "rt6_full"));
        depth.fragment = Some(function(&rt6, "rt6_half"));
        depth.color_attachments[0] = Some(adding(BlendFactor::One));
        depth.depth_format = Some(PixelFormat::Depth32FloatStencil8);
        depth.stencil_format = Some(PixelFormat::Depth32FloatStencil8);
        let mut amplified = tob_pipeline("tob_solid", BlendFactor::One);
        amplified.max_vertex_amplification_count = 2;
        let floats = |values: &[f32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            buffer_with(device, &bytes).unwrap()
        };
        let texture = |pf: PixelFormat, [w, h]: [u64; 2], texels: &[u8]| {
            let mut d = TextureDescriptor::texture_2d(pf, w, h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&d).unwrap();
            let row = texels.len() as u64 / h;
            texture
                .replace_region(Region::new_2d(0, 0, w, h), 0, 0, texels, row, 0)
                .unwrap();
            texture
        };
        let filled = |texel: [u8; 4]| texel.repeat(16);
        // `GREEN` as `RGBA16Float`: 0.25 is 0x3400.
        let half_green: Vec<u8> = [0u16, 0x3400, 0, 0x3400]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let sampler = |mode: SamplerAddressMode| {
            let mut d = SamplerDescriptor::new();
            d.s_address_mode = mode;
            d.t_address_mode = mode;
            device.new_sampler_state(&d).unwrap()
        };
        let state = |adjust: fn(&mut DepthStencilDescriptor)| {
            let mut d = DepthStencilDescriptor::default();
            adjust(&mut d);
            device.new_depth_stencil_state(&d).unwrap()
        };
        let attachment = |pf: PixelFormat, usage: TextureUsage, options: ResourceOptions| {
            let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
            d.usage = usage;
            d.resource_options = options;
            device.new_texture(&d).unwrap()
        };
        Lane {
            queue: device.new_command_queue().unwrap(),
            solid: make(&tob_pipeline("tob_solid", BlendFactor::One)).unwrap(),
            sample: make(&tob_pipeline("tob_sample", BlendFactor::One)).unwrap(),
            constant: make(&tob_pipeline("tob_solid", BlendFactor::BlendColor)).unwrap(),
            depth: make(&depth).unwrap(),
            amplified: make(&amplified).ok(),
            left: floats(&LEFT),
            right: floats(&RIGHT),
            rects: floats(&[LEFT, RIGHT].concat()),
            red: floats(&RED),
            green: floats(&GREEN),
            colors: floats(&[RED, GREEN].concat()),
            red_texture: texture(PixelFormat::RGBA8Unorm, [4, 4], &filled(RED_TEXEL)),
            green_texture: texture(PixelFormat::RGBA8Unorm, [4, 4], &filled(GREEN_TEXEL)),
            half_texture: texture(PixelFormat::RGBA16Float, [4, 4], &half_green.repeat(16)),
            stripes: texture(
                PixelFormat::RGBA8Unorm,
                [2, 1],
                &[RED_TEXEL, GREEN_TEXEL].concat(),
            ),
            clamp: sampler(SamplerAddressMode::ClampToEdge),
            repeat: sampler(SamplerAddressMode::Repeat),
            never: state(|d| d.depth_compare_function = CompareFunction::Never),
            always: state(|d| d.depth_compare_function = CompareFunction::Always),
            equal: state(|d| d.depth_compare_function = CompareFunction::Equal),
            stencil_equal: state(|d| {
                d.front_face_stencil.stencil_compare_function = CompareFunction::Equal;
                d.back_face_stencil.stencil_compare_function = CompareFunction::Equal;
            }),
            target: attachment(
                PixelFormat::RGBA8Unorm,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
                ResourceOptions::shared(),
            ),
            depth_stencil: attachment(
                PixelFormat::Depth32FloatStencil8,
                TextureUsage::RENDER_TARGET,
                ResourceOptions::private(),
            ),
            device: device.clone(),
        }
    }

    /// The pass of `case`, cleared where `first`, else loading what the first stored, counting
    /// into `visibility` where the case does.
    fn pass(&self, case: &Case, first: bool, visibility: &Buffer) -> RenderPassDescriptor {
        let load = match first {
            true => LoadAction::Clear,
            false => LoadAction::Load,
        };
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = load;
        color.store_action = StoreAction::Store;
        if case.pass == Pass::Depth {
            pass.depth_attachment.texture = Some(self.depth_stencil.clone());
            pass.depth_attachment.load_action = load;
            pass.depth_attachment.store_action = StoreAction::Store;
            pass.depth_attachment.clear_depth = 0.5;
            pass.stencil_attachment.texture = Some(self.depth_stencil.clone());
            pass.stencil_attachment.load_action = load;
            pass.stencil_attachment.store_action = StoreAction::Store;
            pass.stencil_attachment.clear_stencil = 0;
        }
        if case.pass == Pass::Visibility {
            pass.visibility_result_buffer = Some(visibility.clone());
        }
        pass
    }

    /// Encode `case` as `encoders` say into one command buffer, commit it and wait: each encoder
    /// takes the shared state, then draws once per entry, the change before an entry that says
    /// so. What it stored.
    fn run(&self, case: &Case, encoders: &[&[bool]]) -> Stored {
        let visibility = buffer_with(&self.device, &[0; 8]).unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        for (at, draws) in encoders.iter().enumerate() {
            let pass = self.pass(case, at == 0, &visibility);
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            (case.shared)(self, &mut encoder);
            for &changed in *draws {
                if changed {
                    (case.change)(self, &mut encoder);
                }
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, case.vertices, 1, 0)
                    .unwrap();
            }
            encoder.end_encoding().unwrap();
        }
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let mut texels = vec![0; (4 * SIDE * SIDE) as usize];
        self.target
            .get_bytes(
                &mut texels,
                4 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        let counted = read_buffer(&visibility, 0, 8).unwrap();
        Stored {
            texels,
            visibility: u64::from_le_bytes(counted.try_into().unwrap()),
        }
    }
}

/// `tob_rect` over `rect` and `tob_solid` adding `RED`.
fn solid_over(lane: &Lane, e: &mut RenderCommandEncoder, rect: [f32; 4]) {
    e.set_render_pipeline_state(&lane.solid).unwrap();
    e.set_vertex_bytes(&floats(&rect), 0).unwrap();
    e.set_fragment_bytes(&floats(&RED), 0).unwrap();
}

fn solid(lane: &Lane, e: &mut RenderCommandEncoder) {
    solid_over(lane, e, FULL);
}

/// `tob_sample` of `texture` through `sampler` over the whole target.
fn sampled(lane: &Lane, e: &mut RenderCommandEncoder, texture: &Texture, sampler: &SamplerState) {
    e.set_render_pipeline_state(&lane.sample).unwrap();
    e.set_vertex_bytes(&floats(&FULL), 0).unwrap();
    e.set_fragment_texture(Some(texture), 0).unwrap();
    e.set_fragment_sampler_state(Some(sampler), 0).unwrap();
}

/// `rt6_full` at depth `z` and `rt6_half`, under `state`.
fn depthed(lane: &Lane, e: &mut RenderCommandEncoder, z: f32, state: &DepthStencilState) {
    e.set_render_pipeline_state(&lane.depth).unwrap();
    e.set_vertex_bytes(&floats(&[z, 0.0]), 0).unwrap();
    e.set_depth_stencil_state(state).unwrap();
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A half of the target, the left or the right, as a viewport.
fn half_viewport(right: bool) -> Viewport {
    let x = if right { SIDE as f64 / 2.0 } else { 0.0 };
    Viewport::new(x, 0.0, SIDE as f64 / 2.0, SIDE as f64, 0.0, 1.0)
}

/// A half of the target, the left or the right, as a scissor rectangle.
fn half_scissor(right: bool) -> ScissorRect {
    let x = if right { SIDE / 2 } else { 0 };
    ScissorRect::new(x, 0, SIDE / 2, SIDE)
}

/// The cases (the module docs).
fn cases() -> Vec<Case> {
    let case = |name, shared, change| Case {
        name,
        pass: Pass::Color,
        vertices: 6,
        shared,
        change,
    };
    let depth_case = |name, shared, change| Case {
        name,
        pass: Pass::Depth,
        vertices: 3,
        shared,
        change,
    };
    vec![
        case(
            "pipeline state",
            |lane, e| {
                solid(lane, e);
                e.set_fragment_texture(Some(&lane.green_texture), 0)
                    .unwrap();
                e.set_fragment_sampler_state(Some(&lane.clamp), 0).unwrap();
            },
            |lane, e| e.set_render_pipeline_state(&lane.sample).unwrap(),
        ),
        case(
            "vertex bytes",
            |lane, e| solid_over(lane, e, LEFT),
            |_, e| e.set_vertex_bytes(&floats(&RIGHT), 0).unwrap(),
        ),
        case(
            "vertex buffer",
            |lane, e| {
                solid(lane, e);
                e.set_vertex_buffer(Some(&lane.left), 0, 0).unwrap();
            },
            |lane, e| e.set_vertex_buffer(Some(&lane.right), 0, 0).unwrap(),
        ),
        case(
            "vertex buffer offset",
            |lane, e| {
                solid(lane, e);
                e.set_vertex_buffer(Some(&lane.rects), 0, 0).unwrap();
            },
            |_, e| e.set_vertex_buffer_offset(16, 0).unwrap(),
        ),
        case("fragment bytes", solid, |_, e| {
            e.set_fragment_bytes(&floats(&GREEN), 0).unwrap()
        }),
        case(
            "fragment buffer",
            |lane, e| {
                solid(lane, e);
                e.set_fragment_buffer(Some(&lane.red), 0, 0).unwrap();
            },
            |lane, e| e.set_fragment_buffer(Some(&lane.green), 0, 0).unwrap(),
        ),
        case(
            "fragment buffer offset",
            |lane, e| {
                solid(lane, e);
                e.set_fragment_buffer(Some(&lane.colors), 0, 0).unwrap();
            },
            |_, e| e.set_fragment_buffer_offset(16, 0).unwrap(),
        ),
        case(
            "fragment texture",
            |lane, e| sampled(lane, e, &lane.red_texture, &lane.clamp),
            |lane, e| {
                e.set_fragment_texture(Some(&lane.green_texture), 0)
                    .unwrap()
            },
        ),
        case(
            "fragment texture of another format",
            |lane, e| sampled(lane, e, &lane.red_texture, &lane.clamp),
            |lane, e| e.set_fragment_texture(Some(&lane.half_texture), 0).unwrap(),
        ),
        case(
            "fragment sampler",
            |lane, e| sampled(lane, e, &lane.stripes, &lane.clamp),
            |lane, e| e.set_fragment_sampler_state(Some(&lane.repeat), 0).unwrap(),
        ),
        case(
            "viewport",
            |lane, e| {
                solid(lane, e);
                e.set_viewport(half_viewport(false)).unwrap();
            },
            |_, e| e.set_viewport(half_viewport(true)).unwrap(),
        ),
        case(
            "scissor rectangle",
            |lane, e| {
                solid(lane, e);
                e.set_scissor_rect(half_scissor(false)).unwrap();
            },
            |_, e| e.set_scissor_rect(half_scissor(true)).unwrap(),
        ),
        case(
            "blend color",
            |lane, e| {
                e.set_render_pipeline_state(&lane.constant).unwrap();
                e.set_vertex_bytes(&floats(&FULL), 0).unwrap();
                e.set_fragment_bytes(&floats(&WHITE), 0).unwrap();
                let [r, g, b, a] = RED;
                e.set_blend_color(r, g, b, a);
            },
            |_, e| {
                let [r, g, b, a] = GREEN;
                e.set_blend_color(r, g, b, a);
            },
        ),
        case(
            "cull mode",
            |lane, e| {
                solid(lane, e);
                e.set_cull_mode(CullMode::Front);
            },
            |_, e| e.set_cull_mode(CullMode::Back),
        ),
        case(
            "front-facing winding",
            |lane, e| {
                solid(lane, e);
                e.set_cull_mode(CullMode::Back);
                e.set_front_facing_winding(Winding::Clockwise);
            },
            |_, e| e.set_front_facing_winding(Winding::CounterClockwise),
        ),
        case(
            "triangle fill mode",
            |lane, e| {
                solid(lane, e);
                e.set_triangle_fill_mode(TriangleFillMode::Fill).unwrap();
            },
            |_, e| e.set_triangle_fill_mode(TriangleFillMode::Lines).unwrap(),
        ),
        depth_case(
            "depth-stencil state",
            |lane, e| depthed(lane, e, 0.5, &lane.never),
            |lane, e| e.set_depth_stencil_state(&lane.always).unwrap(),
        ),
        depth_case(
            "stencil reference",
            |lane, e| {
                depthed(lane, e, 0.5, &lane.stencil_equal);
                e.set_stencil_reference_value(1);
            },
            |_, e| e.set_stencil_reference_value(0),
        ),
        // At a depth of 0.5 under an equal test: a bias of 100 units moves it off the cleared 0.5.
        depth_case(
            "depth bias",
            |lane, e| {
                depthed(lane, e, 0.5, &lane.equal);
                e.set_depth_bias(100.0, 0.0, 0.0).unwrap();
            },
            |_, e| e.set_depth_bias(0.0, 0.0, 0.0).unwrap(),
        ),
        // At a depth of 1.5: clipped, then clamped to 1.
        depth_case(
            "depth clip mode",
            |lane, e| {
                depthed(lane, e, 1.5, &lane.always);
                e.set_depth_clip_mode(DepthClipMode::Clip).unwrap();
            },
            |_, e| e.set_depth_clip_mode(DepthClipMode::Clamp).unwrap(),
        ),
        Case {
            name: "visibility result mode",
            pass: Pass::Visibility,
            vertices: 6,
            shared: |lane, e| {
                solid(lane, e);
                e.set_visibility_result_mode(VisibilityResultMode::Disabled, 0)
                    .unwrap();
            },
            change: |_, e| {
                e.set_visibility_result_mode(VisibilityResultMode::Counting, 0)
                    .unwrap()
            },
        },
        case(
            "vertex amplification count",
            |lane, e| {
                let amplified = lane.amplified.as_ref().unwrap();
                e.set_render_pipeline_state(amplified).unwrap();
                e.set_vertex_bytes(&floats(&FULL), 0).unwrap();
                e.set_fragment_bytes(&floats(&RED), 0).unwrap();
                e.set_vertex_amplification_count(1, None).unwrap();
            },
            |_, e| e.set_vertex_amplification_count(2, None).unwrap(),
        ),
    ]
}

/// Every case's second draw shows its change, together in one encoder as apart (the module docs).
#[test]
fn a_change_between_two_draws_reaches_the_second() {
    let consumer = |options: &mut DeviceOptions| {
        options.record_executions = false;
    };
    on_device_with(consumer, |device| {
        let lane = Lane::new(device);
        let mut checked = Vec::new();
        for case in cases() {
            if case.name == "vertex amplification count" && lane.amplified.is_none() {
                eprintln!(
                    "DRAW-CHANGES {}: {} not checked: the device makes no amplified pipeline",
                    device.name(),
                    case.name
                );
                continue;
            }
            let together = lane.run(&case, &[&[false, true]]);
            let apart = lane.run(&case, &[&[false], &[true]]);
            let unchanged = lane.run(&case, &[&[false], &[false]]);
            assert_ne!(
                apart, unchanged,
                "{}: the change shows nothing in a draw of its own",
                case.name
            );
            assert_eq!(
                together, apart,
                "{}: the second draw of one encoder does not show the change",
                case.name
            );
            checked.push(case.name);
        }
        eprintln!(
            "DRAW-CHANGES {}: {} groups checked: {}",
            device.name(),
            checked.len(),
            checked.join(", ")
        );
    });
}
