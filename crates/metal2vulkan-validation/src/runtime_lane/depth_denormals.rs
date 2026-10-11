//! Whether a device's `Depth32Float` depth image holds a denormal and its depth test compares
//! one as Metal's (the parity rule: Metal keeps one), on the device the cfg selects, under the
//! validation layer.
//!
//! **Metal's answer**: an Apple GPU's `Depth32Float` texture keeps a denormal through a blit in
//! and out and a shader's read (EV FRT4-dm1-m4, an M4 under macOS 27.0.1); MoltenVK on the M4,
//! whose clears, copies, draws and depth tests are Metal's, keeps one a clear, a copy or the CPU
//! stores, stores 0 for a draw at a denormal depth, and its depth test compares a positive one
//! above zero and a negative one not below zero (EV ZIN2-dn2-mvk4,
//! `DriverFact::DepthTestKeepsDenormals`).
//!
//! Each of four denormals (the smallest and largest of each sign) is stored into a 4x4
//! single-sample `Depth32Float` render target by each way a program stores depth: a pass's clear,
//! `replace_region`, a blit from a buffer, and a draw at that depth (compare `Always`, depth
//! writes on; the owned fixture `rt6_render_state`'s fullscreen triangle); the negative ones only
//! by the CPU and the blit (a draw's is clipped, a clear's outside Vulkan's clear range). Each
//! stored texture is then read two ways that do not share a path:
//!
//! - **read**: `get_bytes`, the texel's bits as a transfer reads them;
//! - **tested**: a draw at depth 0 that loads the depth and tests it, `Less` for a positive
//!   denormal and `Greater` for a negative one, writing white where it passes: Metal's passes
//!   against a positive one and fails against a negative one.
//!
//! Each prints a `DEPTH-DENORMAL` line. Checked: the two controls, the smallest normal and 0.5,
//! read and pass as given on every device; a draw's denormal stores zero on every device, as
//! Metal's; where the driver has `DriverFact::DepthImageKeepsDenormals` the clear's, the CPU's and
//! the blit's read back as given; where it has `DriverFact::DepthTestKeepsDenormals` too, each
//! kept one tests as Metal's; and `depth-denormals` is announced exactly where the driver lacks
//! either fact.

use super::degrade_notices::{assert_made_once, clamps_depth_uploads, forced};
use super::degrade_notices::{made_render_kinds, shared_in_system_memory};
use super::render_state::library;
use super::{buffer_with, validated_device, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBufferStatus, CommandQueue, CpuCacheMode, Degradation, Device,
    HazardTrackingMode, Library, LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, Size, StorageMode, StoreAction, Texture,
    TextureDescriptor,
};

/// The kind a depth image that may store a denormal otherwise than Metal's takes.
pub(super) const DEPTH_DENORMALS: [Degradation; 1] = [Degradation::DepthDenormals];

/// The textures' side.
const SIDE: u64 = 4;

/// The denormals: the smallest and largest of each sign.
const DENORMALS: [u32; 4] = [0x0000_0001, 0x007f_ffff, 0x8000_0001, 0x807f_ffff];

/// The controls, which every device stores and tests as given: the smallest normal, and 0.5.
const CONTROLS: [u32; 2] = [0x0080_0000, 0x3f00_0000];

/// The ways a program stores depth.
#[derive(Clone, Copy, Debug)]
enum Route {
    Clear,
    ReplaceRegion,
    BlitFromBuffer,
    Draw,
}

const ROUTES: [Route; 4] = [
    Route::Clear,
    Route::ReplaceRegion,
    Route::BlitFromBuffer,
    Route::Draw,
];

/// The lane's device, queue, and the fixture's pipeline drawing white over a depth attachment.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    white: RenderPipelineState,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let library = library(&device);
        let white = device
            .new_render_pipeline_state(&white_descriptor(&library), &PipelineOptions::default())
            .unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            device,
            library,
            white,
        }
    }

    /// A 4x4 texture of `pf` the passes render to and the CPU reads and writes.
    fn texture(&self, pf: PixelFormat) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        self.device.new_texture(&d).unwrap()
    }

    /// A pass over `depth` (cleared to `clear` or loaded) and a color target cleared to zero,
    /// drawing white at depth `z` under `compare`, with depth writes where `write`, once; the
    /// color target's texels.
    fn pass(
        &self,
        depth: &Texture,
        clear: Option<f64>,
        draw: Option<(f32, CompareFunction, bool)>,
    ) -> Vec<u8> {
        let color = self.texture(PixelFormat::RGBA8Unorm);
        let mut pass = RenderPassDescriptor::new();
        let target = &mut pass.color_attachments[0];
        target.texture = Some(color.clone());
        target.load_action = LoadAction::Clear;
        target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        target.store_action = StoreAction::Store;
        pass.depth_attachment.texture = Some(depth.clone());
        pass.depth_attachment.load_action = match clear {
            Some(_) => LoadAction::Clear,
            None => LoadAction::Load,
        };
        pass.depth_attachment.clear_depth = clear.unwrap_or(1.0);
        pass.depth_attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let mut keep = Vec::new();
        if let Some((z, compare, write)) = draw {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = compare;
            state.depth_write_enabled = write;
            let state = self.device.new_depth_stencil_state(&state).unwrap();
            let words: Vec<u8> = [z, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
            let at = buffer_with(&self.device, &words).unwrap();
            encoder.set_render_pipeline_state(&self.white).unwrap();
            encoder.set_depth_stencil_state(&state).unwrap();
            encoder.set_vertex_buffer(Some(&at), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            keep.push(at);
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        texels(&color)
    }

    /// A depth texture holding `bits` everywhere, stored by `route`.
    fn stored(&self, route: Route, bits: u32) -> Texture {
        let depth = self.texture(PixelFormat::Depth32Float);
        let value = f32::from_bits(bits);
        let bytes: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| bits.to_le_bytes()).collect();
        match route {
            Route::Clear => {
                self.pass(&depth, Some(f64::from(value)), None);
            }
            Route::ReplaceRegion => depth
                .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &bytes, SIDE * 4, 0)
                .unwrap(),
            Route::BlitFromBuffer => {
                let source = buffer_with(&self.device, &bytes).unwrap();
                let command_buffer = self.queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &source,
                    0,
                    SIDE * 4,
                    SIDE * SIDE * 4,
                    Size::new(SIDE, SIDE, 1),
                    &depth,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            Route::Draw => {
                self.pass(
                    &depth,
                    Some(0.5),
                    Some((value, CompareFunction::Always, true)),
                );
            }
        }
        depth
    }
}

/// The descriptor of the lane's pipeline drawing white over a depth attachment, of `library`.
fn white_descriptor(library: &Library) -> RenderPipelineDescriptor {
    let function = |name| library.new_function_with_name(name).unwrap();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
        function("rt6_full").function_descriptor(),
    ));
    descriptor.fragment = Some(StageSource::Function(
        function("rt6_white").function_descriptor(),
    ));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.depth_format = Some(PixelFormat::Depth32Float);
    descriptor
}

/// Every texel of `texture`, 4 bytes each.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut out = vec![0u8; (SIDE * SIDE * 4) as usize];
    texture
        .get_bytes(
            &mut out,
            SIDE * 4,
            SIDE * SIDE * 4,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    out
}

/// Each denormal by each route, read and tested (the module docs).
#[test]
fn depth_images_hold_denormals_as_their_drivers_claim() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let facts = &device.properties().driver_facts;
        let keeps = facts.contains(&DriverFact::DepthImageKeepsDenormals);
        let tests = facts.contains(&DriverFact::DepthTestKeepsDenormals);
        // The uploads into the depth images, the draws' `Shared` buffers, and the white
        // pipeline's making.
        let mut first = forced(&[
            (
                Degradation::RestrictedDepthUpload,
                clamps_depth_uploads(&device),
            ),
            (
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(&device),
            ),
            (Degradation::DepthDenormals, !(keeps && tests)),
        ]);
        let lane = Lane::new(device);
        first.extend(made_render_kinds(
            &lane.device,
            &white_descriptor(&lane.library),
        ));
        for route in ROUTES {
            for bits in CONTROLS.into_iter().chain(DENORMALS) {
                let control = CONTROLS.contains(&bits);
                let negative = bits & 0x8000_0000 != 0;
                // A clear and a draw store no negative depth: a draw's is clipped, and a clear's
                // is outside the range Vulkan clears to without `VK_EXT_depth_range_unrestricted`.
                if negative && matches!(route, Route::Clear | Route::Draw) {
                    continue;
                }
                let depth = lane.stored(route, bits);
                let read: Vec<u32> = texels(&depth)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect();
                let compare = match negative {
                    true => CompareFunction::Greater,
                    false => CompareFunction::Less,
                };
                let color = lane.pass(&depth, None, Some((0.0, compare, false)));
                let passed = color
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|t| t[0] == 255)
                    .count();
                let held = read.iter().filter(|&&w| w == bits).count();
                println!(
                    "DEPTH-DENORMAL {name} {route:?} {bits:08x}: read {:08x} ({held} of 16 kept), \
                     tested {passed} of 16 passed",
                    read[0]
                );
                let at = format!("{name} {route:?} {bits:08x}");
                if control {
                    assert_eq!(held, 16, "{at}: read {read:08x?}");
                    assert_eq!(passed, 16, "{at}: tested");
                    continue;
                }
                // A draw at a denormal depth stores zero, as Metal's rasterizer does.
                if matches!(route, Route::Draw) {
                    assert!(read.iter().all(|&w| w == 0), "{at}: read {read:08x?}");
                    continue;
                }
                if keeps {
                    assert_eq!(held, 16, "{at}: read {read:08x?}");
                }
                if keeps && tests {
                    let metals = if negative { 0 } else { 16 };
                    assert_eq!(passed, metals, "{at}: tested");
                }
            }
        }
        let counts = assert_made_once(&messages, &[], &first);
        let announced = counts.contains_key(&Degradation::DepthDenormals);
        println!("DEPTH-DENORMAL {name} announced depth-denormals: {announced}");
        assert_eq!(
            announced,
            !(keeps && tests),
            "{name}: depth-denormals announced where the driver lacks a fact (keeps {keeps}, \
             tests {tests})"
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
