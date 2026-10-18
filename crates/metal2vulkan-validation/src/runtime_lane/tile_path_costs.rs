//! What a window server's composite on Metal's tile path costs the GPU, part by part, on the
//! device the cfg selects, without the validation layer: report only, in release builds, to weigh
//! the per-pass work of the framebuffer fetch's interlocked storage form, of a memoryless
//! attachment a fetch reads, of extended-range blending and of a tile dispatch's staging before
//! and after a change to them.
//!
//! A frame is [`PASSES`] passes over a 1920x1080 base color (color 0, `BGRA8Unorm` but in the
//! extended-range case, a private texture loaded and stored), each of [`DRAWS`] rectangles
//! (`tgp_composite`'s `tgp_rect`: a fullscreen one, [`WINDOWS`] windows of 800x600 and the rest
//! 240x48 strips), each sampling one of four 256x256 layers with a linear sampler. Every case but
//! the base gives the pass a wide color too (color 1, a memoryless `RGBA16Float`, cleared and
//! stored `DontCare`), the shape of the window server's tile path; each draw's pipeline names both,
//! and a function that writes one color leaves the other unwritten (an empty write mask). The
//! cases, each adding one part to the one before it where it can:
//!
//! - **base**: the base color alone, every draw `tgp_layer` blended `One`, `OneMinusSourceAlpha`
//!   into it: the shape of the window server's composite before its tile path.
//! - **wide attached**: the wide color beside it, unread: what the second attachment costs alone.
//! - **wide fetched once**: the same draws, then one `tgp_resolve` that reads the wide color
//!   through the fetch and writes the base: the memoryless fetch target (its backing image).
//! - **wide composite**: the draws `tgp_layer_wide` into the wide color (blended alike), then the
//!   resolve: the pass splits where the fetch takes the wide color out of the draws' subpass.
//! - **wide blended in the function**: every draw `tgp_blend_wide`, which reads the wide color and
//!   writes it through the fetch, then the resolve: every draw in the interlock, one instance.
//! - **base fetched every fourth draw**: the base case's draws with every fourth one
//!   `tgp_blend_base`, which reads the base color through the fetch: a split at each change.
//! - **window**: [`WINDOW_PASSES`] passes of the two colors at 960x768, of [`WINDOW_DRAWS`]
//!   layers: alone; after a tile dispatch (`tgp_tile`, which reads and writes the base color's
//!   cells); and after the dispatch with every fourth draw `tgp_blend_base`, then the resolve: the
//!   shape of the window server's window passes on its tile path.
//! - **small passes**: [`SMALL_PASSES`] passes of the two colors at 384x64, of four layers, alone
//!   and after a tile dispatch.
//! - **extended range**: the base case's draws over an `RGB10A8_2P_XR10` base color (its blend
//!   computed in the interlock), without and with the wide color; last, since they run longest.
//!
//! Each case times [`RUNS`] frames after one untimed one, each frame one command buffer of its
//! passes, its GPU start to end (the device's timestamps), printed as `TILE-PATH-COST` lines: the median and the
//! fastest frame and the median per pass, with the GPU's clock where Linux's i915 driver reports it
//! (`gt_cur_freq_mhz`, read after each frame; the Iris Xe can sit at its efficient frequency for a
//! whole run), and the notices the case announced (each kind once per device, so at its first
//! case). The cases run in turn [`ROUNDS`] times, so each pair of them is compared within one
//! session.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::degrade_notices::counts;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, CommandBuffer, CommandQueue, Device, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, Size, StorageMode,
    StoreAction, Texture, TextureDescriptor,
};

/// The timed frames of each case per round.
const RUNS: usize = 7;

/// The rounds of every case.
const ROUNDS: usize = 2;

/// The full passes of a frame: the window server's tile path draws about nine a frame.
const PASSES: usize = 9;

/// The draws of a full pass: about 26 on the window server's tile path.
const DRAWS: usize = 26;

/// The draws of a full pass, after the fullscreen one, that are windows.
const WINDOWS: usize = 5;

/// The window passes of a frame, and their draws.
const WINDOW_PASSES: usize = 3;
const WINDOW_DRAWS: usize = 12;

/// The small passes of a frame.
const SMALL_PASSES: usize = 5;

/// A full pass's sides.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;

/// A window pass's sides.
const WINDOW_WIDTH: u64 = 960;
const WINDOW_HEIGHT: u64 = 768;

/// A small pass's sides.
const SMALL_WIDTH: u64 = 384;
const SMALL_HEIGHT: u64 = 64;

/// The layers' side.
const LAYER: u64 = 256;

/// The wide color's format.
const WIDE: PixelFormat = PixelFormat::RGBA16Float;

/// What a draw's function is, by the colors it reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// `tgp_layer`: the layer blended into the base color.
    Layer,
    /// `tgp_layer_wide`: the layer blended into the wide color.
    LayerWide,
    /// `tgp_blend_wide`: the layer over the wide color it reads.
    BlendWide,
    /// `tgp_resolve`: the wide color it reads into the base color.
    Resolve,
    /// `tgp_blend_base`: the layer over the base color it reads.
    BlendBase,
}

impl Step {
    fn function(self) -> &'static str {
        match self {
            Step::Layer => "tgp_layer",
            Step::LayerWide => "tgp_layer_wide",
            Step::BlendWide => "tgp_blend_wide",
            Step::Resolve => "tgp_resolve",
            Step::BlendBase => "tgp_blend_base",
        }
    }

    /// The color it writes, which its pipeline blends where `blended`; it samples a layer unless
    /// it resolves.
    fn writes(self) -> (usize, bool) {
        match self {
            Step::Layer => (0, true),
            Step::LayerWide => (1, true),
            Step::BlendWide => (1, false),
            Step::Resolve => (0, false),
            Step::BlendBase => (0, false),
        }
    }
}

/// The pipelines of one pass shape: the base color's format, with or without the wide color.
struct Pipelines {
    steps: Vec<(Step, RenderPipelineState)>,
}

impl Pipelines {
    fn new(device: &Device, library: &Library, base: PixelFormat, wide: bool) -> Pipelines {
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let steps = [
            Step::Layer,
            Step::LayerWide,
            Step::BlendWide,
            Step::Resolve,
            Step::BlendBase,
        ]
        .into_iter()
        .filter(|step| wide || step.writes().0 == 0 && *step != Step::Resolve)
        .map(|step| {
            let mut descriptor = RenderPipelineDescriptor::new(function("tgp_rect"));
            descriptor.fragment = Some(function(step.function()));
            let formats = if wide { vec![base, WIDE] } else { vec![base] };
            let (written, blended) = step.writes();
            for (index, format) in formats.into_iter().enumerate() {
                let mut color = ColorAttachmentDescriptor::new(format);
                if index == written {
                    if blended {
                        color.blending_enabled = true;
                        color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
                        color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
                    }
                } else {
                    color.write_mask = ColorWriteMask::NONE;
                }
                descriptor.color_attachments[index] = Some(color);
            }
            let state = device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap();
            (step, state)
        })
        .collect();
        Pipelines { steps }
    }

    fn of(&self, step: Step) -> &RenderPipelineState {
        &self.steps.iter().find(|(s, _)| *s == step).unwrap().1
    }
}

/// Everything the cases draw with.
struct Kit {
    device: Device,
    queue: CommandQueue,
    library: Library,
    layers: Vec<Texture>,
    sampler: SamplerState,
}

impl Kit {
    fn new(device: Device) -> Kit {
        let ll = engine_dir().join("tgp_composite.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let layers = (0..4u8)
            .map(|n| {
                let mut d =
                    TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, LAYER, LAYER, false);
                d.usage = TextureUsage::SHADER_READ;
                d.resource_options = ResourceOptions::shared();
                let texture = device.new_texture(&d).unwrap();
                // Premultiplied texels of varied alpha: each row's alpha its index, each channel
                // under it.
                let bytes: Vec<u8> = (0..LAYER * LAYER)
                    .flat_map(|i| {
                        let alpha = (i / LAYER) as u8 | 0x10;
                        let c = alpha / (n + 2);
                        [c, c / 2, c / 3, alpha]
                    })
                    .collect();
                texture
                    .replace_region(
                        Region::new_2d(0, 0, LAYER, LAYER),
                        0,
                        0,
                        &bytes,
                        LAYER * 4,
                        LAYER * LAYER * 4,
                    )
                    .unwrap();
                texture
            })
            .collect();
        Kit {
            queue: device.new_command_queue().unwrap(),
            sampler: device.new_sampler_state(&sampler).unwrap(),
            library,
            layers,
            device,
        }
    }

    /// A private render target of `format` and sides.
    fn target(&self, format: PixelFormat, width: u64, height: u64) -> Texture {
        let mut d = TextureDescriptor::texture_2d(format, width, height, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        self.device.new_texture(&d).unwrap()
    }

    /// A memoryless wide color of sides.
    fn memoryless(&self, width: u64, height: u64) -> Texture {
        let mut d = TextureDescriptor::texture_2d(WIDE, width, height, false);
        d.usage = TextureUsage::RENDER_TARGET;
        d.resource_options = ResourceOptions::new(
            StorageMode::Memoryless,
            metal2vulkan_runtime::CpuCacheMode::DefaultCache,
            metal2vulkan_runtime::HazardTrackingMode::Default,
        );
        self.device.new_texture(&d).unwrap()
    }
}

/// A pass over `base`, loaded and stored, with `wide` cleared and stored `DontCare`.
fn pass(base: &Texture, wide: Option<&Texture>) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(base.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    if let Some(wide) = wide {
        let color = &mut descriptor.color_attachments[1];
        color.texture = Some(wide.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        color.store_action = StoreAction::DontCare;
    }
    descriptor
}

/// Draw `index`'s rectangle in normalized device coordinates: the fullscreen one first, then the
/// windows, then the strips, each placed by its index.
fn rect(index: usize, width: u64, height: u64) -> [f32; 4] {
    let (w, h) = (width as f32, height as f32);
    let (x, y, rw, rh) = match index {
        0 => (0.0, 0.0, w, h),
        i if i <= WINDOWS => {
            let i = i as f32;
            (
                i * 160.0 % (w - 800.0),
                i * 90.0 % (h - 600.0),
                800.0,
                600.0,
            )
        }
        i => {
            let i = (i - WINDOWS) as f32;
            (i * 77.0 % (w - 240.0), i * 53.0 % (h - 48.0), 240.0, 48.0)
        }
    };
    let ndc = |px: f32, side: f32| px / side * 2.0 - 1.0;
    [
        ndc(x, w),
        ndc(y, h),
        ndc(x + rw.min(w), w),
        ndc(y + rh.min(h), h),
    ]
}

/// One pass's shape: its colors, its draws in order, its sides, and the tile function it
/// dispatches before them, if any.
struct Shape<'a> {
    base: &'a Texture,
    wide: Option<&'a Texture>,
    pipelines: &'a Pipelines,
    steps: Vec<Step>,
    tile: Option<&'a RenderPipelineState>,
}

impl Shape<'_> {
    /// Encode the pass into `command_buffer`: the tile dispatch, then one rectangle a step.
    fn encode(&self, kit: &Kit, command_buffer: &CommandBuffer) {
        let descriptor = pass(self.base, self.wide);
        let sides = (self.base.width(), self.base.height());
        let mut encoder = command_buffer.render_command_encoder(&descriptor).unwrap();
        if let Some(tile) = self.tile {
            encoder.set_render_pipeline_state(tile).unwrap();
            let (w, h) = (
                encoder.tile_width().unwrap(),
                encoder.tile_height().unwrap(),
            );
            encoder
                .dispatch_threads_per_tile(Size::new(w, h, 1))
                .unwrap();
        }
        encoder
            .set_fragment_sampler_state(Some(&kit.sampler), 0)
            .unwrap();
        for (index, step) in self.steps.iter().enumerate() {
            encoder
                .set_render_pipeline_state(self.pipelines.of(*step))
                .unwrap();
            let corners = rect(index, sides.0, sides.1);
            let bytes: Vec<u8> = corners.iter().flat_map(|c| c.to_le_bytes()).collect();
            encoder.set_vertex_bytes(&bytes, 0).unwrap();
            encoder
                .set_fragment_texture(Some(&kit.layers[index % kit.layers.len()]), 0)
                .unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
    }
}

/// One case: its name, its pass's shape and the passes of a frame.
struct Case<'a> {
    name: &'static str,
    shape: Shape<'a>,
    passes: usize,
}

/// The GPU's current clock in MHz, where Linux's i915 driver reports it.
fn gpu_clock() -> Option<u32> {
    (0..4).find_map(|card| {
        std::fs::read_to_string(format!("/sys/class/drm/card{card}/gt_cur_freq_mhz"))
            .ok()?
            .trim()
            .parse()
            .ok()
    })
}

/// One frame of `case`: its GPU time in microseconds, and the clock after it.
fn frame(kit: &Kit, case: &Case<'_>) -> (f64, Option<u32>) {
    let command_buffer = kit.queue.command_buffer().unwrap();
    for _ in 0..case.passes {
        case.shape.encode(kit, &command_buffer);
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let start = command_buffer.gpu_start_time().unwrap();
    let end = command_buffer.gpu_end_time().unwrap();
    ((end - start) * 1e6, gpu_clock())
}

/// Every case's GPU times, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn tile_path_cost_parts() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        options.policy = Default::default();
        options.record_executions = false;
    })
    .unwrap();
    messages.take_notices();
    let name = device.name().to_owned();
    let kit = Kit::new(device);
    let extended = PixelFormat::RGB10A82PXR10;
    let base_only = Pipelines::new(&kit.device, &kit.library, PixelFormat::BGRA8Unorm, false);
    let both = Pipelines::new(&kit.device, &kit.library, PixelFormat::BGRA8Unorm, true);
    let extended_only = Pipelines::new(&kit.device, &kit.library, extended, false);
    let extended_both = Pipelines::new(&kit.device, &kit.library, extended, true);
    let mut tile = TileRenderPipelineDescriptor::new(StageSource::Function(
        kit.library
            .new_function_with_name("tgp_tile")
            .unwrap()
            .function_descriptor(),
    ));
    tile.color_attachment_formats[0] = Some(PixelFormat::BGRA8Unorm);
    tile.color_attachment_formats[1] = Some(WIDE);
    tile.threadgroup_size_matches_tile_size = true;
    let tile = kit
        .device
        .new_tile_render_pipeline_state(&tile, &PipelineOptions::default())
        .unwrap();
    let base = kit.target(PixelFormat::BGRA8Unorm, WIDTH, HEIGHT);
    let base_xr = kit.target(extended, WIDTH, HEIGHT);
    let wide = kit.memoryless(WIDTH, HEIGHT);
    let window = kit.target(PixelFormat::BGRA8Unorm, WINDOW_WIDTH, WINDOW_HEIGHT);
    let window_wide = kit.memoryless(WINDOW_WIDTH, WINDOW_HEIGHT);
    let small = kit.target(PixelFormat::BGRA8Unorm, SMALL_WIDTH, SMALL_HEIGHT);
    let small_wide = kit.memoryless(SMALL_WIDTH, SMALL_HEIGHT);
    let layers = vec![Step::Layer; DRAWS];
    let resolved = |mut steps: Vec<Step>| {
        steps.push(Step::Resolve);
        steps
    };
    let every_fourth: Vec<Step> = (0..DRAWS)
        .map(|i| match i % 4 {
            3 => Step::BlendBase,
            _ => Step::Layer,
        })
        .collect();
    let window_draws: Vec<Step> = (0..WINDOW_DRAWS)
        .map(|i| match i % 4 {
            3 => Step::BlendBase,
            _ => Step::Layer,
        })
        .collect();
    let full = |name, base, wide, pipelines, steps| Case {
        name,
        shape: Shape {
            base,
            wide,
            pipelines,
            steps,
            tile: None,
        },
        passes: PASSES,
    };
    let windowed = |name, steps, tile| Case {
        name,
        shape: Shape {
            base: &window,
            wide: Some(&window_wide),
            pipelines: &both,
            steps,
            tile,
        },
        passes: WINDOW_PASSES,
    };
    let small_case = |name, tile| Case {
        name,
        shape: Shape {
            base: &small,
            wide: Some(&small_wide),
            pipelines: &both,
            steps: vec![Step::Layer; 4],
            tile,
        },
        passes: SMALL_PASSES,
    };
    let cases: Vec<Case<'_>> = vec![
        full("base", &base, None, &base_only, layers.clone()),
        full("wide attached", &base, Some(&wide), &both, layers.clone()),
        full(
            "wide fetched once",
            &base,
            Some(&wide),
            &both,
            resolved(layers.clone()),
        ),
        full(
            "wide composite",
            &base,
            Some(&wide),
            &both,
            resolved(vec![Step::LayerWide; DRAWS]),
        ),
        full(
            "wide blended in the function",
            &base,
            Some(&wide),
            &both,
            resolved(vec![Step::BlendWide; DRAWS]),
        ),
        full(
            "base fetched every fourth draw",
            &base,
            Some(&wide),
            &both,
            every_fourth,
        ),
        windowed("window: layers", vec![Step::Layer; WINDOW_DRAWS], None),
        windowed(
            "window: tile dispatch, layers",
            vec![Step::Layer; WINDOW_DRAWS],
            Some(&tile),
        ),
        windowed(
            "window: tile dispatch, base fetched every fourth draw, resolve",
            resolved(window_draws),
            Some(&tile),
        ),
        small_case("small passes", None),
        small_case("small passes, tile dispatch", Some(&tile)),
        full(
            "extended range, one draw",
            &base_xr,
            None,
            &extended_only,
            vec![Step::Layer],
        ),
        full(
            "extended range",
            &base_xr,
            None,
            &extended_only,
            layers.clone(),
        ),
        full(
            "extended range, wide attached",
            &base_xr,
            Some(&wide),
            &extended_both,
            layers,
        ),
    ];
    let announced: Vec<_> = cases
        .iter()
        .map(|case| {
            let _ = frame(&kit, case);
            counts(&messages.take_notices())
        })
        .collect();
    for round in 0..ROUNDS {
        for (case, notices) in cases.iter().zip(&announced) {
            let runs: Vec<(f64, Option<u32>)> = (0..RUNS).map(|_| frame(&kit, case)).collect();
            let mut times: Vec<f64> = runs.iter().map(|r| r.0).collect();
            times.sort_by(f64::total_cmp);
            let clocks: Vec<u32> = runs.iter().filter_map(|r| r.1).collect();
            let median = times[times.len() / 2];
            eprintln!(
                "TILE-PATH-COST on {name}: round {round}: {}: GPU median {median:.0} us a frame \
                 ({:.0} us a pass), fastest {:.0} us; clock {:?} MHz; announced {notices:?}",
                case.name,
                median / case.passes as f64,
                times[0],
                (clocks.iter().min(), clocks.iter().max())
            );
        }
    }
}
