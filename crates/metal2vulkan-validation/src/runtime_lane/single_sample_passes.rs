//! A render pipeline state of more samples than its pass's one on the device the cfg selects, under
//! the validation layer. Metal's validation stops it at `setRenderPipelineState:`; plain Metal
//! (Metal probes; an M4 with macOS 27.0.1) draws a 2- or 4-sample state there as a state of one
//! sample: it covers at the pixel centre, runs once per pixel even reading `[[sample_id]]`, which
//! reads 0, and keeps a pixel by bit 0 of a `[[sample_mask]]` output.
//!
//! - **Drawn as one sample:** a 4² `RGBA8Unorm` pass of one sample, drawn by `rt6_resolves`'
//!   lower-left half (whose diagonal pixels a 4-sample rasterization covers in part) with
//!   `rt6_resolve_white`, `rt6_resolve_samples` (red its `[[sample_id]]` / 3, green 1 on sample 3
//!   alone) and `rt6_resolve_masked` (white on the samples of its `[[sample_mask]]` output) under
//!   masks 1 and 14, stores with each 4-sample state what the 1-sample state stores.
//! - **Refused** `SampleCount`, as plain Metal answers otherwise: alpha to coverage (alphas a
//!   1-sample state drops kept their texels), a fragment function that reads the sample count
//!   (`sample_rate`'s `get_num_samples()` read the state's), and a 1-sample state in a 4-sample
//!   pass (sample 0 alone written, under a coverage rule unmeasured).

use super::resolves::{lane, quiet, Lane, ONE};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, DrawProblem, Error, Invalid, LoadAction, PipelineOptions,
    Problem, RenderPassDescriptor, RenderPipelineState, StoreAction, Texture,
};

/// What a pipeline of the lane draws: `rt6_resolves`' vertex function `vertex`, the fragment
/// function `fragment` of the fixture `stem`, its one color's format, its samples, and alpha to
/// coverage where `coverage`.
struct Pipe<'a> {
    vertex: &'a str,
    fragment: (&'a str, &'a str),
    format: PixelFormat,
    samples: u32,
    coverage: bool,
}

fn pipeline(lane: &Lane, pipe: &Pipe<'_>) -> Result<RenderPipelineState, Error> {
    let function = |stem: &str, name: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        lane.device
            .new_library_with_air(&bitcode)
            .unwrap()
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let (stem, fragment) = pipe.fragment;
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("rt6_resolves", pipe.vertex)));
    descriptor.fragment = Some(StageSource::Function(function(stem, fragment)));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pipe.format));
    descriptor.raster_sample_count = pipe.samples;
    descriptor.alpha_to_coverage = pipe.coverage;
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
}

/// A pass of `target` cleared to zero and stored.
fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let color = &mut descriptor.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    color.store_action = StoreAction::Store;
    descriptor
}

/// The texels `state` draws into a 4² `RGBA8Unorm` target of one sample: its triangle, the
/// fragment's `buffer(0)` holding `mask`.
fn drawn(lane: &Lane, state: &RenderPipelineState, mask: u32) -> Vec<[u8; 4]> {
    let target = lane.texture(PixelFormat::RGBA8Unorm, ONE, TextureUsage::RENDER_TARGET);
    lane.run(&pass(&target), |lane, encoder, held: &mut Vec<Buffer>| {
        let words = super::buffer_with(&lane.device, &mask.to_le_bytes()).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_fragment_buffer(Some(&words), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        held.push(words);
    })
    .unwrap();
    lane.read(&target, (0, 0), 4, BlitOption::None)
        .chunks(4)
        .map(|t| [t[0], t[1], t[2], t[3]])
        .collect()
}

/// A 4-sample state in a 1-sample pass stores what the 1-sample state stores (the module docs).
#[test]
fn a_multisample_state_in_a_single_sample_pass_draws_as_a_single_sample_one() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let fragments = [
        ("rt6_resolve_white", 0),
        ("rt6_resolve_samples", 0),
        ("rt6_resolve_masked", 1),
        ("rt6_resolve_masked", 14),
    ];
    for (fragment, mask) in fragments {
        let state = |samples| {
            pipeline(
                &lane,
                &Pipe {
                    vertex: "rt6_resolve_half",
                    fragment: ("rt6_resolves", fragment),
                    format: PixelFormat::RGBA8Unorm,
                    samples,
                    coverage: false,
                },
            )
            .unwrap()
        };
        let one = drawn(&lane, &state(1), mask);
        let four = drawn(&lane, &state(4), mask);
        assert_eq!(four, one, "{fragment} under mask {mask} on {name}");
        let covered: Vec<&[u8; 4]> = one.iter().filter(|t| **t != [0; 4]).collect();
        let want = if mask == 14 { 0..=0 } else { 6..=10 };
        assert!(
            want.contains(&covered.len()),
            "{fragment} under mask {mask} on {name}: {one:02x?}"
        );
        // Sample 0's run alone: red 0, green 0, alpha 1.
        if fragment == "rt6_resolve_samples" {
            assert!(
                covered
                    .iter()
                    .all(|t| t[0] == 0 && t[1] == 0 && t[3] == 0xff),
                "sample 0 alone on {name}: {one:02x?}"
            );
        }
    }
    eprintln!("a 4-sample state in a 1-sample pass on {name}: drawn as one sample, as Metal");
    quiet(lane, &messages);
}

/// Alpha to coverage, a fragment function reading the sample count, and a state of one sample in
/// a pass of four stay refused `SampleCount` at `set_render_pipeline_state`.
#[test]
fn sample_counts_metal_draws_otherwise_are_refused() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let refused = || {
        Err(Error::Invalid(Invalid::new(
            "set_render_pipeline_state",
            Problem::Draw(DrawProblem::SampleCount),
        )))
    };
    let white = Pipe {
        vertex: "rt6_resolve_full",
        fragment: ("rt6_resolves", "rt6_resolve_white"),
        format: PixelFormat::RGBA8Unorm,
        samples: 4,
        coverage: true,
    };
    let covering = pipeline(&lane, &white).unwrap();
    let single = pipeline(
        &lane,
        &Pipe {
            samples: 1,
            coverage: false,
            ..white
        },
    )
    .unwrap();
    let target = lane.texture(PixelFormat::RGBA8Unorm, ONE, TextureUsage::RENDER_TARGET);
    lane.run(&pass(&target), |_, encoder, _| {
        assert_eq!(encoder.set_render_pipeline_state(&covering), refused());
    })
    .unwrap();
    let msaa = lane.msaa(PixelFormat::RGBA8Unorm);
    let mut descriptor = pass(&msaa);
    descriptor.color_attachments[0].store_action = StoreAction::DontCare;
    lane.run(&descriptor, |_, encoder, _| {
        assert_eq!(encoder.set_render_pipeline_state(&single), refused());
    })
    .unwrap();
    // `get_num_samples()` writes an integer color: a device whose integer colors take no 4
    // samples refuses the state itself.
    let counting = pipeline(
        &lane,
        &Pipe {
            fragment: ("sample_rate", "sample_rate_num_samples"),
            format: PixelFormat::R32Uint,
            coverage: false,
            ..white
        },
    );
    match counting {
        Ok(counting) => {
            let target = lane.texture(PixelFormat::R32Uint, ONE, TextureUsage::RENDER_TARGET);
            lane.run(&pass(&target), |_, encoder, _| {
                assert_eq!(encoder.set_render_pipeline_state(&counting), refused());
            })
            .unwrap();
        }
        Err(Error::Unsupported(u)) => {
            eprintln!("a 4-sample integer color on {name}: unsupported ({u:?}), not drawn");
        }
        Err(e) => panic!("the counting state on {name}: {e:?}"),
    }
    quiet(lane, &messages);
}
