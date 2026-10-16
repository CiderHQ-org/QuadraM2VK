//! Device tests of a patch draw whose only patch is culled, on the device the cfg selects, under
//! the validation layer. The owned fixtures `k7_tess_triangle_domain` and `k7_tess_quad_domain`
//! mark a cell for every domain point their post-tessellation vertex function runs at, so a draw
//! that leaves the marks empty ran it no time and drew no triangle (`patches.rs` has the pass).
//!
//! What Metal gives, as a Metal probe on an M3 Max with macOS 27.0.1 counted it: a draw of one
//! patch with a first edge factor of 0 (quad and triangle, integer and fractional odd
//! partitioning), NaN (quad, integer) or -1 (triangle, fractional even), its other factors 2, runs
//! its post-tessellation vertex function no time. The same draw with that edge factor 2 is the
//! control: it runs the function, so the culling value is what empties the marks.
//!
//! Each test runs on the device's tessellation stages and on the emulation (`patches::Path`).

use super::patches::{cells, on_path, Lane, Path, MARK_WORDS, TWO};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{TessellationFactorStepFunction, TessellationPartitionMode};
use metal2vulkan::pipeline::StageSource;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{PipelineOptions, RenderPipelineState};

/// A patch's domain: the fixture that draws it and its edge and inside factor counts.
#[derive(Clone, Copy, Debug)]
enum Domain {
    Triangle,
    Quad,
}

impl Domain {
    /// The `half`s of one patch's factors, every one 2 but its first edge, `edge`.
    fn halves(self, edge: u16) -> Vec<u16> {
        let factors = match self {
            Domain::Triangle => 3 + 1,
            Domain::Quad => 4 + 2,
        };
        let mut halves = vec![TWO; factors];
        halves[0] = edge;
        halves
    }
}

/// Half-float 0, NaN and -1.
const ZERO: u16 = 0x0000;
const NAN: u16 = 0x7e00;
const MINUS_ONE: u16 = 0xbc00;

/// The draws the M3 counted: each domain, partition mode and culling first edge factor.
const CULLED: [(Domain, TessellationPartitionMode, u16); 6] = [
    (Domain::Quad, TessellationPartitionMode::Integer, ZERO),
    (Domain::Triangle, TessellationPartitionMode::Integer, ZERO),
    (Domain::Quad, TessellationPartitionMode::FractionalOdd, ZERO),
    (
        Domain::Triangle,
        TessellationPartitionMode::FractionalOdd,
        ZERO,
    ),
    (Domain::Quad, TessellationPartitionMode::Integer, NAN),
    (
        Domain::Triangle,
        TessellationPartitionMode::FractionalEven,
        MINUS_ONE,
    ),
];

#[test]
fn a_draw_whose_only_patch_is_culled_runs_no_vertex_function() {
    culled_draws(Path::Native);
}

/// [`a_draw_whose_only_patch_is_culled_runs_no_vertex_function`], emulated.
#[test]
fn emulated_a_draw_whose_only_patch_is_culled_runs_no_vertex_function() {
    culled_draws(Path::Emulated);
}

fn culled_draws(path: Path) {
    on_path(path, |device, name| {
        let quad = llvm_assemble(&engine_dir().join("k7_tess_quad_domain.ll")).unwrap();
        let quad = device
            .new_library_with_air(&quad)
            .unwrap()
            .new_function_with_name("k7_tess_quad_domain")
            .unwrap();
        let lane = Lane::new(device.clone());
        for (domain, partition, edge) in CULLED {
            let mut descriptor = lane.descriptor(TessellationFactorStepFunction::PerPatch, false);
            if let Domain::Quad = domain {
                descriptor.vertex = StageSource::Function(quad.function_descriptor());
            }
            if let Some(tessellation) = descriptor.tessellation.as_mut() {
                tessellation.partition_mode = partition;
            }
            let what = format!("{domain:?} {partition:?} edge {edge:#06x} on {name}");
            let pipeline =
                match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
                    Ok(pipeline) => pipeline,
                    Err(error) => {
                        assert_eq!(path, Path::Native, "{what}: the emulation refused: {error}");
                        eprintln!("{what}: the pipeline is refused, {error}");
                        continue;
                    }
                };
            assert!(
                !marked(&lane, &pipeline, &domain.halves(TWO)).is_empty(),
                "{what}: the control drew nothing"
            );
            assert_eq!(
                marked(&lane, &pipeline, &domain.halves(edge)),
                Vec::<u32>::new(),
                "{what}"
            );
        }
    });
}

/// The cells a draw of one patch of factors `halves` by `pipeline` marked.
fn marked(lane: &Lane, pipeline: &RenderPipelineState, halves: &[u16]) -> Vec<u32> {
    let marks = lane.buffer(&vec![0; MARK_WORDS * 4]);
    let mut bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    bytes.extend([0; 16]);
    let factors = lane.buffer(&bytes);
    lane.pass(pipeline, &marks, |encoder| {
        encoder.set_tessellation_factor_buffer(Some(&factors), 0, 0)?;
        encoder.draw_patches(1, 0, 1, None, 0, 1, 0)
    })
    .unwrap();
    cells(&marks)
}
