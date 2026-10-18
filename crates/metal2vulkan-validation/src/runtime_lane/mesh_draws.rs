//! Mesh draws on the device the cfg selects, under the validation layer: the Metal probe's grid and
//! object cases drawn through `metal2vulkan-runtime` with the probe's functions (the owned fixture
//! `mesh_draws`). Each case is one draw into an 8x8 pass of two `RGBA32Uint` attachments, cleared
//! to zero and stored; each mesh threadgroup draws one triangle over the centre of one pixel, its
//! per-primitive data the builtins it read, which the fragment function writes. Without an object
//! function (`mesh_draws_grid`) the pixel is its threadgroup's position, and the data its position
//! and the grid's threadgroups in the first attachment; the grid's threads, its threadgroup's size
//! packed 10 bits an axis, and its thread 0's position in the grid, `x | y << 16`, in the second.
//! Under the object function (`mesh_draws_object`, whose thread 0 states its own builtins in the
//! payload and launches `ctl.x + group.x` by `ctl.y` mesh threadgroups, `ctl` its buffer 0) the
//! pixel is the mesh threadgroup's x and the object threadgroup's linear index times 2 plus its
//! y, and the data the payload's object words, then the mesh threadgroup's position, grid and
//! size.
//!
//! Every answer is Metal's, which the M3 Max (macOS 26.7) and an M4 (macOS 27.0.1) measured
//! alike, with and without API validation: [`GRID_ANSWERS`] and [`OBJECT_ANSWERS`] hold the probe's
//! lines, which the drawn pixels are printed as and compared with. Among them: `drawMeshThreads`
//! without an object function launches the whole threadgroups its threads fill, whose threads are
//! the grid the builtins report (10 by 3 threads in threadgroups of 4 by 2: 2 by 1 threadgroups of
//! an 8 by 2 grid); a grid with a zero axis draws nothing, and so do threads that fill no
//! threadgroup; an object threadgroup that sets no grid launches nothing.
//!
//! On a device with `VK_EXT_mesh_shader` an indirect draw is refused
//! (`NotImplemented(IndirectDrawArguments)`) until a GPU pass bounds its counts by the device's
//! limits, and a `drawMeshThreads` that leaves a partial object threadgroup is refused
//! (`NotImplemented(MeshShading)`); Metal's answers for them are printed (a partial object
//! threadgroup runs the threads left). What Metal's validation
//! stops draws as plain Metal does: a threadgroup of more than 1024 threads and a misaligned
//! argument offset draw nothing, and a draw of the other kind, which stops plain Metal too, is
//! refused typed.
//!
//! A device without `VK_EXT_mesh_shader` draws every case through the emulation, and the words must
//! be Metal's there too: the object function's compute dispatch writes each object threadgroup's
//! grid and payload, a scan places their mesh threadgroups (or an indirect draw's, from Metal's
//! arguments), and the mesh function's compute dispatch writes records that a synthesized vertex
//! stage draws. There the grid case's indirect draw draws; the object case's indirect draw is
//! refused (`NotImplemented(MeshShading)`), and so is a partial object threadgroup, as on a device
//! with the extension.

use super::degrade_notices::{assert_made_once, made_mesh_kinds};
use super::gpu_times::read_kinds;
use super::render_state::{texels, texture};
use super::shared_kinds;
use super::{buffer_with, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::{BindError, Feature, Needs};
use metal2vulkan::metal::{PixelFormat, PrimitiveType};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, MeshPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, Degradation, Device, DrawProblem, Error, Invalid, Library,
    LoadAction, PipelineOptions, Problem, RenderCommandEncoder, RenderPassDescriptor,
    RenderPipelineState, RuntimeGap, Size, StoreAction,
};

/// the Metal probe's group `g3` (`mesh_draws_grid` without an object function), as it printed it on
/// both Macs: `PX <case> <x> <y> a=<words> b=<words in hex>` for each pixel a case drew, then
/// `RESULT <case> drawn=<pixels>`.
const GRID_ANSWERS: &str = "\
PX grid-groups-3x2-by-4 0 0 a=0,0,3,2 b=c,2,100404,0
PX grid-groups-3x2-by-4 1 0 a=1,0,3,2 b=c,2,100404,4
PX grid-groups-3x2-by-4 2 0 a=2,0,3,2 b=c,2,100404,8
PX grid-groups-3x2-by-4 0 1 a=0,1,3,2 b=c,2,100404,10000
PX grid-groups-3x2-by-4 1 1 a=1,1,3,2 b=c,2,100404,10004
PX grid-groups-3x2-by-4 2 1 a=2,1,3,2 b=c,2,100404,10008
RESULT grid-groups-3x2-by-4 drawn=6
PX grid-groups-2x2-by-2x3 0 0 a=0,0,2,2 b=4,6,100c02,0
PX grid-groups-2x2-by-2x3 1 0 a=1,0,2,2 b=4,6,100c02,2
PX grid-groups-2x2-by-2x3 0 1 a=0,1,2,2 b=4,6,100c02,30000
PX grid-groups-2x2-by-2x3 1 1 a=1,1,2,2 b=4,6,100c02,30002
RESULT grid-groups-2x2-by-2x3 drawn=4
PX grid-threads-12x4-by-4x2 0 0 a=0,0,3,2 b=c,4,100804,0
PX grid-threads-12x4-by-4x2 1 0 a=1,0,3,2 b=c,4,100804,4
PX grid-threads-12x4-by-4x2 2 0 a=2,0,3,2 b=c,4,100804,8
PX grid-threads-12x4-by-4x2 0 1 a=0,1,3,2 b=c,4,100804,20000
PX grid-threads-12x4-by-4x2 1 1 a=1,1,3,2 b=c,4,100804,20004
PX grid-threads-12x4-by-4x2 2 1 a=2,1,3,2 b=c,4,100804,20008
RESULT grid-threads-12x4-by-4x2 drawn=6
PX grid-threads-10x3-by-4x2 0 0 a=0,0,2,1 b=8,2,100804,0
PX grid-threads-10x3-by-4x2 1 0 a=1,0,2,1 b=8,2,100804,4
RESULT grid-threads-10x3-by-4x2 drawn=2
RESULT grid-threads-3x1-by-4 drawn=0
PX grid-indirect-3x1-by-8 0 0 a=0,0,3,1 b=18,1,100408,0
PX grid-indirect-3x1-by-8 1 0 a=1,0,3,1 b=18,1,100408,8
PX grid-indirect-3x1-by-8 2 0 a=2,0,3,1 b=18,1,100408,10
RESULT grid-indirect-3x1-by-8 drawn=3
RESULT grid-groups-0 drawn=0
";

/// the Metal probe's group `g4` (`mesh_draws_object` and `mesh_draws_mesh`), as it printed it on
/// both Macs, in [`GRID_ANSWERS`]' form.
const OBJECT_ANSWERS: &str = "\
PX object-groups-3-by-4-mesh-2 0 0 a=0,65539,1049604,65548 b=0,0,10002,100402
PX object-groups-3-by-4-mesh-2 1 0 a=0,65539,1049604,65548 b=1,0,10002,100402
PX object-groups-3-by-4-mesh-2 0 2 a=1,65539,1049604,65548 b=0,0,10003,100402
PX object-groups-3-by-4-mesh-2 1 2 a=1,65539,1049604,65548 b=1,0,10003,100402
PX object-groups-3-by-4-mesh-2 2 2 a=1,65539,1049604,65548 b=2,0,10003,100402
PX object-groups-3-by-4-mesh-2 0 4 a=2,65539,1049604,65548 b=0,0,10004,100402
PX object-groups-3-by-4-mesh-2 1 4 a=2,65539,1049604,65548 b=1,0,10004,100402
PX object-groups-3-by-4-mesh-2 2 4 a=2,65539,1049604,65548 b=2,0,10004,100402
PX object-groups-3-by-4-mesh-2 3 4 a=2,65539,1049604,65548 b=3,0,10004,100402
RESULT object-groups-3-by-4-mesh-2 drawn=9
PX object-groups-2-mesh-2x2 0 0 a=0,65538,1049602,65540 b=0,0,20001,100802
PX object-groups-2-mesh-2x2 0 1 a=0,65538,1049602,65540 b=0,1,20001,100802
PX object-groups-2-mesh-2x2 0 2 a=1,65538,1049602,65540 b=0,0,20002,100802
PX object-groups-2-mesh-2x2 1 2 a=1,65538,1049602,65540 b=1,0,20002,100802
PX object-groups-2-mesh-2x2 0 3 a=1,65538,1049602,65540 b=0,1,20002,100802
PX object-groups-2-mesh-2x2 1 3 a=1,65538,1049602,65540 b=1,1,20002,100802
RESULT object-groups-2-mesh-2x2 drawn=6
PX object-threads-6-by-4 0 0 a=0,65538,1049604,65542 b=0,0,10002,100402
PX object-threads-6-by-4 1 0 a=0,65538,1049604,65542 b=1,0,10002,100402
PX object-threads-6-by-4 0 2 a=1,65538,1049602,65542 b=0,0,10003,100402
PX object-threads-6-by-4 1 2 a=1,65538,1049602,65542 b=1,0,10003,100402
PX object-threads-6-by-4 2 2 a=1,65538,1049602,65542 b=2,0,10003,100402
RESULT object-threads-6-by-4 drawn=5
PX object-threads-5x3-by-4x2 0 0 a=0,131074,1050628,196613 b=0,0,10001,100402
PX object-threads-5x3-by-4x2 0 2 a=1,131074,1050625,196613 b=0,0,10002,100402
PX object-threads-5x3-by-4x2 1 2 a=1,131074,1050625,196613 b=1,0,10002,100402
PX object-threads-5x3-by-4x2 0 4 a=65536,131074,1049604,196613 b=0,0,10001,100402
PX object-threads-5x3-by-4x2 0 6 a=65537,131074,1049601,196613 b=0,0,10002,100402
PX object-threads-5x3-by-4x2 1 6 a=65537,131074,1049601,196613 b=1,0,10002,100402
RESULT object-threads-5x3-by-4x2 drawn=6
PX object-grid-0-then-1 0 2 a=1,65538,1049601,65538 b=0,0,10001,100402
RESULT object-grid-0-then-1 drawn=1
PX object-indirect-2 0 0 a=0,65538,1049604,65544 b=0,0,10001,100402
PX object-indirect-2 0 2 a=1,65538,1049604,65544 b=0,0,10002,100402
PX object-indirect-2 1 2 a=1,65538,1049604,65544 b=1,0,10002,100402
RESULT object-indirect-2 drawn=3
";

/// The kind [`object_draws_store_metals_words`] takes on a device without `VK_EXT_mesh_shader`,
/// whose mesh draws are emulated, and not elsewhere.
pub(super) const EMULATED_MESH: [Degradation; 1] = [Degradation::EmulatedMeshDraws];

/// Whether `device` emulates mesh draws: its profile lacks `VK_EXT_mesh_shader` or its
/// `meshShader` feature, as the compile layer decides it.
pub(super) fn emulates(device: &Device) -> bool {
    let enabled = device.compile_device().profile().enabled();
    !enabled.extensions.contains(&VkExtension::ExtMeshShader)
        || !enabled.features.contains(&VkFeature::MeshShader)
}

/// The pass's side in pixels.
const SIDE: u64 = 8;

/// One mesh draw of the probe: its grid, then its threads per object threadgroup and per mesh
/// threadgroup.
#[derive(Clone, Copy)]
enum Draw {
    /// `drawMeshThreadgroups` of these threadgroups.
    Groups([u64; 3], [u64; 3], [u64; 3]),
    /// `drawMeshThreads` of these threads.
    Threads([u64; 3], [u64; 3], [u64; 3]),
    /// `drawMeshThreadgroupsWithIndirectBuffer` of these counts at this offset of a zeroed
    /// 64-byte buffer.
    Indirect([u32; 3], u64, [u64; 3], [u64; 3]),
}

/// The probe's cases of `mesh_draws_grid`, its threads per object threadgroup 1.
const GRID_CASES: [(&str, Draw); 7] = [
    (
        "grid-groups-3x2-by-4",
        Draw::Groups([3, 2, 1], [1; 3], [4, 1, 1]),
    ),
    (
        "grid-groups-2x2-by-2x3",
        Draw::Groups([2, 2, 1], [1; 3], [2, 3, 1]),
    ),
    (
        "grid-threads-12x4-by-4x2",
        Draw::Threads([12, 4, 1], [1; 3], [4, 2, 1]),
    ),
    (
        "grid-threads-10x3-by-4x2",
        Draw::Threads([10, 3, 1], [1; 3], [4, 2, 1]),
    ),
    (
        "grid-threads-3x1-by-4",
        Draw::Threads([3, 1, 1], [1; 3], [4, 1, 1]),
    ),
    (
        "grid-indirect-3x1-by-8",
        Draw::Indirect([3, 1, 1], 8, [1; 3], [8, 1, 1]),
    ),
    ("grid-groups-0", Draw::Groups([0, 2, 1], [1; 3], [4, 1, 1])),
];

/// The probe's cases of `mesh_draws_object` and `mesh_draws_mesh`, each with the object
/// function's `ctl` (bound at object buffer 0).
const OBJECT_CASES: [(&str, [u32; 4], Draw); 6] = [
    (
        "object-groups-3-by-4-mesh-2",
        [2, 1, 0, 0],
        Draw::Groups([3, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    (
        "object-groups-2-mesh-2x2",
        [1, 2, 0, 0],
        Draw::Groups([2, 1, 1], [2, 1, 1], [2, 2, 1]),
    ),
    (
        "object-threads-6-by-4",
        [2, 1, 0, 0],
        Draw::Threads([6, 1, 1], [4, 1, 1], [2, 1, 1]),
    ),
    (
        "object-threads-5x3-by-4x2",
        [1, 1, 0, 0],
        Draw::Threads([5, 3, 1], [4, 2, 1], [2, 1, 1]),
    ),
    (
        "object-grid-0-then-1",
        [0, 1, 0, 0],
        Draw::Groups([2, 1, 1], [1; 3], [2, 1, 1]),
    ),
    (
        "object-indirect-2",
        [1, 1, 0, 0],
        Draw::Indirect([2, 1, 1], 4, [4, 1, 1], [2, 1, 1]),
    ),
];

fn size(axes: [u64; 3]) -> Size {
    Size::new(axes[0], axes[1], axes[2])
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("mesh_draws.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    /// The state of `mesh` (and `object`, where given) with the fragment function, over the
    /// pass's two attachments.
    fn pipeline(&self, object: Option<&str>, mesh: &str) -> Result<RenderPipelineState, Error> {
        self.device.new_mesh_render_pipeline_state(
            &self.descriptor(object, mesh),
            &PipelineOptions::default(),
        )
    }

    /// The descriptor of [`pipeline`](Self::pipeline)'s state.
    fn descriptor(&self, object: Option<&str>, mesh: &str) -> MeshPipelineDescriptor {
        let function = |name| {
            StageSource::Function(
                self.library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = MeshPipelineDescriptor::new(function(mesh));
        descriptor.object = object.map(function);
        descriptor.fragment = Some(function("mesh_draws_fragment"));
        for attachment in &mut descriptor.color_attachments[..2] {
            *attachment = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        }
        descriptor
    }

    /// A pass of `state` whose commands `body` encodes: the probe's lines of the pixels it drew,
    /// as case `name`, or the first refusal `body` returns (the pass then ends without it).
    fn run(
        &self,
        name: &str,
        state: &RenderPipelineState,
        body: impl FnOnce(&mut RenderCommandEncoder) -> Result<(), Error>,
    ) -> Result<Vec<String>, Error> {
        let targets = [0, 1].map(|_| texture(&self.device, PixelFormat::RGBA32Uint, SIDE));
        let mut pass = RenderPassDescriptor::new();
        for (attachment, target) in pass.color_attachments.iter_mut().zip(&targets) {
            attachment.texture = Some(target.clone());
            attachment.load_action = LoadAction::Clear;
            attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            attachment.store_action = StoreAction::Store;
        }
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        let encoded = body(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        encoded?;
        // Each `VkPipeline` the draws created (an emulated mesh draw's compute pipeline beside
        // its graphics one) within the creation ceiling, and the pass's GPU time, printed.
        let created: Vec<std::time::Duration> = command_buffer
            .executions()
            .iter()
            .flat_map(|e| e.pipeline_creation().to_vec())
            .collect();
        for made in &created {
            crate::budgets::creation_within_ceiling(name, *made).unwrap();
        }
        let gpu = match (
            command_buffer.gpu_start_time(),
            command_buffer.gpu_end_time(),
        ) {
            (Ok(start), Ok(end)) => format!("{:.1} us", (end - start) * 1e6),
            _ => "untimed".to_owned(),
        };
        eprintln!(
            "TIMING {name}: created {:?} ms, gpu {gpu}",
            created.iter().map(|d| d.as_millis()).collect::<Vec<_>>()
        );
        let words = |t| -> Vec<u32> {
            texels(t, 16)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        };
        let (a, b) = (words(&targets[0]), words(&targets[1]));
        let mut lines = Vec::new();
        for y in 0..SIDE as usize {
            for x in 0..SIDE as usize {
                let at = 4 * (y * SIDE as usize + x);
                let (wa, wb) = (&a[at..at + 4], &b[at..at + 4]);
                if wa.iter().chain(wb).all(|&w| w == 0) {
                    continue;
                }
                let decimal: Vec<String> = wa.iter().map(u32::to_string).collect();
                let hex: Vec<String> = wb.iter().map(|w| format!("{w:x}")).collect();
                lines.push(format!(
                    "PX {name} {x} {y} a={} b={}",
                    decimal.join(","),
                    hex.join(",")
                ));
            }
        }
        lines.push(format!("RESULT {name} drawn={}", lines.len()));
        Ok(lines)
    }
}

/// `draw` on `e`, its indirect buffer one of `device`'s, kept in `held`.
fn encode(
    device: &Device,
    e: &mut RenderCommandEncoder,
    draw: Draw,
    held: &mut Vec<Buffer>,
) -> Result<(), Error> {
    match draw {
        Draw::Groups(groups, object, mesh) => {
            e.draw_mesh_threadgroups(size(groups), size(object), size(mesh))
        }
        Draw::Threads(threads, object, mesh) => {
            e.draw_mesh_threads(size(threads), size(object), size(mesh))
        }
        Draw::Indirect(counts, offset, object, mesh) => {
            let mut bytes = vec![0u8; 64];
            for (i, count) in counts.iter().enumerate() {
                let at = offset as usize + 4 * i;
                bytes[at..at + 4].copy_from_slice(&count.to_le_bytes());
            }
            let buffer = buffer_with(device, &bytes).unwrap();
            let drawn =
                e.draw_mesh_threadgroups_indirect(&buffer, offset, size(object), size(mesh));
            held.push(buffer);
            drawn
        }
    }
}

/// The probe's lines of case `name` among `answers`.
fn metals(answers: &str, name: &str) -> Vec<String> {
    answers
        .lines()
        .filter(|line| {
            let mut words = line.split(' ');
            words.next();
            words.next() == Some(name)
        })
        .map(str::to_owned)
        .collect()
}

/// Whether `refused` is the pipeline refusal of a device without mesh shading, printed.
fn mesh_refused(refused: &Error, device: &str) -> bool {
    match refused {
        Error::Unsupported(u) if u.feature == Feature::MeshShading => {
            eprintln!("refused: mesh render pipelines on {device}: {u}");
            true
        }
        _ => false,
    }
}

/// Each grid case draws the words Metal drew (the module docs), but the indirect one where the
/// device has `VK_EXT_mesh_shader`, refused.
#[test]
fn mesh_draws_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let started = std::time::Instant::now();
        let state = match lane.pipeline(None, "mesh_draws_grid") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_grid on {name}: {error}"),
        };
        eprintln!(
            "TIMING state on {name}: {} ms",
            started.elapsed().as_millis()
        );
        for (case, draw) in GRID_CASES {
            let mut held = Vec::new();
            let drawn = lane.run(case, &state, |e| encode(&lane.device, e, draw, &mut held));
            match (draw, drawn) {
                (
                    Draw::Indirect(..),
                    Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments)),
                ) => {
                    let metals = metals(GRID_ANSWERS, case);
                    eprintln!("{case} on {name}: refused; Metal's: {metals:?}");
                }
                (_, Ok(lines)) => assert_eq!(lines, metals(GRID_ANSWERS, case), "{case} on {name}"),
                (_, Err(error)) => panic!("{case} on {name}: {error}"),
            }
        }
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// Each object case draws the words Metal drew (the module docs), but the indirect one and those
/// that leave a partial object threadgroup, refused; a device without `VK_EXT_mesh_shader`
/// announces its emulated draws once ([`EMULATED_MESH`]), and another none.
#[test]
fn object_draws_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let state = match lane.pipeline(Some("mesh_draws_object"), "mesh_draws_mesh") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_object on {name}: {error}"),
        };
        // Each case with `ctl` in a buffer, then set as inline bytes, which it reads alike.
        let both = OBJECT_CASES.iter().flat_map(|&c| [(c, false), (c, true)]);
        for ((case, ctl, draw), inline) in both {
            let mut held = Vec::new();
            let bytes: Vec<u8> = ctl.iter().flat_map(|w| w.to_le_bytes()).collect();
            let ctl = buffer_with(&lane.device, &bytes).unwrap();
            let drawn = lane.run(case, &state, |e| {
                match inline {
                    true => e.set_object_bytes(&bytes, 0)?,
                    false => e.set_object_buffer(Some(&ctl), 0, 0)?,
                }
                encode(&lane.device, e, draw, &mut held)
            });
            let metals = metals(OBJECT_ANSWERS, case);
            match (draw, drawn) {
                (
                    Draw::Indirect(..),
                    Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments)),
                ) => eprintln!("{case} on {name}: refused; Metal's: {metals:?}"),
                // The emulation's indirect draw with an object function.
                (Draw::Indirect(..), Err(Error::Bind(BindError::Unsupported(u))))
                    if u.feature == Feature::MeshShading && u.needs == Needs::NotImplemented =>
                {
                    eprintln!("{case} on {name}: refused, emulated; Metal's: {metals:?}")
                }
                (Draw::Threads(..), Err(Error::Bind(BindError::Unsupported(u))))
                    if u.feature == Feature::MeshShading && u.needs == Needs::NotImplemented =>
                {
                    eprintln!(
                        "{case} on {name}: refused, a partial object threadgroup; Metal's: \
                         {metals:?}"
                    )
                }
                (_, Ok(lines)) => assert_eq!(lines, metals, "{case} on {name}"),
                (_, Err(error)) => panic!("{case} on {name}: {error}"),
            }
        }
        // Exactly the emulation's kind where it draws, beside what the state's making forces
        // (its stages' notes: `FastMathRewrites` on ANV), the `Shared` buffers, and the reads of
        // each pass's GPU times its timing line prints (`gpu_times::read_kinds`:
        // `GpuTimesFromHost` on MoltenVK); before, a device that does not emulate let any kind
        // pass.
        let descriptor = lane.descriptor(Some("mesh_draws_object"), "mesh_draws_mesh");
        let mut made = made_mesh_kinds(&lane.device, &descriptor);
        made.extend(shared_kinds(&lane.device));
        made.extend(read_kinds(&lane.device));
        let emulated: &[Degradation] = match emulates(&lane.device) {
            true => &EMULATED_MESH,
            false => &[],
        };
        assert_made_once(&messages, emulated, &made);
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// The pixels Metal draws for `cases` drawn in order in one pass, as case `name`: each case's
/// pixels from its probe lines (`answers`), a later case's over an earlier one's, since Metal
/// rasterizes an encoder's primitives in order and the pass neither blends nor tests depth.
fn overlaid(cases: &[(&str, &str)], name: &str) -> Vec<String> {
    let mut pixels = std::collections::BTreeMap::new();
    for (answers, case) in cases {
        for line in metals(answers, case) {
            let words: Vec<&str> = line.split(' ').collect();
            if let ["PX", _, x, y, a, b] = words[..] {
                let at: (u32, u32) = (y.parse().unwrap(), x.parse().unwrap());
                pixels.insert(at, format!("{a} {b}"));
            }
        }
    }
    let mut lines: Vec<String> = pixels
        .iter()
        .map(|((y, x), data)| format!("PX {name} {x} {y} {data}"))
        .collect();
    lines.push(format!("RESULT {name} drawn={}", lines.len()));
    lines
}

/// Draws of both pipelines in one pass draw the words Metal draws for each in order (the
/// module docs): two draws of the object function, whose second's mesh threadgroups and, on a
/// device without `VK_EXT_mesh_shader`, records follow the first's in the command buffer's
/// records pool, and then a draw without one.
#[test]
fn draws_of_both_pipelines_in_one_pass_store_metals_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let (object, grid) = match (
            lane.pipeline(Some("mesh_draws_object"), "mesh_draws_mesh"),
            lane.pipeline(None, "mesh_draws_grid"),
        ) {
            (Ok(object), Ok(grid)) => (object, grid),
            (Err(refused), _) | (_, Err(refused)) if mesh_refused(&refused, &name) => return,
            (Err(error), _) | (_, Err(error)) => panic!("the mesh pipelines on {name}: {error}"),
        };
        let ctl = |words: [u32; 4]| {
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            buffer_with(&lane.device, &bytes).unwrap()
        };
        let (first, second) = (ctl([1, 2, 0, 0]), ctl([2, 1, 0, 0]));
        let one = [1; 3];
        let lines = lane
            .run("both", &object, |e| {
                e.set_object_buffer(Some(&first), 0, 0)?;
                e.draw_mesh_threadgroups(size([2, 1, 1]), size([2, 1, 1]), size([2, 2, 1]))?;
                e.set_object_buffer(Some(&second), 0, 0)?;
                e.draw_mesh_threadgroups(size([3, 1, 1]), size([4, 1, 1]), size([2, 1, 1]))?;
                e.set_render_pipeline_state(&grid)?;
                e.draw_mesh_threadgroups(size([2, 2, 1]), size(one), size([2, 3, 1]))
            })
            .unwrap();
        let metals = overlaid(
            &[
                (OBJECT_ANSWERS, "object-groups-2-mesh-2x2"),
                (OBJECT_ANSWERS, "object-groups-3-by-4-mesh-2"),
                (GRID_ANSWERS, "grid-groups-2x2-by-2x3"),
            ],
            "both",
        );
        assert_eq!(lines, metals, "on {name}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// What Metal's validation stops draws as plain Metal does (a probe): more than 1024 threads in a
/// threadgroup and a misaligned argument offset draw nothing; what stops plain Metal too, and
/// arguments past the buffer, are refused typed. The pass draws nothing of it.
#[test]
fn mesh_draws_answer_as_plain_metal_and_refuse_the_rest() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        let state = match lane.pipeline(None, "mesh_draws_grid") {
            Ok(state) => state,
            Err(refused) if mesh_refused(&refused, &name) => return,
            Err(error) => panic!("mesh_draws_grid on {name}: {error}"),
        };
        let invalid =
            |call, problem| Err(Error::Invalid(Invalid::new(call, Problem::Draw(problem))));
        let one = size([1; 3]);
        let arguments = buffer_with(&lane.device, &[0; 16]).unwrap();
        let lines = lane
            .run("refused", &state, |e| {
                // Probe 1b: more than 1024 threads in a mesh threadgroup draw nothing.
                assert_eq!(
                    e.draw_mesh_threadgroups(one, one, size([1025, 1, 1])),
                    Ok(())
                );
                assert_eq!(
                    e.draw_mesh_threads(size([64, 1, 1]), one, size([32, 33, 1])),
                    Ok(())
                );
                // Probe 1b: a misaligned argument offset draws nothing; arguments past the
                // buffer are refused, as every indirect draw's.
                let indirect = "draw_mesh_threadgroups_indirect";
                assert_eq!(
                    e.draw_mesh_threadgroups_indirect(&arguments, 2, one, one),
                    Ok(())
                );
                assert_eq!(
                    e.draw_mesh_threadgroups_indirect(&arguments, 8, one, one),
                    invalid(indirect, DrawProblem::IndirectBufferRange)
                );
                // Probe 1b: a draw of primitives with a mesh render pipeline's state.
                assert_eq!(
                    e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0),
                    invalid("draw_primitives", DrawProblem::DrawKind)
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(lines, ["RESULT refused drawn=0"], "on {name}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
