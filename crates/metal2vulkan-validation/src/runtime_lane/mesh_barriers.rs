//! Mesh draws behind memory barriers on the device the cfg selects, under the validation layer: the
//! Metal probe's group `g2`, drawn through `metal2vulkan-runtime` with the probe's functions (the
//! owned fixture `mesh_barrier`). Each case is one pass of one `RGBA32Uint` attachment, 8x8,
//! cleared to zero and stored, and a control buffer of four words, `(1, 1, 1, 0)` from the CPU. A
//! writer draw covers pixel (7, 7) and stores the control words from its mesh function (`(3, 2, 1,
//! 7)`) or its fragment function (`(3, 2, 1, 9)`); a memory barrier over buffers (or over the
//! control buffer) from the writer's stage to the reader's follows; then a reader draw reads them:
//! its object function launches `ctl.x` by `ctl.y` mesh threadgroups, or each threadgroup of its 4
//! by 3 grid below them draws, or an indirect draw takes its grid from them. Each drawing mesh
//! threadgroup covers the pixel of its position with `(x, y, ctl.z, ctl.w)`.
//!
//! [`ANSWERS`] holds the probe's lines, as an M4 (macOS 27.0.1) printed them without API
//! validation (our probe runs them on the M3 Max), which the drawn pixels
//! and the control words are printed as and compared with. Behind a barrier after the mesh stage
//! the reader reads the writer's words, with and without validation. A barrier after the
//! fragment stage is one Metal's validation stops on an Apple GPU ("afterStages (0x2) can not
//! contain MTLRenderStageTile or MTLRenderStageFragment on this device"), and plain Metal does
//! not order the fragment's write before the reader, which reads the words the CPU wrote. The
//! runtime's reader reads them too, from a copy taken before the pass
//! (`render/pass_start_reads.rs`, as a vertex stage's), across the split the barrier makes: the
//! lane compares each case with Metal's lines.
//!
//! On a device without `VK_EXT_mesh_shader` the reader's compute work (its object function, the
//! scan, its mesh function) reads what the writer wrote behind the barrier, so it records before
//! the reader's own render pass instance, which the barrier began, after the writer's. There the
//! indirect draw of the object reader is refused (`NotImplemented(MeshShading)`); on a device with
//! the extension every indirect draw is (`NotImplemented(IndirectDrawArguments)`).

use super::render_state::{texels, texture};
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::{BindError, Feature, Needs};
use metal2vulkan::metal::{BarrierScope, PixelFormat, RenderStages};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, MeshPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandQueue, Device, Error, Library, LoadAction, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceRef, RuntimeGap, Size, StoreAction,
};

/// the Metal probe's group `g2`, as an M4 printed it without API validation: `PX <case> <x> <y>
/// a=<words>` for each pixel a case drew, `RESULT <case> drawn=<pixels>`, then `CTL <case>
/// <words>`, the control words after the pass.
const ANSWERS: &str = "\
PX mesh-to-object 0 0 a=0,0,1,7
PX mesh-to-object 1 0 a=1,0,1,7
PX mesh-to-object 2 0 a=2,0,1,7
PX mesh-to-object 0 1 a=0,1,1,7
PX mesh-to-object 1 1 a=1,1,1,7
PX mesh-to-object 2 1 a=2,1,1,7
PX mesh-to-object 7 7 a=43690,0,0,0
RESULT mesh-to-object drawn=7
CTL mesh-to-object 3,2,1,7
PX fragment-to-object 0 0 a=0,0,1,0
PX fragment-to-object 7 7 a=48059,0,0,0
RESULT fragment-to-object drawn=2
CTL fragment-to-object 3,2,1,9
PX fragment-to-object-resources 0 0 a=0,0,1,0
PX fragment-to-object-resources 7 7 a=48059,0,0,0
RESULT fragment-to-object-resources drawn=2
CTL fragment-to-object-resources 3,2,1,9
PX fragment-to-mesh 0 0 a=0,0,1,0
PX fragment-to-mesh 7 7 a=48059,0,0,0
RESULT fragment-to-mesh drawn=2
CTL fragment-to-mesh 3,2,1,9
PX mesh-to-mesh 0 0 a=0,0,1,7
PX mesh-to-mesh 1 0 a=1,0,1,7
PX mesh-to-mesh 2 0 a=2,0,1,7
PX mesh-to-mesh 0 1 a=0,1,1,7
PX mesh-to-mesh 1 1 a=1,1,1,7
PX mesh-to-mesh 2 1 a=2,1,1,7
PX mesh-to-mesh 7 7 a=43690,0,0,0
RESULT mesh-to-mesh drawn=7
CTL mesh-to-mesh 3,2,1,7
PX fragment-to-indirect 0 0 a=0,0,1,0
PX fragment-to-indirect 7 7 a=48059,0,0,0
RESULT fragment-to-indirect drawn=2
CTL fragment-to-indirect 3,2,1,9
PX fragment-to-indirect-object 0 0 a=0,0,1,0
PX fragment-to-indirect-object 7 7 a=48059,0,0,0
RESULT fragment-to-indirect-object drawn=2
CTL fragment-to-indirect-object 3,2,1,9
";

/// The pass's side in pixels.
const SIDE: u64 = 8;

/// The writer draw's stage.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Writer {
    Mesh,
    Fragment,
}

/// The reader draw.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reader {
    /// The object function's, one object threadgroup.
    Object,
    /// Without an object function, a 4 by 3 grid.
    Grid,
    /// Without an object function, its grid the control words.
    GridIndirect,
    /// The object function's, its object threadgroups the control words.
    ObjectIndirect,
}

/// The barrier between them: over buffers, or over the control buffer, after and before these
/// stages.
#[derive(Clone, Copy)]
enum Barrier {
    Scope(RenderStages, RenderStages),
    Resources(RenderStages, RenderStages),
}

/// The probe's cases.
const CASES: [(&str, Writer, Barrier, Reader); 7] = [
    (
        "mesh-to-object",
        Writer::Mesh,
        Barrier::Scope(RenderStages::MESH, RenderStages::OBJECT),
        Reader::Object,
    ),
    (
        "fragment-to-object",
        Writer::Fragment,
        Barrier::Scope(RenderStages::FRAGMENT, RenderStages::OBJECT),
        Reader::Object,
    ),
    (
        "fragment-to-object-resources",
        Writer::Fragment,
        Barrier::Resources(RenderStages::FRAGMENT, RenderStages::OBJECT),
        Reader::Object,
    ),
    (
        "fragment-to-mesh",
        Writer::Fragment,
        Barrier::Scope(RenderStages::FRAGMENT, RenderStages::MESH),
        Reader::Grid,
    ),
    (
        "mesh-to-mesh",
        Writer::Mesh,
        Barrier::Scope(RenderStages::MESH, RenderStages::MESH),
        Reader::Grid,
    ),
    (
        "fragment-to-indirect",
        Writer::Fragment,
        Barrier::Scope(
            RenderStages::FRAGMENT,
            RenderStages::OBJECT.union(RenderStages::MESH),
        ),
        Reader::GridIndirect,
    ),
    (
        "fragment-to-indirect-object",
        Writer::Fragment,
        Barrier::Scope(
            RenderStages::FRAGMENT,
            RenderStages::OBJECT.union(RenderStages::MESH),
        ),
        Reader::ObjectIndirect,
    ),
];

/// The probe's lines of case `name`.
fn metals(name: &str) -> Vec<String> {
    ANSWERS
        .lines()
        .filter(|line| line.split(' ').nth(1) == Some(name))
        .map(str::to_owned)
        .collect()
}

/// The lane's device, queue and the fixture's library.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("mesh_barrier.ll")).unwrap();
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            device,
        }
    }

    /// The state of `mesh` (and `object`, where given) with `fragment`, over the pass's
    /// attachment.
    fn pipeline(
        &self,
        object: Option<&str>,
        mesh: &str,
        fragment: &str,
    ) -> Result<RenderPipelineState, Error> {
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
        descriptor.fragment = Some(function(fragment));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
        self.device
            .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
    }

    /// Case `name`: the probe's lines of the pixels it drew and its control words, or the first
    /// refusal of the reader's draw (the pass then ends without it).
    fn run(
        &self,
        name: &str,
        (writer, barrier, reader): (Writer, Barrier, Reader),
    ) -> Result<Vec<String>, Error> {
        let one = Size::new(1, 1, 1);
        let writes = match writer {
            Writer::Mesh => self.pipeline(None, "mesh_barrier_writer", "mesh_barrier_fragment"),
            Writer::Fragment => {
                self.pipeline(None, "mesh_barrier_cover", "mesh_barrier_fragment_writer")
            }
        }
        .unwrap();
        let reads = match reader {
            Reader::Object | Reader::ObjectIndirect => self.pipeline(
                Some("mesh_barrier_object"),
                "mesh_barrier_mesh",
                "mesh_barrier_fragment",
            ),
            Reader::Grid | Reader::GridIndirect => {
                self.pipeline(None, "mesh_barrier_grid", "mesh_barrier_fragment")
            }
        }
        .unwrap();
        let words: Vec<u8> = [1u32, 1, 1, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let ctl: Buffer = buffer_with(&self.device, &words).unwrap();
        let target = texture(&self.device, PixelFormat::RGBA32Uint, SIDE);
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        attachment.store_action = StoreAction::Store;
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut e = command_buffer.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&writes).unwrap();
        match writer {
            Writer::Mesh => e.set_mesh_buffer(Some(&ctl), 0, 0).unwrap(),
            Writer::Fragment => e.set_fragment_buffer(Some(&ctl), 0, 0).unwrap(),
        }
        e.draw_mesh_threadgroups(one, one, one).unwrap();
        match barrier {
            Barrier::Scope(after, before) => {
                e.memory_barrier_with_scope(BarrierScope::BUFFERS, after, before)
            }
            Barrier::Resources(after, before) => e
                .memory_barrier_with_resources(&[ResourceRef::Buffer(&ctl)], after, before)
                .unwrap(),
        }
        e.set_render_pipeline_state(&reads).unwrap();
        let drawn = match reader {
            Reader::Object => e
                .set_object_buffer(Some(&ctl), 0, 0)
                .and_then(|()| e.draw_mesh_threadgroups(one, one, one)),
            Reader::Grid => e
                .set_mesh_buffer(Some(&ctl), 0, 0)
                .and_then(|()| e.draw_mesh_threadgroups(Size::new(4, 3, 1), one, one)),
            Reader::GridIndirect => e
                .set_mesh_buffer(Some(&ctl), 0, 0)
                .and_then(|()| e.draw_mesh_threadgroups_indirect(&ctl, 0, one, one)),
            Reader::ObjectIndirect => e
                .set_object_buffer(Some(&ctl), 0, 0)
                .and_then(|()| e.draw_mesh_threadgroups_indirect(&ctl, 0, one, one)),
        };
        e.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{name}");
        drawn?;
        let texel_words: Vec<u32> = texels(&target, 16)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let mut lines = Vec::new();
        for y in 0..SIDE as usize {
            for x in 0..SIDE as usize {
                let at = 4 * (y * SIDE as usize + x);
                let pixel = &texel_words[at..at + 4];
                if pixel.iter().all(|&w| w == 0) {
                    continue;
                }
                let decimal: Vec<String> = pixel.iter().map(u32::to_string).collect();
                lines.push(format!("PX {name} {x} {y} a={}", decimal.join(",")));
            }
        }
        lines.push(format!("RESULT {name} drawn={}", lines.len()));
        let after: Vec<String> = read_buffer(&ctl, 0, 16)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w).to_string())
            .collect();
        lines.push(format!("CTL {name} {}", after.join(",")));
        Ok(lines)
    }
}

/// Each case draws the pixels and leaves the control words Metal did, behind a barrier after the
/// mesh stage and after the fragment stage (the module docs), but the indirect draws a device
/// refuses, as the module docs say.
#[test]
fn readers_behind_a_barrier_read_the_writers_words() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let lane = Lane::new(device);
        for (case, writer, barrier, reader) in CASES {
            let metals = metals(case);
            match (reader, lane.run(case, (writer, barrier, reader))) {
                (_, Ok(lines)) => assert_eq!(lines, metals, "{case} on {name}"),
                (
                    Reader::GridIndirect | Reader::ObjectIndirect,
                    Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments)),
                ) => eprintln!("{case} on {name}: refused; Metal's: {metals:?}"),
                (Reader::ObjectIndirect, Err(Error::Bind(BindError::Unsupported(u))))
                    if u.feature == Feature::MeshShading && u.needs == Needs::NotImplemented =>
                {
                    eprintln!("{case} on {name}: refused, emulated; Metal's: {metals:?}")
                }
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
