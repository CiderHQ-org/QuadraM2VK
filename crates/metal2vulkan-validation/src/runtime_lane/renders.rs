//! Device tests of render command encoders (its first part, the pass descriptor, load and store
//! actions, the deferred begin, and direct draws; its second, the depth/stencil state and the rest
//! of the render state) on the device the cfg selects, under the validation layer.
//!
//! - **The census.** Every committed public render case whose inputs these parts of the runtime
//!   bind draws through the runtime as the Metal oracle draws it (`metal::platform`), by the case
//!   runner the engine executor's runtime path shares (`candidate::runtime_case`'s
//!   `commit_render`, which states the draw): a library of its AIR and one of its generated
//!   companion's, each stage's function specialized with its constant values and the case's own
//!   linking its functions, a render pipeline state of the oracle's descriptor, its attachments,
//!   state, vertex buffers and resources, its one draw, and its output read back after the command
//!   buffer completed, a multisampled pass's sample by sample; a fragment case's explicit
//!   imageblock between the oracle's generated initializer and resolver, and a rate-mapped pass
//!   drawn region by region. A case the executor's route keeps for a later step
//!   (`runtime_run::render_prerequisites`: argument buffers, visible-function tables, out-of-bounds
//!   targets) is counted, not run. So is a case of out-of-bounds targets the route runs on the
//!   runtime: its output is judged by its containment, which the runtime path decides from what its
//!   draw executed and the census does not, so the runtime path's tests run it
//!   (`runtime_run::tests`, `runs_as_the_executor_compiles_it`; a fetch past a vertex buffer placed
//!   in a heap reads the heap's bytes, which no byte comparison admits). So is a case whose
//!   argument buffers hold pipeline states or command buffers: their templates' states are the
//!   runtime path's to make (`runtime_run::template_states`), and so are the command buffers'
//!   executions after the draw; its tests run it. Where the runtime's output on this device is not
//!   Metal's, the engine candidate runs the case afresh, as the kernel census does
//!   (`kernel_textures.rs`), and the case passes only when the runtime's bytes are the candidate's.
//!   A case with no current Metal row (one awaiting Metal qualification) goes the same way, and
//!   says so.
//! - **Clears** write Metal's measured bytes: each color format, depth and stencil value an M4 was
//!   measured with, the NaNs of half, small-float and float formats among them (a Metal probe, the
//!   M3 Max alike). A format the device cannot render is counted.
//! - **Load and store actions**, the encoder's late store action, the passes' order against blits
//!   in one command buffer, and Metal's refusals of passes and draws, typed. What a `DontCare`
//!   load or store leaves is printed: Metal leaves it undefined, and Apple's GPU keeps the old
//!   texels (measured).

use super::{
    buffer_with, metal_output, read_buffer, texture_usage, validated_device, Messages, RuntimeRun,
    ScratchCache, WAIT,
};
use crate::air::reflect::FragmentImageblock;
use crate::candidate::engine::graphics::{render_case, RenderCase};
use crate::candidate::engine::runtime_run::render_prerequisites;
use crate::candidate::engine::{linked_modules, LinkedModule, ENGINE_EXECUTOR_ABI};
use crate::candidate::runtime_case::{
    commit_render, render_pipeline_descriptor, render_pipeline_state, RenderInputs, TemplateState,
};
use crate::candidate::runtime_samples::encode as samples_of;
use crate::candidate::tests::engine_render;
use crate::case::{AuthoredCase, Stage};
use crate::check::interface::AirInterface;
use crate::executor_contract::multisample::{ReadbackView, SampleRead};
use crate::gpu::TEST_DEVICE;
use crate::store::CorpusStore;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan::pipeline::RenderPipelineDescriptor;
use metal2vulkan_runtime::{
    AppleOnly, Attachment, BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus,
    Device, DrawProblem, Error, Invalid, LoadAction, Origin, Problem, Region, RenderCommandEncoder,
    RenderPassDescriptor, RenderPassProblem, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};
use std::collections::BTreeMap;
use std::path::Path;

/// A committed public render case for the runtime: the case, its render mapping, its AIR
/// bitcode and its generated companion's, the usage the Metal oracle gives each of its textures,
/// its AIR interface (whose argument-buffer members give its argument-buffer textures' usage),
/// and the output its current Metal observation records, or why it has none (a case awaiting
/// Metal qualification, such as the macOS 27 cases).
pub(super) struct PublicRender {
    pub(super) case: AuthoredCase,
    pub(super) render: RenderCase,
    bitcode: Vec<u8>,
    companion: Option<Vec<u8>>,
    linked: Vec<LinkedModule>,
    texture_usage: BTreeMap<u32, TextureUsage>,
    interface: AirInterface,
    /// A fragment case's imageblock master, the checked reflection's.
    imageblock: Option<FragmentImageblock>,
    pub(super) metal_output: Result<Vec<u8>, String>,
}

/// The template states of a lane case, which names no pipeline template (the census leaves those
/// to the runtime path's tests, the module's note).
static NO_TEMPLATES: BTreeMap<u64, TemplateState> = BTreeMap::new();

impl PublicRender {
    /// What the case runner draws of the case.
    pub(super) fn inputs(&self) -> RenderInputs<'_> {
        RenderInputs {
            case: &self.case,
            render: &self.render,
            bitcode: &self.bitcode,
            companion: self.companion.as_deref(),
            linked: &self.linked,
            texture_usage: &self.texture_usage,
            interface: &self.interface,
            imageblock: self.imageblock.as_ref(),
            templates: &NO_TEMPLATES,
        }
    }
}

/// Every committed render case whose AIR is a public fixture, in shard order: mapped, or why
/// not (a later runtime step's input among them, the executor's route's
/// `runtime_run::render_prerequisites`).
pub(super) fn render_cases() -> Vec<(String, Result<PublicRender, String>)> {
    render_cases_where(is_public_render)
}

/// Whether `case` is a committed render case whose AIR is a public fixture.
pub(super) fn is_public_render(case: &AuthoredCase) -> bool {
    matches!(case.stage, Stage::Vertex | Stage::Fragment) && case.public_source.is_some()
}

/// The committed cases `select` takes (public render cases, [`is_public_render`], or some of
/// them), in shard order, mapped as [`render_cases`] maps them; only those are mapped.
pub(super) fn render_cases_where(
    select: impl Fn(&AuthoredCase) -> bool,
) -> Vec<(String, Result<PublicRender, String>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let mut cases = Vec::new();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            if select(&case) {
                let name = case.name.clone();
                cases.push((name, map_render(&root, &store, case)));
            }
        }
    }
    assert!(
        !cases.is_empty(),
        "no committed public render case selected"
    );
    cases
}

/// `case` mapped as [`render_cases`] maps it: refused where the executor's route keeps it for a
/// later runtime step.
pub(super) fn map_render(
    root: &Path,
    store: &CorpusStore,
    case: AuthoredCase,
) -> Result<PublicRender, String> {
    map_render_where(root, store, case, true)
}

/// The committed public render case named `name`, mapped whatever runtime step its inputs wait
/// for: a lane test that encodes those inputs itself draws it.
pub(super) fn public_render(name: &str) -> PublicRender {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let case = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .find(|case| case.name == name && case.public_source.is_some())
        .unwrap_or_else(|| panic!("no committed public render case {name}"));
    map_render_where(&root, &store, case, false).unwrap()
}

/// `case` mapped, refused where `routed` and the executor's route keeps it for a later runtime
/// step.
fn map_render_where(
    root: &Path,
    store: &CorpusStore,
    case: AuthoredCase,
    routed: bool,
) -> Result<PublicRender, String> {
    let name = case.name.clone();
    let metal_output = metal_output(store, &case);
    let checked = crate::check::check_case(root, case)
        .map_err(|errors| format!("{name}: the checker: {}", errors.join("; ")))?;
    let primary = &checked.primary;
    let render = render_case(
        &primary.case,
        &primary.interface,
        primary.entry_ll(),
        primary.companion.as_ref(),
        primary.reflection.fragment_imageblock.as_ref(),
    )
    .map_err(|error| format!("{name}: render_case: {error}"))?;
    let later = render_prerequisites(&primary.case, &render);
    if routed && !later.is_empty() {
        let names: Vec<_> = later
            .iter()
            .map(|prerequisite| prerequisite.name())
            .collect();
        return Err(format!(
            "{name}: {}, a later runtime step",
            names.join(", ")
        ));
    }
    let scratch = crate::ScratchDir::new("runtime-lane-render")?;
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &primary.source.air_ll).map_err(|error| format!("{name}: {error}"))?;
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll)
        .map_err(|error| format!("{name}: llvm-as: {error:?}"))?;
    let companion = render
        .companion
        .as_ref()
        .map(crate::companion::Companion::opaque_bitcode)
        .transpose()
        .map_err(|error| format!("{name}: the companion: {error}"))?;
    let texture_usage = texture_usage(&primary.case, &primary.interface)
        .map_err(|error| format!("{name}: {error}"))?;
    let linked = linked_modules(&primary.linked_functions)
        .map_err(|error| format!("{name}: the linked functions: {error}"))?;
    Ok(PublicRender {
        case: primary.case.clone(),
        render,
        bitcode,
        companion,
        linked,
        texture_usage,
        interface: primary.interface.clone(),
        imageblock: primary.reflection.fragment_imageblock.clone(),
        metal_output,
    })
}

/// Whether `why` is an honest refusal of a run: a gap the runtime names (`NotImplemented`), or
/// what the device lacks (`Unsupported`, the runtime's or the binder's: a bind state the device
/// cannot run, such as a shader-side stencil's write that the depth test decides).
pub(super) fn refusal(why: &str) -> bool {
    why.contains("not implemented in the runtime yet")
        || why.contains("unsupported:")
        || why.contains("the bind state was refused: Unsupported(")
}

/// The render pipeline descriptor of `case`, its functions made on `device`, as the case runner
/// makes it (`runtime_case::render_pipeline_descriptor`).
pub(super) fn descriptor_of(
    device: &Device,
    case: &PublicRender,
) -> Result<RenderPipelineDescriptor, String> {
    render_pipeline_descriptor(device, &case.inputs())
        .map_err(|error| format!("{}: {error}", case.case.name))
}

/// Run `case` once on `device` by the case runner the executor's runtime path shares
/// (`runtime_case::commit_render`): one command buffer with one render encoder, waited for.
pub(super) fn run_render(device: &Device, case: &PublicRender) -> Result<RuntimeRun, String> {
    let name = &case.case.name;
    let inputs = case.inputs();
    let committed = commit_render(device, &inputs).map_err(|error| format!("{name}: {error}"))?;
    let command_buffer = &committed.command_buffer;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("{name}: the command buffer did not complete"))?;
    if command_buffer.status() != CommandBufferStatus::Completed {
        return Err(format!(
            "{name}: the command buffer failed: {:?}",
            command_buffer.error()
        ));
    }
    let report = command_buffer
        .execution_report()
        .ok_or("a completed command buffer has an execution report")?;
    let output = committed
        .output(&inputs)
        .map_err(|error| format!("{name}: {error}"))?;
    Ok(RuntimeRun {
        output,
        report,
        executions: committed.executions,
    })
}

/// The engine candidate's fresh run of `case` on the lane's device: the device it opened, its
/// output and its execution report, or why it gave none.
fn fresh_candidate(case: &PublicRender) -> Result<(String, Vec<u8>, StatusReport), String> {
    let run = engine_render::run_mapped(&case.render, &case.bitcode)
        .map_err(|error| error.to_string())?;
    let device = run.environment["device"]
        .as_str()
        .unwrap_or("an unnamed device")
        .to_owned();
    Ok((device, run.output, run.status))
}

/// Every committed public render case this part of the runtime binds draws through it with a
/// clean execution report and Metal's output, or the output of the candidate run afresh on the
/// case (the module's note), or is refused honestly (`refusal`); the validation layer reports
/// nothing, leaks included. A case of out-of-bounds targets, or of pipeline states or command
/// buffers, is left to the runtime path's tests (the module's note), and listed.
#[test]
fn public_render_cases_match_metal() {
    let cases = render_cases();
    let messages = Messages::default();
    let (mut matched, mut differing, mut refused, mut later, mut failed) =
        (0usize, Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut contained, mut templated) = (Vec::new(), Vec::new());
    let device_name;
    {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        device_name = device.name().to_owned();
        for (name, case) in &cases {
            let case = match case {
                Ok(case) => case,
                Err(why) if why.contains("later runtime step") => {
                    later.push(why.clone());
                    continue;
                }
                Err(why) => {
                    failed.push(why.clone());
                    continue;
                }
            };
            if case.case.out_of_bounds.is_some() {
                contained.push(name);
                continue;
            }
            if !case.case.pipeline_states.is_empty() || !case.case.command_buffers.is_empty() {
                templated.push(name);
                continue;
            }
            match run_render(&device, case) {
                Ok(run) if run.report != StatusReport::default() => {
                    failed.push(format!("{name}: the execution report is {:?}", run.report))
                }
                Ok(run) if Ok(&run.output) == case.metal_output.as_ref() => matched += 1,
                Ok(_) => differing.push((name, case)),
                Err(why) if why.contains("later runtime step") => later.push(why),
                Err(why) if refusal(&why) => refused.push(why),
                Err(why) => failed.push(why),
            }
            let reported = messages.take();
            if !reported.is_empty() {
                failed.push(format!(
                    "{name}: the validation layer reported {reported:#?}"
                ));
            }
        }
    }
    let leaked = messages.take();
    let mut as_candidate = Vec::new();
    for (name, case) in differing {
        let (candidate_device, candidate, candidate_report) = match fresh_candidate(case) {
            Ok(run) => run,
            Err(why) => {
                failed.push(format!("{name}: the fresh candidate gave no output: {why}"));
                continue;
            }
        };
        let rerun = {
            let cache = ScratchCache::new().unwrap();
            let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
            run_render(&device, case)
        };
        let reported = messages.take();
        if !reported.is_empty() {
            failed.push(format!(
                "{name}: the validation layer reported {reported:#?}"
            ));
        }
        match rerun {
            _ if candidate_device != device_name => failed.push(format!(
                "{name}: the candidate ran on {candidate_device}, the runtime on {device_name}"
            )),
            Ok(run) if run.report != StatusReport::default() => {
                failed.push(format!("{name}: the execution report is {:?}", run.report))
            }
            Ok(run) if run.output == candidate => as_candidate.push(match &case.metal_output {
                Ok(_) => format!("{name}: the candidate's execution report {candidate_report:?}"),
                Err(why) => format!(
                    "{name}: the candidate's execution report {candidate_report:?}; no Metal \
                     answer to compare: {why}"
                ),
            }),
            Ok(run) => failed.push(format!(
                "{name}: differs from Metal and from the fresh candidate \
                 ({ENGINE_EXECUTOR_ABI}):\n runtime   {:02x?}\n candidate {:02x?}\n \
                 metal     {:02x?}",
                run.output, candidate, case.metal_output
            )),
            Err(why) => failed.push(format!("{name}: the run after the candidate's: {why}")),
        }
    }
    eprintln!(
        "runtime render cases on {device_name}: {matched} match Metal, {} match the fresh \
         candidate ({ENGINE_EXECUTOR_ABI}), {} refused, {} of later steps, {} of \
         out-of-bounds targets and {} of pipeline states or command buffers left to the runtime \
         path's tests, {} failed, of {}",
        as_candidate.len(),
        refused.len(),
        later.len(),
        contained.len(),
        templated.len(),
        failed.len(),
        cases.len()
    );
    for name in &contained {
        eprintln!("judged by containment: {name}");
    }
    for name in &templated {
        eprintln!("of pipeline states or command buffers: {name}");
    }
    for line in &as_candidate {
        eprintln!("as the candidate: {line}");
    }
    for why in &refused {
        eprintln!("refused: {why}");
    }
    for why in &later {
        eprintln!("later: {why}");
    }
    assert!(
        failed.is_empty() && leaked.is_empty(),
        "failed:\n{}\nat device destruction: {leaked:#?}",
        failed.join("\n")
    );
    assert!(matched > 0, "no case ran");
}

/// Destroy `device`, then check the validation layer reported nothing, leaks included.
fn quiet(device: Device, messages: &Messages) {
    drop(device);
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// Commit `command_buffer` and wait for it to complete without an error.
pub(super) fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// A `width × height` texture of `pf` with `RenderTarget` usage (and `ShaderRead`), of
/// `options`.
fn attachment(
    device: &Device,
    pf: PixelFormat,
    [width, height]: [u64; 2],
    options: ResourceOptions,
) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, width, height, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    device.new_texture(&descriptor)
}

/// The bytes of `texture`'s first level and slice, `pixel` bytes a texel, copied by a blit into a
/// `Shared` buffer.
pub(super) fn blit_read(device: &Device, texture: &Texture, pixel: u64) -> Vec<u8> {
    let (width, height) = (texture.width(), texture.height());
    let buffer = device
        .new_buffer(width * height * pixel, ResourceOptions::shared())
        .unwrap();
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(width, height, 1),
        &buffer,
        0,
        width * pixel,
        width * height * pixel,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    read_buffer(&buffer, 0, buffer.length()).unwrap()
}

/// One color clear Metal was measured with: the format, the clear color, and the texel's bytes.
type ColorClear = (PixelFormat, [f64; 4], &'static [u8]);

/// Color clears measured on an M4 (our Metal probes; the NaNs on an M4 and the M3 Max alike), each
/// format's bytes as Metal stored them.
const COLOR_CLEARS: &[ColorClear] = &[
    (
        PixelFormat::RGBA8Unorm,
        [0.5, 0.25, 1.5, -0.5],
        &[0x80, 0x40, 0xff, 0x00],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [
            0.00196078431372549,
            0.0058823529411764705,
            0.00980392156862745,
            0.501960784,
        ],
        &[0x01, 0x02, 0x03, 0x80],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [0.0019, 0.0021, 0.998, 0.996],
        &[0x00, 0x01, 0xfe, 0xfe],
    ),
    (
        PixelFormat::RGBA8Unorm,
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.75],
        &[0x00, 0xff, 0x00, 0xbf],
    ),
    (
        PixelFormat::BGRA8Unorm,
        [0.25, 0.5, 0.75, 1.0],
        &[0xbf, 0x80, 0x40, 0xff],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.5, 0.214, 0.0031308, 1.0],
        &[0xbc, 0x7f, 0x0a, 0xff],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.001, 0.01, 0.1, 0.5],
        &[0x03, 0x19, 0x59, 0x80],
    ),
    (
        PixelFormat::RGBA8UnormSRGB,
        [0.2158605, 0.2122308, 0.214, 0.25],
        &[0x80, 0x7f, 0x7f, 0x40],
    ),
    (
        PixelFormat::BGRA8UnormSRGB,
        [0.5, 0.214, 0.0031308, 0.5],
        &[0x0a, 0x7f, 0xbc, 0x80],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [-1.5, -0.5, 0.5, 0.011811023622047244],
        &[0x81, 0xc0, 0x40, 0x01],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [-1.0, -0.0039, 0.0039, 0.502],
        &[0x81, 0x00, 0x00, 0x40],
    ),
    (
        PixelFormat::RGBA8Snorm,
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0],
        &[0x00, 0x7f, 0x81, 0x00],
    ),
    (PixelFormat::R8Uint, [3.7, 0.0, 0.0, 0.0], &[0x03]),
    (PixelFormat::R8Uint, [300.0, 0.0, 0.0, 0.0], &[0xff]),
    (PixelFormat::R8Uint, [-1.0, 0.0, 0.0, 0.0], &[0x00]),
    (PixelFormat::R8Uint, [f64::NAN, 0.0, 0.0, 0.0], &[0x00]),
    (PixelFormat::R8Sint, [-3.7, 0.0, 0.0, 0.0], &[0xfd]),
    (PixelFormat::R8Sint, [200.0, 0.0, 0.0, 0.0], &[0x7f]),
    (PixelFormat::R8Sint, [-200.0, 0.0, 0.0, 0.0], &[0x80]),
    (
        PixelFormat::R16Uint,
        [65535.7, 0.0, 0.0, 0.0],
        &[0xff, 0xff],
    ),
    (PixelFormat::R16Sint, [-2.5, 0.0, 0.0, 0.0], &[0xfe, 0xff]),
    (
        PixelFormat::R32Uint,
        [4294967296.0, 0.0, 0.0, 0.0],
        &[0xff, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::R32Uint,
        [16777217.0, 0.0, 0.0, 0.0],
        &[0x01, 0x00, 0x00, 0x01],
    ),
    (
        PixelFormat::R32Uint,
        [4294967040.5, 0.0, 0.0, 0.0],
        &[0x00, 0xff, 0xff, 0xff],
    ),
    (
        PixelFormat::R32Sint,
        [-16777217.0, 0.0, 0.0, 0.0],
        &[0xff, 0xff, 0xff, 0xfe],
    ),
    (
        PixelFormat::R32Sint,
        [-3000000000.0, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0x00, 0x80],
    ),
    (PixelFormat::R16Float, [0.1, 0.0, 0.0, 0.0], &[0x66, 0x2e]),
    (
        PixelFormat::R16Float,
        [70000.0, 0.0, 0.0, 0.0],
        &[0x00, 0x7c],
    ),
    (PixelFormat::R16Float, [-0.0, 0.0, 0.0, 0.0], &[0x00, 0x80]),
    (
        PixelFormat::R16Float,
        [1.001220703125, 0.0, 0.0, 0.0],
        &[0x01, 0x3c],
    ),
    (
        PixelFormat::R16Float,
        [1.0004882812509095, 0.0, 0.0, 0.0],
        &[0x00, 0x3c],
    ),
    (
        PixelFormat::R16Float,
        [1.00146484375, 0.0, 0.0, 0.0],
        &[0x02, 0x3c],
    ),
    (
        PixelFormat::R32Float,
        [0.1, 0.0, 0.0, 0.0],
        &[0xcd, 0xcc, 0xcc, 0x3d],
    ),
    (
        PixelFormat::R32Float,
        [1e-40, 0.0, 0.0, 0.0],
        &[0xc2, 0x16, 0x01, 0x00],
    ),
    (
        PixelFormat::R32Float,
        [1e39, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0x80, 0x7f],
    ),
    (
        PixelFormat::R32Float,
        [f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0xc0, 0x7f],
    ),
    (
        PixelFormat::R32Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0xc0, 0xff],
    ),
    (
        PixelFormat::R16Float,
        [f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x7e],
    ),
    (
        PixelFormat::R16Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x7e],
    ),
    (
        PixelFormat::RG16Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c],
    ),
    (
        PixelFormat::RG16Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x00, 0x3c, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x00, 0x3c, 0x00, 0x7e, 0x00, 0x40, 0x00, 0x34],
    ),
    (
        PixelFormat::RGBA16Float,
        [0.1, -2.5, 65504.0, 1e-05],
        &[0x66, 0x2e, 0x00, 0xc1, 0xff, 0x7b, 0xa8, 0x00],
    ),
    (
        PixelFormat::RG16Snorm,
        [-1.5, 0.5, 0.0, 0.0],
        &[0x01, 0x80, 0x00, 0x40],
    ),
    (
        PixelFormat::RG16Snorm,
        [-1e-05, 0.99999, 0.0, 0.0],
        &[0x00, 0x00, 0xff, 0x7f],
    ),
    (
        PixelFormat::RGBA16Unorm,
        [0.5, 7.629510948348211e-06, 2.2888532845044633e-05, 1.2],
        &[0x00, 0x80, 0x00, 0x00, 0x01, 0x00, 0xff, 0xff],
    ),
    (
        PixelFormat::RGB10A2Unorm,
        [0.5, 0.25, 1.0, 0.5],
        &[0x00, 0x02, 0xf4, 0xbf],
    ),
    (
        PixelFormat::RGB10A2Unorm,
        [0.0005, 0.9995, 0.33, 0.17],
        &[0x01, 0xf8, 0x2f, 0x55],
    ),
    (
        PixelFormat::BGR10A2Unorm,
        [0.5, 0.25, 0.75, 0.4],
        &[0xff, 0x02, 0x04, 0x60],
    ),
    (
        PixelFormat::RGBA32Float,
        [0.1, -1e39, 1e-40, 3.0],
        &[
            0xcd, 0xcc, 0xcc, 0x3d, 0x00, 0x00, 0x80, 0xff, 0xc2, 0x16, 0x01, 0x00, 0x00, 0x00,
            0x40, 0x40,
        ],
    ),
    (
        PixelFormat::RGBA32Uint,
        [1.5, 4294967296.0, -1.0, 7.0],
        &[1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 7, 0, 0, 0],
    ),
    (
        PixelFormat::RG11B10Float,
        [0.1, -1.0, 70000.0, 0.0],
        &[0xe6, 0x02, 0xc0, 0xf7],
    ),
    (
        PixelFormat::RG11B10Float,
        [65000.0, 0.5, 1e-05, 0.0],
        &[0xbe, 0x07, 0x5c, 0x01],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        &[0xe0, 0x07, 0x3f, 0xfc],
    ),
    (
        PixelFormat::RG11B10Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0xe0, 0x07, 0x3f, 0xfc],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0xe0, 0x07, 0x1e, 0x70],
    ),
    (
        PixelFormat::RG11B10Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0xc0, 0x03, 0x3f, 0x80],
    ),
    (
        PixelFormat::RG11B10Float,
        [1.0078125, 1.015625, 1.0078125, 0.0],
        &[0xc0, 0x0b, 0x1e, 0x78],
    ),
    (
        PixelFormat::RGB9E5Float,
        [0.1, 0.5, 2.0, 0.0],
        &[0x0c, 0x80, 0x00, 0x8c],
    ),
    (
        PixelFormat::RGB9E5Float,
        [-1.0, 70000.0, 0.001, 0.0],
        &[0x00, 0xfe, 0x03, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [65408.0, 1e-08, 3.0, 0.0],
        &[0xff, 0x01, 0x00, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [f64::NAN, f64::INFINITY, 1.0, 0.0],
        &[0x00, 0xfe, 0x03, 0xf8],
    ),
    (
        PixelFormat::RGB9E5Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0x00, 0x00, 0x00, 0x00],
    ),
    (
        PixelFormat::RGB9E5Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x00, 0x02, 0x82],
    ),
    (
        PixelFormat::RGB9E5Float,
        [1.0, f64::NAN, 2.0, 0.25],
        &[0x80, 0x00, 0x00, 0x8c],
    ),
    (
        PixelFormat::B5G6R5Unorm,
        [0.5, 0.25, 0.75, 1.0],
        &[0x17, 0x82],
    ),
    (
        PixelFormat::A1BGR5Unorm,
        [0.5, 0.25, 0.75, 0.5],
        &[0x2e, 0x82],
    ),
    (
        PixelFormat::ABGR4Unorm,
        [0.5, 0.25, 0.75, 0.1],
        &[0xb1, 0x84],
    ),
    (
        PixelFormat::BGR5A1Unorm,
        [0.5, 0.25, 0.75, 0.6],
        &[0x17, 0xc1],
    ),
];

/// Clear `texture` (attached as the pass's only color attachment) to `color`, and store.
fn clear_color(device: &Device, texture: &Texture, color: [f64; 4]) -> Result<(), Error> {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    let [red, green, blue, alpha] = color;
    attachment.clear_color = ClearColor::new(red, green, blue, alpha);
    let command_buffer = device.new_command_queue()?.command_buffer()?;
    command_buffer
        .render_command_encoder(&pass)?
        .end_encoding()?;
    run(&command_buffer);
    Ok(())
}

/// Whether `channel`'s bits, `exponent` exponent bits above `mantissa` mantissa bits, are a NaN.
pub(super) fn small_nan(channel: u32, exponent: u32, mantissa: u32) -> bool {
    let all = (1 << exponent) - 1;
    (channel >> mantissa) & all == all && channel & ((1 << mantissa) - 1) != 0
}

/// Each channel of a `pf` texel (`RGBA16Float`'s four halves, `RG11B10Float`'s three fields) as
/// its bits, its exponent's width and its mantissa's.
fn channels(pf: PixelFormat, texel: &[u8]) -> Vec<(u32, u32, u32)> {
    match pf {
        PixelFormat::RGBA16Float => texel
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| (u32::from(u16::from_le_bytes([h[0], h[1]])) & 0x7fff, 5, 10))
            .collect(),
        PixelFormat::RG11B10Float => {
            let word = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
            vec![
                (word & 0x7ff, 5, 6),
                ((word >> 11) & 0x7ff, 5, 6),
                (word >> 22, 5, 5),
            ]
        }
        _ => Vec::new(),
    }
}

/// A NaN clear: its format and color, the texel where Metal drops a `half` NaN's sign, and the
/// one where it keeps it ([`NAN_CLEARS`]).
type NanClear = (PixelFormat, [f64; 4], &'static [u8], &'static [u8]);

/// Metal's NaN clears by format and clear color (an M4 under macOS 27.0.1; a Metal probe; the
/// consumer's M3 Max alike at 4x4): the texel a level stores in its texels or every sample where
/// Metal drops a `half` NaN's sign and payload (an uncompressed texture, or a level under 16 texels
/// a side), then where it keeps them.
const NAN_CLEARS: &[NanClear] = &[
    (
        PixelFormat::RGBA16Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x7e],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x7e],
    ),
    (
        PixelFormat::RGBA16Float,
        [-f64::NAN, -f64::NAN, -f64::NAN, -f64::NAN],
        &[0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e, 0x00, 0x7e],
        &[0x00, 0xfe, 0x00, 0xfe, 0x00, 0xfe, 0x00, 0xfe],
    ),
    (
        PixelFormat::RGBA16Float,
        [f64::NAN, 1.0, 0.5, -f64::NAN],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0x7e],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x38, 0x00, 0xfe],
    ),
    (
        PixelFormat::R16Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x7e],
        &[0x00, 0xfe],
    ),
    (
        PixelFormat::RGBA16Float,
        [
            f64::from_bits(0x7ffc_0000_0000_0000),
            1.0,
            f64::from_bits(0x7ff4_0000_0000_0000),
            f64::from_bits(0xffff_ffff_ffff_ffff),
        ],
        &[0x00, 0x7e, 0x00, 0x3c, 0x00, 0x7e, 0x00, 0x7e],
        &[0x00, 0x7f, 0x00, 0x3c, 0x00, 0x7f, 0xff, 0xff],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, 1.0, 0.5, f64::NAN],
        &[0xe0, 0x07, 0x1e, 0x70],
        &[0xe0, 0x07, 0x1e, 0x70],
    ),
    (
        PixelFormat::RG11B10Float,
        [f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        &[0xe0, 0x07, 0x3f, 0xfc],
        &[0xe0, 0x07, 0x3f, 0xfc],
    ),
    (
        PixelFormat::R32Float,
        [f64::from_bits(0x7ff0_0000_2000_0000), 0.0, 0.0, 0.0],
        &[0x01, 0x00, 0xc0, 0x7f],
        &[0x01, 0x00, 0xc0, 0x7f],
    ),
    (
        PixelFormat::R32Float,
        [-f64::NAN, 0.0, 0.0, 0.0],
        &[0x00, 0x00, 0xc0, 0xff],
        &[0x00, 0x00, 0xc0, 0xff],
    ),
];

/// `texel` with each NaN channel as Metal's 4x resolve makes it where it drops the clear's NaN
/// (a level whose clear drops a `half` NaN's sign): zero in a small-float channel, and
/// `0x7fc00000` in a 32-bit float one.
fn nans_resolved(pf: PixelFormat, texel: &[u8]) -> Vec<u8> {
    match pf {
        PixelFormat::R32Float => texel
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(
                |f| match f32::from_le_bytes([f[0], f[1], f[2], f[3]]).is_nan() {
                    true => 0x7fc0_0000u32.to_le_bytes(),
                    false => [f[0], f[1], f[2], f[3]],
                },
            )
            .collect(),
        PixelFormat::RGBA16Float | PixelFormat::R16Float => texel
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|h| {
                match small_nan(u32::from(u16::from_le_bytes([h[0], h[1]])) & 0x7fff, 5, 10) {
                    true => [0, 0],
                    false => [h[0], h[1]],
                }
            })
            .collect(),
        PixelFormat::RG11B10Float => {
            let word = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
            let fields = [(0, 11, 6), (11, 11, 6), (22, 10, 5)];
            let kept = fields.iter().fold(0u32, |out, &(at, width, mantissa)| {
                let field = (word >> at) & ((1 << width) - 1);
                match small_nan(field, 5, mantissa) {
                    true => out,
                    false => out | (field << at),
                }
            });
            kept.to_le_bytes().to_vec()
        }
        _ => texel.to_vec(),
    }
}

/// A NaN `Clear` of a new `Private` attachment (`nan_clear.rs`) of 4x4, 16x16 and 64x64, 1x and
/// 4x, plain (a usage Metal compresses) and with `PixelFormatView` usage (kept uncompressed),
/// in a pass without a draw call and in one whose draw (`rt6_restart`, `clear_reach::Lane`)
/// lies wholly off the target, stores Metal's texel ([`NAN_CLEARS`]): the one with a `half`
/// NaN's sign where Metal keeps it, at a plain level of 16 or more in a pass without a draw
/// call, and without it elsewhere. A 1x attachment is read whole by a blit. Each sample of a 4x one
/// is read through an unsigned view of its texel size where its format has one (`RGBA16Float`,
/// `R16Float` and `R32Float`, `runtime_samples::encode`): its own with `PixelFormatView` usage,
/// else a same-format copy's with it (Metal's and the devices' copies keep every sample's bits, but
/// for a device whose multisampled copies convert float NaNs, `MultisampledCopiesConvertFloatNans`,
/// on which a copy's samples are left unread). Every 4x one is resolved into a new plain 1x texture
/// read by a blit. In a pass without a draw call Metal's resolve keeps the texel where it keeps the
/// sign and stores zero in each small-float NaN channel and `0x7fc00000` in each 32-bit float one
/// elsewhere, which the runtime writes on every device (an `R32Float` clear is Metal's conversion
/// of the double at every level). In a pass with a draw Metal's resolve averages the samples, each
/// NaN zero: a device's resolve whose NaN channels differ only so is its resolve rule, printed
/// (`device-rounded-resolve`), not this clear's. A `BGR10_XR` NaN clear is printed beside them, or
/// the device's refusal of the format. Each prints `MSAA-NAN-CLEAR`.
#[test]
fn multisampled_nan_clears_store_metal_s_texel() {
    let (messages, cache, lane) = super::clear_reach::lane();
    let device = &lane.device;
    let name = device.name().to_owned();
    let xr: &[NanClear] = &[(PixelFormat::BGR10XR, [f64::NAN, 0.5, 1.0, 1.0], &[], &[])];
    // A triangle wholly off the target, the vertex function's positions.
    let off: Vec<u8> = [2.0f32, 2.0, 3.0, 2.0, 2.0, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let off = buffer_with(device, &off).unwrap();
    let (mut wrong, mut checked) = (Vec::new(), 0usize);
    // A device whose multisampled copies convert float NaNs (RADV) cannot copy the samples of an
    // attachment without `PixelFormatView` usage out to be read.
    let copies_convert = device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::MultisampledCopiesConvertFloatNans);
    for &(pf, color, small, large) in NAN_CLEARS.iter().chain(xr) {
        let bytes = match pf {
            PixelFormat::RGBA16Float => 8,
            PixelFormat::R16Float => 2,
            _ => 4,
        };
        let view = match pf {
            PixelFormat::RGBA16Float => Some(ReadbackView::Rg32Uint),
            PixelFormat::R16Float => Some(ReadbackView::R16Uint),
            PixelFormat::R32Float => Some(ReadbackView::R32Uint),
            _ => None,
        };
        let measured = !small.is_empty();
        for (side, samples, viewed, drawn) in [4u32, 16, 64]
            .into_iter()
            .flat_map(|side| [(side, 1u64), (side, 4)])
            .flat_map(|(side, samples)| [(side, samples, false), (side, samples, true)])
            .flat_map(|(side, samples, viewed)| {
                [
                    (side, samples, viewed, false),
                    (side, samples, viewed, true),
                ]
            })
            .filter(|&(_, _, _, drawn)| measured || !drawn)
        {
            let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let new = |samples: u64, viewed: bool| {
                let mut descriptor =
                    TextureDescriptor::texture_2d(pf, side.into(), side.into(), false);
                if samples > 1 {
                    descriptor.texture_type = TextureType::Type2DMultisample;
                    descriptor.sample_count = samples;
                }
                descriptor.usage = match viewed {
                    true => usage | TextureUsage::PIXEL_FORMAT_VIEW,
                    false => usage,
                };
                descriptor.resource_options = ResourceOptions::private();
                device.new_texture(&descriptor)
            };
            let label = format!(
                "{pf:?} {color:?} {samples}x {side}x{side}{}{}",
                if viewed { " PixelFormatView" } else { "" },
                if drawn { " drawn off the target" } else { "" }
            );
            let texture = match new(samples, viewed) {
                Ok(texture) => texture,
                Err(Error::Unsupported(why)) => {
                    println!("MSAA-NAN-CLEAR {label} on {name}: unsupported {why:?}");
                    continue;
                }
                Err(error) => panic!("{label}: {error}"),
            };
            let resolved = (samples > 1).then(|| {
                attachment(
                    device,
                    pf,
                    [side.into(), side.into()],
                    ResourceOptions::private(),
                )
                .unwrap()
            });
            let mut pass = RenderPassDescriptor::new();
            let target = &mut pass.color_attachments[0];
            target.texture = Some(texture.clone());
            target.load_action = LoadAction::Clear;
            target.store_action = match &resolved {
                Some(resolved) => {
                    target.resolve_texture = Some(resolved.clone());
                    StoreAction::StoreAndMultisampleResolve
                }
                None => StoreAction::Store,
            };
            let [red, green, blue, alpha] = color;
            target.clear_color = ClearColor::new(red, green, blue, alpha);
            let pipeline = drawn.then(|| lane.pipeline(&[pf], (None, None), samples));
            let command_buffer = lane.queue.command_buffer().unwrap();
            let encoded = command_buffer
                .render_command_encoder(&pass)
                .and_then(|mut encoder| {
                    if let Some(pipeline) = &pipeline {
                        encoder.set_render_pipeline_state(pipeline)?;
                        encoder.set_vertex_buffer(Some(&off), 0, 0)?;
                        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
                    }
                    encoder.end_encoding()
                });
            match encoded {
                Ok(()) => {}
                // A device that cannot render the format refuses the pass (`BGR10_XR` on x86
                // lavapipe and on MoltenVK).
                Err(Error::Unsupported(why)) => {
                    println!("MSAA-NAN-CLEAR {label} on {name}: unsupported {why:?}");
                    continue;
                }
                Err(error) => {
                    println!("MSAA-NAN-CLEAR {label} on {name}: refused {error:?}");
                    wrong.push(format!("{label}: refused {error:?}"));
                    continue;
                }
            }
            // The samples, through the unsigned view of a texture with `PixelFormatView` usage:
            // the attachment's own, or a copy's.
            let read = match (samples > 1, view) {
                (true, Some(_)) if !viewed && copies_convert => None,
                (true, Some(view)) => {
                    let readable = match viewed {
                        true => texture.clone(),
                        false => {
                            let copy = new(samples, true).unwrap();
                            let mut blit = command_buffer.blit_command_encoder().unwrap();
                            blit.copy_texture(&texture, &copy).unwrap();
                            blit.end_encoding().unwrap();
                            copy
                        }
                    };
                    Some(
                        samples_of(
                            device,
                            &command_buffer,
                            &readable,
                            SampleRead::Color(view),
                            [side, side],
                            samples as u32,
                        )
                        .unwrap(),
                    )
                }
                _ => None,
            };
            run(&command_buffer);
            let distinct = |all: &[u8]| {
                let mut seen: Vec<Vec<u8>> = Vec::new();
                for texel in all.chunks_exact(bytes) {
                    if !seen.iter().any(|s| s == texel) {
                        seen.push(texel.to_vec());
                    }
                }
                seen
            };
            let stored = distinct(&blit_read(
                device,
                resolved.as_ref().unwrap_or(&texture),
                bytes as u64,
            ));
            let sampled = read.map(|read| distinct(&read.bytes().unwrap()));
            let sampled_line = match &sampled {
                Some(sampled) => format!(" samples {sampled:02x?}"),
                None if samples > 1 && view.is_some() => {
                    " samples unread (the device's copy converts NaNs)".to_owned()
                }
                None => String::new(),
            };
            let what = if samples > 1 { "resolved" } else { "stored" };
            if !measured {
                println!(
                    "MSAA-NAN-CLEAR {label} on {name}:{sampled_line} {what} {stored:02x?} \
                     (printed: no measured Metal texel)"
                );
                continue;
            }
            checked += 1;
            let signed = side >= 16 && !viewed && !drawn;
            let texel = if signed { large } else { small };
            let samples_exact = sampled.iter().flatten().all(|sample| sample == texel);
            let verdict = if samples == 1 {
                match stored.iter().all(|got| got == texel) {
                    true => Ok("Metal's texel".to_owned()),
                    false => Err(()),
                }
            } else {
                let metal = match signed {
                    true => texel.to_vec(),
                    false => nans_resolved(pf, texel),
                };
                // A resolve keeping or dropping a NaN, of any bits, beside Metal's other channels.
                let resolve_rule = |got: &[u8]| {
                    nans_resolved(pf, got) == nans_resolved(pf, texel)
                        && channels(pf, got).iter().zip(channels(pf, texel)).all(
                            |(&(g, e, m), (t, _, _))| match small_nan(t, e, m) {
                                true => g == 0 || small_nan(g, e, m),
                                false => g == t,
                            },
                        )
                };
                match (samples_exact, stored.iter().all(|got| *got == metal)) {
                    (true, true) => Ok("Metal's texel, Metal's resolve".to_owned()),
                    (true, false) if drawn && stored.iter().all(|got| resolve_rule(got)) => {
                        Ok(format!(
                            "Metal's texel; the device's resolve rule, Metal's resolve \
                             {metal:02x?}"
                        ))
                    }
                    _ => Err(()),
                }
            };
            match verdict {
                Ok(verdict) => println!(
                    "MSAA-NAN-CLEAR {label} on {name}:{sampled_line} {what} {stored:02x?}: {verdict}"
                ),
                Err(()) => {
                    println!(
                        "MSAA-NAN-CLEAR {label} on {name}:{sampled_line} {what} {stored:02x?}: \
                         WRONG, Metal {texel:02x?}"
                    );
                    wrong.push(format!(
                        "{label}:{sampled_line} {what} {stored:02x?}, Metal {texel:02x?}"
                    ));
                }
            }
        }
    }
    println!("MSAA-NAN-CLEAR {checked} cases checked on {name}");
    drop(off);
    drop(lane);
    drop(cache);
    let _ = messages.take_notices();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Each color clear Metal was measured with stores Metal's bytes, depth and stencil clears
/// included; a format the device cannot render to is counted, not run.
#[test]
fn clears_store_the_bytes_metal_stores() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let (mut checked, mut unsupported, mut wrong) = (0usize, Vec::new(), Vec::new());
    for &(pf, color, expected) in COLOR_CLEARS {
        let texture = match attachment(&device, pf, [1, 1], ResourceOptions::private()) {
            Ok(texture) => texture,
            Err(Error::Unsupported(why)) => {
                unsupported.push(format!("{pf:?}: {why:?}"));
                continue;
            }
            Err(error) => panic!("{pf:?}: {error}"),
        };
        // A texture the device makes but cannot render to is refused at the pass.
        match clear_color(&device, &texture, color) {
            Ok(()) => {}
            Err(Error::Unsupported(why)) => {
                unsupported.push(format!("{pf:?}: {why:?}"));
                continue;
            }
            Err(error) => panic!("{pf:?}: {error}"),
        }
        let bytes = blit_read(&device, &texture, expected.len() as u64);
        if bytes != expected {
            wrong.push(format!(
                "{pf:?} {color:?}: {bytes:02x?}, Metal {expected:02x?}"
            ));
        }
        checked += 1;
    }
    // Depth: through `float`, `Depth16Unorm` truncating, and outside [0, 1] the nearer end (Metal
    // probe case `depth_clear_range_load` on an M4, which Metal's validation stops); stencil: the
    // low 8 bits.
    let depth_clears: &[(PixelFormat, f64, &[u8])] = &[
        (PixelFormat::Depth32Float, 2.0, &[0x00, 0x00, 0x80, 0x3f]),
        (PixelFormat::Depth32Float, -1.0, &[0x00, 0x00, 0x00, 0x00]),
        (PixelFormat::Depth32Float, 0.3, &[0x9a, 0x99, 0x99, 0x3e]),
        (PixelFormat::Depth32Float, 0.1, &[0xcd, 0xcc, 0xcc, 0x3d]),
        (PixelFormat::Depth32Float, 1.0, &[0x00, 0x00, 0x80, 0x3f]),
        (PixelFormat::Depth16Unorm, 0.5, &[0xff, 0x7f]),
        (PixelFormat::Depth16Unorm, 0.3, &[0xcc, 0x4c]),
        (PixelFormat::Depth16Unorm, 0.1, &[0x99, 0x19]),
        (
            PixelFormat::Depth16Unorm,
            7.629510948348211e-06,
            &[0x00, 0x00],
        ),
        (
            PixelFormat::Depth16Unorm,
            2.2888532845044633e-05,
            &[0x01, 0x00],
        ),
        (PixelFormat::Depth16Unorm, 1.0, &[0xff, 0xff]),
    ];
    for &(pf, depth, expected) in depth_clears {
        let texture = attachment(&device, pf, [1, 1], ResourceOptions::private()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.depth_attachment.texture = Some(texture.clone());
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::Store;
        pass.depth_attachment.clear_depth = depth;
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        run(&command_buffer);
        let bytes = blit_read(&device, &texture, expected.len() as u64);
        if bytes != expected {
            wrong.push(format!(
                "{pf:?} {depth}: {bytes:02x?}, Metal {expected:02x?}"
            ));
        }
        checked += 1;
    }
    for (stencil, expected) in [(90u32, 0x5au8), (511, 0xff), (256, 0x00), (u32::MAX, 0xff)] {
        let texture = attachment(
            &device,
            PixelFormat::Stencil8,
            [1, 1],
            ResourceOptions::private(),
        )
        .unwrap();
        let mut pass = RenderPassDescriptor::new();
        pass.stencil_attachment.texture = Some(texture.clone());
        pass.stencil_attachment.load_action = LoadAction::Clear;
        pass.stencil_attachment.store_action = StoreAction::Store;
        pass.stencil_attachment.clear_stencil = stencil;
        let command_buffer = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        run(&command_buffer);
        let bytes = blit_read(&device, &texture, 1);
        if bytes != [expected] {
            wrong.push(format!(
                "Stencil8 {stencil}: {bytes:02x?}, Metal {expected:02x}"
            ));
        }
        checked += 1;
    }
    eprintln!(
        "clears on {}: {checked} checked, {} formats the device cannot render to",
        device.name(),
        unsupported.len()
    );
    for why in &unsupported {
        eprintln!("unsupported: {why}");
    }
    assert!(
        wrong.is_empty(),
        "clears Metal stores otherwise:\n{}",
        wrong.join("\n")
    );
    quiet(device, &messages);
}

/// A 4×4 `RGBA8Unorm` pattern.
fn pattern(seed: u8) -> Vec<u8> {
    (0..64u8)
        .map(|i| i.wrapping_mul(7).wrapping_add(seed))
        .collect()
}

/// A 4×4 `RGBA8Unorm` `Shared` attachment holding `bytes`.
fn holding(device: &Device, bytes: &[u8]) -> Texture {
    let texture = attachment(
        device,
        PixelFormat::RGBA8Unorm,
        [4, 4],
        ResourceOptions::shared(),
    )
    .unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, bytes, 16, 64)
        .unwrap();
    texture
}

/// A 4×4 `RGBA32Float` `Shared` attachment (the format of [`CONSTANT_COLOR`]'s pipeline) holding
/// `bytes` (256 of them).
fn holding_float(device: &Device, bytes: &[u8]) -> Texture {
    let texture = attachment(
        device,
        PixelFormat::RGBA32Float,
        [4, 4],
        ResourceOptions::shared(),
    )
    .unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, 4, 4), 0, 0, bytes, 64, 256)
        .unwrap();
    texture
}

/// The 4×4 `RGBA32Float` texels of `texture`.
fn float_texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0u8; 256];
    texture
        .get_bytes(&mut bytes, 64, 256, Region::new_2d(0, 0, 4, 4), 0, 0)
        .unwrap();
    bytes
}

/// The 4×4 `RGBA8Unorm` texels of `texture`.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut bytes = vec![0u8; 64];
    texture
        .get_bytes(&mut bytes, 16, 64, Region::new_2d(0, 0, 4, 4), 0, 0)
        .unwrap();
    bytes
}

/// A pass whose only color attachment is `texture`, with `load` and `store`, clearing to a
/// color whose bytes are `[0x40, 0x80, 0xc0, 0xff]`.
fn one_target(texture: &Texture, load: LoadAction, store: StoreAction) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = load;
    attachment.store_action = store;
    attachment.clear_color = ClearColor::new(64.0 / 255.0, 128.0 / 255.0, 192.0 / 255.0, 1.0);
    pass
}

/// The clear color of [`one_target`], every texel.
fn cleared() -> Vec<u8> {
    [0x40u8, 0x80, 0xc0, 0xff].repeat(16)
}

/// Run one pass of `descriptor` without draws, `encode` applied to its encoder first.
fn empty_pass(
    device: &Device,
    descriptor: &RenderPassDescriptor,
    encode: impl FnOnce(&mut RenderCommandEncoder),
) {
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    run(&command_buffer);
}

/// `Load` keeps the texels and `Clear` writes the clear color; a store action the descriptor left
/// `Unknown` is the one the encoder sets; passes order against blits in one command buffer (a
/// pass loads what a blit before it wrote, a blit after it reads what it stored). What `DontCare`
/// leaves is printed.
#[test]
fn load_and_store_actions_keep_and_clear_as_metal() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let texture = holding(&device, &pattern(1));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Load, StoreAction::Store),
        |_| {},
    );
    assert_eq!(texels(&texture), pattern(1), "Load, Store");
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Clear, StoreAction::Store),
        |_| {},
    );
    assert_eq!(texels(&texture), cleared(), "Clear, Store");

    // The late store action.
    let texture = holding(&device, &pattern(2));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Clear, StoreAction::Unknown),
        |encoder| {
            encoder
                .set_color_store_action(StoreAction::Store, 0)
                .unwrap()
        },
    );
    assert_eq!(texels(&texture), cleared(), "Clear, a late Store");

    // In one command buffer: a blit writes the attachment, a pass loads and stores it, a blit
    // reads it; then a pass clears it and a blit reads it again.
    let texture = attachment(
        &device,
        PixelFormat::RGBA8Unorm,
        [4, 4],
        ResourceOptions::private(),
    )
    .unwrap();
    let source = buffer_with(&device, &pattern(3)).unwrap();
    let loaded = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let after_clear = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let command_buffer = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    let blit_copy = |command_buffer: &CommandBuffer, into: Option<&Buffer>| {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let (origin, size) = (Origin::new(0, 0, 0), Size::new(4, 4, 1));
        match into {
            None => blit.copy_from_buffer_to_texture(
                &source,
                0,
                16,
                64,
                size,
                &texture,
                0,
                0,
                origin,
                BlitOption::None,
            ),
            Some(buffer) => blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                origin,
                size,
                buffer,
                0,
                16,
                64,
                BlitOption::None,
            ),
        }
        .unwrap();
        blit.end_encoding().unwrap();
    };
    blit_copy(&command_buffer, None);
    command_buffer
        .render_command_encoder(&one_target(&texture, LoadAction::Load, StoreAction::Store))
        .unwrap()
        .end_encoding()
        .unwrap();
    blit_copy(&command_buffer, Some(&loaded));
    command_buffer
        .render_command_encoder(&one_target(&texture, LoadAction::Clear, StoreAction::Store))
        .unwrap()
        .end_encoding()
        .unwrap();
    blit_copy(&command_buffer, Some(&after_clear));
    run(&command_buffer);
    assert_eq!(
        read_buffer(&loaded, 0, 64).unwrap(),
        pattern(3),
        "a blit, then Load"
    );
    assert_eq!(
        read_buffer(&after_clear, 0, 64).unwrap(),
        cleared(),
        "then Clear"
    );

    // DontCare: undefined in Metal; printed.
    let texture = holding(&device, &pattern(4));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::Load, StoreAction::DontCare),
        |_| {},
    );
    let after_store = texels(&texture);
    let texture = holding(&device, &pattern(5));
    empty_pass(
        &device,
        &one_target(&texture, LoadAction::DontCare, StoreAction::Store),
        |_| {},
    );
    let after_load = texels(&texture);
    let verdict = |bytes: &[u8], old: &[u8]| {
        if bytes == old {
            "keeps the old texels".to_owned()
        } else {
            format!("leaves {:02x?}", &bytes[..16])
        }
    };
    eprintln!(
        "DontCare on {}: a DontCare store {}; a DontCare load with no draws {}",
        device.name(),
        verdict(&after_store, &pattern(4)),
        verdict(&after_load, &pattern(5))
    );
    quiet(device, &messages);
}

/// The public case whose pipeline the refusal test draws with: a constant fragment color through
/// its generated passthrough vertex, one `RGBA8Unorm` target.
const CONSTANT_COLOR: &str = "authored-fragment-constant-color";

/// Metal's refusals of passes and draws, typed: a NaN clear depth; a store action never set; a
/// pipeline state of other formats than the pass's; a draw without a pipeline (one of another
/// topology class draws as plain Metal's: `topology_classes`). A refused draw records nothing, and
/// a draw of no vertices or no instances records nothing; a misaligned index buffer offset reads
/// from the offset rounded down. A color store action set late over a known one, or for no
/// attachment, and a depth one for no attachment change nothing, as plain Metal's (Metal probes on
/// an M4, which Metal's validation stops). The render encoder's Apple-only setters answer by step
/// 12's table. (A draw whose indices run past its index buffer draws as Metal's runtime does:
/// `index_fetches`.)
#[test]
fn passes_and_draws_are_refused_as_metal_refuses_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let case = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .find(|case| case.name == CONSTANT_COLOR)
        .expect("the constant color case");
    let case = map_render(&root, &store, case).unwrap();
    let pipeline = render_pipeline_state(&device, &case.inputs()).unwrap();
    let queue = device.new_command_queue().unwrap();

    // A NaN clear depth of a format whose store of one is not measured, where the device makes
    // it (a value outside [0, 1] clears to the nearer end:
    // `clears_store_the_bytes_metal_stores`; a NaN of a measured format clears as Metal's:
    // `depth_denormals::a_nan_clear_depth_stores_and_tests_as_metal_s`).
    match attachment(
        &device,
        PixelFormat::Depth24UnormStencil8,
        [4, 4],
        ResourceOptions::private(),
    ) {
        Ok(depth) => {
            let mut pass = RenderPassDescriptor::new();
            pass.depth_attachment.texture = Some(depth.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.clear_depth = f64::NAN;
            let command_buffer = queue.command_buffer().unwrap();
            assert_eq!(
                command_buffer.render_command_encoder(&pass).unwrap_err(),
                invalid(
                    "render_command_encoder",
                    Problem::RenderPass(RenderPassProblem::ClearDepth)
                )
            );
        }
        Err(error) => println!("NAN-CLEAR-DEPTH Depth24UnormStencil8 not made: {error}"),
    }

    // Late store actions, in a pass of the pipeline's formats: over the descriptor's `Store` (the
    // texels stay stored, `nothing drew` below), of a color attachment without a texture, and of no
    // depth attachment, each changes nothing (Metal probe cases `late_store_known`,
    // `late_store_no_texture` and `late_store_depth_no_texture` on an M4).
    let target = holding_float(&device, &pattern(6).repeat(4));
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&one_target(&target, LoadAction::Load, StoreAction::Store))
        .unwrap();
    assert_eq!(
        encoder.set_color_store_action(StoreAction::DontCare, 0),
        Ok(())
    );
    assert_eq!(
        encoder.set_color_store_action(StoreAction::Store, 1),
        Ok(())
    );
    assert_eq!(encoder.set_depth_store_action(StoreAction::Store), Ok(()));

    // Draws.
    let draw = |encoder: &mut RenderCommandEncoder, instances| {
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, instances, 0)
    };
    assert_eq!(
        draw(&mut encoder, 1),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    // An empty draw without a pipeline state stops plain Metal too: refused alike.
    assert_eq!(
        draw(&mut encoder, 0),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    encoder.set_render_pipeline_state(&pipeline).unwrap();
    let draw_problem = |call, problem| Err(invalid(call, Problem::Draw(problem)));
    // No instances: plain Metal draws nothing, where its validation stops the draw (Metal probe
    // case `draw_zero` on an M4).
    assert_eq!(draw(&mut encoder, 0), Ok(()));
    // An empty draw is checked only as plain Metal checks one: past a pipeline state of its
    // kind, it draws nothing whatever the rest of the encoder's state, here an amplification
    // count above the pipeline's maximum of 1, which refuses a draw that draws (its validation
    // stops both).
    encoder.set_vertex_amplification_count(2, None).unwrap();
    assert_eq!(draw(&mut encoder, 0), Ok(()));
    assert_eq!(
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0),
        Ok(())
    );
    assert_eq!(
        draw(&mut encoder, 1),
        draw_problem(
            "draw_primitives",
            DrawProblem::AmplificationCount { count: 2, max: 1 }
        )
    );
    encoder.set_vertex_amplification_count(1, None).unwrap();
    let indices = buffer_with(&device, &[0, 0, 1, 0, 2, 0, 0, 0]).unwrap();
    let indexed = |encoder: &mut RenderCommandEncoder, ty, count, ty_index, offset| {
        encoder.draw_indexed_primitives(ty, count, ty_index, &indices, offset, 1, 0, 0)
    };
    // An index buffer offset that is not a multiple of the index size reads from the offset
    // rounded down, as plain Metal reads it: one index, no triangle.
    assert_eq!(
        indexed(
            &mut encoder,
            PrimitiveType::Triangle,
            1,
            IndexType::UInt32,
            2
        ),
        Ok(())
    );
    // The render encoder's Apple-only setters (step 12's table): Metal's defaults are accepted.
    let apple = |feature| Err(Error::AppleOnly(feature));
    assert_eq!(encoder.set_alpha_test_reference_value(0.5), Ok(()));
    assert_eq!(encoder.set_line_width(1.0), Ok(()));
    assert_eq!(encoder.set_line_width(2.0), apple(AppleOnly::LineWidth));
    // `setPointSize:` changes nothing Metal draws, at every size (088029d75).
    assert_eq!(encoder.set_point_size(4.0), Ok(()));
    assert_eq!(encoder.set_provoking_vertex_mode(0), Ok(()));
    assert_eq!(
        encoder.set_provoking_vertex_mode(1),
        apple(AppleOnly::ProvokingVertexMode)
    );
    assert_eq!(encoder.set_viewport_transform_enabled(true), Ok(()));
    assert_eq!(
        encoder.set_viewport_transform_enabled(false),
        apple(AppleOnly::ViewportTransformDisabled)
    );
    assert_eq!(
        encoder.set_vertex_amplification_mode(1, 0),
        apple(AppleOnly::VertexAmplificationMode)
    );
    assert_eq!(encoder.set_depth_cleared(), apple(AppleOnly::DepthCleared));
    assert_eq!(
        encoder.set_stencil_cleared(),
        apple(AppleOnly::StencilCleared)
    );
    // Nothing above drew; a draw of no vertices records nothing.
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 0, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(float_texels(&target), pattern(6).repeat(4), "nothing drew");

    // A pipeline of another color format than the pass's (measured: Metal's validation stops
    // `setRenderPipelineState:`, and plain Metal stores the pipeline's encoding). The refused
    // call keeps the state set before. (A color the pipeline lacks is drawn, zeroed where
    // covered, and a depth or stencil format the pipeline names otherwise is the pass's, as
    // plain Metal draws them: `extra_colors.rs` and `hidden_depth.rs`.)
    let other = attachment(
        &device,
        PixelFormat::R32Float,
        [4, 4],
        ResourceOptions::private(),
    )
    .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&one_target(&other, LoadAction::Clear, StoreAction::Store))
        .unwrap();
    assert_eq!(
        encoder.set_render_pipeline_state(&pipeline),
        draw_problem(
            "set_render_pipeline_state",
            DrawProblem::AttachmentFormat(Attachment::Color(0))
        )
    );
    assert_eq!(
        draw(&mut encoder, 1),
        Err(invalid("draw_primitives", Problem::NoPipelineState))
    );
    encoder.end_encoding().unwrap();
    run(&command_buffer);

    // A color store action still Unknown at end_encoding stores as `DontCare` (Metal probe case
    // `store_unknown` on an M4, which Metal's validation stops: plain Metal kept the old texels
    // under a clear); a `DontCare` store's texels are the device's
    // (`load_and_store_actions_keep_and_clear_as_metal`).
    let command_buffer = queue.command_buffer().unwrap();
    let encoder = command_buffer
        .render_command_encoder(&one_target(
            &target,
            LoadAction::Clear,
            StoreAction::Unknown,
        ))
        .unwrap();
    assert_eq!(encoder.end_encoding(), Ok(()));
    run(&command_buffer);
    // So does a depth or stencil store action still Unknown without a resolve texture: plain
    // Metal left the texels under a clear, as its `DontCare` (Metal's
    // validation stops it).
    for (format, depth, stencil) in [
        (PixelFormat::Depth32Float, true, false),
        (PixelFormat::Stencil8, false, true),
        (PixelFormat::Depth32FloatStencil8, true, true),
    ] {
        let texture = attachment(&device, format, [4, 4], ResourceOptions::private()).unwrap();
        let mut pass = RenderPassDescriptor::new();
        if depth {
            pass.depth_attachment.texture = Some(texture.clone());
            pass.depth_attachment.load_action = LoadAction::Clear;
            pass.depth_attachment.store_action = StoreAction::Unknown;
        }
        if stencil {
            pass.stencil_attachment.texture = Some(texture.clone());
            pass.stencil_attachment.load_action = LoadAction::Clear;
            pass.stencil_attachment.store_action = StoreAction::Unknown;
        }
        let command_buffer = queue.command_buffer().unwrap();
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        assert_eq!(encoder.end_encoding(), Ok(()), "{format:?}");
        run(&command_buffer);
    }
    quiet(device, &messages);
}
