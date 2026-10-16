//! The engine executor's runtime path: a case of a family the runtime runs goes through
//! `metal2vulkan_runtime` as a Metal program would run it, by the case runner the runtime lane
//! shares ([`crate::candidate::runtime_case`]).
//!
//! [`route`] decides by the case's own inputs, device-free, which path runs it: the runtime, or the
//! executor's own context while the runtime path waits for the [`Prerequisite`]s later steps add. A
//! run the runtime takes opens a device of its own behind the hang tripwire
//! (`runtime_case::open_device`) among the physical devices of the run's device type, the
//! validation layer reporting to a sink that fails the row (with the executor's settings: every
//! occurrence of a message reported, synchronization validation on), a shader cache directory of
//! the run's own, empty when made, so no warm cache stands in for a translation, the robustness
//! features withheld in the robustness-off leg, and this process's policy. A run this process
//! restricts to a committed profile (`--profiles-layer`) opens it as the executor's restricted
//! context opens its own: a device without the profile's capabilities refused, typed, before it
//! opens; the profiles layer below the validation layer with the context's settings, the instance
//! at the profile's version (`DeviceOptions::instance_api_version`); the context's second check
//! made on the opened device through its raw handles, and the row stored in the profile's slot.
//! Where the runtime would compile for the device's own kept members rather than the profile's, the
//! case goes to the executor's context (`translation`, the bridge), but for a patch draw the
//! profile emulates, whose emulation only the runtime runs: the runtime runs it on those members,
//! and the row records them (`kept_members`). A kernel case's AIR becomes a
//! library, its entry a function, the function with the functions it links (each linked module a
//! library of its own) a pipeline state, which makes the case's visible-function tables (each entry
//! the handle the state gives the function it holds, `runtime_case::visible_function_tables`), and
//! its resources, its tables, one compute encoder, and one dispatch a command buffer, committed
//! once what the dispatch executed has decided how it contains the case's out-of-bounds targets
//! (`kernel_out_of_bounds`, the executor's own decision; a target the device leaves uncontained
//! refuses the case uncommitted, the refusal recording what the dispatch compiled; a buffer target
//! lies in a heap of its own between guards, `runtime_case::guarded_buffer`), and each indirect
//! command buffer the case executes, in key order, a command buffer of its own committed after it
//! on its queue (`runtime_case::encoded_execution`, the case's command buffers made as the Metal
//! oracle makes them, a command buffer of render commands included, its host commands encoded, its
//! templates' states, compute or render, made with `support_indirect_command_buffers`,
//! `template_states`); a tile kernel's function a tile pipeline state, which makes its tables for
//! the tile stage (`runtime_case::tile_tables`), and its resources, its argument buffers, its
//! tables, its pass's attachments, one render encoder of the pass and one tile dispatch a command
//! buffer, committed (`runtime_case::encoded_tile`); a render case draws once as the Metal oracle
//! draws it (`runtime_case::encoded_render`: its functions and its generated companion's, the
//! case's own linking its functions, a render pipeline state of the oracle's descriptor, which
//! makes each stage's tables for that stage, its attachments, state, resources, argument buffers
//! (with the pipeline states they hold, each its template's, `template_states`) and tables, and one
//! draw), committed once what the draw executed has decided how it contains the case's vertex and
//! index buffer targets as the executor decides it (`draw_out_of_bounds`; each target in a heap of
//! its own between guards, and an index buffer target's draw holding its index positions past the
//! buffer to that containment, `engine::index_padding_agrees`). The wait for the command buffer
//! is bounded under the hang protocol (`gpu::wait::runtime`), the status is its execution report,
//! and the output is read back after it completed. The runtime compiles in this process, the device
//! worker (the compile boundary, for these runs, is the device worker's, with the runtime's own
//! compile budget).
//!
//! What the row records comes from what the runtime executed (its execution record): the census
//! of the compiled pipeline's modules, the artifact digest of the pipeline planned for the bind
//! shape it ran with the bytes of the tables it bound, the SPIR-V hash (a render pipeline's stages'
//! words in order, as the executor hashes them), the pipeline-creation ceiling of each `VkPipeline`
//! the dispatch or draw recorded, and its out-of-bounds facts, with what the run changed around
//! each target it placed between guards (`guards::with_escaped`, the heap's cause recorded where
//! one explains an escape or a read past a target). The device's identity, driver, and active tools
//! are read through the device's raw handles (`candidate::device_facts`).
//!
//! The bridge (temporary): a typed runtime gap (`Error::NotImplemented`) raised before the command
//! buffer commits (the device's creation's among them), so nothing ran, sends the case to the
//! executor's own context, and the row records the gap. So do two refusals that are not the case's
//! own, each keyed on the error's origin and type: a refusal of a mechanism of the runtime's own
//! that the device's driver runs wrong (`Needs::RuntimeDriverDefect`; the gap `<call>: driver
//! defect <defect> (<feature>)`), and the runtime's refusal of a helper's pipeline, the runner's
//! own work around the case's pass (`CaseError::Helper`: a sample fill's, a sample readback's; the
//! gap `<call>: helper refused <refusal>`). Both are the runtime path's limitations, not the
//! device's lack: the executor's context does that work its own way.

use super::artifact::CaseArtifact;
use super::artifact::Planned;
use super::graphics::{self, RenderCase};
use super::guards::{with_escaped, Placement};
use super::tables::TableBuffers;
use super::{
    bind_refusal, companion_source, compute_case, draw_out_of_bounds_run, engine_fingerprint,
    index_padding_agrees, linked_modules, out_of_bounds_run, product_refusal, report_verdict,
    CompileSubject, ComputeCase, ComputeRun, Mapped, PreparedCase, RunAttempt,
};
use crate::air_library::AirLibrary;
use crate::candidate::device_facts::{DeviceFacts, RECORDED_LAYERS};
use crate::candidate::runtime_case::{
    argument_texture_usage, compute_descriptor, encoded_case, encoded_execution, encoded_mesh,
    encoded_render, encoded_tile, executor_layer_settings, linked_functions, mesh_pipeline_state,
    open_device, render_pipeline_descriptor, texture_usage, tile_descriptor, tile_tables,
    validated_options, visible_function_tables, CaseError, MeshInputs, Messages, RenderInputs,
    ScratchCache, TemplateState,
};
use crate::candidate::{runtime_restriction, RestrictedDevice, Restricting, RuntimeRestriction};
use crate::case::out_of_bounds::ContainmentForm;
use crate::case::{AuthoredCase, Stage};
use crate::check::admission::ExecutorError;
use crate::companion::Spelling;
use crate::coverage::{ExecutedCensus, Leg, ModuleCensus, ROBUSTNESS_FEATURES};
use crate::engine::RowSource;
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun};
use crate::gpu::wait::{with_submission_label, SubmissionBackend, SubmissionLabel};
use crate::hash::sha256_bytes;
use crate::observation::{Backend, TRANSLATOR_FINGERPRINT};
use crate::worker::device::OpenedDevice;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Needs, Unsupported};
use metal2vulkan::pipeline::{Stage as ShaderStage, StageModule};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, Device, Error, ExecutedPipeline, Execution, Function, PipelineOptions,
};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// What a case needs that the runtime path does not run yet: the steps that add a case's
/// prerequisites decide when its family moves. Until then the executor's own context runs the case,
/// and its row records them (`runtime_waits_for`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prerequisite {
    /// A layered pass's multisampled color targets, which the render runner does not make:
    /// step 6's multisample arrays.
    LayeredMultisampledTargets,
    /// Acceleration structures and intersection function tables: they stay on the executor in
    /// this phase.
    RayTracing,
    /// A tile dispatch's command buffers and the pipeline states its argument buffers hold: the
    /// tile runner (`runtime_case::encoded_tile`) makes none. A kernel's and a draw's run on the
    /// indirect render commands, executed in a render encoder.
    IndirectRenderCommands,
    /// A draw's out-of-bounds target the runtime path does not place between guards yet: a stage
    /// buffer's, or a tile dispatch's out-of-bounds target. Vertex and index buffer targets run on
    /// the runtime, and so do a patch draw's control point and patch input buffers,
    /// vertex buffers its post-tessellation vertex function fetches.
    DrawOutOfBoundsTargets,
}

impl Prerequisite {
    /// The name a row records.
    pub fn name(self) -> &'static str {
        match self {
            Prerequisite::LayeredMultisampledTargets => "layered-multisampled-targets",
            Prerequisite::RayTracing => "ray-tracing",
            Prerequisite::IndirectRenderCommands => "indirect-render-commands",
            Prerequisite::DrawOutOfBoundsTargets => "draw-out-of-bounds-targets",
        }
    }
}

/// Which path runs a prepared case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// The runtime ([`run`]).
    Runtime,
    /// The executor's own context, while the runtime path waits for these prerequisites (one or
    /// more).
    Executor(BTreeSet<Prerequisite>),
}

/// The path that runs `prepared`: the runtime, unless the case needs a [`Prerequisite`]: a kernel
/// case's own inputs ([`kernel_prerequisites`]) or a render case's ([`render_prerequisites`]).
/// The functions a case links run on the runtime too (its pipeline state's descriptor links
/// them), and so do its compute commands and the pipeline states its argument buffers hold, a
/// render template's included (their templates' states made with
/// `support_indirect_command_buffers`, `template_states`).
pub fn route(prepared: &PreparedCase) -> Route {
    let authored = &prepared.checked.primary.case;
    let needs = match &prepared.mapped {
        Mapped::Compute(compute) => kernel_prerequisites(compute),
        Mapped::Render(render) => render_prerequisites(authored, render),
        // Only the runtime draws a mesh render pipeline (`runtime_case::encoded_mesh`).
        Mapped::Mesh(_) => BTreeSet::new(),
    };
    if needs.is_empty() {
        Route::Runtime
    } else {
        Route::Executor(needs)
    }
}

/// The prerequisites whose condition holds.
fn needed<const N: usize>(conditions: [(bool, Prerequisite); N]) -> BTreeSet<Prerequisite> {
    conditions
        .into_iter()
        .filter_map(|(needed, prerequisite)| needed.then_some(prerequisite))
        .collect()
}

/// The [`Prerequisite`]s the inputs of a kernel case, mapped as `compute`, need: none for a case
/// of the runtime's families.
pub fn kernel_prerequisites(compute: &ComputeCase) -> BTreeSet<Prerequisite> {
    needed([
        (
            !compute.acceleration_structures.is_empty(),
            Prerequisite::RayTracing,
        ),
        // A tile kernel dispatches on a render encoder (`runtime_case::encoded_tile`), which
        // makes no command buffer or pipeline state.
        (
            compute.tile.is_some()
                && (!compute.command_buffers.is_empty() || !compute.objects.is_empty()),
            Prerequisite::IndirectRenderCommands,
        ),
        (
            compute.tile.is_some() && !compute.out_of_bounds.is_empty(),
            Prerequisite::DrawOutOfBoundsTargets,
        ),
    ])
}

/// The [`Prerequisite`]s the inputs of render case `case`, mapped as `render`, need: none for a
/// case the render runner draws (`runtime_case::encoded_render`): one draw of primitives, direct
/// or indexed, or of a tessellation pipeline's patches with their factor buffer, of a pipeline its
/// functions and their linked functions make, with its attachments, vertex buffers (a vertex
/// input's, a control point's or a patch input's target between guards), each stage's buffers,
/// textures (multisampled and sparse ones included), samplers, argument buffers of buffers,
/// textures, samplers, pipeline states and command buffers (the commands each executes run
/// after the draw), and visible-function tables, a fragment case's explicit imageblock (between the
/// oracle's helpers' draws), and a rasterization-rate map, the pass's or a buffer's.
pub fn render_prerequisites(case: &AuthoredCase, render: &RenderCase) -> BTreeSet<Prerequisite> {
    needed([
        (
            !case.acceleration_structures.is_empty()
                || !case.intersection_function_tables.is_empty()
                || !case.argument_buffer_intersection_function_tables.is_empty(),
            Prerequisite::RayTracing,
        ),
        (
            render.out_of_bounds.iter().any(|target| {
                !matches!(
                    target.form,
                    ContainmentForm::VertexInput
                        | ContainmentForm::IndexBuffer
                        | ContainmentForm::ControlPoint
                )
            }),
            Prerequisite::DrawOutOfBoundsTargets,
        ),
        (
            render.render_target_array_length > 0
                && render
                    .color_targets
                    .values()
                    .any(|target| target.samples != 1),
            Prerequisite::LayeredMultisampledTargets,
        ),
    ])
}

/// What the runtime path came to: an attempt, or a typed runtime gap (or a refusal of a runtime
/// mechanism the driver runs wrong, or of a helper's pipeline) raised before anything was
/// committed, which sends the case to the executor (the bridge).
pub enum RuntimeOutcome {
    Ran(Box<RunAttempt>),
    Gap(String),
}

impl RuntimeOutcome {
    /// The attempt `result` came to, with no device a refusal names.
    fn of(result: Result<ComputeRun, ExecutorError>) -> RuntimeOutcome {
        RuntimeOutcome::Ran(Box::new(RunAttempt::of(result)))
    }

    /// The failed attempt `message` describes.
    fn failed(message: String) -> RuntimeOutcome {
        RuntimeOutcome::of(Err(ExecutorError::Failed(message)))
    }
}

/// Why a run on the runtime's device produced no row.
enum Stop {
    /// A typed runtime gap before the command buffer committed.
    Gap(String),
    /// A typed refusal after the dispatch or draw took its pipeline, before anything was
    /// committed.
    Refused(Box<Refused>),
    /// The run's error.
    Error(ExecutorError),
}

/// A typed refusal after the dispatch or draw took its pipeline, before anything was committed (a
/// target the device leaves uncontained), with what it had compiled, which the refusal records as
/// the executor's does.
struct Refused {
    error: ExecutorError,
    artifact: CaseArtifact,
}

impl From<ExecutorError> for Stop {
    fn from(error: ExecutorError) -> Stop {
        Stop::Error(error)
    }
}

impl From<&str> for Stop {
    fn from(error: &str) -> Stop {
        Stop::Error(ExecutorError::Failed(error.into()))
    }
}

impl From<String> for Stop {
    fn from(error: String) -> Stop {
        Stop::Error(ExecutorError::Failed(error))
    }
}

/// A runtime call's refusal as the executor reports it. Before the command buffer committed, a
/// typed runtime gap, a refusal of a runtime mechanism the driver runs wrong
/// (`Needs::RuntimeDriverDefect`), and a refusal of a helper's pipeline ([`CaseError::Helper`])
/// are the bridge's; a compile or device refusal is the product's
/// ([`product_refusal`]), the binder's refusal of the bind state is [`bind_refusal`]'s, and
/// anything else, or a literal the case runner cannot run, fails the run.
fn stop(error: CaseError, committed: bool) -> Stop {
    match error {
        CaseError::Runtime { call, error } => match *error {
            Error::NotImplemented(gap) if !committed => Stop::Gap(format!("{call}: {gap:?}")),
            Error::Unsupported(Unsupported {
                feature,
                needs: Needs::RuntimeDriverDefect(defect),
                ..
            }) if !committed => {
                Stop::Gap(format!("{call}: driver defect {defect:?} ({feature:?})"))
            }
            Error::Compile(error) => Stop::Error(product_refusal(&error)),
            Error::Unsupported(unsupported) => Stop::Error(product_refusal(
                &metal2vulkan::error::Error::Unsupported(unsupported),
            )),
            Error::Bind(error) => Stop::Error(bind_refusal(&error)),
            other => Stop::Error(ExecutorError::Failed(format!(
                "the runtime's {call}: {other}"
            ))),
        },
        CaseError::Helper { call, error } if !committed => match *error {
            Error::Unsupported(unsupported)
            | Error::Compile(metal2vulkan::error::Error::Unsupported(unsupported)) => {
                Stop::Gap(format!("{call}: helper refused {unsupported}"))
            }
            other => stop(
                CaseError::Runtime {
                    call,
                    error: Box::new(other),
                },
                committed,
            ),
        },
        CaseError::Helper { call, error } => stop(CaseError::Runtime { call, error }, committed),
        CaseError::Case(why) => {
            Stop::Error(ExecutorError::Failed(format!("the runtime path: {why}")))
        }
    }
}

/// What a completed run read: the output, the execution report's verdict, and the execution
/// record, with the device's facts. Holds no runtime object.
struct Ran {
    output: Vec<u8>,
    status: metal2vulkan::bind::StatusReport,
    executions: Vec<Execution>,
    /// The bytes of the visible-function tables the dispatch bound, as the artifact records them
    /// (`runtime_case::CaseTables::words`); none for a draw.
    tables: TableBuffers,
    /// How the dispatch or draw contained the case's out-of-bounds targets
    /// ([`kernel_out_of_bounds`], [`draw_out_of_bounds`]), with what the run changed around them;
    /// `None` for a case without targets, and for a tile dispatch (whose targets wait,
    /// [`Prerequisite::DrawOutOfBoundsTargets`]).
    out_of_bounds: Option<OutOfBoundsRun>,
    /// What the commands of the case's executed indirect command buffers ran
    /// (`runtime_case::encoded_execution`), in execution order: none for a case that executes
    /// none.
    children: Vec<Execution>,
}

/// Run `prepared`, a case [`route`] gives the runtime, on a runtime device of its device type, in
/// this process's leg and policy: the attempt, or the bridge's gap. A typed refusal after the
/// device opened names that device ([`RunAttempt::refused_on`]).
pub fn run(prepared: &PreparedCase, backend: Backend) -> RuntimeOutcome {
    let leg = crate::candidate::requested_leg();
    // A run this process restricts to a committed profile (`--profiles-layer`) opens the runtime's
    // device under the profiles layer as the executor's context opens its own: a device that lacks
    // capabilities the profile states is refused, typed, before any device opens.
    let executor = Kind::of(&prepared.mapped).executor();
    let restriction = match runtime_restriction(prepared.device, executor) {
        Ok(None) => None,
        Ok(Some(Restricting::Takes(restriction))) => Some(restriction),
        Ok(Some(Restricting::Refused(mut attempt))) => {
            let environment = attempt
                .refused_on
                .as_mut()
                .and_then(|opened| opened.environment.as_object_mut());
            if let Some(object) = environment {
                object.insert("executed_by".into(), serde_json::json!("runtime"));
            }
            return RuntimeOutcome::Ran(attempt);
        }
        Err(error) => return RuntimeOutcome::failed(format!("the profile restriction: {error}")),
    };
    let messages = Messages::default();
    let cache = match ScratchCache::new() {
        Ok(cache) => cache,
        Err(error) => {
            return RuntimeOutcome::failed(format!("the runtime's shader cache directory: {error}"))
        }
    };
    let mut options = validated_options(&messages, cache.path());
    options.layer_settings = executor_layer_settings();
    options.policy = crate::candidate::requested_policy();
    // A run without generated commands (`--without-generated-commands`) opens the runtime's device
    // without `VK_EXT_device_generated_commands`, as the executor's context opens its own.
    options.withhold_generated_commands = !crate::candidate::requested_generated_commands();
    if leg == Leg::RobustnessOff {
        options.withheld_features = ROBUSTNESS_FEATURES.into_iter().collect();
    }
    if let Some(restriction) = &restriction {
        if let Err(error) = restriction.restrict(&mut options) {
            return RuntimeOutcome::failed(format!("the profile restriction: {error}"));
        }
    }
    let device = match open_device(prepared.device, options) {
        Ok(device) => device,
        Err(error) => {
            let failed = format!(
                "the runtime opened no {} device: {error}",
                prepared.device.as_str()
            );
            // A typed runtime gap at the device's creation ran nothing: the bridge's, once the
            // sink holds no message. A typed refusal of the device itself (one below Vulkan's
            // timeline-semaphore minimum, say) is the device's lack: under a restriction it is
            // recorded with the device as it is and the restriction, as a refusal of the
            // restriction is; unrestricted, it names no device.
            return match (stop(error, false), messages_check(&messages)) {
                (Stop::Gap(gap), Ok(())) => RuntimeOutcome::Gap(gap),
                (_, Err(messages)) => RuntimeOutcome::failed(format!("{messages} ({failed})")),
                (Stop::Error(error @ ExecutorError::NotSupported(_)), Ok(())) => {
                    match restriction.as_deref().map(|r| r.refused_on(executor)) {
                        Some(Err(why)) => RuntimeOutcome::failed(format!("{failed}, and {why}")),
                        refused_on => {
                            let mut refused_on = refused_on.and_then(Result::ok);
                            let environment = refused_on
                                .as_mut()
                                .and_then(|opened| opened.environment.as_object_mut());
                            if let Some(object) = environment {
                                object.insert("executed_by".into(), serde_json::json!("runtime"));
                            }
                            RuntimeOutcome::Ran(Box::new(RunAttempt {
                                result: Err(error),
                                refused_on,
                            }))
                        }
                    }
                }
                (Stop::Refused(_) | Stop::Error(_), Ok(())) => RuntimeOutcome::failed(failed),
            };
        }
    };
    let opened = Opened::of(&device, leg, &prepared.mapped, restriction.as_deref());
    let ran = match &opened {
        Ok(Opened {
            untranslated: Some(gap),
            ..
        }) => Err(Stop::Gap(gap.clone())),
        Ok(opened) => run_on(&device, opened, prepared, backend),
        Err(error) => Err(Stop::Error(ExecutorError::Failed(error.clone()))),
    };
    // Every object of the run is destroyed with the device, whose instance reports its
    // destruction to the sink too; then the sink has no other holder. The teardown may run after
    // this last handle goes, on the runtime's thread that lets go of the device's last share
    // (`messages_check`).
    drop(device);
    let checked = messages_check(&messages);
    let degrades = degrades(&messages);
    let opened = match opened {
        Ok(opened) => opened,
        Err(error) => return RuntimeOutcome::failed(error),
    };
    let ran = match (ran, checked) {
        (Err(Stop::Gap(gap)), Ok(())) => return RuntimeOutcome::Gap(gap),
        (ran, Err(messages)) => {
            let also = match &ran {
                Ok(_) => String::new(),
                Err(Stop::Gap(gap)) => format!(" (the runtime also raised the gap {gap})"),
                Err(Stop::Refused(refused)) => format!(" (the run also failed: {})", refused.error),
                Err(Stop::Error(error)) => format!(" (the run also failed: {error})"),
            };
            return RuntimeOutcome::failed(format!("{messages}{also}"));
        }
        (Ok(ran), Ok(())) => Ok(ran),
        (Err(Stop::Refused(refused)), Ok(())) => Err((refused.error, refused.artifact)),
        (Err(Stop::Error(error)), Ok(())) => Err((error, CaseArtifact::default())),
    };
    match ran {
        Ok(ran) => RuntimeOutcome::of(opened.row(ran, degrades)),
        Err((error @ ExecutorError::NotSupported(_), artifact)) => {
            RuntimeOutcome::Ran(Box::new(RunAttempt {
                result: Err(error),
                refused_on: Some(opened.refused_on(&artifact)),
            }))
        }
        Err((error, _)) => RuntimeOutcome::of(Err(error)),
    }
}

/// The codes of the degrade notices the run's device delivered to `messages`: each kind the runtime
/// announced, once, for this case on this device, in code order.
fn degrades(messages: &Messages) -> Vec<&'static str> {
    let codes: BTreeSet<&'static str> = messages
        .take_notices()
        .iter()
        .filter_map(|message| message.degradation.as_ref())
        .map(|notice| notice.degradation.code())
        .collect();
    codes.into_iter().collect()
}

/// The failure of a run whose validation layer reported a message, or whose device's teardown
/// did not end within the wait bound after its last handle dropped (its messages may be
/// incomplete). The teardown may end on another of the runtime's threads a little after the last
/// handle goes: one that held the device's last share as the handle dropped (a pipeline making
/// still running) runs it once it lets go, as the runtime's teardown lane waits for
/// (`runtime_lane::teardown_holders`).
fn messages_check(messages: &Messages) -> Result<(), String> {
    if !messages.unshared_within(crate::gpu::wait::wait_timeout()) {
        return Err(format!(
            "the runtime's device was not torn down within {:?} of its last handle's drop; its \
             validation messages are incomplete, so the run is not evidence",
            crate::gpu::wait::wait_timeout()
        ));
    }
    let reported = messages.take();
    if reported.is_empty() {
        return Ok(());
    }
    Err(format!(
        "VK_LAYER_KHRONOS_validation reported {} message(s); the run is not evidence: {}",
        reported.len(),
        reported
            .iter()
            .map(|message| format!("{} ({}): {}", message.id_name, message.id, message.text))
            .collect::<Vec<_>>()
            .join(" | ")
    ))
}

/// Which of the runtime's paths runs a case: a kernel's dispatch or a render case's draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Compute,
    Render,
}

impl Kind {
    /// The path of a case mapped as `mapped`.
    fn of(mapped: &Mapped) -> Kind {
        match mapped {
            Mapped::Compute(_) => Kind::Compute,
            // A mesh draw is a render encoder's draw of a render pipeline state.
            Mapped::Render(_) | Mapped::Mesh(_) => Kind::Render,
        }
    }

    /// The executor a row of this path names.
    fn executor(self) -> &'static str {
        match self {
            Kind::Compute => "engine-runtime-compute",
            Kind::Render => "engine-runtime-render",
        }
    }
}

/// What a row or refusal records of the runtime's opened device: its facts, its environment, and
/// what it compiles for.
struct Opened {
    facts: DeviceFacts,
    kind: Kind,
    environment: serde_json::Value,
    translator_fingerprint: String,
    profile: metal2vulkan::device::DeviceProfile,
    compile: metal2vulkan::device::Device,
    leg: Leg,
    /// The committed profile the device is restricted to, which the census names.
    restricted_to: Option<String>,
    /// Under a restriction, why the device's compile device does not translate for the profile
    /// as the executor's context does ([`translation`]): the bridge's gap.
    untranslated: Option<String>,
    /// The bound the case's own pipeline's creation is held to: a kernel case's
    /// (`ComputeCase::creation_bound`), the shared ceiling for a render or mesh case.
    creation_bound: std::time::Duration,
}

impl Opened {
    /// The facts of `device`, opened in `leg` for a case mapped as `mapped` (restricted to
    /// `restriction`'s profile, whose second check it makes), read through its raw handles.
    fn of(
        device: &Device,
        leg: Leg,
        mapped: &Mapped,
        restriction: Option<&RuntimeRestriction>,
    ) -> Result<Opened, String> {
        let kind = Kind::of(mapped);
        let restricted = restriction
            .map(|restriction| restriction.check(device))
            .transpose()?;
        let translation = restricted
            .as_ref()
            .map(|restricted| translation(device, restricted, mapped))
            .transpose()?;
        let untranslated = match &translation {
            Some(Translation::Bridged(gap)) => Some(gap.clone()),
            Some(Translation::Profile | Translation::Kept(_)) | None => None,
        };
        // SAFETY: only physical-device queries of the device's own instance; nothing is created,
        // destroyed, or submitted, and the handles are not kept past the borrow.
        let facts = unsafe {
            let handles = device.raw();
            DeviceFacts::query(handles.entry, handles.instance, handles.physical_device)
        }?;
        let compile = device.compile_device().clone();
        let profile = compile.profile().clone();
        let enabled = &device.enabled().features;
        let mut environment = facts.environment(enabled.contains(&VkFeature::SamplerAnisotropy));
        let fingerprint: String = profile
            .fingerprint()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if let Some(object) = environment.as_object_mut() {
            for (key, value) in [
                ("executor", serde_json::json!(kind.executor())),
                ("executed_by", serde_json::json!("runtime")),
                ("layers", serde_json::json!(RECORDED_LAYERS)),
                ("synchronization_validation", serde_json::json!(true)),
                (
                    "profiles_layer",
                    restricted
                        .as_ref()
                        .map_or(serde_json::Value::Null, |restricted| {
                            restricted.record.clone()
                        }),
                ),
                (
                    "robustness",
                    serde_json::json!({
                        "leg": leg,
                        "enabled": ROBUSTNESS_FEATURES
                            .iter()
                            .filter(|feature| enabled.contains(feature))
                            .map(|feature| format!("{feature:?}"))
                            .collect::<Vec<_>>(),
                    }),
                ),
                ("profile_fingerprint", serde_json::json!(fingerprint)),
                ("profile", serde_json::json!(format!("{profile:?}"))),
            ] {
                object.insert(key.into(), value);
            }
            if let Some(Translation::Kept(kept)) = &translation {
                object.insert("kept_members".into(), serde_json::json!(kept));
            }
        }
        Ok(Opened {
            facts,
            kind,
            environment,
            translator_fingerprint: engine_fingerprint(&profile),
            profile,
            compile,
            leg,
            restricted_to: restriction.map(|restriction| restriction.stem().to_owned()),
            untranslated,
            creation_bound: match mapped {
                Mapped::Compute(case) => case.creation_bound,
                Mapped::Render(_) | Mapped::Mesh(_) => crate::budgets::PIPELINE_CREATION_CEILING,
            },
        })
    }

    /// The opened device a typed refusal names, with what the run had compiled when it refused
    /// (`artifact`: empty for a refusal before the dispatch took its pipeline, which compiled
    /// nothing the census can cover).
    fn refused_on(&self, artifact: &CaseArtifact) -> OpenedDevice {
        OpenedDevice {
            environment: self.environment.clone(),
            translator_fingerprint: self.translator_fingerprint.clone(),
            artifact: Some(artifact.executed()),
        }
    }

    /// The one pipeline `execution` (the case's one dispatch or draw) executed, of this run's
    /// kind: what a row names it, its stage modules in pipeline order, and its plan.
    fn executed<'a>(
        &self,
        execution: &'a Execution,
    ) -> Result<(&'static str, &'a [StageModule], &'a dyn Planned), ExecutorError> {
        match (execution.pipeline(), self.kind) {
            (ExecutedPipeline::Compute(pipeline), Kind::Compute) => Ok((
                "the compute pipeline",
                std::slice::from_ref(&pipeline.stage),
                pipeline.as_ref(),
            )),
            (ExecutedPipeline::Render(pipeline), Kind::Render) => {
                Ok(("the render pipeline", &pipeline.stages, pipeline.as_ref()))
            }
            (_, kind) => Err(ExecutorError::Failed(format!(
                "the case's {} run executed a pipeline of the other kind",
                kind.executor()
            ))),
        }
    }

    /// The artifact of `execution`'s pipeline, planned for the bind shape it ran with, with the
    /// bytes of the visible-function `tables` it bound.
    fn artifact(
        &self,
        execution: &Execution,
        tables: &TableBuffers,
    ) -> Result<CaseArtifact, ExecutorError> {
        let (_, _, planned) = self.executed(execution)?;
        let mut artifact = CaseArtifact::default();
        artifact.record(
            CompileSubject::Case,
            planned,
            tables,
            &self.compile,
            execution.shape(),
        )?;
        Ok(artifact)
    }

    /// The out-of-bounds facts a dispatch's or draw's containment decision came to (`decision`,
    /// taken before anything was committed), or the refusal of a target the device leaves
    /// uncontained, recording what `execution` compiled with the bytes of the `tables` it binds.
    fn decided(
        &self,
        execution: &Execution,
        tables: &TableBuffers,
        decision: Result<Option<OutOfBoundsRun>, ExecutorError>,
    ) -> Result<Option<OutOfBoundsRun>, Stop> {
        match decision {
            Err(error @ ExecutorError::NotSupported(_)) => {
                let artifact = self.artifact(execution, tables)?;
                Err(Stop::Refused(Box::new(Refused { error, artifact })))
            }
            decided => decided.map_err(Stop::from),
        }
    }

    /// The row of `ran`: the one dispatch's or draw's compiled pipeline, its census, artifact (with
    /// the bytes of the tables it bound), and SPIR-V hash (a render pipeline's stage words in
    /// pipeline order, as the executor hashes them), each `VkPipeline` it recorded held to the
    /// creation ceiling, and the codes of the degrade notices the run's device delivered
    /// (`runtime_degrades`, [`degrades`]).
    fn row(&self, ran: Ran, degrades: Vec<&'static str>) -> Result<ComputeRun, ExecutorError> {
        let [execution] = ran.executions.as_slice() else {
            return Err(ExecutorError::Failed(format!(
                "the runtime recorded {} executions of the case's one dispatch or draw",
                ran.executions.len()
            )));
        };
        let (what, stages, planned) = self.executed(execution)?;
        for &created in execution.pipeline_creation() {
            crate::budgets::creation_within(what, created, self.creation_bound)
                .map_err(ExecutorError::Failed)?;
        }
        let mut artifact = self.artifact(execution, &ran.tables)?;
        // The pipeline's notes under the specialization its plan for the bind shape it ran with
        // gives: the bound formats' classes and samplers' words.
        let plan = planned.plan_for(&self.compile, execution.shape()).ok();
        let mut modules =
            ModuleCensus::of_pipeline_modules_under(stages, &self.profile, plan.as_ref())
                .map_err(ExecutorError::Failed)?;
        // The census covers every module the run executed, the producer's and each executed
        // command's, as the executor's does. The authored compile census cannot compile what the
        // runtime ran for those commands (its own execution of the records), so the row has no
        // artifact digest and stays on its build (`uncovered`).
        for child in &ran.children {
            let stages = match child.pipeline() {
                ExecutedPipeline::Compute(pipeline) => std::slice::from_ref(&pipeline.stage),
                ExecutedPipeline::Render(pipeline) => &pipeline.stages[..],
            };
            for &created in child.pipeline_creation() {
                crate::budgets::creation_within_ceiling("an executed command's pipeline", created)
                    .map_err(ExecutorError::Failed)?;
            }
            modules.extend(
                ModuleCensus::of_pipeline_modules(stages, &self.profile)
                    .map_err(ExecutorError::Failed)?,
            );
            artifact.record_uncovered();
        }
        let census = ExecutedCensus {
            device: Some(self.facts.execution_class()),
            product: TRANSLATOR_FINGERPRINT.into(),
            restricted_to: self.restricted_to.clone(),
            leg: Some(self.leg),
            modules,
            out_of_bounds: ran.out_of_bounds,
            rate_map_pass: None,
            artifact: Some(artifact.executed()),
        };
        let mut environment = self.environment.clone();
        environment[crate::observation::RUNTIME_DEGRADES] = serde_json::json!(degrades);
        Ok(ComputeRun {
            output: ran.output,
            spv_sha256: spirv_sha256(stages),
            translator_fingerprint: self.translator_fingerprint.clone(),
            environment,
            status: ran.status,
            comparison: census.comparison_class(),
            census,
        })
    }
}

/// The SHA-256 of `stages`' SPIR-V words in pipeline order: a compute pipeline's one stage's, and
/// every stage of a render pipeline, as the executor hashes them.
fn spirv_sha256(stages: &[StageModule]) -> String {
    sha256_bytes(
        &stages
            .iter()
            .flat_map(StageModule::spirv_bytes)
            .collect::<Vec<_>>(),
    )
}

/// The run of `prepared` on `device`: a kernel case's library, function and pipeline state and
/// its committed dispatch, or a render case's committed draw (`runtime_case::encoded_render`); the
/// bounded wait for it under the hang protocol (its hang record names the case and the modules it
/// executed), the execution report's verdict, and the output.
fn run_on(
    device: &Device,
    opened: &Opened,
    prepared: &PreparedCase,
    backend: Backend,
) -> Result<Ran, Stop> {
    let moltenvk = opened.facts.is_moltenvk();
    match backend {
        Backend::Moltenvk if !moltenvk => {
            return Err(format!(
                "the moltenvk backend opened {} (driver id {}), not MoltenVK",
                opened.facts.driver_name, opened.facts.driver_id
            )
            .into())
        }
        Backend::Vulkan if moltenvk => {
            return Err(
                "the native vulkan backend opened MoltenVK; use the moltenvk backend"
                    .to_owned()
                    .into(),
            )
        }
        Backend::Moltenvk | Backend::Vulkan => {}
    }
    let primary = &prepared.checked.primary;
    let bitcode = RowSource::of_row(&primary.source)
        .bitcode()
        .map_err(|error| format!("the case's AIR: {error:?}"))?;
    let usage = texture_usage(&primary.case, &primary.interface)?;
    match &prepared.mapped {
        Mapped::Compute(compute) if compute.tile.is_some() => {
            let argument_usage =
                argument_texture_usage(&compute.argument_buffer_textures, &primary.interface)?;
            // A tile kernel's tile pipeline state, then its visible-function tables for the tile
            // stage, dispatched on a render encoder of its pass (`runtime_case::encoded_tile`).
            let (pipeline, tables) = (|| {
                let function = function(device, &bitcode, compute)?;
                let linked = linked_functions(device, &prepared.linked, &compute.constants)?;
                let pass = compute
                    .tile
                    .as_ref()
                    .ok_or("the case runs in no tile pass")?;
                let pipeline = device
                    .new_tile_render_pipeline_state(
                        &tile_descriptor(&function, linked.functions.clone(), pass),
                        &PipelineOptions::default(),
                    )
                    .map_err(CaseError::runtime("new_tile_render_pipeline_state"))?;
                let tables = tile_tables(&pipeline, compute, &linked)?;
                Ok((pipeline, tables))
            })()
            .map_err(|error| stop(error, false))?;
            let committed = encoded_tile(
                device,
                compute,
                &usage,
                &argument_usage,
                &pipeline,
                tables.tables,
            )
            .and_then(|encoded| encoded.commit())
            .map_err(|error| stop(error, false))?;
            let status = completed(prepared, &committed.command_buffer, &committed.executions)?;
            let output = committed
                .output(compute)
                .map_err(|error| stop(error, true))?;
            Ok(Ran {
                output,
                status,
                executions: committed.executions,
                tables: tables.words,
                out_of_bounds: None,
                children: Vec::new(),
            })
        }
        Mapped::Compute(compute) => {
            let argument_usage =
                argument_texture_usage(&compute.argument_buffer_textures, &primary.interface)?;
            // The pipeline state, then its visible-function tables, each entry the handle the
            // state gives the function it holds; and the state of each template its commands and
            // argument buffers name.
            let templates = template_states(device, prepared)?;
            let (pipeline, tables) = (|| {
                let function = function(device, &bitcode, compute)?;
                let linked = linked_functions(device, &prepared.linked, &compute.constants)?;
                let pipeline = device
                    .new_compute_pipeline_state(
                        &compute_descriptor(&function, linked.functions.clone(), compute),
                        &PipelineOptions::default(),
                    )
                    .map_err(CaseError::runtime("new_compute_pipeline_state"))?;
                let tables = visible_function_tables(&pipeline, compute, &linked)?;
                Ok((pipeline, tables))
            })()
            .map_err(|error| stop(error, false))?;
            let encoded = encoded_case(
                device,
                compute,
                &usage,
                &argument_usage,
                &pipeline,
                (tables.tables, &templates),
            )
            .map_err(|error| stop(error, false))?;
            // How the dispatch's module contains each out-of-bounds target on this device, from
            // what the dispatch executed, before anything runs: a target the device leaves
            // uncontained refuses the case uncommitted, recording what the dispatch compiled.
            let [execution] = encoded.executions.as_slice() else {
                return Err(format!(
                    "the case's one dispatch recorded {} executions",
                    encoded.executions.len()
                )
                .into());
            };
            let out_of_bounds = opened.decided(
                execution,
                &tables.words,
                kernel_out_of_bounds(opened, compute, execution),
            )?;
            // Each command buffer the case executes, in key order, encoded before anything is
            // committed and committed after the producer on its queue.
            let resident: Vec<_> = encoded
                .buffers
                .values()
                .chain(encoded.resources.argument_buffers.values())
                .map(Buffer::as_resource)
                .collect();
            let executions = compute
                .command_buffers
                .iter()
                .filter(|(_, resource)| resource.execute.is_some())
                .map(|(&key, resource)| {
                    encoded_execution(
                        encoded.command_buffer.command_queue(),
                        (&encoded.buffers, &resident),
                        &encoded.resources.commands,
                        (key, resource),
                        &templates,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| stop(error, false))?;
            let committed = encoded.commit().map_err(|error| stop(error, false))?;
            for execution in &executions {
                execution
                    .commit()
                    .map_err(|error| stop(CaseError::runtime("commit")(error), true))?;
            }
            let status = completed(prepared, &committed.command_buffer, &committed.executions)?;
            // The commands the producer left, read back, then what each execution ran.
            committed
                .resources
                .commands
                .check_commands()
                .map_err(|error| stop(error, true))?;
            let mut children = Vec::new();
            for execution in &executions {
                completed(prepared, execution, &execution.executions())?;
                children.extend(execution.executions());
            }
            let output = committed
                .output(compute)
                .map_err(|error| stop(error, true))?;
            // What the run changed around each target placed between guards, read once it
            // completed.
            let guards = committed
                .guard_counts(compute)
                .map_err(|error| stop(error, true))?;
            let out_of_bounds =
                out_of_bounds.map(|run| with_escaped(run, &guards, Placement::Heap));
            Ok(Ran {
                output,
                status,
                executions: committed.executions,
                tables: tables.words,
                out_of_bounds,
                children,
            })
        }
        Mapped::Render(render) => {
            let companion = companion_source(render)
                .map(|source| source.bitcode())
                .transpose()
                .map_err(|error| format!("the generated companion's AIR: {error:?}"))?;
            // The state of each template its argument buffers name.
            let templates = template_states(device, prepared)?;
            let inputs = RenderInputs {
                case: &primary.case,
                render,
                bitcode: &bitcode,
                companion: companion.as_deref(),
                linked: &prepared.linked,
                texture_usage: &usage,
                interface: &primary.interface,
                imageblock: primary.reflection.fragment_imageblock.as_ref(),
                templates: &templates,
            };
            let encoded = encoded_render(device, &inputs).map_err(|error| stop(error, false))?;
            // How the draw's vertex module contains each out-of-bounds target on this device,
            // from what the draw executed, before anything runs: a target the device leaves
            // uncontained refuses the case uncommitted, recording what the draw compiled.
            let [execution] = encoded.executions.as_slice() else {
                return Err(format!(
                    "the case's one draw recorded {} executions",
                    encoded.executions.len()
                )
                .into());
            };
            let out_of_bounds = opened.decided(
                execution,
                &encoded.tables,
                draw_out_of_bounds(opened, render, execution),
            )?;
            // Each command buffer the draw executes, encoded before anything is committed and
            // committed after the draw on its queue.
            let executions = encoded
                .executions(render, &templates)
                .map_err(|error| stop(error, false))?;
            let committed = encoded.commit().map_err(|error| stop(error, false))?;
            for execution in &executions {
                execution
                    .commit()
                    .map_err(|error| stop(CaseError::runtime("commit")(error), true))?;
            }
            let status = completed(prepared, &committed.command_buffer, &committed.executions)?;
            // The commands the draw left, read back, then what each execution ran.
            committed
                .check_commands()
                .map_err(|error| stop(error, true))?;
            let mut children = Vec::new();
            for execution in &executions {
                completed(prepared, execution, &execution.executions())?;
                children.extend(execution.executions());
            }
            let output = committed
                .output(&inputs)
                .map_err(|error| stop(error, true))?;
            // What the draw changed around each vertex and index buffer target, read once it
            // completed.
            let guards = committed
                .guard_counts()
                .map_err(|error| stop(error, true))?;
            let out_of_bounds =
                out_of_bounds.map(|run| with_escaped(run, &guards, Placement::Heap));
            Ok(Ran {
                output,
                status,
                executions: committed.executions,
                tables: committed.tables,
                out_of_bounds,
                children,
            })
        }
        Mapped::Mesh(mesh) => {
            let companion = mesh
                .companion
                .as_ref()
                .map(|generated| {
                    RowSource::Text(Cow::Owned(generated.air_text(Spelling::Opaque))).bitcode()
                })
                .transpose()
                .map_err(|error| format!("the generated companion's AIR: {error:?}"))?;
            let inputs = MeshInputs {
                case: &primary.case,
                mesh,
                bitcode: &bitcode,
                companion: companion.as_deref(),
                texture_usage: &usage,
                interface: &primary.interface,
            };
            let encoded = encoded_mesh(device, &inputs).map_err(|error| stop(error, false))?;
            let committed = encoded.commit().map_err(|error| stop(error, false))?;
            let status = completed(prepared, &committed.command_buffer, &committed.executions)?;
            let output = committed
                .mesh_output(mesh)
                .map_err(|error| stop(error, true))?;
            Ok(Ran {
                output,
                status,
                executions: committed.executions,
                tables: committed.tables,
                out_of_bounds: None,
                children: Vec::new(),
            })
        }
    }
}

/// How the module `compute`'s one dispatch executed (`execution`, its encoded command buffer's)
/// contains each of the case's out-of-bounds targets on `opened`'s device, as the executor's own
/// context decides it (`engine::out_of_bounds_run`: the module's slots, stage-input fetches
/// and strategies, the device's profile and execution class), or `None` for a case without
/// targets: the facts before what the targets' guards count (`guards::with_escaped`).
fn kernel_out_of_bounds(
    opened: &Opened,
    compute: &ComputeCase,
    execution: &Execution,
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    if compute.out_of_bounds.is_empty() {
        return Ok(None);
    }
    let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
        return Err("the case's dispatch executed a render pipeline".into());
    };
    let attributes = out_of_bounds::Fetch::of_stage_input(compute.stage_input.as_ref());
    out_of_bounds_run(
        Some(opened.facts.execution_class()),
        &opened.profile,
        &out_of_bounds::ModuleFacts {
            stage: ShaderStage::Kernel,
            slots: &pipeline.interface.slots,
            attributes: &attributes,
            resident_roots: pipeline.interface.needs_pointer_translation,
            handle_table: pipeline.interface.needs_handle_translation,
            strategies: &pipeline.stage.strategies,
        },
        &compute.out_of_bounds,
    )
}

/// The pipeline state of each pipeline token `prepared`'s case names: each pipeline state's an
/// argument buffer holds and each execution's parent pipeline's, made from its template's AIR as
/// the template's own case makes it (its constant values, the functions it links), with
/// `support_indirect_command_buffers`, as the Metal oracle makes the template states its commands
/// name: a kernel template's compute state, a vertex or fragment template's render state of the
/// oracle's descriptor, as the render runner makes its own case's
/// (`runtime_case::render_pipeline_descriptor`), and an object or mesh template's mesh render
/// state, as the mesh runner makes its own case's (`runtime_case::mesh_pipeline_state`).
fn template_states(
    device: &Device,
    prepared: &PreparedCase,
) -> Result<BTreeMap<u64, TemplateState>, Stop> {
    let checked = &prepared.checked;
    let case = &checked.primary.case;
    let parents = case
        .command_buffers
        .iter()
        .filter_map(|resource| resource.execute.as_ref()?.pipeline.as_ref())
        .map(|template| (template.token, &template.case_id));
    let mut states = BTreeMap::new();
    for (token, case_id) in case
        .pipeline_states
        .iter()
        .map(|state| (state.token, &state.case_id))
        .chain(parents)
    {
        if states.contains_key(&token) {
            continue;
        }
        let template = checked
            .pipeline_templates
            .get(case_id)
            .ok_or_else(|| format!("pipeline template {case_id} was not checked"))?;
        let bitcode = RowSource::of_row(&template.source)
            .bitcode()
            .map_err(|error| format!("pipeline template {case_id}'s AIR: {error:?}"))?;
        let linked = linked_modules(&template.linked_functions)?;
        let state = match template.case.stage {
            Stage::Kernel => {
                let mapped = compute_case(&template.case, &template.interface)
                    .map_err(|refused| Stop::Error(refused.into()))?;
                (|| {
                    let function = function(device, &bitcode, &mapped)?;
                    let linked = linked_functions(device, &linked, &mapped.constants)?;
                    let mut descriptor = compute_descriptor(&function, linked.functions, &mapped);
                    descriptor.support_indirect_command_buffers = true;
                    device
                        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
                        .map(TemplateState::Compute)
                        .map_err(CaseError::runtime(format!(
                            "pipeline template {case_id}: new_compute_pipeline_state"
                        )))
                })()
            }
            Stage::Vertex | Stage::Fragment => {
                let render = graphics::render_case(
                    &template.case,
                    &template.interface,
                    template.entry_ll(),
                    template.companion.as_ref(),
                    template.reflection.fragment_imageblock.as_ref(),
                )?;
                let companion = companion_source(&render)
                    .map(|source| source.bitcode())
                    .transpose()
                    .map_err(|error| {
                        format!("pipeline template {case_id}'s generated companion: {error:?}")
                    })?;
                let usage = texture_usage(&template.case, &template.interface)?;
                let inputs = RenderInputs {
                    case: &template.case,
                    render: &render,
                    bitcode: &bitcode,
                    companion: companion.as_deref(),
                    linked: &linked,
                    texture_usage: &usage,
                    interface: &template.interface,
                    imageblock: template.reflection.fragment_imageblock.as_ref(),
                    templates: &BTreeMap::new(),
                };
                (|| {
                    let mut descriptor = render_pipeline_descriptor(device, &inputs)?;
                    descriptor.support_indirect_command_buffers = true;
                    device
                        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                        .map(TemplateState::Render)
                        .map_err(CaseError::runtime(format!(
                            "pipeline template {case_id}: new_render_pipeline_state"
                        )))
                })()
            }
            Stage::Object | Stage::Mesh => {
                let mesh = super::mesh::mesh_case(&template.case, template.companion.as_ref())
                    .map_err(Stop::Error)?;
                let companion = mesh
                    .companion
                    .as_ref()
                    .map(|generated| {
                        RowSource::Text(Cow::Owned(generated.air_text(Spelling::Opaque))).bitcode()
                    })
                    .transpose()
                    .map_err(|error| {
                        format!("pipeline template {case_id}'s generated companion: {error:?}")
                    })?;
                mesh_pipeline_state(device, &mesh, (&bitcode, companion.as_deref()), true)
                    .map(TemplateState::Render)
            }
        }
        .map_err(|error| stop(error, false))?;
        states.insert(token, state);
    }
    Ok(states)
}

/// The function of kernel case `compute`'s entry, made with its constant values, from a library
/// of `bitcode`, its AIR, on `device`.
fn function(device: &Device, bitcode: &[u8], compute: &ComputeCase) -> Result<Function, CaseError> {
    device
        .new_library_with_air(bitcode)
        .map_err(CaseError::runtime("new_library_with_air"))?
        .new_function_with_name_constant_values(&compute.entry, &compute.constants)
        .map_err(CaseError::runtime("new_function_with_name_constant_values"))
}

/// How the render pipeline `render`'s one draw executed (`execution`, its encoded command
/// buffer's) contains each of the case's out-of-bounds targets on `opened`'s device, as the
/// executor's own context decides it (`engine::draw_out_of_bounds_run`: the module that runs
/// the vertex function, its attributes' fetches, the device's profile and execution class), or
/// `None` for a case without targets: the facts before what the targets' guards count
/// (`guards::with_escaped`). The draw contains its index positions past an index buffer target as
/// that containment states (its execution record's `index_padding`), or the run fails, as the
/// executor's context fails its draw (`engine::index_padding_agrees`).
fn draw_out_of_bounds(
    opened: &Opened,
    render: &RenderCase,
    execution: &Execution,
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    if render.out_of_bounds.is_empty() {
        return Ok(None);
    }
    let ExecutedPipeline::Render(pipeline) = execution.pipeline() else {
        return Err("the case's draw executed a compute pipeline".into());
    };
    let run = draw_out_of_bounds_run(
        Some(opened.facts.execution_class()),
        &opened.profile,
        pipeline,
        render,
    )?;
    index_padding_agrees(run.as_ref(), execution.index_padding())?;
    Ok(run)
}

/// What a device restricted to a profile translates a case for.
enum Translation {
    /// The profile: the device's kept members (`profiles::KEPT`, read as the device's own through
    /// the layer) are the profile's values.
    Profile,
    /// The profile's members but the kept ones, `kept`, which are the device's: the case is a
    /// patch draw the profile emulates ([`emulated_patches`]), which the runtime runs on them.
    Kept(Vec<String>),
    /// The same, for any other case, which the bridge sends to the executor's context: the gap.
    Bridged(String),
}

/// What a device restricted to a profile translates `mapped` for: the runtime compiles for the
/// description it read through the profiles layer, whose kept members (`profiles::KEPT`) are the
/// device's own, where the executor's context translates for the profile's (`check_restricted`).
/// [`Translation::Profile`] where the two make the same profile (the profile states the device's
/// own values there), so the runtime translates for the profile; else the bridge sends the case to
/// the executor's context (item 3), whose restricted row means what it always has, but for a patch
/// draw the profile emulates, whose emulation that context does not run: the runtime runs it on the
/// kept members, which its row records.
fn translation(
    device: &Device,
    restricted: &RestrictedDevice,
    mapped: &Mapped,
) -> Result<Translation, String> {
    let translated = restricted
        .translated
        .clone()
        .with_enabled(device.enabled())
        .and_then(|description| {
            metal2vulkan::device::Device::new(description, crate::candidate::requested_policy())
        })
        .map_err(|error| format!("the profile's description on the enabled set: {error}"))?;
    if translated.profile().fingerprint() == device.compile_device().profile().fingerprint() {
        return Ok(Translation::Profile);
    }
    if emulated_patches(mapped, translated.profile()) {
        return Ok(Translation::Kept(restricted.kept.clone()));
    }
    Ok(Translation::Bridged(format!(
        "profile restriction: the runtime compiles for the device's kept members ({}), not the \
         profile's",
        restricted.kept.join(", ")
    )))
}

/// Whether `mapped` is a patch draw (`graphics::Assembly::Patches`) that `profile`, without
/// `tessellationShader`, compiles emulated (`metal2vulkan`'s `lower::tessellation`): its pipeline
/// binds the emulation's work and records, which the runtime makes and the executor's context
/// does not (it refuses their descriptor source, `NotImplemented`).
fn emulated_patches(mapped: &Mapped, profile: &metal2vulkan::device::DeviceProfile) -> bool {
    let Mapped::Render(render) = mapped else {
        return false;
    };
    matches!(render.assembly, graphics::Assembly::Patches { .. })
        && !profile
            .enabled()
            .features
            .contains(&VkFeature::TessellationShader)
}

/// Wait for `command_buffer`, which runs `prepared`'s `executions`, bounded under the hang
/// protocol (its hang record names the case and the SPIR-V the first execution ran), then its
/// execution report's verdict.
fn completed(
    prepared: &PreparedCase,
    command_buffer: &CommandBuffer,
    executions: &[Execution],
) -> Result<metal2vulkan::bind::StatusReport, Stop> {
    let module_sha256 = executions
        .first()
        .map(|execution| match execution.pipeline() {
            ExecutedPipeline::Compute(pipeline) => {
                spirv_sha256(std::slice::from_ref(&pipeline.stage))
            }
            ExecutedPipeline::Render(pipeline) => spirv_sha256(&pipeline.stages),
        })
        .unwrap_or_default();
    let label = SubmissionLabel {
        case_id: prepared.checked.primary.case.case_id.clone(),
        module_sha256,
    };
    let waited = with_submission_label(label, || {
        crate::gpu::wait::runtime::wait_for_command_buffer(
            command_buffer,
            SubmissionBackend::vulkan(prepared.device),
            "the runtime's command buffer",
        )
    });
    // A read of an argument the function constants removed completes the command buffer, its
    // report stating the read: `report_verdict` gives the translation's typed refusal.
    waited?;
    let report = command_buffer
        .execution_report()
        .ok_or("the completed command buffer has no execution report")?;
    Ok(report_verdict(report)?)
}

/// The members a row or refusal of the executor's own context records of the runtime path:
/// that the executor ran it, and why the runtime did not: the prerequisites its family waits for,
/// or the gap the runtime raised before it committed (the bridge).
pub fn executor_members(why: &ExecutorRun) -> [(&'static str, serde_json::Value); 2] {
    let reason = match why {
        ExecutorRun::WaitsFor(prerequisites) => (
            "runtime_waits_for",
            serde_json::json!(prerequisites
                .iter()
                .map(|prerequisite| prerequisite.name())
                .collect::<Vec<_>>()),
        ),
        ExecutorRun::Gap(gap) => ("runtime_gap", serde_json::json!(gap)),
    };
    [("executed_by", serde_json::json!("executor")), reason]
}

/// Why the executor's own context ran a case.
pub enum ExecutorRun {
    /// The runtime path waits for the prerequisites of the case's family.
    WaitsFor(BTreeSet<Prerequisite>),
    /// The runtime raised a typed gap before it committed (the bridge).
    Gap(String),
}

#[cfg(test)]
mod tests;
