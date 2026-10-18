//! The Vulkan objects of one compiled variant of a pipeline state: its shader modules (one per
//! stage), immutable samplers, descriptor-set layouts and pipeline layout, taken when the variant
//! compiles; a `VkPipeline` per region specialization of a compute variant, or per graphics key
//! of a render variant (`graphics.rs`), made when a command first needs it; a command's set 0
//! from its command buffer's pools (`descriptor_arena.rs`); and its hold on the device's heap set
//! of its set-1 layout (`heap_sets.rs`), taken at its first command.
//!
//! Each object owns its device handle and is destroyed when its last holder drops it: a command
//! buffer retains what its commands use until it completes. A shader module is the device's,
//! shared by every variant whose stage has its exact SPIR-V (`content_keyed.rs`); so are the
//! immutable samplers, by their descriptions (`sampler_set`), and the set layouts and pipeline
//! layout, by their exact definition (`shared_layouts.rs`). Only the device's maps destroy those:
//! a variant's are shares.

use super::content_keyed::{program_call, ContentKeyed, Owned};
use super::graphics::Notices;
use super::kept::{Compiled, Instance, KeptObjects, KeptPipeline, Place};
use super::libraries::Linked;
use super::shared_layouts::{settle_layouts, shared_layout, SharedLayout};
use super::RegionKey;
use crate::binding::{Info, SetWrite};
use crate::degradation::{
    tested_slots_of, BoundFormats, BoundNotice, CompiledPipeline, Kinds, Occurrence, TestedFormats,
};
use crate::descriptor_arena::{DescriptorArena, SetNeeds};
use crate::device::{lock, DeviceShared};
use crate::error::{internal, vulkan, Error};
use crate::exit_guard;
use crate::generated_commands::INDIRECT_BINDABLE;
use crate::heap_sets::{self, HeapSet};
use crate::release;
use crate::shader_cache::driver::Making;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::DispatchRegion;
use metal2vulkan::cache::{Hash256, SpecializationKey};
use metal2vulkan::pipeline::{
    DescriptorSetLayout, PipelineLayout, RenderPipeline, SlotRef, StageModule,
};
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// What a compute `VkPipeline` is made of: a stage's module, and the specialization, stage
/// flags and required subgroup size it is created under, with where its specialized notices
/// are announced (`Device::degrade_specialized`; by the command that makes it, naming what it binds
/// where it knows that, or by the dispatch that handed it to the pool).
struct ComputeStage<'a> {
    object: &'a StageObject,
    specialization: &'a metal2vulkan::cache::SpecializationKey,
    stage_flags: vk::PipelineShaderStageCreateFlags,
    required_subgroup_size: Option<u32>,
    notices: Notices<'a>,
}

/// One stage's shader module and entry point.
#[derive(Clone)]
pub(crate) struct StageObject {
    pub(super) stage: vk::ShaderStageFlags,
    pub(super) module: Arc<ShaderModule>,
    pub(super) entry: CString,
}

/// The immutable samplers of a layout, one per binding that bakes one, in the layout's order of
/// sets and bindings: the device's set of those samplers, shared by every variant instance whose
/// layout bakes the same ones ([`SamplerSets`]), or a host step's own. Counted against the
/// device's limit from their creation. It keeps no device functions and destroys nothing itself:
/// only the device's map destroys it, through the device, its count given back, where a program's
/// call settles the map once nothing else holds it, at a sweep for a host step's own, or at the
/// device's teardown (`content_keyed.rs`).
pub(crate) struct SamplerSet {
    samplers: Vec<vk::Sampler>,
}

impl Owned<DeviceShared> for SamplerSet {
    fn destroy(&self, shared: &DeviceShared) {
        self.release(shared);
        if !self.samplers.is_empty() {
            shared.counts.sampler_set_destroyed();
        }
    }
}

impl SamplerSet {
    /// Its samplers, one per binding that bakes one, in its layout's order.
    pub(super) fn samplers(&self) -> &[vk::Sampler] {
        &self.samplers
    }

    /// Destroy its samplers, giving their count back: the set's destruction, or a set whose
    /// creation failed part way (never made, so never counted destroyed).
    fn release(&self, shared: &DeviceShared) {
        for &sampler in &self.samplers {
            // SAFETY: the map lets a set go only when nothing else shares it, so no layout uses
            // it (a layout's objects hold the set until they destroyed the layout) and no pending
            // command does (a command buffer retains the objects that hold it until it
            // completes); or at the device's teardown, when no variant and no command buffer is
            // left (each holds the device).
            if unsafe { release::object(shared.vk(), sampler) } {
                shared.sampler_destroyed();
            }
        }
    }
}

/// The device's immutable sampler sets by the descriptors of their samplers
/// (`DeviceShared::sampler_sets`).
pub(crate) type SamplerSets = ContentKeyed<Vec<metal2vulkan::vk::SamplerDesc>, SamplerSet>;

/// The device's set of the samplers `descs` describes, made now where it holds none, and whether
/// it was made. In a program's call (`content_keyed::program_call`) a set made here enters the
/// device's map; a host step, a pool piece or another runtime thread only looks it up, and on a
/// miss makes a set of its own that no lookup finds, gone at a later sweep once nothing else holds
/// it: each call here sweeps. A layout that bakes no sampler takes an empty set, counted nowhere
/// and never in the map.
pub(crate) fn sampler_set(
    device: &Device,
    descs: Vec<metal2vulkan::vk::SamplerDesc>,
) -> Result<(Arc<SamplerSet>, bool), Error> {
    if descs.is_empty() {
        let empty = SamplerSet {
            samplers: Vec::new(),
        };
        return Ok((Arc::new(empty), false));
    }
    let shared = device.shared();
    let program = program_call();
    // The lock is held across the creation, so two of the program's calls never make two sets of
    // the same samplers.
    let mut sets = lock(&shared.sampler_sets);
    let made = sets.get_or_make(descs.clone(), program, || {
        let mut set = SamplerSet {
            samplers: Vec::with_capacity(descs.len()),
        };
        for desc in &descs {
            let mut reduction = vk::SamplerReductionModeCreateInfo::default();
            let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
            let info =
                metal2vulkan::ash_interop::sampler_create_info(desc, &mut reduction, &mut border);
            // SAFETY: `info` and its chain are live for the call. Counted against the device's
            // limit (`DeviceShared::create_sampler`).
            match unsafe { shared.create_sampler(&info) } {
                Ok(sampler) => set.samplers.push(sampler),
                Err(error) => {
                    // Those made go back, and their count with them: no holder has the set.
                    set.release(shared);
                    return Err(error);
                }
            }
        }
        Ok(set)
    });
    let gone = sets.sweep();
    drop(sets);
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
    let (set, made) = made?;
    match made {
        true => shared.counts.sampler_set_made(),
        false => shared.counts.sampler_set_shared(),
    }
    Ok((set, made))
}

/// Let go of the device's sampler sets no variant holds any more (`content_keyed.rs`): in a
/// program's call only, where the kept variants applied their queue.
fn settle_sampler_sets(device: &Device) {
    debug_assert!(
        !super::pool::in_piece(),
        "the device's sampler sets settled in a pipeline pool's piece"
    );
    // A host step's variant settles nothing, as it takes nothing back (`Kept::take`).
    if !program_call() {
        return;
    }
    let shared = device.shared();
    let gone = lock(&shared.sampler_sets).settle();
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
}

impl StageObject {
    /// The stage's `VkShaderModule`.
    pub(super) fn module(&self) -> vk::ShaderModule {
        self.module.raw
    }
}

/// One `VkShaderModule`: the device's module of its SPIR-V, shared by every variant whose stage
/// has those exact words ([`ShaderModules`]), or a host step's own. It keeps no device functions
/// and has no destroying `Drop`: a holder's `Arc` is a share, and only the device's map destroys
/// it ([`Owned::destroy`]), once nothing else shares it or at the device's teardown
/// (`content_keyed.rs`), so a holder that outlives the device's teardown (a variant whose last
/// device handle goes before its stages) calls nothing. On MoltenVK a module keeps the libraries
/// its SPIR-V converted to, so every variant and state that shares it converts and compiles its
/// MSL once.
pub(crate) struct ShaderModule {
    raw: vk::ShaderModule,
}

impl Owned<DeviceShared> for ShaderModule {
    fn destroy(&self, shared: &DeviceShared) {
        // SAFETY: the map lets a module go only when nothing else shares it, so no pipeline is
        // being made from it (a making holds its variant's objects) and no pending command needs
        // it (a pipeline does not need its module once made); or at the device's teardown, when
        // no variant and no command buffer is left (each holds the device).
        unsafe { release::object(shared.vk(), self.raw) };
    }
}

/// The device's shader modules by the SHA-256 of their exact SPIR-V ([`words_digest`],
/// `DeviceShared::shader_modules`).
pub(crate) type ShaderModules = ContentKeyed<Hash256, ShaderModule>;

/// The SHA-256 of SPIR-V `words` (their length, then their bytes): what names a stage's module in
/// the device's map and its words in its makings' records, taken once per stage of a variant.
pub(crate) fn words_digest(words: &[u32]) -> Hash256 {
    Making::new("words").words(words).finish()
}

/// The digest of what every `VkPipeline` of a variant of `layout` and `stages` reads besides its
/// making's own state, which its makings' records are named by (`shader_cache/driver.rs`): each
/// stage's stage, entry point, stage flags, required subgroup size, specialization layout and
/// SPIR-V words (By their digest, `words`, in stage order), and the layout, its immutable samplers
/// among it.
fn variant_content(layout: &PipelineLayout, stages: &[StageModule], words: &[Hash256]) -> Hash256 {
    let mut making = Making::new("variant").text(layout);
    for (stage, words) in stages.iter().zip(words) {
        making = making
            .text(&(
                stage.stage,
                &stage.entry_point,
                stage.stage_flags,
                stage.required_subgroup_size,
                &stage.specialization,
            ))
            .digest(words);
    }
    making.finish()
}

impl VariantObjects {
    /// The description of a making of `kind` of one of its `VkPipeline`s, begun with its
    /// content ([`variant_content`]); the making adds its own state.
    pub(super) fn making(&self, kind: &str) -> Making {
        Making::new(kind).digest(&self.content)
    }

    /// The flight of a making of one of its `VkPipeline`s whose own part of the driver's compile
    /// is `own` (its specialization, a region's stage): its content ([`variant_content`]) and
    /// that, which a twin's making shares (`flight.rs`).
    pub(super) fn flight(&self, own: &dyn std::fmt::Debug) -> super::flight::FlightKey {
        Making::new("flight")
            .digest(&self.content)
            .text(own)
            .finish()
    }

    /// The digests of `pipeline`'s (its compiled pipeline's) layout and stages
    /// ([`PartsContent`]), taken once.
    pub(super) fn parts_content(&self, pipeline: &RenderPipeline) -> &PartsContent {
        self.parts_content.get_or_init(|| PartsContent {
            layout: Making::new("layout").text(&pipeline.layout).finish(),
            stages: pipeline
                .stages
                .iter()
                .zip(self.words.iter())
                .map(|(stage, words)| {
                    Making::new("stage")
                        .text(&(stage.stage, &stage.entry_point, &stage.specialization))
                        .digest(words)
                        .finish()
                })
                .collect(),
        })
    }
}

/// The device's module of `spirv`, made now where it holds none. In a program's call
/// (`content_keyed::program_call`) a module made here enters the device's map; a host step or a
/// pool piece only looks it up, and on a miss makes a module of its own that no lookup finds, so
/// the map's lookups change only in the program's calls. Such a module of its own goes at a later
/// sweep, once nothing else holds it (`content_keyed.rs`, the exception): each call here sweeps.
pub(crate) fn shader_module(
    device: &Device,
    spirv: &[u32],
    digest: &Hash256,
) -> Result<Arc<ShaderModule>, Error> {
    let shared = device.shared();
    let program = program_call();
    // The lock is held across the creation, so two of the program's calls never make two
    // modules of the same words; `vkCreateShaderModule` copies the words and compiles nothing.
    let mut modules = lock(&shared.shader_modules);
    let made = modules.get_or_make(*digest, program, || {
        let info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the code is live SPIR-V words for the call.
        let raw = unsafe { shared.vk().create_shader_module(&info, None) }.map_err(vulkan)?;
        Ok::<_, Error>(ShaderModule { raw })
    });
    let gone = modules.sweep();
    drop(modules);
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
    let (module, made) = made?;
    match made {
        true => shared.counts.shader_module_made(),
        false => shared.counts.shader_module_shared(),
    }
    Ok(module)
}

/// Let go of the device's modules no variant holds any more (`content_keyed.rs`): on a
/// program's thread only, where the kept variants applied their queue.
fn settle_modules(device: &Device) {
    debug_assert!(
        !super::pool::in_piece(),
        "the device's shader modules settled in a pipeline pool's piece"
    );
    // A host step's variant settles nothing, as it takes nothing back (`Kept::take`).
    if !program_call() {
        return;
    }
    let shared = device.shared();
    let gone = lock(&shared.shader_modules).settle();
    // Destroyed here, the map's lock released.
    gone.destroy(shared);
}

/// Let go of the device's graphics pipeline libraries no pipeline holds any more
/// (`libraries.rs`): in a program's call only, as for its modules.
fn settle_libraries(device: &Device) {
    if !program_call() {
        return;
    }
    let shared = device.shared();
    if let Some(libraries) = &shared.libraries {
        libraries.settle(shared);
    }
}

/// The layouts and shader modules of one variant.
pub(crate) struct VariantObjects {
    pub(super) device: Device,
    /// The stages' modules, in the pipeline's stage order.
    pub(super) stages: Vec<StageObject>,
    samplers: Arc<SamplerSet>,
    /// Its set layouts and pipeline layout: the device's of their definition
    /// (`shared_layouts.rs`).
    layout: Arc<SharedLayout>,
    /// What one set of each layout set takes from a pool, by set number.
    set_needs: Vec<SetNeeds>,
    /// Whether its draws push set 0 rather than allocate it (`pushes_set_zero`).
    push_set_zero: bool,
    /// The shape of its set 0 and push-constant ranges ([`set_zero_shape`]), which a draw's
    /// record compares with the last push's (`render/bound.rs`).
    set_zero_shape: u64,
    /// Where they do, a write of each element of each set-0 sampler binding that bakes its sampler
    /// (`immutable_pushes`), pushed with every draw's writes.
    immutable_pushes: Vec<SetWrite>,
    /// The stages with notes a specialization decides, announced as each `VkPipeline` of the
    /// variant is made (`Device::degrade_specialized`).
    pub(super) specialized: Vec<StageModule>,
    /// The kinds its making may announce, which a bind tests (`announce_bound`).
    kinds: Kinds,
    /// The texture slots its specialized notes test (`degradation::tested_slots_of`): those
    /// whose bound formats a bind's notices may name.
    tested: Vec<SlotRef>,
    /// The notices its last bind announced at every occurrence, and what they were made for
    /// (`announce_bound`).
    bound: Mutex<Option<BoundNotices>>,
    /// The layout's set 1, the heap, where it has one.
    heap_layout: Option<DescriptorSetLayout>,
    /// The device's heap set of that layout, from the variant's first command that binds it
    /// (`heap_sets.rs`).
    heap: Mutex<Option<Arc<HeapSet>>>,
    /// The variant instance its objects are, where its last release keeps them for a variant of
    /// the same compiled pipeline made again (`kept.rs`); `None` destroys them.
    keep: Option<Instance>,
    /// The digest its makings' records are named by, with each making's own state
    /// ([`variant_content`]).
    content: Hash256,
    /// Each stage's words' digest ([`words_digest`]), in stage order: taken once, for its
    /// modules, its content and its parts' content.
    words: Arc<[Hash256]>,
    /// The digests its graphics pipeline libraries are named by (`graphics.rs`), from its first
    /// making linked from libraries.
    parts_content: OnceLock<PartsContent>,
}

/// The digests of a variant's layout and of each of its stages (its stage, entry point,
/// specialization layout and words), which name the libraries of its pipelines' parts
/// (`GraphicsContent::names`).
pub(crate) struct PartsContent {
    pub(crate) layout: Hash256,
    pub(crate) stages: Vec<Hash256>,
}

/// The notices a bind of a variant announced at every occurrence
/// (`VariantObjects::announce_bound`), and the specializations and tested formats they were made
/// for.
struct BoundNotices {
    made_for: Vec<(SpecializationKey, Option<TestedFormats>)>,
    notices: Arc<[BoundNotice]>,
}

impl BoundNotices {
    /// Whether the notices were made for `specializations`: the same specializations in the
    /// same order, each with the same formats at the variant's tested slots, or with none known
    /// where none were.
    fn made_for(&self, specializations: &[(&SpecializationKey, Option<BoundFormats<'_>>)]) -> bool {
        self.made_for.len() == specializations.len()
            && self
                .made_for
                .iter()
                .zip(specializations)
                .all(|((made, tested), (key, formats))| {
                    made == *key
                        && match (tested, formats) {
                            (None, None) => true,
                            (Some(tested), Some(formats)) => {
                                tested.iter().all(|&(slot, format)| formats(slot) == format)
                            }
                            _ => false,
                        }
                })
    }
}

impl Drop for VariantObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // A kept instance's objects go to the device's kept variants. The objects own no Vulkan
        // object of their own: their shares of the device's modules, layouts and samplers go
        // with them, and only the device's maps destroy those (`content_keyed.rs`).
        let Some(instance) = self.keep else {
            return;
        };
        let objects = KeptObjects {
            stages: self.stages.clone(),
            samplers: Arc::clone(&self.samplers),
            layout: Arc::clone(&self.layout),
            content: self.content,
            words: Arc::clone(&self.words),
        };
        lock(&self.device.shared().kept).objects_released(instance, objects);
    }
}

/// One `VkPipeline` of a variant: the one a compute variant's regions of one specialization
/// share, or a render variant's of one graphics key.
pub(crate) struct RawPipeline {
    pub(super) device: Device,
    pub(crate) raw: vk::Pipeline,
    /// How long the driver took to create it (an execution record's, `execution.rs`); zero for
    /// one taken back from a released variant (`kept.rs`).
    pub(crate) created_in: std::time::Duration,
    /// The variant instance whose layout made it and its place there, where its last release
    /// keeps it for that instance (`kept.rs`); `None` destroys it.
    pub(super) keep: Option<(Instance, Place)>,
    /// Made at its state's creation, its specialized notices not announced yet: the first draw
    /// that takes it announces them, with what it binds (`RenderVariant::first_use`).
    pub(super) unannounced: AtomicBool,
    /// The libraries it was linked from, held while it lives (`libraries.rs`); `None` for a
    /// pipeline made whole.
    pub(super) linked: Option<Linked>,
}

impl Drop for RawPipeline {
    fn drop(&mut self) {
        // As for `VariantObjects`.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        if let Some((instance, place)) = self.keep.clone() {
            let shared = self.device.shared();
            let kept = KeptPipeline {
                place,
                raw: self.raw,
                unannounced: self.unannounced.load(Ordering::Acquire),
                linked: self.linked.clone(),
            };
            lock(&shared.kept).pipeline_released(instance, kept);
            return;
        }
        // SAFETY: as for `VariantObjects`.
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

impl RawPipeline {
    /// `kept`, a `VkPipeline` of `instance` a released variant kept, taken back.
    pub(crate) fn taken_back(
        device: &Device,
        kept: KeptPipeline,
        instance: Instance,
    ) -> RawPipeline {
        RawPipeline {
            device: device.clone(),
            raw: kept.raw,
            created_in: std::time::Duration::ZERO,
            keep: Some((instance, kept.place)),
            unannounced: AtomicBool::new(kept.unannounced),
            linked: kept.linked,
        }
    }

    /// Whether its specialized notices wait for this, its first draw: once true, then false.
    pub(crate) fn first_drawn(&self) -> bool {
        self.unannounced.swap(false, Ordering::AcqRel)
    }

    /// The pipeline, kept at `place` of `objects`' instance once released, where `objects` are
    /// kept (`kept.rs`).
    pub(crate) fn kept_at(mut self, objects: &VariantObjects, place: Place) -> RawPipeline {
        self.keep = objects.keep.map(|instance| (instance, place));
        self
    }
}

impl VariantObjects {
    /// The shader modules of `stages` and the layouts of `layout`, a compiled pipeline's. Sets
    /// are numbered by position (the layout's set `n` is at index `n`), which the compile layer
    /// guarantees. Once they are made, the device announces what the stages record: their
    /// precision notes (but those a specialization decides, announced with each `VkPipeline`), and
    /// reconvergence that is not Metal's (`degradation.rs`), but the kinds it announces at every
    /// occurrence, which each bind announces (`announce_bound`); what the variant's `pipeline`
    /// records beside its stages counts among its kinds (`Device::degrade_pipeline`).
    pub(crate) fn create(
        device: &Device,
        layout: &PipelineLayout,
        stages: &[StageModule],
        pipeline: CompiledPipeline<'_>,
    ) -> Result<VariantObjects, Error> {
        let push_set_zero = pushes_set_zero(device, layout, stages, pipeline);
        let set_needs = needs_of(layout, push_set_zero)?;
        // One sampler per binding that bakes one, in the layout's order: the live set of the same
        // samplers where it holds one, else one made now (`sampler_set`).
        let descs: Vec<metal2vulkan::vk::SamplerDesc> = layout
            .sets
            .iter()
            .flat_map(|set| &set.bindings)
            .filter_map(|binding| binding.immutable_sampler)
            .collect();
        let (samplers, made) = sampler_set(device, descs)?;
        if made && exit_guard::in_host_step() {
            // A host step's variant whose samplers the device's map holds no set of: counted
            // where the step runs (the open break).
            device.shared().counts.host_step_sampler_set_made();
        }
        let words: Arc<[Hash256]> = stages
            .iter()
            .map(|stage| words_digest(&stage.spirv))
            .collect();
        let shared_layout = shared_layout(device, layout, &samplers, push_set_zero)?;
        let mut objects = VariantObjects {
            device: device.clone(),
            stages: Vec::with_capacity(stages.len()),
            samplers,
            layout: shared_layout,
            set_needs,
            push_set_zero,
            set_zero_shape: set_zero_shape(layout, push_set_zero),
            immutable_pushes: Vec::new(),
            specialized: stages
                .iter()
                .filter(|s| !s.specialized_notes.is_empty())
                .cloned()
                .collect(),
            heap_layout: layout.sets.get(1).cloned(),
            heap: Mutex::new(None),
            kinds: Kinds::of(stages, pipeline),
            tested: tested_slots_of(stages),
            bound: Mutex::new(None),
            keep: None,
            content: variant_content(layout, stages, &words),
            words: Arc::clone(&words),
            parts_content: OnceLock::new(),
        };
        for (stage, words) in stages.iter().zip(words.iter()) {
            let entry = CString::new(stage.entry_point.as_str())
                .map_err(|_| internal("an entry point name holds NUL"))?;
            let module = shader_module(device, &stage.spirv, words)?;
            objects.stages.push(StageObject {
                stage: stage.stage.into(),
                module,
                entry,
            });
        }
        if push_set_zero {
            objects.immutable_pushes = immutable_pushes(layout, &objects.samplers.samplers)?;
        }
        device.degrade_stages(stages, Occurrence::Made);
        Ok(objects)
    }

    /// [`create`](Self::create)'s objects of a variant of `compiled`, kept when released: those
    /// of a released instance of it taken back from the device's kept variants with the
    /// `VkPipeline`s that instance made (each at its place), else made now as a new instance
    /// (`kept.rs`).
    pub(crate) fn kept_or_create(
        device: &Device,
        compiled: Compiled,
        layout: &PipelineLayout,
        stages: &[StageModule],
        pipeline: CompiledPipeline<'_>,
    ) -> Result<(VariantObjects, Vec<(Place, RawPipeline)>), Error> {
        let push_set_zero = pushes_set_zero(device, layout, stages, pipeline);
        let set_needs = needs_of(layout, push_set_zero)?;
        let shared = device.shared();
        let back = lock(&shared.kept).take(shared, compiled);
        // The kept variants' queue applied (an evicted instance's modules let go): the device's
        // modules nothing holds now go, in the program's order.
        settle_modules(device);
        // And the device's pipeline layouts nothing holds now, then the sampler sets nothing
        // holds now, the evicted instances' layouts gone.
        settle_layouts(device);
        settle_sampler_sets(device);
        // And the device's graphics pipeline libraries nothing holds now.
        settle_libraries(device);
        let Some(back) = back else {
            let mut objects = VariantObjects::create(device, layout, stages, pipeline)?;
            objects.keep = Some(Instance::new(compiled));
            return Ok((objects, Vec::new()));
        };
        device.shared().counts.variant_taken_back();
        let instance = back.instance;
        let immutable_pushes = match push_set_zero {
            true => immutable_pushes(layout, &back.objects.samplers.samplers)?,
            false => Vec::new(),
        };
        let objects = VariantObjects {
            device: device.clone(),
            stages: back.objects.stages,
            samplers: back.objects.samplers,
            layout: back.objects.layout,
            set_needs,
            push_set_zero,
            set_zero_shape: set_zero_shape(layout, push_set_zero),
            immutable_pushes,
            specialized: stages
                .iter()
                .filter(|s| !s.specialized_notes.is_empty())
                .cloned()
                .collect(),
            heap_layout: layout.sets.get(1).cloned(),
            heap: Mutex::new(None),
            kinds: Kinds::of(stages, pipeline),
            tested: tested_slots_of(stages),
            bound: Mutex::new(None),
            keep: Some(instance),
            content: back.objects.content,
            words: back.objects.words,
            parts_content: OnceLock::new(),
        };
        let pipelines = back
            .pipelines
            .into_iter()
            .map(|kept| {
                let place = kept.place.clone();
                (place, RawPipeline::taken_back(device, kept, instance))
            })
            .collect();
        device.degrade_stages(stages, Occurrence::Made);
        Ok((objects, pipelines))
    }

    /// The kinds its making may announce (`Kinds::of`), which a bind tests.
    pub(crate) fn kinds(&self) -> &Kinds {
        &self.kinds
    }

    /// The format `formats` gives at each texture slot a specialized note of its stages tests:
    /// what a bind's notices name, taken while the bind's tables live.
    pub(crate) fn tested_formats(&self, formats: BoundFormats<'_>) -> TestedFormats {
        self.tested
            .iter()
            .map(|&slot| (slot, formats(slot)))
            .collect()
    }

    /// Announce, at a dispatch's or draw's bind of the variant, the notices its making announces
    /// whose kinds the device's options ask for at every occurrence (`DeviceOptions::
    /// announce_each`), on the calling thread, with the making's object and reason
    /// (`Device::bound_notices`): `stages` are the variant's, `pipeline` its compiled pipeline,
    /// and `specializations` the specializations and bound formats of the `VkPipeline`s the bind
    /// takes. Its kinds are tested first (`Device::binds_each`), so a variant none of whose kinds
    /// is asked for costs an AND a word. The notices are made once and announced at each bind
    /// until the specializations, or the formats bound at the slots its notes test, change:
    /// they are a function of those alone, the variant's stages and pipeline fixed.
    pub(crate) fn announce_bound<'a>(
        &self,
        (stages, pipeline): (&[StageModule], CompiledPipeline<'_>),
        specializations: impl IntoIterator<Item = (&'a SpecializationKey, Option<BoundFormats<'a>>)>,
    ) {
        let device = &self.device;
        if !device.binds_each(&self.kinds) {
            return;
        }
        let specializations: Vec<_> = specializations.into_iter().collect();
        let notices = {
            let mut bound = lock(&self.bound);
            match bound.as_ref().filter(|b| b.made_for(&specializations)) {
                Some(b) => Arc::clone(&b.notices),
                None => {
                    device.shared().counts.bound_notices_made();
                    let notices: Arc<[BoundNotice]> = device
                        .bound_notices((stages, pipeline), &specializations)
                        .into();
                    let made_for = specializations
                        .iter()
                        .map(|&(s, f)| (s.clone(), f.map(|f| self.tested_formats(f))))
                        .collect();
                    *bound = Some(BoundNotices {
                        made_for,
                        notices: Arc::clone(&notices),
                    });
                    notices
                }
            }
        };
        // Announced with no lock held: the device's sink may call back.
        device.announce_bound(&notices);
    }

    /// The pipeline layout: the device's of its definition, which every variant of the same
    /// definition shares (`shared_layouts.rs`).
    pub(crate) fn layout(&self) -> vk::PipelineLayout {
        self.layout.layout()
    }

    /// How many sets the layout has.
    pub(crate) fn set_count(&self) -> usize {
        self.layout.set_layouts().len()
    }

    /// Whether its draws push set 0 rather than allocate it (`pushes_set_zero`).
    pub(crate) fn pushes_set_zero(&self) -> bool {
        self.push_set_zero
    }

    /// The shape of its set 0 and push-constant ranges ([`set_zero_shape`]).
    pub(crate) fn set_zero_shape(&self) -> u64 {
        self.set_zero_shape
    }

    /// The writes its draws push for set 0's baked samplers, where they push set 0.
    pub(crate) fn immutable_pushes(&self) -> &[SetWrite] {
        &self.immutable_pushes
    }

    /// Whether its layout bakes an immutable sampler.
    pub(crate) fn bakes_samplers(&self) -> bool {
        !self.samplers.samplers.is_empty()
    }

    /// A command's set 0, from its command buffer's pools `arena`; none when the layout has no
    /// set, set 0 no descriptor (a pipeline that uses no set needs none bound), or its commands
    /// push set 0.
    pub(crate) fn set_zero(
        &self,
        arena: &mut DescriptorArena,
    ) -> Result<Option<vk::DescriptorSet>, Error> {
        match (self.layout.set_layouts().first(), self.set_needs.first()) {
            (Some(&layout), Some(needs)) if !needs.is_empty() => {
                Ok(Some(arena.allocate(&self.device, layout, needs)?))
            }
            _ => Ok(None),
        }
    }

    /// The device's heap set of the layout's set 1 (`heap_sets.rs`), which the variant holds
    /// from its first command that binds it; refused for a layout without one (`Internal`).
    pub(crate) fn heap_set(&self) -> Result<Arc<HeapSet>, Error> {
        let mut held = lock(&self.heap);
        if let Some(set) = held.as_ref() {
            return Ok(Arc::clone(set));
        }
        let layout = self
            .heap_layout
            .as_ref()
            .ok_or_else(|| internal("a heap set of a layout without one"))?;
        let set = heap_sets::heap_set(&self.device, layout)?;
        *held = Some(Arc::clone(&set));
        Ok(set)
    }

    /// The `VkPipeline` of `key`'s specialization, stage flags, and required subgroup size, of a
    /// compute variant (one stage), its specialized notices announced as `notices` says.
    pub(crate) fn pipeline(
        &self,
        key: &RegionKey,
        notices: Notices<'_>,
    ) -> Result<RawPipeline, Error> {
        self.compute_pipeline(key, false, notices)
    }

    /// The `VkPipeline` of a render variant's `COMPUTE` stage at `index` of its stages, an
    /// emulated object or mesh function's (`module`, the variant's stage there), under a mesh
    /// draw's complete `specialization` (`DrawCommands::specialization`), which binds `formats`.
    pub(crate) fn emulated_pipeline(
        &self,
        index: usize,
        module: &StageModule,
        (specialization, formats): (&metal2vulkan::cache::SpecializationKey, BoundFormats<'_>),
    ) -> Result<RawPipeline, Error> {
        let object = self
            .stages
            .get(index)
            .filter(|s| s.stage == vk::ShaderStageFlags::COMPUTE)
            .ok_or_else(|| internal("an emulated mesh pipeline without its compute stage"))?;
        let stage = ComputeStage {
            object,
            specialization,
            stage_flags: module.stage_flags.into(),
            required_subgroup_size: module.required_subgroup_size,
            notices: Notices::Now(Some(formats)),
        };
        self.made_pipeline(&stage, false)
    }

    /// [`pipeline`](Self::pipeline), created with `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`
    /// for an indirect execution set (`generated_execute.rs`): on a device that enabled
    /// `VK_EXT_device_generated_commands` (and `maintenance5`, whose flags it takes). The
    /// replay that makes it does not know the formats its commands bind.
    pub(crate) fn indirect_bindable_pipeline(
        &self,
        region: &DispatchRegion,
    ) -> Result<RawPipeline, Error> {
        self.compute_pipeline(&RegionKey::of(region), true, Notices::Now(None))
    }

    /// The compute pipeline of `key`, indirect-bindable where `bindable` says so, its specialized
    /// notices announced as `notices` says.
    fn compute_pipeline(
        &self,
        key: &RegionKey,
        bindable: bool,
        notices: Notices<'_>,
    ) -> Result<RawPipeline, Error> {
        let [kernel] = self.stages.as_slice() else {
            return Err(internal("a compute pipeline of other than one stage"));
        };
        let stage = ComputeStage {
            object: kernel,
            specialization: &key.specialization,
            stage_flags: key.stage_flags.into(),
            required_subgroup_size: key.required_subgroup_size,
            notices,
        };
        self.made_pipeline(&stage, bindable)
    }

    /// The compute pipeline of `c`, indirect-bindable where `bindable` says so.
    fn made_pipeline(&self, c: &ComputeStage<'_>, bindable: bool) -> Result<RawPipeline, Error> {
        let map: Vec<vk::SpecializationMapEntry> = c
            .specialization
            .entries
            .iter()
            .map(|&entry| entry.into())
            .collect();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&map)
            .data(&c.specialization.data);
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(c.required_subgroup_size.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(c.object.module())
            .name(&c.object.entry)
            .flags(c.stage_flags);
        if !map.is_empty() {
            stage = stage.specialization_info(&spec);
        }
        if c.required_subgroup_size.is_some() {
            stage = stage.push_next(&mut subgroup);
        }
        let mut flags = vk::PipelineCreateFlags2CreateInfoKHR::default().flags(INDIRECT_BINDABLE);
        let mut info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.layout.layout());
        if bindable {
            info = info.push_next(&mut flags);
        }
        // Whether the making's cache held it and its one stage, where the making reports them
        // (`shader_cache::found_in_cache`; one stage's feedback, as
        // VUID-VkComputePipelineCreateInfo-pipelineStageCreationFeedbackCount-06566 allows).
        let mut whole = vk::PipelineCreationFeedback::default();
        let mut stage = [vk::PipelineCreationFeedback::default()];
        let mut feedback = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut whole)
            .pipeline_stage_creation_feedbacks(&mut stage);
        let shared = self.device.shared();
        let driver_reports = shared.shader_cache.driver_reports();
        if driver_reports {
            info = info.push_next(&mut feedback);
        }
        // Its record is named by the variant's content and what this making adds to it.
        let making = self.making("compute").text(&(
            c.object.stage.as_raw(),
            &c.object.entry,
            c.specialization,
            c.stage_flags.as_raw(),
            c.required_subgroup_size,
            bindable,
        ));
        // The turn first (none where a piece's thread holds one), once no twin's making is in the
        // air (`flight.rs`): the driver's time alone is timed, not the wait for a turn or a twin.
        let flight = self.flight(&(
            c.object.stage.as_raw(),
            &c.object.entry,
            c.specialization,
            c.stage_flags.as_raw(),
            c.required_subgroup_size,
        ));
        let (turn, leader, followed) = shared.flights.board(flight);
        if followed {
            shared.counts.flight_followed();
        }
        let started = std::time::Instant::now();
        let created = shared
            .make_pipelines(making, |cache| {
                // SAFETY: every pointer in `info` outlives the call; the pipeline cache is the
                // making's own, destroyed once the call returned.
                unsafe { shared.vk().create_compute_pipelines(cache, &[info], None) }
            })
            .map_err(|(_, error)| vulkan(error))?;
        let created_in = started.elapsed();
        drop(leader);
        drop(turn);
        if driver_reports {
            shared.shader_cache.driver_reported(&whole, &stage);
        }
        let raw = created
            .first()
            .copied()
            .ok_or_else(|| internal("the driver made no pipeline"))?;
        self.device.shared().counts.variant_pipeline_made();
        let pipeline = RawPipeline {
            device: self.device.clone(),
            raw,
            created_in,
            keep: None,
            unannounced: AtomicBool::new(false),
            linked: None,
        };
        if let Notices::Now(formats) = c.notices {
            self.device.degrade_specialized(
                &self.specialized,
                c.specialization,
                formats,
                Occurrence::Made,
            );
        }
        Ok(pipeline)
    }
}

/// The `VkDescriptorSetLayout` of `set`, each binding with its `immutable` samplers (none where
/// its list is empty or missing).
pub(crate) fn create_set_layout(
    device: &Device,
    set: &DescriptorSetLayout,
    immutable: &[&[vk::Sampler]],
    push: bool,
) -> Result<vk::DescriptorSetLayout, Error> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = set
        .bindings
        .iter()
        .enumerate()
        .map(|(n, binding)| {
            let info = vk::DescriptorSetLayoutBinding::default()
                .binding(binding.binding)
                .descriptor_type(binding.ty.into())
                .descriptor_count(binding.count)
                .stage_flags(binding.stages.into());
            match immutable.get(n).filter(|samplers| !samplers.is_empty()) {
                Some(samplers) => info.immutable_samplers(samplers),
                None => info,
            }
        })
        .collect();
    let flags: Vec<vk::DescriptorBindingFlags> =
        set.bindings.iter().map(|b| b.flags.into()).collect();
    let mut flags_info =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let mut set_flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
    if push {
        set_flags |= vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
    }
    let mut info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(set_flags)
        .bindings(&bindings);
    if flags.iter().any(|f| !f.is_empty()) {
        info = info.push_next(&mut flags_info);
    }
    // SAFETY: `info` and every array it points at are live for the call.
    unsafe {
        device
            .shared()
            .vk()
            .create_descriptor_set_layout(&info, None)
    }
    .map_err(vulkan)
}

/// The shape of `layout`'s set 0 and push-constant ranges, where its draws push set 0
/// (`push_set_zero`) or not: a hash of what Vulkan's compatibility for set 0 compares (the set-0
/// `VkDescriptorSetLayout`'s create info, each binding's number, type, count, stages, baked sampler
/// and flags and the set's flags, and the push-constant ranges), which two variants of the same
/// shape would create identically defined, compatible for set 0 (Pipeline Layout
/// Compatibility). Variants share a layout only where it is identically defined whole
/// (`shared_layouts.rs`), so a draw's record compares shapes to count the pushes that carry set 0
/// whole across a switch between two layouts of equal shapes (which differ in a later set, or
/// bake a host step's own samplers; `WorkCounts::set_zero_equal_shape_switches`). Taken once, as
/// the variant's objects are made.
fn set_zero_shape(layout: &PipelineLayout, push_set_zero: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    push_set_zero.hash(&mut hasher);
    if let Some(set) = layout.sets.first() {
        set.flags.hash(&mut hasher);
        for binding in &set.bindings {
            binding.binding.hash(&mut hasher);
            binding.ty.hash(&mut hasher);
            binding.count.hash(&mut hasher);
            binding.stages.hash(&mut hasher);
            binding.immutable_sampler.hash(&mut hasher);
            binding.flags.hash(&mut hasher);
        }
    }
    layout.push_constants.hash(&mut hasher);
    hasher.finish()
}

/// What one set of each of `layout`'s sets takes from a pool, by set number: nothing for set 0
/// where it is pushed (`push_set_zero`).
fn needs_of(layout: &PipelineLayout, push_set_zero: bool) -> Result<Vec<SetNeeds>, Error> {
    layout
        .sets
        .iter()
        .enumerate()
        .map(|(index, set)| match push_set_zero && index == 0 {
            true => Ok(SetNeeds {
                counts: Vec::new(),
                flags: vk::DescriptorPoolCreateFlags::empty(),
            }),
            false => set_needs(set),
        })
        .collect()
}

/// The descriptor types a pushed set 0 takes: those a set-0 write (`binding::Info`) carries, but
/// the dynamic buffers, inline uniform blocks and mutable descriptors a push descriptor set layout
/// cannot hold (`VkDescriptorSetLayoutCreateInfo`'s valid usage).
const PUSHED_TYPES: [vk::DescriptorType; 9] = [
    vk::DescriptorType::SAMPLER,
    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    vk::DescriptorType::SAMPLED_IMAGE,
    vk::DescriptorType::STORAGE_IMAGE,
    vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
    vk::DescriptorType::STORAGE_TEXEL_BUFFER,
    vk::DescriptorType::UNIFORM_BUFFER,
    vk::DescriptorType::STORAGE_BUFFER,
    vk::DescriptorType::INPUT_ATTACHMENT,
];

/// Whether a variant of `layout` and `stages` pushes its set 0 (`vkCmdPushDescriptorSet`) at
/// each draw or dispatch, rather than allocating a set from its command buffer's pools, writing it
/// and binding it: the same descriptors, at the same bindings. Where the device has the push
/// command (Vulkan 1.4, or `VK_KHR_push_descriptor`, `binding::push_descriptor_command`), the
/// compiled layout's set 0 fits `maxPushDescriptors` (`push_descriptor_eligible`), takes no set or
/// binding flag and only descriptor types a push descriptor set layout takes (`PUSHED_TYPES`),
/// and the variant binds set 0 at one bind point: a render variant of no compute stage at the
/// graphics bind point, its draws' (`render::Recording`), where one that emulates a mesh or
/// tessellation stage dispatches with set 0 too; a kernel's variant, of compute stages alone
/// (a `CompiledPipeline::Kernel`), at the compute bind point, its dispatches' and tile
/// dispatches' (`pending_dispatch::Recording`, `render::tile_dispatch`).
fn pushes_set_zero(
    device: &Device,
    layout: &PipelineLayout,
    stages: &[StageModule],
    pipeline: CompiledPipeline<'_>,
) -> bool {
    let compute = |stage: &StageModule| {
        let flags: vk::ShaderStageFlags = stage.stage.into();
        flags.contains(vk::ShaderStageFlags::COMPUTE)
    };
    let one_point = match pipeline {
        CompiledPipeline::Render(_) => !stages.iter().any(compute),
        CompiledPipeline::Kernel(_) => !stages.is_empty() && stages.iter().all(compute),
    };
    device.shared().push_descriptor.is_some()
        && layout.push_descriptor_eligible
        && one_point
        && layout.sets.first().is_some_and(|set| {
            let flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
            flags.is_empty()
                && !set.bindings.is_empty()
                && set.bindings.iter().all(|binding| {
                    let flags: vk::DescriptorBindingFlags = binding.flags.into();
                    flags.is_empty() && PUSHED_TYPES.contains(&binding.ty.into())
                })
        })
}

/// A write of each element of each sampler binding of `layout`'s set 0 that bakes its sampler, of
/// the sampler `samplers` holds for it (one per binding that bakes one, in the layout's order of
/// sets and bindings, as `VariantObjects::create` takes them). A pushed set takes them, which a
/// pool-allocated one holds from its layout. The Vulkan specification, `vkCmdPushDescriptorSet`:
/// "This includes immutable sampler descriptors, which must be pushed before they are accessed by
/// a pipeline (the immutable samplers are pushed, rather than the samplers in pDescriptorWrites)."
/// A combined image sampler binding that bakes its sampler takes the draw's image write, whose
/// sampler Vulkan ignores there.
fn immutable_pushes(
    layout: &PipelineLayout,
    samplers: &[vk::Sampler],
) -> Result<Vec<SetWrite>, Error> {
    let mut baked = samplers.iter().copied();
    let mut pushes = Vec::new();
    for (index, set) in layout.sets.iter().enumerate() {
        for binding in set
            .bindings
            .iter()
            .filter(|b| b.immutable_sampler.is_some())
        {
            let sampler = baked
                .next()
                .ok_or_else(|| internal("a sampler set shorter than its layout"))?;
            let ty: vk::DescriptorType = binding.ty.into();
            if index == 0 && ty == vk::DescriptorType::SAMPLER {
                pushes.extend((0..binding.count).map(|element| SetWrite {
                    binding: binding.binding,
                    element,
                    ty,
                    info: Info::Image(vk::DescriptorImageInfo::default().sampler(sampler)),
                }));
            }
        }
    }
    Ok(pushes)
}

/// What a pool that allocates one set of `set` needs: each binding's count summed per
/// descriptor type, since a set takes every binding's whole count; and `UPDATE_AFTER_BIND` when
/// the set layout is `UPDATE_AFTER_BIND_POOL`.
pub(crate) fn set_needs(set: &DescriptorSetLayout) -> Result<SetNeeds, Error> {
    let mut counts: std::collections::BTreeMap<i32, u32> = std::collections::BTreeMap::new();
    let mut flags = vk::DescriptorPoolCreateFlags::empty();
    let set_flags: vk::DescriptorSetLayoutCreateFlags = set.flags.into();
    if set_flags.contains(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL) {
        flags |= vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND;
    }
    for binding in &set.bindings {
        let ty: vk::DescriptorType = binding.ty.into();
        let count = counts.entry(ty.as_raw()).or_default();
        *count = count
            .checked_add(binding.count)
            .ok_or_else(|| internal("the layout's descriptor count overflows"))?;
    }
    Ok(SetNeeds {
        counts: counts.into_iter().filter(|&(_, count)| count > 0).collect(),
        flags,
    })
}
