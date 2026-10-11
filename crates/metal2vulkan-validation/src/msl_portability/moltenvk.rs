//! What MoltenVK 1.4.1 hands SPIRV-Cross and the Metal compiler when it creates a pipeline from
//! one module on the `moltenvk-m3` device: the MSL options, the resource bindings of every layout
//! binding with its descriptor count, the discrete descriptor sets, and the dynamic buffers,
//! derived from the pipeline layout the product's link gave the module.
//!
//! Sources, MoltenVK v1.4.1: `MVKPipeline.mm` (`MVKPipelineLayout` creation and
//! `populateShaderConversionConfig`, `addResourceBindingToShaderConfig`, the compute pipeline's
//! `getMTLFunction`, the graphics pipeline's `initShaderConversionConfig` and
//! `add{Vertex,Fragment}ShaderToPipeline`), `MVKDescriptorSet.mm` (`pickArgumentBufferMode`,
//! `pickGPULayout`, `perDescriptorResourceCount`, `MVKDescriptorSetLayout::Create`),
//! `MVKDevice.mm` (`initMetalFeatures`, `getMTLCompileOptions`), and MoltenVKShaderConverter's
//! `SPIRVToMSLConverter.cpp` (the options' constructor and `convert`, which the helper applies).
//!
//! The device facts are those of an Apple M3 Max under macOS 26 (`initMetalFeatures`): Metal
//! argument buffers in their Metal 3 mode (tier 2, no argument encoders), native texture atomics
//! (so a storage image is one texture), 31 buffers per stage, 2D textures up to 16384, a SIMD
//! group of 32 with permutes (no emulated subgroups), the cube-gradient workaround, depth-compare
//! samplers (so no immutable sampler becomes a constexpr sampler), base vertex and instance
//! drawing, and layered rendering.

use metal2vulkan::pipeline::PipelineLayout;
use metal2vulkan::vk;
use serde::{Deserialize, Serialize};

/// `MTLLanguageVersion4_0` (macOS 26 with MoltenVK built by Xcode 26), as SPIRV-Cross numbers it.
const MSL_VERSION: u32 = 40000;
/// `maxPerStageBufferCount`.
const MAX_PER_STAGE_BUFFER_COUNT: u32 = 31;
/// `maxTextureDimension` (Apple 7 through 9).
const MAX_TEXTURE_DIMENSION: u32 = 16384;
/// `maxSubgroupSize`: MoltenVK fixes it unless the stage allows a varying size, which the
/// product never asks for.
const SUBGROUP_SIZE: u32 = 32;
/// `getVkFormatTexelBufferAlignment(VK_FORMAT_R32_UINT)`: the profile's
/// `minTexelBufferOffsetAlignment`, MoltenVK's fallback.
const R32UI_LINEAR_TEXTURE_ALIGNMENT: u32 = 16;
/// `kMVKMaxDescriptorSetCount` (SPIRV-Cross's `kMaxArgumentBuffers`): the Metal buffer indices a
/// pipeline layout reserves for argument buffers once any set uses one.
const MAX_ARGUMENT_BUFFERS: u32 = 8;
/// SPIRV-Cross's `ResourceBindingPushConstantDescriptorSet` and `...Binding`.
const PUSH_CONSTANT_SET: u32 = u32::MAX;
const PUSH_CONSTANT_BINDING: u32 = 0;
/// SPIRV-Cross's `kBufferSizeBufferBinding`.
const BUFFER_SIZE_BUFFER_BINDING: u32 = !2;
/// `MTLArgumentBuffersTier2` as SPIRV-Cross's `ArgumentBuffersTier::Tier2`.
const ARGUMENT_BUFFERS_TIER_2: u32 = 1;
/// `CompilerMSL::Options::macOS`.
const PLATFORM_MACOS: u32 = 1;

/// A module's pipeline stage, as MoltenVK creates a pipeline for it alone: a compute pipeline, or
/// the vertex or fragment stage of a graphics pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    /// The SPIR-V execution model.
    pub fn execution_model(self) -> u32 {
        match self {
            Stage::Vertex => 0,
            Stage::Fragment => 4,
            Stage::Compute => 5,
        }
    }

    fn flag(self) -> vk::ShaderStageFlags {
        match self {
            Stage::Vertex => vk::ShaderStageFlags::VERTEX,
            Stage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            Stage::Compute => vk::ShaderStageFlags::COMPUTE,
        }
    }
}

/// The module's one entry point: what a pipeline stage names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub stage: Stage,
}

/// The entry point of the module `words`. An error (a module MoltenVK's pipeline options are not
/// derived for, a harness limit) for no or several entry points, or another execution model.
pub fn entry_point(words: &[u32]) -> Result<Entry, String> {
    const OP_ENTRY_POINT: u32 = 15;
    let mut at = words.get(5..).ok_or("no SPIR-V header")?;
    let mut entries = Vec::new();
    while let Some(&first) = at.first() {
        let count = (first >> 16) as usize;
        let inst = at
            .get(..count)
            .filter(|_| count > 0)
            .ok_or("truncated instruction")?;
        if first & 0xffff == OP_ENTRY_POINT {
            entries.push(inst);
        }
        at = &at[count..];
    }
    let [entry] = entries.as_slice() else {
        return Err(format!(
            "{} entry points (a MoltenVK pipeline stage names one)",
            entries.len()
        ));
    };
    let (model, name) = match entry {
        [_, model, _id, name @ ..] if !name.is_empty() => (*model, name),
        _ => return Err("truncated OpEntryPoint".to_owned()),
    };
    let bytes = name
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .take_while(|b| *b != 0)
        .collect::<Vec<_>>();
    let name = String::from_utf8(bytes).map_err(|error| format!("entry name: {error}"))?;
    let stage = match model {
        0 => Stage::Vertex,
        4 => Stage::Fragment,
        5 => Stage::Compute,
        other => return Err(format!("execution model {other}")),
    };
    Ok(Entry { name, stage })
}

/// What MoltenVK reads of a pipeline layout for a shader conversion: each set's bindings (type,
/// count, stages) in set order, the stages with push constants, and whether set 0 may be a
/// push-descriptor set. The product's immutable samplers change nothing on this device (no sampler
/// becomes constexpr, and a binding they sit on already has a sampler), and neither do its binding
/// and set flags (partially bound, update after bind), so they are not kept.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub sets: Vec<SetLayout>,
    pub push_constant_stages: vk::ShaderStageFlags,
    pub push_descriptor_eligible: bool,
}

/// One descriptor set layout of a [`Layout`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetLayout {
    pub set: u32,
    pub bindings: Vec<Binding>,
}

/// One `VkDescriptorSetLayoutBinding`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
}

impl From<&PipelineLayout> for Layout {
    fn from(layout: &PipelineLayout) -> Layout {
        Layout {
            sets: layout
                .sets
                .iter()
                .map(|set| SetLayout {
                    set: set.set,
                    bindings: set
                        .bindings
                        .iter()
                        .map(|b| Binding {
                            binding: b.binding,
                            ty: b.ty,
                            count: b.count,
                            stages: b.stages,
                        })
                        .collect(),
                })
                .collect(),
            push_constant_stages: layout
                .push_constants
                .iter()
                .fold(vk::ShaderStageFlags::empty(), |all, range| {
                    all | range.stages
                }),
            push_descriptor_eligible: layout.push_descriptor_eligible,
        }
    }
}

/// How the pipeline's consumer creates descriptor set 0. The product's layout is one pooled set,
/// or a push-descriptor set when `push_descriptor_eligible`; MoltenVK turns argument buffers off
/// for a push-descriptor set layout, so its resources become discrete MSL arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Set0 {
    Pooled,
    Push,
}

impl Set0 {
    pub fn as_str(self) -> &'static str {
        match self {
            Set0::Pooled => "pooled",
            Set0::Push => "push",
        }
    }
}

/// SPIRV-Cross's resource base type of a binding (`SPIRType::BaseType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseType {
    Void,
    Image,
    Sampler,
    SampledImage,
}

impl BaseType {
    fn as_str(self) -> &'static str {
        match self {
            BaseType::Void => "void",
            BaseType::Image => "image",
            BaseType::Sampler => "sampler",
            BaseType::SampledImage => "sampled_image",
        }
    }
}

/// One `MSLResourceBinding` of the entry's stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceBinding {
    pub basetype: BaseType,
    pub set: u32,
    pub binding: u32,
    pub count: u32,
    pub msl_buffer: u32,
    pub msl_texture: u32,
    pub msl_sampler: u32,
}

/// Everything MoltenVK configures SPIRV-Cross with for one stage: what the helper applies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conversion {
    pub entry: Entry,
    pub set0: Set0,
    /// `CompilerMSL::Options` fields and values, in the order MoltenVK assigns them.
    pub options: Vec<(&'static str, u32)>,
    pub flip_vert_y: bool,
    pub fixup_clipspace: bool,
    pub bindings: Vec<ResourceBinding>,
    pub discrete_sets: Vec<u32>,
    /// `(set, binding, index)` of each dynamic buffer.
    pub dynamic_buffers: Vec<(u32, u32, u32)>,
}

impl Conversion {
    /// The helper's configuration text (`src/msl_portability/helper.cpp`).
    pub fn config_text(&self) -> String {
        use std::fmt::Write as _;
        let model = self.entry.stage.execution_model();
        let mut text = String::new();
        let _ = writeln!(text, "entry {model} {}", self.entry.name);
        let _ = writeln!(text, "common flip_vert_y {}", u32::from(self.flip_vert_y));
        let _ = writeln!(
            text,
            "common fixup_clipspace {}",
            u32::from(self.fixup_clipspace)
        );
        for (name, value) in &self.options {
            let _ = writeln!(text, "option {name} {value}");
        }
        for b in &self.bindings {
            let _ = writeln!(
                text,
                "binding {model} {} {} {} {} {} {} {}",
                b.basetype.as_str(),
                b.set,
                b.binding,
                b.count,
                b.msl_buffer,
                b.msl_texture,
                b.msl_sampler
            );
        }
        for set in &self.discrete_sets {
            let _ = writeln!(text, "discrete {set}");
        }
        for (set, binding, index) in &self.dynamic_buffers {
            let _ = writeln!(text, "dynamic {model} {set} {binding} {index}");
        }
        text
    }
}

/// The conversions a pipeline of the module can see: set 0 pooled, and pushed when the layout is
/// push-descriptor eligible and that changes the conversion.
pub fn conversions(entry: &Entry, layout: &Layout) -> Result<Vec<Conversion>, String> {
    let pooled = conversion(entry, layout, Set0::Pooled)?;
    let mut out = Vec::with_capacity(2);
    if layout.push_descriptor_eligible {
        let push = conversion(entry, layout, Set0::Push)?;
        let same = push.bindings == pooled.bindings
            && push.discrete_sets == pooled.discrete_sets
            && push.dynamic_buffers == pooled.dynamic_buffers;
        out.push(pooled);
        if !same {
            out.push(push);
        }
    } else {
        out.push(pooled);
    }
    Ok(out)
}

/// `MVKDescriptorGPULayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuLayout {
    None,
    Texture,
    Sampler,
    Buffer,
    BufferAuxSize,
    TexBufSoA,
    TexSampSoA,
}

/// `MVKDescriptorResourceCount` (per descriptor) and `MVKShaderStageResourceBinding` (indices).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Resources {
    buffer: u32,
    texture: u32,
    sampler: u32,
    dynamic_offset: u32,
}

impl Resources {
    fn times(self, count: u32) -> Resources {
        Resources {
            buffer: self.buffer * count,
            texture: self.texture * count,
            sampler: self.sampler * count,
            dynamic_offset: self.dynamic_offset * count,
        }
    }

    fn add(&mut self, other: Resources) {
        self.buffer += other.buffer;
        self.texture += other.texture;
        self.sampler += other.sampler;
        self.dynamic_offset += other.dynamic_offset;
    }
}

/// The descriptor types the product's layouts use (`vk::DescriptorType`s link and lower emit).
/// Any other is refused rather than given semantics MoltenVK's tables were not read for.
fn check_type(ty: vk::DescriptorType) -> Result<(), String> {
    use vk::DescriptorType as T;
    match ty {
        T::SAMPLER
        | T::COMBINED_IMAGE_SAMPLER
        | T::SAMPLED_IMAGE
        | T::STORAGE_IMAGE
        | T::UNIFORM_TEXEL_BUFFER
        | T::STORAGE_TEXEL_BUFFER
        | T::UNIFORM_BUFFER
        | T::STORAGE_BUFFER
        | T::UNIFORM_BUFFER_DYNAMIC
        | T::STORAGE_BUFFER_DYNAMIC
        | T::INPUT_ATTACHMENT => Ok(()),
        other => Err(format!("descriptor type {other:?}")),
    }
}

fn is_dynamic(ty: vk::DescriptorType) -> bool {
    ty == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        || ty == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
}

/// `pickGPULayout` on this device (one-plane samplers, native texture atomics).
fn gpu_layout(ty: vk::DescriptorType, count: u32, argument_buffer: bool) -> GpuLayout {
    use vk::DescriptorType as T;
    if !argument_buffer || count == 0 {
        return GpuLayout::None;
    }
    match ty {
        T::COMBINED_IMAGE_SAMPLER => GpuLayout::TexSampSoA,
        T::SAMPLER => GpuLayout::Sampler,
        T::SAMPLED_IMAGE
        | T::STORAGE_IMAGE
        | T::UNIFORM_TEXEL_BUFFER
        | T::STORAGE_TEXEL_BUFFER
        | T::INPUT_ATTACHMENT => GpuLayout::Texture,
        _ => GpuLayout::BufferAuxSize,
    }
}

/// `perDescriptorResourceCount` on this device.
fn per_descriptor(ty: vk::DescriptorType, gpu: GpuLayout, argument_buffer: bool) -> Resources {
    use vk::DescriptorType as T;
    let dynamic_offset = u32::from(is_dynamic(ty));
    let (buffer, texture, sampler) = if argument_buffer {
        match gpu {
            GpuLayout::None => (0, 0, 0),
            GpuLayout::Texture => (0, 1, 0),
            GpuLayout::Sampler => (0, 0, 1),
            GpuLayout::Buffer | GpuLayout::BufferAuxSize => (1, 0, 0),
            GpuLayout::TexBufSoA => (1, 1, 0),
            GpuLayout::TexSampSoA => (0, 1, 1),
        }
    } else {
        match ty {
            T::COMBINED_IMAGE_SAMPLER => (0, 1, 1),
            T::SAMPLER => (0, 0, 1),
            // A storage image has no atomic buffer with native texture atomics.
            T::INPUT_ATTACHMENT
            | T::SAMPLED_IMAGE
            | T::UNIFORM_TEXEL_BUFFER
            | T::STORAGE_IMAGE
            | T::STORAGE_TEXEL_BUFFER => (0, 1, 0),
            _ => (1, 0, 0),
        }
    };
    Resources {
        buffer,
        texture,
        sampler,
        dynamic_offset,
    }
}

/// `getBindingLayout`: the layout of a binding bound with the Metal binding API.
fn binding_layout(per: Resources) -> GpuLayout {
    match (per.sampler > 0, per.texture > 0, per.buffer > 0) {
        (true, true, _) => GpuLayout::TexSampSoA,
        (true, false, _) => GpuLayout::Sampler,
        (false, true, true) => GpuLayout::TexBufSoA,
        (false, false, true) => GpuLayout::Buffer,
        (false, true, false) => GpuLayout::Texture,
        (false, false, false) => GpuLayout::None,
    }
}

/// One set as MoltenVK lays it out: its argument-buffer mode and each binding's facts.
struct SetPlan<'a> {
    argument_buffer: bool,
    needs_size_buffer: bool,
    bindings: Vec<(&'a Binding, GpuLayout, Resources)>,
}

/// MoltenVK's conversion configuration of `entry` under `layout` with set 0 created as `set0`.
pub fn conversion(entry: &Entry, layout: &Layout, set0: Set0) -> Result<Conversion, String> {
    let stage = entry.stage;
    let flag = stage.flag();
    let mut plans = Vec::with_capacity(layout.sets.len());
    for (position, set) in layout.sets.iter().enumerate() {
        // MoltenVK numbers a set by its position in `pSetLayouts`.
        if usize::try_from(set.set) != Ok(position) {
            return Err(format!("set {} at layout position {position}", set.set));
        }
        let push = set0 == Set0::Push && set.set == 0;
        // `pickArgumentBufferMode`: Metal 3 argument buffers, off for a push-descriptor layout
        // and for a layout without a binding (Metal refuses empty argument encoders).
        let argument_buffer = !push && set.bindings.iter().any(|b| b.count > 0);
        let mut bindings = Vec::with_capacity(set.bindings.len());
        for b in &set.bindings {
            check_type(b.ty)?;
            let gpu = gpu_layout(b.ty, b.count, argument_buffer);
            bindings.push((b, gpu, per_descriptor(b.ty, gpu, argument_buffer)));
        }
        // `MVKDescriptorSetLayout::Create` sorts by binding, then stage flags.
        bindings.sort_by_key(|(b, _, _)| (b.binding, b.stages.as_raw()));
        let needs_size_buffer = bindings
            .iter()
            .any(|(_, gpu, _)| *gpu == GpuLayout::BufferAuxSize);
        plans.push(SetPlan {
            argument_buffer,
            needs_size_buffer,
            bindings,
        });
    }

    // `MVKPipelineLayout` creation: the Metal indices of the entry's stage. Argument buffers take
    // the first buffer indices, then the push constants, then each discrete set in order.
    let mut counts = Resources::default();
    if plans.iter().any(|plan| plan.argument_buffer) {
        counts.buffer += MAX_ARGUMENT_BUFFERS;
    }
    let push_constant_index = counts.buffer;
    let uses_push_constants = layout.push_constant_stages.intersects(flag);
    if uses_push_constants {
        counts.buffer += 1;
    }
    let mut offsets = Vec::with_capacity(plans.len());
    for plan in &plans {
        offsets.push(counts);
        let mut total = Resources::default();
        for (b, _, per) in &plan.bindings {
            if b.count > 0 && b.stages.intersects(flag) {
                total.add(per.times(b.count));
            }
        }
        if plan.argument_buffer {
            total = Resources {
                dynamic_offset: total.dynamic_offset,
                ..Resources::default()
            };
        }
        counts.add(total);
    }

    // `populateShaderConversionConfig`, for the entry's stage (the helper passes SPIRV-Cross only
    // that stage's bindings, as `convert` does).
    let mut bindings = Vec::new();
    let mut discrete_sets = Vec::new();
    let mut dynamic_buffers = Vec::new();
    if uses_push_constants {
        bindings.push(ResourceBinding {
            basetype: BaseType::Void,
            set: PUSH_CONSTANT_SET,
            binding: PUSH_CONSTANT_BINDING,
            count: 1,
            msl_buffer: push_constant_index,
            msl_texture: 0,
            msl_sampler: 0,
        });
    }
    for ((set, plan), offset) in (0u32..).zip(&plans).zip(&offsets) {
        let mut at = *offset;
        let mut argument_index = 0u32;
        if plan.argument_buffer && plan.needs_size_buffer {
            argument_index += 1;
            bindings.push(ResourceBinding {
                basetype: BaseType::Void,
                set,
                binding: BUFFER_SIZE_BUFFER_BINDING,
                count: 1,
                msl_buffer: 0,
                msl_texture: 0,
                msl_sampler: 0,
            });
        }
        for (b, gpu, per) in &plan.bindings {
            let resources = per.times(b.count);
            let used = b.stages.intersects(flag);
            if plan.argument_buffer {
                at.texture = argument_index;
                at.buffer = argument_index + resources.texture;
                at.sampler = argument_index + resources.texture;
            }
            if plan.argument_buffer || used {
                let layout = if plan.argument_buffer {
                    *gpu
                } else {
                    binding_layout(*per)
                };
                let basetypes: &[BaseType] = match layout {
                    GpuLayout::None => &[],
                    GpuLayout::Texture => &[BaseType::Image],
                    GpuLayout::Sampler => &[BaseType::Sampler],
                    GpuLayout::Buffer | GpuLayout::BufferAuxSize => &[BaseType::Void],
                    GpuLayout::TexBufSoA => &[BaseType::Image, BaseType::Void],
                    GpuLayout::TexSampSoA => &[BaseType::SampledImage],
                };
                if b.count > 0 {
                    for basetype in basetypes {
                        bindings.push(ResourceBinding {
                            basetype: *basetype,
                            set,
                            binding: b.binding,
                            count: b.count,
                            msl_buffer: at.buffer,
                            msl_texture: at.texture,
                            msl_sampler: at.sampler,
                        });
                    }
                }
                if per.dynamic_offset != 0 && used {
                    dynamic_buffers.push((set, b.binding, at.dynamic_offset));
                }
                if plan.argument_buffer {
                    if used && per.dynamic_offset != 0 {
                        at.dynamic_offset += b.count;
                    }
                } else {
                    at.add(resources);
                }
            }
            argument_index += resources.texture + resources.buffer + resources.sampler;
        }
        if !plan.argument_buffer {
            discrete_sets.push(set);
        }
    }

    Ok(Conversion {
        entry: entry.clone(),
        set0,
        options: options(stage),
        // `shouldFlipVertexY` (`MVK_CONFIG_SHADER_CONVERSION_FLIP_VERTEX_Y`, default on); no
        // depth clip control asks for clip-space fixup.
        flip_vert_y: true,
        fixup_clipspace: false,
        bindings,
        discrete_sets,
        dynamic_buffers,
    })
}

/// The MSL options of `stage`'s pipeline, in the order MoltenVK assigns them. Pipeline state the
/// module does not carry keeps SPIRV-Cross's defaults: the point-size, fragment depth, and
/// stencil-reference outputs and the fragment output mask stay enabled; there is no vertex input,
/// previous-stage output, multisample state, or multiview.
fn options(stage: Stage) -> Vec<(&'static str, u32)> {
    // `getImplicitBufferIndex`: counted down from the last buffer index. A graphics stage counts
    // below its vertex buffers, and a module linked alone has none.
    let implicit = |offset: u32| MAX_PER_STAGE_BUFFER_COUNT - 1 - offset;
    // `SPIRVToMSLConversionOptions`'s constructor.
    let mut options = vec![
        ("platform", PLATFORM_MACOS),
        ("pad_fragment_output_components", 1),
    ];
    let argument_buffers = [
        ("argument_buffers", 1),
        ("force_active_argument_buffer_resources", 0),
        ("pad_argument_buffer_resources", 1),
        ("argument_buffers_tier", ARGUMENT_BUFFERS_TIER_2),
    ];
    match stage {
        Stage::Compute => {
            options.extend([
                ("msl_version", MSL_VERSION),
                ("texel_buffer_texture_width", MAX_TEXTURE_DIMENSION),
                (
                    "r32ui_linear_texture_alignment",
                    R32UI_LINEAR_TEXTURE_ALIGNMENT,
                ),
                ("texture_buffer_native", 1),
                ("dispatch_base", 0),
                ("texture_1D_as_2D", 1),
                ("fixed_subgroup_size", SUBGROUP_SIZE),
            ]);
            options.extend(argument_buffers);
            options.extend([
                ("emulate_subgroups", 0),
                ("buffer_size_buffer_index", implicit(1)),
                ("dynamic_offsets_buffer_index", implicit(0)),
                ("indirect_params_buffer_index", implicit(2)),
                ("replace_recursive_inputs", 1),
            ]);
        }
        Stage::Vertex | Stage::Fragment => {
            options.extend([
                ("msl_version", MSL_VERSION),
                ("texel_buffer_texture_width", MAX_TEXTURE_DIMENSION),
                (
                    "r32ui_linear_texture_alignment",
                    R32UI_LINEAR_TEXTURE_ALIGNMENT,
                ),
                ("texture_buffer_native", 1),
            ]);
            options.extend(argument_buffers);
            options.extend([
                ("agx_manual_cube_grad_fixup", 1),
                ("ios_support_base_vertex_instance", 1),
                ("texture_1D_as_2D", 1),
                ("tess_domain_origin_lower_left", 0),
                ("multiview", 0),
                ("multiview_layered_rendering", 1),
                ("view_index_from_device_index", 0),
                ("replace_recursive_inputs", 1),
                ("emulate_subgroups", 0),
                // The vertex stage's, which a fragment conversion inherits: MoltenVK converts
                // both with one configuration, the vertex stage first.
                ("buffer_size_buffer_index", implicit(1)),
                ("dynamic_offsets_buffer_index", implicit(0)),
                ("shader_output_buffer_index", implicit(3)),
                ("view_mask_buffer_index", implicit(2)),
                ("capture_output_to_buffer", 0),
                ("disable_rasterization", 0),
            ]);
            if stage == Stage::Fragment {
                options.extend([
                    ("fixed_subgroup_size", SUBGROUP_SIZE),
                    ("check_discarded_frag_stores", 1),
                    ("force_fragment_with_side_effects_execution", 0),
                    ("input_attachment_is_ds_attachment", 0),
                ]);
            }
        }
    }
    options
}

/// What the helper reports for the Metal compile (`SPIRVToMSLConversionResultInfo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reported {
    /// `get_fp_fast_math_flags(true)` of the entry point.
    pub fp_fast_math_flags: u32,
    /// `is_position_invariant()`.
    pub position_invariant: bool,
}

impl Reported {
    /// The helper's standard output on success.
    pub fn parse(stdout: &str) -> Result<Reported, String> {
        let mut flags = None;
        let mut invariant = None;
        for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
            match line.trim().split_once('=') {
                Some(("fp_fast_math_flags", value)) => flags = value.parse::<u32>().ok(),
                Some(("position_invariant", "0")) => invariant = Some(false),
                Some(("position_invariant", "1")) => invariant = Some(true),
                _ => return Err(format!("unexpected helper output line {line:?}")),
            }
        }
        match (flags, invariant) {
            (Some(fp_fast_math_flags), Some(position_invariant)) => Ok(Reported {
                fp_fast_math_flags,
                position_invariant,
            }),
            _ => Err(format!("incomplete helper output {stdout:?}")),
        }
    }

    /// `xcrun` arguments compiling the MSL as `MVKDevice::getMTLCompileOptions` configures
    /// `newLibraryWithSource` under the default on-demand fast math: language 4.0, the math mode
    /// the reported flags allow (fast with fast fp32 functions when NotNaN, NotInf, NSZ,
    /// AllowRecip, AllowReassoc, and AllowContract are all set, relaxed when the last four are,
    /// safe with precise functions otherwise), and preserved invariance for an invariant
    /// position. The caller appends `<msl> -o <air>`.
    pub fn metal_arguments(self) -> Vec<String> {
        const NOT_NAN: u32 = 0x1;
        const NOT_INF: u32 = 0x2;
        const NSZ: u32 = 0x4;
        const ALLOW_RECIP: u32 = 0x8;
        const ALLOW_CONTRACT: u32 = 0x0001_0000;
        const ALLOW_REASSOC: u32 = 0x0002_0000;
        let relaxed = NSZ | ALLOW_RECIP | ALLOW_REASSOC | ALLOW_CONTRACT;
        let flags = self.fp_fast_math_flags;
        let (mode, functions) = if flags & relaxed != relaxed {
            ("safe", "precise")
        } else if flags & (NOT_NAN | NOT_INF) == NOT_NAN | NOT_INF {
            ("fast", "fast")
        } else {
            ("relaxed", "precise")
        };
        let mut args = [
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            &format!("-fmetal-math-mode={mode}"),
            &format!("-fmetal-math-fp32-functions={functions}"),
        ]
        .map(str::to_owned)
        .to_vec();
        if self.position_invariant {
            args.push("-fpreserve-invariance".to_owned());
        }
        args.push("-c".to_owned());
        args
    }
}
