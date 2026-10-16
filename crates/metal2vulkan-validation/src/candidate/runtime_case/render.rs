//! A render case's draw through the Metal-like runtime, as the Metal oracle draws it
//! (`metal::platform`): the one copy the runtime lane's render census (`runtime_lane::renders`) and
//! the engine executor's runtime path (`runtime_run`) share.
//!
//! A library of the case's AIR and one of its generated companion's, each stage's function
//! specialized with its constant values, the case's own function linking the functions of its
//! linked libraries as the oracle links them, a render pipeline state of the oracle's descriptor
//! (`graphics::pipeline_descriptor`), its color targets and its separate `Depth32Float` and
//! `Stencil8` attachments loaded and stored, its depth/stencil state, references, winding and cull
//! mode, its vertex buffers, buffers, textures and sampler states bound at their slots, a patch
//! draw's factor buffer as both executors bind it (`executor_contract::tessellation`), its one
//! draw, and its output read back after the command buffer completed. A multisampled pass's
//! attachments and textures are filled sample by sample and its output read sample by sample, never
//! resolved, in the same command buffer as the oracle does (`candidate::runtime_samples`), a
//! uniform stencil cleared to its value. A case without attachments draws into the oracle's
//! one-pixel pass.
//!
//! A vertex buffer that is an out-of-bounds target (the buffer feeding an attribute the draw
//! fetches past: a vertex input's, or a patch draw's control point's or patch input's, which its
//! post-tessellation vertex function fetches), and an index buffer that is one (the draw's
//! indices run past it), lie in a
//! placement heap of its own between guards, as a kernel's buffer target does
//! (`runtime_case::guarded_buffer`), and what the draw changed there is counted once it completed
//! ([`CommittedRender::guard_counts`]). The draw is encoded before it is committed
//! ([`encoded_render`]), so the runtime path decides how the draw contains its targets from what it
//! executed before anything runs.
//!
//! A fragment case's explicit imageblock is drawn as the oracle draws it, in one pass with the
//! case's draw (`imageblock.rs`): its pass's imageblock sample length is the case pipeline's and
//! its tile 16x16, the generated initializer draws before the case and the generated resolver after
//! it where the output is an imageblock member; the case's executions are its own draw's.
//!
//! A case's rasterization-rate map is made as the oracle makes it: the pass's through
//! `new_rasterization_rate_map`, its viewport the map's screen, and a buffer of the case's own
//! stage that holds a map's parameter data through `rate_map_buffer`.
//!
//! A stage's argument buffers are made as a kernel's are (`runtime_case::bound_buffers`): each
//! member as the oracle makes it, a texture's usage its role united with the declared access of the
//! case's AIR member at its field (`runtime_case::argument_texture_usage`), written into its owner
//! at its field as Metal's argument encoder writes it, and the buffers and textures made resident
//! for reading and writing in the vertex and fragment stages (`use_resources`), as the oracle's
//! `useResource:usage:stages:` does. A stage's visible-function tables are made by the render
//! pipeline state for that stage (`newVisibleFunctionTableWithDescriptor:stage:`), each entry the
//! handle the state gives the function it holds for that stage
//! (`functionHandleWithFunction:stage:`), and bound at their index of that stage, as the oracle
//! makes and binds them; the bytes the runtime wrote to each are keyed by stage and index
//! ([`EncodedRender::tables`]). A pipeline state an argument buffer holds is its token's template's
//! state, made with `support_indirect_command_buffers` (`RenderInputs::templates`), its owner
//! holding its `gpu_resource_id` at the field as Metal's argument encoder writes it. A stage's
//! indirect command buffers are made, encoded by the host, made resident, read back and executed as
//! a kernel's are (the `commands` module): the commands each executes run after the draw, each in a
//! command buffer of its own ([`EncodedRender::executions`]).
//!
//! What a later runtime step adds (a layered multisampled target) is refused as a
//! [`CaseError::Case`] naming "a later runtime step"; the executor's route sends no such case
//! here (`runtime_run::render_prerequisites`).

use super::commands::{encode_host_commands, encode_twin_copies, encoded_execution, CaseCommands};
use super::imageblock::Helpers;
use super::{
    argument_samplers, argument_texture_usage, argument_textures, buffer_with, filled_texture,
    guard_count, guarded_buffer, linked_functions, member_words, pipeline_states, rate_map_buffer,
    read_buffer, read_texture, tables_of, CaseError, LinkedSet, TemplateState,
};
use crate::air::reflect::FragmentImageblock;
use crate::air_library::AirLibrary;
use crate::candidate::engine::argument_buffers::owner_bytes;
use crate::candidate::engine::graphics::{
    pipeline_descriptor, Assembly, ColorTarget, RenderCase, RenderFunction, RenderOutput,
};
use crate::candidate::engine::guards::{declared_reach, GuardCount};
use crate::candidate::engine::resources::{pixel_format, sampler_descriptor, texture_type};
use crate::candidate::engine::tables::TableBuffers;
use crate::candidate::engine::LinkedModule;
use crate::candidate::runtime_samples::{self, SampleFill};
use crate::case::indirect_commands::metal_primitive_type;
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::case::TextureFormat as CaseFormat;
use crate::case::{AuthoredCase, Stage, TextureType};
use crate::check::interface::AirInterface;
use crate::executor_contract::multisample::{sample_read, uniform_stencil, SampleRead};
use crate::executor_contract::tessellation as patches;
use crate::literal::{select_tightly_packed_2d, texture_layout, LiteralDepthStencil};
use metal2vulkan::bind::DrawCall;
use metal2vulkan::metal::{IndexType, PixelFormat, RenderStages, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::RenderPipelineDescriptor;
use metal2vulkan::pipeline::Stage as ShaderStage;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, Device, Execution, Function, LoadAction, PipelineOptions, Region,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions, ResourceRef,
    SamplerState, StoreAction, Texture, TextureDescriptor, Viewport, VisibleFunctionTable,
};
use std::collections::BTreeMap;

mod mesh;
pub use mesh::{encoded_mesh, mesh_pipeline_state, MeshInputs};

/// What a render case's draw needs: the authored case, its render mapping, its AIR bitcode and its
/// generated companion's, the libraries its own function links (`engine::linked_modules`),
/// the usage the Metal oracle gives each of its textures (`runtime_case::texture_usage`), its own
/// function's AIR interface, whose argument-buffer members give each argument-buffer texture's
/// usage (`runtime_case::argument_texture_usage`), and a fragment case's explicit imageblock's
/// master, the checked reflection's (`CheckedShader::reflection`), whose members the imageblock
/// helpers write and read; and the state of each pipeline template its argument buffers name, by
/// token (`runtime_run::template_states`).
pub struct RenderInputs<'a> {
    pub case: &'a AuthoredCase,
    pub render: &'a RenderCase,
    pub bitcode: &'a [u8],
    pub companion: Option<&'a [u8]>,
    pub linked: &'a [LinkedModule],
    pub texture_usage: &'a BTreeMap<u32, TextureUsage>,
    pub interface: &'a AirInterface,
    pub imageblock: Option<&'a FragmentImageblock>,
    pub templates: &'a BTreeMap<u64, TemplateState>,
}

/// Work a multisampled case's command buffer runs before its pass, as the Metal oracle's does: a
/// texture's fill, or a stencil attachment cleared to its one value by a pass of its own.
enum Prep {
    Fill(SampleFill),
    ClearStencil(Texture, u32),
}

impl Prep {
    fn encode(&self, command_buffer: &CommandBuffer) -> Result<(), CaseError> {
        match self {
            Prep::Fill(fill) => fill.encode(command_buffer),
            Prep::ClearStencil(texture, value) => {
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.stencil_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.clear_stencil = *value;
                attachment.store_action = StoreAction::Store;
                command_buffer
                    .render_command_encoder(&pass)
                    .map_err(CaseError::runtime(
                        "the stencil clear's render_command_encoder",
                    ))?
                    .end_encoding()
                    .map_err(CaseError::runtime("the stencil clear's end_encoding"))
            }
        }
    }
}

/// A 2D multisample texture of `pixel_format`, `dimensions` and `samples` on `device`, as the
/// Metal oracle makes a multisampled attachment (`metal::platform::attachments`): `Private`, a
/// render target the per-sample readback reads (`ShaderRead`), through a view of another format
/// for a color one (`PixelFormatView`).
fn multisampled(
    device: &Device,
    pixel_format: PixelFormat,
    [width, height]: [u32; 2],
    samples: u32,
) -> Result<Texture, CaseError> {
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = texture_type(TextureType::D2Multisample);
    descriptor.pixel_format = pixel_format;
    descriptor.width = width.into();
    descriptor.height = height.into();
    descriptor.sample_count = samples.into();
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    if !matches!(
        pixel_format,
        PixelFormat::Depth32Float | PixelFormat::Stencil8
    ) {
        descriptor.usage |= TextureUsage::PIXEL_FORMAT_VIEW;
    }
    descriptor.resource_options = ResourceOptions::private();
    device
        .new_texture(&descriptor)
        .map_err(CaseError::runtime(format!(
            "a multisampled {pixel_format:?} attachment: new_texture"
        )))
}

/// A color target of `target` on `device` holding its bytes: 2D, or a 2D array of one slice in a
/// layered pass, `Shared`, `RenderTarget` usage (the Metal oracle's,
/// `metal::platform::attachments`). A multisampled one is [`multisampled`], filled sample by
/// sample by its fill in `prep`.
fn target_with(
    device: &Device,
    target: &ColorTarget,
    layered: bool,
    prep: &mut Vec<Prep>,
) -> Result<Texture, CaseError> {
    if target.samples != 1 {
        if layered {
            return Err("a layered multisampled target, a later runtime step".into());
        }
        let texture = multisampled(
            device,
            pixel_format(target.format),
            target.dimensions,
            target.samples,
        )?;
        let [width, height] = target.dimensions;
        let layout = texture_layout(
            TextureType::D2Multisample,
            [width, height, 1],
            target.samples,
        )?;
        let fill = SampleFill::new(device, &texture, target.format, layout, &target.bytes)?;
        prep.push(Prep::Fill(fill));
        return Ok(texture);
    }
    let [width, height] = target.dimensions;
    let mut descriptor = TextureDescriptor::default();
    descriptor.texture_type = texture_type(if layered {
        TextureType::D2Array
    } else {
        TextureType::D2
    });
    descriptor.pixel_format = pixel_format(target.format);
    descriptor.width = width.into();
    descriptor.height = height.into();
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device
        .new_texture(&descriptor)
        .map_err(CaseError::runtime("a color target: new_texture"))?;
    let row = u64::from(width) * target.format.bytes_per_pixel() as u64;
    texture
        .replace_region(
            Region::new_2d(0, 0, width.into(), height.into()),
            0,
            0,
            &target.bytes,
            row,
            row * u64::from(height),
        )
        .map_err(CaseError::runtime("a color target: replace_region"))?;
    Ok(texture)
}

/// The depth and stencil attachments of `literal` on `device`, each authored aspect its own
/// texture holding its bytes: `Depth32Float` and `Stencil8`, 2D, `Shared`, `RenderTarget` usage
/// (the Metal oracle's separate attachments, `metal::platform::attachments`). Multisampled ones
/// are [`multisampled`]: the depth filled sample by sample, the stencil, whose samples hold one
/// value, cleared to it, each by its work in `prep`.
fn depth_stencil_with(
    device: &Device,
    literal: &LiteralDepthStencil,
    prep: &mut Vec<Prep>,
) -> Result<(Option<Texture>, Option<Texture>), CaseError> {
    let [width, height] = literal.dimensions;
    if literal.samples != 1 {
        let samples = literal.samples;
        let depth = literal
            .depth
            .as_deref()
            .map(|bytes| {
                let texture = multisampled(
                    device,
                    PixelFormat::Depth32Float,
                    literal.dimensions,
                    samples,
                )?;
                let layout =
                    texture_layout(TextureType::D2Multisample, [width, height, 1], samples)?;
                let fill =
                    SampleFill::new(device, &texture, CaseFormat::Depth32Float, layout, bytes)?;
                prep.push(Prep::Fill(fill));
                Ok::<Texture, CaseError>(texture)
            })
            .transpose()?;
        let stencil = literal
            .stencil
            .as_deref()
            .map(|bytes| {
                let value = uniform_stencil(bytes)?;
                let texture =
                    multisampled(device, PixelFormat::Stencil8, literal.dimensions, samples)?;
                prep.push(Prep::ClearStencil(texture.clone(), value.into()));
                Ok::<Texture, CaseError>(texture)
            })
            .transpose()?;
        return Ok((depth, stencil));
    }
    let aspect = |pixel_format: PixelFormat, bytes: &[u8], texel: u64| {
        let mut descriptor = TextureDescriptor::default();
        descriptor.pixel_format = pixel_format;
        descriptor.width = width.into();
        descriptor.height = height.into();
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        let texture = device
            .new_texture(&descriptor)
            .map_err(CaseError::runtime(format!(
                "a {pixel_format:?} attachment: new_texture"
            )))?;
        let row = u64::from(width) * texel;
        texture
            .replace_region(
                Region::new_2d(0, 0, width.into(), height.into()),
                0,
                0,
                bytes,
                row,
                row * u64::from(height),
            )
            .map_err(CaseError::runtime(format!(
                "a {pixel_format:?} attachment: replace_region"
            )))?;
        Ok::<Texture, CaseError>(texture)
    };
    let depth = literal
        .depth
        .as_deref()
        .map(|bytes| aspect(PixelFormat::Depth32Float, bytes, 4))
        .transpose()?;
    let stencil = literal
        .stencil
        .as_deref()
        .map(|bytes| aspect(PixelFormat::Stencil8, bytes, 1))
        .transpose()?;
    Ok((depth, stencil))
}

/// A multisampled pass's render-target, depth or stencil output, as the Metal oracle reads it: the
/// attachment, how it is read sample by sample, its extent, and the region selected of every
/// sample, never resolved. `None` for a pass of one sample, whose outputs are copied, and for any
/// other output.
type SampleOutput<'a> = (&'a Texture, SampleRead, [u32; 2], [u32; 2], [u32; 2]);

fn multisampled_output<'a>(
    inputs: &RenderInputs<'_>,
    targets: &'a BTreeMap<u32, Texture>,
    depth: Option<&'a Texture>,
    stencil: Option<&'a Texture>,
) -> Result<Option<SampleOutput<'a>>, CaseError> {
    if inputs.case.raster_samples() == 1 {
        return Ok(None);
    }
    let render = inputs.render;
    let depth_stencil = || {
        render
            .depth_stencil
            .as_ref()
            .ok_or("the output's depth/stencil is not declared")
    };
    let attached = |texture: Option<&'a Texture>, what: &str| {
        texture.ok_or_else(|| format!("the output's {what} is not attached"))
    };
    Ok(Some(match render.output {
        RenderOutput::RenderTarget {
            index,
            origin,
            dimensions,
        } => {
            let target = render
                .color_targets
                .get(&index)
                .ok_or_else(|| format!("the output target {index} is not declared"))?;
            (
                attached(targets.get(&index), "target")?,
                sample_read(target.format)?,
                target.dimensions,
                origin,
                dimensions,
            )
        }
        RenderOutput::Depth { origin, dimensions } => (
            attached(depth, "depth")?,
            SampleRead::Depth,
            depth_stencil()?.dimensions,
            origin,
            dimensions,
        ),
        RenderOutput::Stencil { origin, dimensions } => (
            attached(stencil, "stencil")?,
            SampleRead::Stencil,
            depth_stencil()?.dimensions,
            origin,
            dimensions,
        ),
        _ => return Ok(None),
    }))
}

/// `length` bytes of each of `dimensions` texels from `origin` of `texture`'s level 0, rows in
/// order.
fn texels_of(
    texture: &Texture,
    origin: [u32; 2],
    dimensions: [u32; 2],
    texel: u64,
) -> Result<Vec<u8>, CaseError> {
    let [width, height] = dimensions.map(u64::from);
    let mut bytes = vec![0u8; (width * height * texel) as usize];
    texture
        .get_bytes(
            &mut bytes,
            width * texel,
            width * height * texel,
            Region::new_2d(origin[0].into(), origin[1].into(), width, height),
            0,
            0,
        )
        .map_err(CaseError::runtime("get_bytes"))?;
    Ok(bytes)
}

/// The functions of `inputs`' pipeline on `device`, vertex then fragment: its own from its
/// library, linking the functions of its linked libraries (each made with the own function's
/// constant values, [`LinkedSet`]); its companion's from the companion's.
fn functions(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<(Function, Option<Function>, LinkedSet), CaseError> {
    let library = |bitcode: &[u8]| {
        device
            .new_library_with_air(bitcode)
            .map_err(CaseError::runtime("new_library_with_air"))
    };
    let own = library(inputs.bitcode)?;
    let companion = inputs.companion.map(library).transpose()?;
    let render = inputs.render;
    let generated = render.companion_stage();
    let function = |stage: ShaderStage, function: &RenderFunction| {
        let library = match generated {
            Some(generated) if generated == stage => companion
                .as_ref()
                .ok_or("a companion without its library")?,
            _ => &own,
        };
        library
            .new_function_with_name_constant_values(&function.entry, &function.constants)
            .map_err(CaseError::runtime(format!(
                "function {}: new_function_with_name_constant_values",
                function.entry
            )))
    };
    let vertex = function(ShaderStage::Vertex, &render.vertex)?;
    let fragment = render
        .fragment
        .as_ref()
        .map(|fragment| function(ShaderStage::Fragment, fragment))
        .transpose()?;
    // The case's own function is the stage the companion did not generate.
    let own_stage = match (generated, &render.fragment) {
        (Some(ShaderStage::Vertex), Some(fragment)) => fragment,
        _ => &render.vertex,
    };
    let linked = linked_functions(device, inputs.linked, &own_stage.constants)?;
    Ok((vertex, fragment, linked))
}

/// The render pipeline descriptor of `inputs`, its functions made on `device`: the oracle's
/// descriptor of its functions, the case's own one linking its linked functions.
pub fn render_pipeline_descriptor(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<RenderPipelineDescriptor, CaseError> {
    Ok(descriptor_and_links(device, inputs)?.0)
}

/// [`render_pipeline_descriptor`], with the functions its own function links, which its
/// visible-function tables' entries name ([`LinkedSet`]).
fn descriptor_and_links(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<(RenderPipelineDescriptor, LinkedSet), CaseError> {
    let (vertex, fragment, linked) = functions(device, inputs)?;
    let generated = inputs.render.companion_stage();
    let descriptor_of = |stage: ShaderStage, function: &Function| {
        let descriptor = function.function_descriptor();
        if generated == Some(stage) {
            descriptor
        } else {
            descriptor.with_linked(linked.functions.clone())
        }
    };
    let descriptor = pipeline_descriptor(
        inputs.render,
        descriptor_of(ShaderStage::Vertex, &vertex),
        fragment
            .as_ref()
            .map(|fragment| descriptor_of(ShaderStage::Fragment, fragment)),
    )
    .map_err(|error| format!("the descriptor: {error}"))?;
    Ok((descriptor, linked))
}

/// The render pipeline state of `inputs` on `device`, of [`render_pipeline_descriptor`]'s
/// descriptor under the default pipeline options.
pub fn render_pipeline_state(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<RenderPipelineState, CaseError> {
    let descriptor = render_pipeline_descriptor(device, inputs)?;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(CaseError::runtime("new_render_pipeline_state"))
}

/// What one stage of a case binds, made on a device: its buffers and textures by Metal slot, the
/// members of its argument buffers by owner slot and field offset, and its visible-function tables
/// by Metal buffer index.
#[derive(Default)]
struct StageBound {
    buffers: BTreeMap<u32, Buffer>,
    textures: BTreeMap<u32, Texture>,
    argument_buffers: BTreeMap<(u32, u32), Buffer>,
    argument_textures: BTreeMap<(u32, u32), Texture>,
    argument_samplers: BTreeMap<(u32, u32), SamplerState>,
    commands: CaseCommands,
    tables: BTreeMap<u32, VisibleFunctionTable>,
}

impl StageBound {
    /// The word each argument-buffer member's owner holds at its field, as Metal's argument
    /// encoder writes it (`runtime_case::bound_buffers`): a buffer's `gpu_address`, a texture's,
    /// sampler state's, command buffer's or pipeline state's `gpu_resource_id`.
    fn member_words(&self) -> Result<BTreeMap<(u32, u32), u64>, CaseError> {
        let buffers = self
            .argument_buffers
            .iter()
            .map(|(&key, buffer)| (key, buffer.gpu_address().0));
        let textures = self
            .argument_textures
            .iter()
            .map(|(&key, texture)| (key, texture.gpu_resource_id().0));
        let samplers = self
            .argument_samplers
            .iter()
            .map(|(&key, sampler)| (key, sampler.gpu_resource_id().0));
        member_words(
            buffers
                .chain(textures)
                .chain(samplers)
                .chain(self.commands.words()),
        )
    }

    /// The members the draw makes resident: each argument-buffer buffer, texture and command
    /// buffer (a sampler or pipeline state is no resource).
    fn members(&self) -> impl Iterator<Item = ResourceRef<'_>> {
        self.argument_buffers
            .values()
            .map(Buffer::as_resource)
            .chain(self.argument_textures.values().map(Texture::as_resource))
            .chain(self.commands.resources())
    }
}

/// What each stage of one pipeline makes its resources with: the authored case, the usage the Metal
/// oracle gives each of the case's own textures, its own function's AIR interface, the state of
/// each pipeline template its argument buffers name, and the pipeline state and linked functions
/// that make each stage's visible-function tables.
struct StageContext<'a> {
    case: &'a AuthoredCase,
    texture_usage: &'a BTreeMap<u32, TextureUsage>,
    interface: &'a AirInterface,
    templates: &'a BTreeMap<u64, TemplateState>,
    pipeline: &'a RenderPipelineState,
    linked: &'a LinkedSet,
}

/// One stage's resources made on a device ([`StageContext::bound`]): what it binds, its sampler
/// states by slot, and the bytes the runtime wrote to its visible-function tables.
struct MadeStage {
    bound: StageBound,
    samplers: Vec<(u32, SamplerState)>,
    words: TableBuffers,
}

impl StageContext<'_> {
    /// `function`'s resources at `stage`, made on `device` as the Metal oracle makes them: its
    /// argument buffers' members, its command buffers and the pipeline states its argument buffers
    /// hold, its visible-function tables, its buffers holding their members' words, the host's
    /// commands encoded, its textures and its sampler states. The fills its textures need join
    /// `prep`.
    fn bound(
        &self,
        device: &Device,
        stage: ShaderStage,
        function: &RenderFunction,
        prep: &mut Vec<Prep>,
    ) -> Result<MadeStage, CaseError> {
        let resources = &function.resources;
        let mut stage_bound = StageBound::default();
        let mut fills = Vec::new();
        // Its argument buffers' members, as a kernel's are made (`runtime_case::CaseResources`):
        // an argument-buffer texture's usage is its role united with the declared access of the
        // case's own function's AIR member at its field.
        let argument_usage = if Some(stage) == case_stage(self.case) {
            argument_texture_usage(&resources.argument_buffer_textures, self.interface)?
        } else {
            BTreeMap::new()
        };
        // A command buffer's field holds the command buffer, not the record storage the mapped
        // case states for the executor.
        stage_bound.argument_buffers = resources
            .argument_buffer_buffers
            .iter()
            .filter(|(key, _)| !resources.command_buffers.contains_key(key))
            .map(|(&key, bytes)| Ok((key, buffer_with(device, bytes)?)))
            .collect::<Result<_, CaseError>>()?;
        stage_bound.argument_textures = argument_textures(
            device,
            &resources.argument_buffer_textures,
            &argument_usage,
            &mut fills,
        )?;
        stage_bound.argument_samplers =
            argument_samplers(device, &resources.argument_buffer_samplers)?;
        stage_bound.commands = CaseCommands::of(device, &resources.command_buffers)?;
        stage_bound.commands.states = pipeline_states(&resources.objects, self.templates)?;
        let fields = stage_bound.member_words()?;
        // Its visible-function tables, made by the pipeline state for the stage, as the oracle
        // makes them (`newVisibleFunctionTableWithDescriptor:stage:`).
        let made = tables_of(
            (stage, &resources.function_tables),
            self.linked,
            |descriptor| {
                self.pipeline
                    .new_visible_function_table(descriptor, render_stage(stage))
            },
            |function| self.pipeline.function_handle(function, render_stage(stage)),
        )?;
        stage_bound.tables = made.tables;
        for &index in resources.buffers.keys() {
            // A buffer of the case's own stage that holds a rasterization-rate map's parameter
            // data is made as the oracle makes it (`rate_map_buffer`).
            let map = (Some(stage) == case_stage(self.case))
                .then(|| {
                    self.case
                        .buffers
                        .iter()
                        .find(|buffer| buffer.binding == index)
                })
                .flatten()
                .and_then(|buffer| buffer.rasterization_rate_map.as_ref());
            // An argument buffer's owner holds each member's word at its field.
            let buffer = match map {
                Some(map) => rate_map_buffer(device, map)?,
                None => buffer_with(device, &owner_bytes(&resources.buffers, index, &fields)?)?,
            };
            stage_bound.buffers.insert(index, buffer);
        }
        // The host's commands, before the draw runs.
        encode_host_commands(
            &resources.command_buffers,
            &stage_bound.commands,
            &stage_bound.buffers,
            self.templates,
        )?;
        for (&slot, literal) in &resources.textures {
            let usage = self
                .texture_usage
                .get(&slot)
                .copied()
                .ok_or_else(|| format!("texture {slot} has no usage"))?;
            let texture = filled_texture(device, literal, usage, &mut fills)?;
            stage_bound.textures.insert(slot, texture);
        }
        prep.extend(fills.into_iter().map(Prep::Fill));
        let samplers = resources
            .samplers
            .iter()
            .map(|(&slot, sampler)| {
                let state = device
                    .new_sampler_state(&sampler_descriptor(sampler))
                    .map_err(CaseError::runtime("new_sampler_state"))?;
                Ok((slot, state))
            })
            .collect::<Result<_, CaseError>>()?;
        Ok(MadeStage {
            bound: stage_bound,
            samplers,
            words: made.words,
        })
    }
}

/// Bind each stage's buffers, textures, visible-function tables and sampler states at their
/// indices through that stage's setters (`setVertex*`, `setFragment*`, `setObject*`, `setMesh*`).
/// Metal has no object or mesh setter for a visible-function table, so neither stage binds one.
fn bind_stages(
    encoder: &mut RenderCommandEncoder,
    bound: &BTreeMap<ShaderStage, StageBound>,
    samplers: &[(ShaderStage, u32, SamplerState)],
) -> Result<(), CaseError> {
    for (&stage, stage_bound) in bound {
        for (&index, buffer) in &stage_bound.buffers {
            let buffer = Some(buffer);
            match stage {
                ShaderStage::Fragment => encoder.set_fragment_buffer(buffer, 0, index),
                ShaderStage::Object => encoder.set_object_buffer(buffer, 0, index),
                ShaderStage::Mesh => encoder.set_mesh_buffer(buffer, 0, index),
                _ => encoder.set_vertex_buffer(buffer, 0, index),
            }
            .map_err(CaseError::runtime("set_*_buffer"))?;
        }
        for (&index, texture) in &stage_bound.textures {
            let texture = Some(texture);
            match stage {
                ShaderStage::Fragment => encoder.set_fragment_texture(texture, index),
                ShaderStage::Object => encoder.set_object_texture(texture, index),
                ShaderStage::Mesh => encoder.set_mesh_texture(texture, index),
                _ => encoder.set_vertex_texture(texture, index),
            }
            .map_err(CaseError::runtime("set_*_texture"))?;
        }
        for (&index, table) in &stage_bound.tables {
            match stage {
                ShaderStage::Fragment => {
                    encoder.set_fragment_visible_function_table(Some(table), index)
                }
                ShaderStage::Object | ShaderStage::Mesh => {
                    return Err(format!(
                    "{stage:?} visible-function table {index}: no object or mesh setter binds one"
                )
                    .into())
                }
                _ => encoder.set_vertex_visible_function_table(Some(table), index),
            }
            .map_err(CaseError::runtime("set_*_visible_function_table"))?;
        }
    }
    for &(stage, slot, ref state) in samplers {
        let state = Some(state);
        match stage {
            ShaderStage::Fragment => encoder.set_fragment_sampler_state(state, slot),
            ShaderStage::Object => encoder.set_object_sampler_state(state, slot),
            ShaderStage::Mesh => encoder.set_mesh_sampler_state(state, slot),
            _ => encoder.set_vertex_sampler_state(state, slot),
        }
        .map_err(CaseError::runtime("set_*_sampler_state"))?;
    }
    Ok(())
}

/// The render stage of `stage`'s function, whose tables the pipeline state makes and whose handles
/// it gives.
fn render_stage(stage: ShaderStage) -> RenderStages {
    match stage {
        ShaderStage::Fragment => RenderStages::FRAGMENT,
        ShaderStage::Object => RenderStages::OBJECT,
        ShaderStage::Mesh => RenderStages::MESH,
        _ => RenderStages::VERTEX,
    }
}

/// What a render case's draw reads its output from, and the guards around its out-of-bounds
/// targets, once its command buffer completed.
struct Drawn {
    targets: BTreeMap<u32, Texture>,
    depth: Option<Texture>,
    stencil: Option<Texture>,
    bound: BTreeMap<ShaderStage, StageBound>,
    /// A multisampled output's readback, with its extent, origin, dimensions and texel bytes.
    readback: Option<(runtime_samples::SampleReadback, [[u32; 2]; 3], usize)>,
    /// The vertex or index buffer of each out-of-bounds target placed between guards, by target.
    guarded: BTreeMap<OutOfBoundsResource, Buffer>,
    /// A fragment case's imageblock helpers, whose resolver's buffer holds an imageblock output.
    imageblock: Option<Helpers>,
}

/// A render case's draw encoded on its device ([`encoded_render`]), not yet committed: its
/// command buffer, what the draw executed (the command buffer's executions after its
/// preparation's and before its readback's), and the bytes the runtime wrote to each
/// visible-function table it binds, by stage and Metal buffer index, as the executor's artifact
/// records a table's buffer (`tables::TableBuffers`).
pub struct EncodedRender {
    pub command_buffer: CommandBuffer,
    pub executions: Vec<Execution>,
    pub tables: TableBuffers,
    drawn: Drawn,
}

/// A render case's draw committed on its device ([`EncodedRender::commit`]): its command buffer,
/// what the draw executed, its tables' bytes, and what its output is read from once the command
/// buffer completed ([`CommittedRender::output`]).
pub struct CommittedRender {
    pub command_buffer: CommandBuffer,
    pub executions: Vec<Execution>,
    pub tables: TableBuffers,
    drawn: Drawn,
}

/// Encode `inputs`' draw on `device`: its pipeline state, attachments, vertex and index buffers (a
/// target's between guards), resources holding their bytes, argument buffers holding their
/// members' words and tables holding their entries' handles, one command buffer with the fills its
/// multisampled textures and attachments need, one render encoder with the case's state and its
/// one draw (between a fragment imageblock's helpers' draws), ended, and a multisampled output's
/// readback.
pub fn encoded_render(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<EncodedRender, CaseError> {
    let render = inputs.render;
    let (descriptor, linked) = descriptor_and_links(device, inputs)?;
    let pipeline = device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(CaseError::runtime("new_render_pipeline_state"))?;
    // A fragment imageblock's initializer and resolver, as the oracle draws them.
    let imageblock = match (&render.imageblock, inputs.imageblock) {
        (None, _) => None,
        (Some(cells), Some(master)) => {
            let resolves = matches!(render.output, RenderOutput::Imageblock { .. });
            Some(Helpers::new(device, &descriptor, master, cells, resolves)?)
        }
        (Some(_), None) => return Err("an imageblock without its reflected master".into()),
    };
    let layered = render.render_target_array_length > 0;
    let mut prep = Vec::new();
    let targets = render
        .color_targets
        .iter()
        .map(|(&index, target)| Ok((index, target_with(device, target, layered, &mut prep)?)))
        .collect::<Result<BTreeMap<_, _>, CaseError>>()?;
    let (depth, stencil) = match &render.depth_stencil {
        Some(literal) => depth_stencil_with(device, literal, &mut prep)?,
        None => (None, None),
    };
    let mut pass = RenderPassDescriptor::new();
    for (&index, texture) in &targets {
        let attachment = &mut pass.color_attachments[index as usize];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    // Each aspect loaded and stored, as the Metal oracle's are.
    pass.depth_attachment.texture = depth.clone();
    pass.depth_attachment.load_action = LoadAction::Load;
    pass.depth_attachment.store_action = StoreAction::Store;
    pass.stencil_attachment.texture = stencil.clone();
    pass.stencil_attachment.load_action = LoadAction::Load;
    pass.stencil_attachment.store_action = StoreAction::Store;
    pass.render_target_array_length = render.render_target_array_length.into();
    let samples = inputs.case.raster_samples();
    if targets.is_empty() && depth.is_none() && stencil.is_none() {
        // The Metal oracle's pass without attachments (`metal::platform`): one pixel, of the
        // case's sample count.
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = samples.into();
    }
    // The case's rasterization-rate map, made as the oracle makes it.
    if let Some(literal) = &inputs.case.render_pass_rate_map {
        let map = device
            .new_rasterization_rate_map(&literal.descriptor())
            .map_err(CaseError::runtime("new_rasterization_rate_map"))?;
        pass.rasterization_rate_map = Some(map);
    }
    if let Some(helpers) = &imageblock {
        helpers.prepare_pass(&mut pass, &pipeline)?;
    }
    // A vertex buffer that is an out-of-bounds target (a vertex input's, a control point's or a
    // patch input's) lies in a heap of its own between guards.
    let mut vertex_buffers = Vec::new();
    let mut guarded = BTreeMap::new();
    for input in &render.vertex_buffers {
        let target = OutOfBoundsResource::AttributeBuffer {
            location: input.location,
        };
        let buffer = match declared_reach(&render.out_of_bounds, target) {
            Some(reach) => {
                let buffer = guarded_buffer(device, &input.bytes, reach)?;
                guarded.insert(target, buffer.clone());
                buffer
            }
            None => buffer_with(device, &input.bytes)?,
        };
        // A dynamic layout's buffer is bound with the stride form.
        let stride = input.dynamic.then_some(input.stride);
        vertex_buffers.push((input.buffer_index, buffer, stride));
    }
    let mut bound: BTreeMap<ShaderStage, StageBound> = BTreeMap::new();
    let mut samplers: Vec<(ShaderStage, u32, SamplerState)> = Vec::new();
    let mut tables = TableBuffers::new();
    let context = StageContext {
        case: inputs.case,
        texture_usage: inputs.texture_usage,
        interface: inputs.interface,
        templates: inputs.templates,
        pipeline: &pipeline,
        linked: &linked,
    };
    for (stage, function) in render.stages() {
        let made = context.bound(device, stage, function, &mut prep)?;
        tables.extend(made.words);
        samplers.extend(
            made.samplers
                .into_iter()
                .map(|(slot, state)| (stage, slot, state)),
        );
        bound.insert(stage, made.bound);
    }
    // An index buffer that is an out-of-bounds target lies in a heap of its own between guards
    // too.
    let index_buffer = match &render.index_buffer {
        Some(bytes) => Some(
            match declared_reach(&render.out_of_bounds, OutOfBoundsResource::IndexBuffer) {
                Some(reach) => {
                    let buffer = guarded_buffer(device, bytes, reach)?;
                    guarded.insert(OutOfBoundsResource::IndexBuffer, buffer.clone());
                    buffer
                }
                None => buffer_with(device, bytes)?,
            },
        ),
        None => None,
    };
    let queue = device
        .new_command_queue()
        .map_err(CaseError::runtime("new_command_queue"))?;
    let command_buffer = queue
        .command_buffer()
        .map_err(CaseError::runtime("command_buffer"))?;
    for work in &prep {
        work.encode(&command_buffer)?;
    }
    // The case's own executions, between its preparation's and its readback's (each ended
    // encoder's join the command buffer's in encoding order).
    let prepared = command_buffer.executions().len();
    let mut encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(CaseError::runtime("render_command_encoder"))?;
    // A rate-mapped pass's viewport is the map's screen, as the oracle sets it.
    if let Some(literal) = &inputs.case.render_pass_rate_map {
        let [width, height] = literal.screen_size.map(f64::from);
        encoder
            .set_viewport(Viewport::new(0.0, 0.0, width, height, 0.0, 1.0))
            .map_err(CaseError::runtime("set_viewport"))?;
    }
    if let Some(helpers) = &imageblock {
        helpers.initialize(&mut encoder)?;
    }
    encoder
        .set_render_pipeline_state(&pipeline)
        .map_err(CaseError::runtime("set_render_pipeline_state"))?;
    // The argument buffers' members, resident for reading and writing in the vertex and fragment
    // stages, as the oracle's `useResource:usage:stages:` makes them.
    let members: Vec<ResourceRef<'_>> = bound.values().flat_map(StageBound::members).collect();
    if !members.is_empty() {
        encoder
            .use_resources(
                &members,
                ResourceUsage::READ | ResourceUsage::WRITE,
                RenderStages::VERTEX | RenderStages::FRAGMENT,
            )
            .map_err(CaseError::runtime("use_resources"))?;
    }
    // The draw's depth/stencil state, references, winding and cull mode, as the Metal oracle
    // sets them.
    let state = &render.depth_stencil_state;
    let depth_stencil_state = device
        .new_depth_stencil_state(&state.descriptor)
        .map_err(CaseError::runtime("new_depth_stencil_state"))?;
    encoder
        .set_depth_stencil_state(&depth_stencil_state)
        .map_err(CaseError::runtime("set_depth_stencil_state"))?;
    encoder.set_stencil_reference_values(state.front_reference, state.back_reference);
    encoder.set_front_facing_winding(state.winding);
    encoder.set_cull_mode(render.cull_mode);
    for (slot, buffer, stride) in &vertex_buffers {
        match stride {
            Some(stride) => encoder
                .set_vertex_buffer_with_stride(Some(buffer), 0, (*stride).into(), *slot)
                .map_err(CaseError::runtime("set_vertex_buffer_with_stride"))?,
            None => encoder
                .set_vertex_buffer(Some(buffer), 0, *slot)
                .map_err(CaseError::runtime("set_vertex_buffer"))?,
        }
    }
    bind_stages(&mut encoder, &bound, &samplers)?;
    let primitive = match &render.assembly {
        Assembly::Primitives(primitive) => Some(metal_primitive_type(*primitive)),
        Assembly::Patches { factors, .. } => {
            let factors = buffer_with(device, factors)?;
            encoder
                .set_tessellation_factor_buffer(
                    Some(&factors),
                    patches::FACTOR_OFFSET,
                    patches::FACTOR_INSTANCE_STRIDE,
                )
                .map_err(CaseError::runtime("set_tessellation_factor_buffer"))?;
            None
        }
    };
    match (primitive, render.draw, &index_buffer) {
        (
            None,
            DrawCall::Patches {
                control_points,
                patch_start,
                patch_count,
                patch_index: None,
                instance_count,
                base_instance,
                control_point_index: None,
            },
            None,
        ) => encoder
            .draw_patches(
                control_points,
                patch_start,
                patch_count,
                None,
                0,
                instance_count,
                base_instance,
            )
            .map_err(CaseError::runtime("draw_patches"))?,
        (
            Some(primitive_type),
            DrawCall::Primitives {
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            },
            None,
        ) => encoder
            .draw_primitives(
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            )
            .map_err(CaseError::runtime("draw_primitives"))?,
        (
            Some(primitive_type),
            DrawCall::Indexed {
                index_count,
                index_offset,
                instance_count,
                base_vertex,
                base_instance,
                ..
            },
            Some(buffer),
        ) => {
            let index_type = match inputs
                .case
                .draw
                .as_ref()
                .and_then(|d| d.index_buffer.as_ref())
            {
                Some(literal) => match literal.index_type {
                    crate::case::IndexType::Uint16 => IndexType::UInt16,
                    crate::case::IndexType::Uint32 => IndexType::UInt32,
                },
                None => return Err("an indexed draw without its literal".into()),
            };
            encoder
                .draw_indexed_primitives(
                    primitive_type,
                    index_count,
                    index_type,
                    buffer,
                    index_offset,
                    instance_count,
                    base_vertex,
                    base_instance,
                )
                .map_err(CaseError::runtime("draw_indexed_primitives"))?
        }
        (_, draw, _) => return Err(format!("the draw {draw:?}, a later runtime step").into()),
    }
    if let Some(helpers) = &imageblock {
        helpers.resolve(&mut encoder)?;
    }
    encoder
        .end_encoding()
        .map_err(CaseError::runtime("end_encoding"))?;
    // Each `Private` command buffer copied into its twin, so its commands read back.
    for stage_bound in bound.values() {
        encode_twin_copies(&command_buffer, &stage_bound.commands)?;
    }
    let mut executions = command_buffer.executions().split_off(prepared);
    // The case's own draw's, not its imageblock helpers'.
    if let Some(helpers) = &imageblock {
        let (before, after) = helpers.draws();
        if executions.len() != before + 1 + after {
            return Err(format!(
                "the pass recorded {} executions of the case's draw and {} helper draws",
                executions.len(),
                before + after
            )
            .into());
        }
        executions = executions.split_off(before);
        executions.truncate(1);
    }
    let readback = match multisampled_output(inputs, &targets, depth.as_ref(), stencil.as_ref())? {
        Some((texture, read, extent, origin, dimensions)) => Some((
            runtime_samples::encode(device, &command_buffer, texture, read, extent, samples)?,
            [extent, origin, dimensions],
            read.texel_bytes() * samples as usize,
        )),
        None => None,
    };
    Ok(EncodedRender {
        command_buffer,
        executions,
        tables,
        drawn: Drawn {
            targets,
            depth,
            stencil,
            bound,
            readback,
            guarded,
            imageblock,
        },
    })
}

impl EncodedRender {
    /// Each command buffer the draw's stages execute (`render`'s), in stage and key order, encoded
    /// on a command buffer of its own of the draw's queue and not committed
    /// ([`encoded_execution`]): with the buffers of the stage that holds it, and every buffer and
    /// argument-buffer buffer of the draw resident; each parent pipeline its token's template's
    /// among `templates`. Committed after the draw's command buffer, each runs once the draw
    /// completed.
    pub fn executions(
        &self,
        render: &RenderCase,
        templates: &BTreeMap<u64, TemplateState>,
    ) -> Result<Vec<CommandBuffer>, CaseError> {
        let bound = &self.drawn.bound;
        let resident: Vec<ResourceRef<'_>> = bound
            .values()
            .flat_map(|stage| {
                stage
                    .buffers
                    .values()
                    .chain(stage.argument_buffers.values())
            })
            .map(Buffer::as_resource)
            .collect();
        let mut executions = Vec::new();
        for (stage, function) in render.stages() {
            let Some(stage_bound) = bound.get(&stage) else {
                continue;
            };
            for (&key, resource) in &function.resources.command_buffers {
                if resource.execute.is_some() {
                    executions.push(encoded_execution(
                        self.command_buffer.command_queue(),
                        (&stage_bound.buffers, &resident),
                        &stage_bound.commands,
                        (key, resource),
                        templates,
                    )?);
                }
            }
        }
        Ok(executions)
    }

    /// Commit the encoded draw.
    pub fn commit(self) -> Result<CommittedRender, CaseError> {
        let EncodedRender {
            command_buffer,
            executions,
            tables,
            drawn,
        } = self;
        command_buffer
            .commit()
            .map_err(CaseError::runtime("commit"))?;
        Ok(CommittedRender {
            command_buffer,
            executions,
            tables,
            drawn,
        })
    }
}

/// [`encoded_render`], committed. Nothing is committed when a call is refused.
pub fn commit_render(
    device: &Device,
    inputs: &RenderInputs<'_>,
) -> Result<CommittedRender, CaseError> {
    encoded_render(device, inputs)?.commit()
}

impl CommittedRender {
    /// Read back each stage's command buffers once the draw completed
    /// (`CaseCommands::check_commands`).
    pub fn check_commands(&self) -> Result<(), CaseError> {
        self.drawn
            .bound
            .values()
            .try_for_each(|stage| stage.commands.check_commands())
    }

    /// What the draw changed around each of its out-of-bounds targets it placed between guards
    /// (`runtime_case::guard_count`), by target, once the command buffer completed: each vertex
    /// and index buffer target's.
    pub fn guard_counts(&self) -> Result<BTreeMap<OutOfBoundsResource, GuardCount>, CaseError> {
        self.drawn
            .guarded
            .iter()
            .map(|(&target, buffer)| Ok((target, guard_count(buffer)?)))
            .collect()
    }

    /// The bytes `inputs`' output selects, once the command buffer completed.
    pub fn output(&self, inputs: &RenderInputs<'_>) -> Result<Vec<u8>, CaseError> {
        let render = inputs.render;
        if let Some((readback, [extent, origin, dimensions], texel)) = &self.drawn.readback {
            return Ok(select_tightly_packed_2d(
                &readback.bytes()?,
                *extent,
                *origin,
                *dimensions,
                *texel,
            )?);
        }
        self.drawn.read(
            &render.output,
            |stage| render.function(stage),
            &render.color_targets,
        )
    }
}

impl Drawn {
    /// The bytes `output` selects of what the draw left, once the command buffer completed: a
    /// stage's resource, of the function `function_of` gives at its stage, or a region of a color
    /// target of `color_targets`, the depth or the stencil attachment, or the imageblock.
    fn read<'a>(
        &self,
        output: &RenderOutput,
        function_of: impl Fn(ShaderStage) -> Option<&'a RenderFunction>,
        color_targets: &BTreeMap<u32, ColorTarget>,
    ) -> Result<Vec<u8>, CaseError> {
        let stage_of = |stage: ShaderStage| {
            self.bound
                .get(&stage)
                .ok_or_else(|| format!("the output's stage {stage:?} binds nothing"))
        };
        match *output {
            RenderOutput::None => Ok(Vec::new()),
            RenderOutput::Buffer {
                stage,
                binding,
                offset,
                length,
            } => read_buffer(
                stage_of(stage)?
                    .buffers
                    .get(&binding)
                    .ok_or_else(|| format!("the output buffer {binding} is not bound"))?,
                offset,
                length,
            ),
            RenderOutput::Texture {
                stage,
                binding,
                origin,
                dimensions,
            } => read_texture(
                stage_of(stage)?
                    .textures
                    .get(&binding)
                    .ok_or_else(|| format!("the output texture {binding} is not bound"))?,
                function_of(stage)
                    .and_then(|function| function.resources.textures.get(&binding))
                    .ok_or_else(|| format!("the output texture {binding} is not declared"))?,
                origin,
                dimensions,
            ),
            RenderOutput::ArgumentBufferBuffer {
                stage,
                owner,
                field_offset,
                offset,
                length,
            } => read_buffer(
                stage_of(stage)?
                    .argument_buffers
                    .get(&(owner, field_offset))
                    .ok_or_else(|| {
                        format!("no argument-buffer buffer at {owner}+{field_offset}")
                    })?,
                offset,
                length,
            ),
            RenderOutput::ArgumentBufferTexture {
                stage,
                owner,
                field_offset,
                origin,
                dimensions,
            } => read_texture(
                stage_of(stage)?
                    .argument_textures
                    .get(&(owner, field_offset))
                    .ok_or_else(|| {
                        format!("no argument-buffer texture at {owner}+{field_offset}")
                    })?,
                function_of(stage)
                    .and_then(|function| {
                        function
                            .resources
                            .argument_buffer_textures
                            .get(&(owner, field_offset))
                    })
                    .ok_or_else(|| {
                        format!("argument-buffer texture {owner}+{field_offset} undeclared")
                    })?,
                origin,
                dimensions,
            ),
            RenderOutput::RenderTarget {
                index,
                origin,
                dimensions,
            } => {
                let texture = self
                    .targets
                    .get(&index)
                    .ok_or_else(|| format!("the output target {index} is not attached"))?;
                let pixel = color_targets
                    .get(&index)
                    .ok_or_else(|| format!("the output target {index} is not declared"))?
                    .format
                    .bytes_per_pixel() as u64;
                texels_of(texture, origin, dimensions, pixel)
            }
            RenderOutput::Depth { origin, dimensions } => {
                let texture = self
                    .depth
                    .as_ref()
                    .ok_or("the output's depth is not attached")?;
                texels_of(texture, origin, dimensions, 4)
            }
            RenderOutput::Stencil { origin, dimensions } => {
                let texture = self
                    .stencil
                    .as_ref()
                    .ok_or("the output's stencil is not attached")?;
                texels_of(texture, origin, dimensions, 1)
            }
            RenderOutput::Imageblock {
                member,
                origin,
                dimensions,
            } => Ok(self
                .imageblock
                .as_ref()
                .ok_or("an imageblock output without its helpers")?
                .resolved()?
                .select(member, origin, dimensions)?),
        }
    }
}

/// The stage of `case`'s own function, whose buffers the case authors: none for a kernel.
fn case_stage(case: &AuthoredCase) -> Option<ShaderStage> {
    match case.stage {
        Stage::Vertex => Some(ShaderStage::Vertex),
        Stage::Fragment => Some(ShaderStage::Fragment),
        Stage::Object => Some(ShaderStage::Object),
        Stage::Mesh => Some(ShaderStage::Mesh),
        Stage::Kernel => None,
    }
}
