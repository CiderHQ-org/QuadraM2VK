//! An object or mesh case's draw through the Metal-like runtime, as the Metal oracle draws it
//! (`metal::platform::mesh`): a library of the case's AIR, the function of each stage its draw has
//! (its own and the partners it names) specialized with the case's constant values, the generated
//! stages of a case that states an observation or an object launch from a library of the
//! companion's AIR, a mesh render pipeline state of the oracle's descriptor (its color formats, or
//! the one-pixel sink where it does not rasterize, and its rasterization), its color targets loaded
//! and stored, its own stage's
//! resources made and bound as a render case's stage's are ([`StageContext::bound`],
//! [`bind_stages`]: buffers, argument buffers and their members, textures and sampler states,
//! through that stage's `setObject*` or `setMesh*` setters), and its one `draw_mesh_threadgroups`
//! or `draw_mesh_threads`. A device without `VK_EXT_mesh_shader` draws through the runtime's
//! emulation, which announces `emulated-mesh-draws`.

use super::*;
use crate::candidate::engine::mesh::MeshCase;
use crate::case::MeshGrid;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, MeshPipelineDescriptor, StageSource};
use metal2vulkan_runtime::Size;

/// What an object or mesh case's draw needs: the authored case, its mesh mapping, its AIR bitcode,
/// its generated companion's (the generated stages of a case that states an observation or an
/// object launch), the
/// usage the Metal oracle gives each of its own textures (`runtime_case::texture_usage`), and its
/// own function's AIR interface, whose argument-buffer members give each argument-buffer texture's
/// usage.
pub struct MeshInputs<'a> {
    pub case: &'a AuthoredCase,
    pub mesh: &'a MeshCase,
    pub bitcode: &'a [u8],
    pub companion: Option<&'a [u8]>,
    pub texture_usage: &'a BTreeMap<u32, TextureUsage>,
    pub interface: &'a AirInterface,
}

/// The mesh render pipeline state of `mesh`'s draw on `device`, of the oracle's descriptor: the
/// function of each stage its draw has from a library of `bitcode`, its AIR, or of `companion`, the
/// generated companion's, for the generated stages, specialized with the case's constant values;
/// its color formats (or the one-pixel sink's where it does not rasterize) and its rasterization;
/// and `support_indirect_command_buffers` where `indirect`, as a pipeline template's state.
pub fn mesh_pipeline_state(
    device: &Device,
    mesh: &MeshCase,
    (bitcode, companion): (&[u8], Option<&[u8]>),
    indirect: bool,
) -> Result<RenderPipelineState, CaseError> {
    let library = device
        .new_library_with_air(bitcode)
        .map_err(CaseError::runtime("new_library_with_air"))?;
    // The generated stages' library, which a draw that states an observation or an object launch
    // has.
    let generated = match (companion, mesh.generated.is_empty()) {
        (Some(bitcode), false) => Some(
            device
                .new_library_with_air(bitcode)
                .map_err(CaseError::runtime("new_library_with_air"))?,
        ),
        (None, true) => None,
        _ => return Err("a generated companion without its library, or its stages".into()),
    };
    let function = |stage: ShaderStage, function: &RenderFunction| {
        let (library, what) = match &generated {
            Some(generated) if mesh.generated.contains(&stage) => {
                (generated, "the generated companion's ")
            }
            _ => (&library, ""),
        };
        library
            .new_function_with_name_constant_values(&function.entry, &function.constants)
            .map_err(CaseError::runtime(format!(
                "{what}function {}: new_function_with_name_constant_values",
                function.entry
            )))
    };
    let source = |function: Function| StageSource::Function(function.function_descriptor());
    let mut descriptor =
        MeshPipelineDescriptor::new(source(function(ShaderStage::Mesh, &mesh.mesh)?));
    descriptor.object = mesh
        .object
        .as_ref()
        .map(|object| function(ShaderStage::Object, object))
        .transpose()?
        .map(source);
    descriptor.fragment = mesh
        .fragment
        .as_ref()
        .map(|fragment| function(ShaderStage::Fragment, fragment))
        .transpose()?
        .map(source);
    descriptor.rasterization_enabled = mesh.rasterization;
    for (&index, target) in &mesh.color_targets {
        let attachment = descriptor
            .color_attachments
            .get_mut(index as usize)
            .ok_or_else(|| format!("color target {index} is past the descriptor's attachments"))?;
        *attachment = Some(ColorAttachmentDescriptor::new(pixel_format(target.format)));
    }
    descriptor.support_indirect_command_buffers = indirect;
    device
        .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(CaseError::runtime("new_mesh_render_pipeline_state"))
}

/// Encode `inputs`' draw on `device`: its pipeline state ([`mesh_pipeline_state`]), color targets,
/// its own stage's resources, one command buffer with the fills its textures need, and one render
/// encoder with its one draw, ended. Committed ([`EncodedRender::commit`]), its output is read by
/// [`CommittedRender::mesh_output`].
pub fn encoded_mesh(device: &Device, inputs: &MeshInputs<'_>) -> Result<EncodedRender, CaseError> {
    let mesh = inputs.mesh;
    let pipeline = mesh_pipeline_state(device, mesh, (inputs.bitcode, inputs.companion), false)?;
    let mut prep = Vec::new();
    let targets = mesh
        .color_targets
        .iter()
        .map(|(&index, target)| Ok((index, target_with(device, target, false, &mut prep)?)))
        .collect::<Result<BTreeMap<_, _>, CaseError>>()?;
    let mut pass = RenderPassDescriptor::new();
    for (&index, texture) in &targets {
        let attachment = &mut pass.color_attachments[index as usize];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
    }
    // The case's own function links nothing: the schema refuses an object or mesh case's
    // visible-function references and tables.
    let own = mesh
        .function(mesh.stage)
        .ok_or("the case's own stage has no function")?;
    let linked = linked_functions(device, &[], &own.constants)?;
    let templates = BTreeMap::new();
    let context = StageContext {
        case: inputs.case,
        texture_usage: inputs.texture_usage,
        interface: inputs.interface,
        templates: &templates,
        pipeline: &pipeline,
        linked: &linked,
    };
    let mut bound: BTreeMap<ShaderStage, StageBound> = BTreeMap::new();
    let mut samplers = Vec::new();
    let mut tables = TableBuffers::new();
    for (stage, function) in mesh.stages() {
        let made = context.bound(device, stage, function, &mut prep)?;
        tables.extend(made.words);
        samplers.extend(
            made.samplers
                .into_iter()
                .map(|(slot, state)| (stage, slot, state)),
        );
        bound.insert(stage, made.bound);
    }
    let queue = device
        .new_command_queue()
        .map_err(CaseError::runtime("new_command_queue"))?;
    let command_buffer = queue
        .command_buffer()
        .map_err(CaseError::runtime("command_buffer"))?;
    for work in &prep {
        work.encode(&command_buffer)?;
    }
    // The case's own execution, after its preparation's.
    let prepared = command_buffer.executions().len();
    let mut encoder = command_buffer
        .render_command_encoder(&pass)
        .map_err(CaseError::runtime("render_command_encoder"))?;
    encoder
        .set_render_pipeline_state(&pipeline)
        .map_err(CaseError::runtime("set_render_pipeline_state"))?;
    // The argument buffers' members, resident for reading and writing in the case's own stage, as
    // the oracle's `useResource:usage:stages:` makes them.
    let members: Vec<ResourceRef<'_>> = bound.values().flat_map(StageBound::members).collect();
    if !members.is_empty() {
        encoder
            .use_resources(
                &members,
                ResourceUsage::READ | ResourceUsage::WRITE,
                render_stage(mesh.stage),
            )
            .map_err(CaseError::runtime("use_resources"))?;
    }
    bind_stages(&mut encoder, &bound, &samplers)?;
    // An object function's threadgroup memory, as the oracle sets it; the case schema refuses a
    // mesh case's.
    if inputs.case.stage == crate::case::Stage::Object {
        for resource in &inputs.case.threadgroup_memory {
            encoder
                .set_object_threadgroup_memory_length(resource.length, resource.binding)
                .map_err(CaseError::runtime("set_object_threadgroup_memory_length"))?;
        }
    }
    let size = |[x, y, z]: [u32; 3]| Size::new(x.into(), y.into(), z.into());
    let (object, threads) = (
        size(mesh.threads_per_object_threadgroup),
        size(mesh.threads_per_mesh_threadgroup),
    );
    match mesh.grid {
        MeshGrid::Threadgroups(groups) => {
            encoder.draw_mesh_threadgroups(size(groups), object, threads)
        }
        MeshGrid::Threads(grid) => encoder.draw_mesh_threads(size(grid), object, threads),
    }
    .map_err(CaseError::runtime("draw_mesh_*"))?;
    encoder
        .end_encoding()
        .map_err(CaseError::runtime("end_encoding"))?;
    let executions = command_buffer.executions().split_off(prepared);
    Ok(EncodedRender {
        command_buffer,
        executions,
        tables,
        drawn: Drawn {
            targets,
            depth: None,
            stencil: None,
            bound,
            readback: None,
            guarded: BTreeMap::new(),
            imageblock: None,
        },
    })
}

impl CommittedRender {
    /// The bytes `mesh`'s output selects, once the command buffer completed: a resource of its own
    /// stage, or a region of a render target.
    pub fn mesh_output(&self, mesh: &MeshCase) -> Result<Vec<u8>, CaseError> {
        self.drawn.read(
            &mesh.output,
            |stage| mesh.function(stage),
            &mesh.color_targets,
        )
    }
}
