//! An object or mesh case's draw on the Metal device: its mesh render pipeline
//! (`MTLMeshRenderPipelineDescriptor`) of the case's own function at its stage and the partners
//! its draw names ([`MeshDraw`]), each specialized with the case's constant values from the case's
//! own library, or, for a case that states an observation or an object launch, the generated
//! stages in its partners' places (`companion::MeshStages`: a mesh case's observer fragment and
//! payload object, an object case's launch observer and its observer fragment) from the
//! companion's library, its resources bound through
//! its stage's setters (`setObjectBuffer:`, `setMeshBuffer:`, ...), and its one
//! `drawMeshThreadgroups:` or `drawMeshThreads:`.
//!
//! The payload length is the descriptor's default (`payloadMemoryLength` 0), under which the M3
//! Max (macOS 27.0.1) runs a mesh function reading the object function's whole payload, as with
//! any explicit length that holds it. A draw without rasterization has no fragment function and
//! draws into a private one-pixel sink, as a rasterization-disabled vertex case does; its mesh
//! function's device writes land (measured on the same device). A rasterizing pipeline always has
//! its fragment function: the case schema refuses a rasterizing draw that names none, since Metal's
//! descriptor validation aborts the process on such a pipeline, and this executor refuses one too
//! before Metal sees it.

use super::*;
use crate::case::{MeshDraw, MeshGrid, Stage};
use crate::companion::Companion;
use objc2_metal::MTLMeshRenderPipelineDescriptor;

/// The functions of a mesh render pipeline, by stage.
struct MeshFunctions {
    object: Option<Function>,
    mesh: Function,
    fragment: Option<Function>,
}

/// `draw`'s pipeline functions for `case`: its own `entry` function at its stage, each partner the
/// draw names from the case's own `library`, specialized with the case's constant values, and the
/// generated stages (`companion`) of a draw that states an observation or an object launch, from
/// the companion's library.
fn make_mesh_functions(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    draw: &MeshDraw,
    (library, entry): (&ProtocolObject<dyn MTLLibrary>, &Function),
    constants: &[crate::literal::LiteralFunctionConstant],
    companion: Option<&Companion>,
) -> Result<MeshFunctions, String> {
    let generated = match (draw.generates(), companion) {
        (false, None) => None,
        (true, Some(stages @ Companion::MeshStages(_))) => {
            Some((stages, companion_library(device, stages)?))
        }
        (generates, companion) => {
            return Err(format!(
                "a mesh draw that takes generated stages ({generates}) has companion {:?}",
                companion.map(Companion::entry)
            ))
        }
    };
    let function = |stage| -> Result<Option<Function>, String> {
        if stage == case.stage {
            return Ok(Some(entry.clone()));
        }
        let made = generated
            .as_ref()
            .and_then(|(companion, library)| Some((companion.stage_entry(stage)?, library)));
        match (draw.partner(stage), made) {
            (named, None) => named
                .map(|name| make_function(library, &NSString::from_str(name), constants))
                .transpose(),
            (None, Some((name, library))) => library
                .newFunctionWithName(&NSString::from_str(name))
                .map(Some)
                .ok_or_else(|| format!("the generated companion has no entry {name}")),
            (Some(named), Some((name, _))) => Err(format!(
                "a mesh draw names {named:?} at the {} stage, and its companion generates {name}",
                stage.entry_list()
            )),
        }
    };
    Ok(MeshFunctions {
        object: function(Stage::Object)?,
        mesh: function(Stage::Mesh)?.ok_or_else(|| {
            "an object case's mesh draw names no mesh function (mesh_companion)".to_string()
        })?,
        fragment: function(Stage::Fragment)?,
    })
}

/// The mesh render pipeline of `functions` for `case`'s `draw`: rasterizing into the case's render
/// targets through its fragment function, or, without rasterization, into the one-pixel sink;
/// supporting indirect command buffers where `indirect`.
fn make_mesh_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    draw: &MeshDraw,
    functions: &MeshFunctions,
    indirect: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalRunError> {
    if draw.rasterization && functions.fragment.is_none() {
        // Metal's descriptor validation aborts the process here ("fragmentFunction must not be
        // nil"); the case schema refuses such a draw first.
        return Err("a rasterizing mesh render pipeline needs a fragment function".into());
    }
    let descriptor = MTLMeshRenderPipelineDescriptor::new();
    unsafe {
        descriptor.setObjectFunction(functions.object.as_deref());
        descriptor.setMeshFunction(Some(&functions.mesh));
        descriptor.setFragmentFunction(functions.fragment.as_deref());
    }
    descriptor.setRasterizationEnabled(draw.rasterization);
    descriptor.setSupportIndirectCommandBuffers(indirect);
    let attachments = descriptor.colorAttachments();
    for target in &case.render_targets {
        let attachment = unsafe { attachments.objectAtIndexedSubscript(target.index as usize) };
        attachment.setPixelFormat(metal_pixel_format(target.format));
    }
    if !draw.rasterization {
        let attachment = unsafe { attachments.objectAtIndexedSubscript(0) };
        attachment.setPixelFormat(MTLPixelFormat::R8Unorm);
    }
    compile::bounded("mesh render pipeline creation", || {
        device.newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
    })
    .map_err(|error| pipelines::pipeline_failure(device, "create mesh render pipeline", &error))
}

/// The mesh render pipeline state of pipeline template `case`, an object or mesh case whose draw
/// does not rasterize (the checker admits no other), supporting indirect command buffers: its own
/// function `entry` from `library`, its partners and generated stages as its own draw takes them
/// ([`make_mesh_functions`]), and the one-pixel sink's format, which the execution's pass has.
pub(super) fn make_template_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    (library, entry): (&ProtocolObject<dyn MTLLibrary>, &Function),
    constants: &[crate::literal::LiteralFunctionConstant],
    companion: Option<&Companion>,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, MetalRunError> {
    let draw = case
        .mesh_draw
        .as_ref()
        .ok_or_else(|| "an object or mesh template has no mesh draw".to_string())?;
    if draw.rasterization {
        return Err("a mesh template rasterizes: its execution's pass has no target for it".into());
    }
    let functions =
        make_mesh_functions(device, case, draw, (library, entry), constants, companion)?;
    make_mesh_pipeline(device, case, draw, &functions, true)
}

/// The setters of the case's own stage, object or mesh.
#[derive(Clone, Copy)]
struct StageSetters<'a> {
    encoder: &'a ProtocolObject<dyn MTLRenderCommandEncoder>,
    stage: Stage,
}

impl StageSetters<'_> {
    fn buffer(self, buffer: &Buffer, index: u32) {
        let index = index as usize;
        match self.stage {
            Stage::Object => unsafe {
                self.encoder
                    .setObjectBuffer_offset_atIndex(Some(&**buffer), 0, index)
            },
            _ => unsafe {
                self.encoder
                    .setMeshBuffer_offset_atIndex(Some(&**buffer), 0, index)
            },
        }
    }

    fn texture(self, texture: &Texture, index: u32) {
        let index = index as usize;
        match self.stage {
            Stage::Object => unsafe {
                self.encoder
                    .setObjectTexture_atIndex(Some(&**texture), index)
            },
            _ => unsafe { self.encoder.setMeshTexture_atIndex(Some(&**texture), index) },
        }
    }

    fn sampler(self, sampler: &ProtocolObject<dyn objc2_metal::MTLSamplerState>, index: u32) {
        let index = index as usize;
        match self.stage {
            Stage::Object => unsafe {
                self.encoder
                    .setObjectSamplerState_atIndex(Some(sampler), index)
            },
            _ => unsafe {
                self.encoder
                    .setMeshSamplerState_atIndex(Some(sampler), index)
            },
        }
    }

    fn render_stages(self) -> MTLRenderStages {
        match self.stage {
            Stage::Object => MTLRenderStages::Object,
            _ => MTLRenderStages::Mesh,
        }
    }
}

/// Execute an object or mesh case: its draw, then its selected output.
pub(super) fn execute_mesh(
    context: &MetalContext,
    case: &AuthoredCase,
    source: OracleSource<'_>,
    resources: &LiteralResources,
    facts: ShaderFacts<'_>,
) -> Result<Vec<u8>, MetalRunError> {
    let ShaderFacts {
        reflection,
        interface,
    } = facts;
    let draw = case
        .mesh_draw
        .as_ref()
        .ok_or_else(|| "an object or mesh case has no mesh draw".to_string())?;
    // The case schema refuses what no object or mesh stage binds (acceleration structures,
    // function tables, out-of-bounds targets, indirect command buffers and pipeline states).
    let device = &context.device;
    let NativeBuffers {
        mut buffers,
        device_buffer_array_elements,
    } = make_buffers(device, resources, reflection, interface)?;
    let indirect_commands = make_indirect_commands(device, case)?;
    let (library, mut fuel) = load_producer_library(
        device,
        &source,
        &indirect_commands,
        interface,
        &mut buffers,
        &[],
    )?;
    let entry = make_function(
        &library,
        &NSString::from_str(&case.entry),
        &resources.function_constants,
    )?;
    let functions = make_mesh_functions(
        device,
        case,
        draw,
        (&library, &entry),
        &resources.function_constants,
        source.companion,
    )?;
    let mut render_targets = make_render_targets(device, &context.queue, resources, false)?;
    if !draw.rasterization {
        // Metal still requires a render-pass attachment to create an encoder. Rasterization is
        // disabled on the pipeline, so this private 1x1 sink is API scaffolding and cannot
        // participate in the authored observation.
        render_targets.push((0, make_rasterization_sink(device)?));
    }
    let pipeline = make_mesh_pipeline(device, case, draw, &functions, false)?;
    if functions.object.is_some() {
        threads_per_threadgroup_within_limit(
            draw.threads_per_object_threadgroup,
            pipeline.maxTotalThreadsPerObjectThreadgroup(),
            "Metal object",
        )?;
    }
    threads_per_threadgroup_within_limit(
        draw.threads_per_mesh_threadgroup,
        pipeline.maxTotalThreadsPerMeshThreadgroup(),
        "Metal mesh",
    )?;
    let argument_buffer_buffers = make_argument_buffer_buffers(device, case, resources)?;
    let textures = make_textures(device, &context.queue, resources, interface)?;
    let texture_arrays = make_texture_arrays(device, &context.queue, resources, interface)?;
    let argument_buffer_textures =
        make_argument_buffer_textures(device, &context.queue, resources, interface)?;
    encode_argument_buffer_buffers(&entry, case, &buffers, &argument_buffer_buffers, interface)?;
    encode_argument_buffer_textures(&entry, &buffers, &argument_buffer_textures, interface)?;
    let _argument_buffer_samplers =
        encode_argument_buffer_samplers(device, &entry, case, &buffers, interface)?;
    let samplers = make_samplers(device, case)?;

    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let pass_attachments = pass.colorAttachments();
    for (index, texture) in &render_targets {
        let attachment = unsafe { pass_attachments.objectAtIndexedSubscript(*index as usize) };
        attachment.setTexture(Some(&**texture));
        attachment.setLoadAction(MTLLoadAction::Load);
        attachment.setStoreAction(MTLStoreAction::Store);
    }
    let command_buffer = context
        .queue
        .commandBuffer()
        .ok_or_else(|| "MTLCommandQueue::commandBuffer returned nil".to_string())?;
    let encoder = command_buffer
        .renderCommandEncoderWithDescriptor(&pass)
        .ok_or_else(|| "renderCommandEncoderWithDescriptor returned nil".to_string())?;
    encoder.setRenderPipelineState(&pipeline);
    let stage = StageSetters {
        encoder: &encoder,
        stage: case.stage,
    };
    argument_buffer_resources_resident_on_render(
        &encoder,
        &argument_buffer_textures,
        &argument_buffer_buffers,
        &indirect_commands.commands,
        stage.render_stages(),
    );
    for (binding, buffer) in &buffers {
        stage.buffer(buffer, *binding);
    }
    for (binding, texture) in &textures {
        stage.texture(texture, *binding);
    }
    for (binding, elements) in &texture_arrays {
        for (element, texture) in elements.iter().enumerate() {
            stage.texture(texture, *binding + element as u32);
        }
    }
    for (binding, sampler) in &samplers {
        stage.sampler(sampler, *binding);
    }
    // Only an object function's threadgroup memory has a setter; the case schema refuses a mesh
    // case's.
    if case.stage == Stage::Object {
        for resource in &case.threadgroup_memory {
            unsafe {
                encoder.setObjectThreadgroupMemoryLength_atIndex(
                    resource.length as usize,
                    resource.binding as usize,
                );
            }
        }
    }
    let (object, mesh) = (
        mtl_size(draw.threads_per_object_threadgroup),
        mtl_size(draw.threads_per_mesh_threadgroup),
    );
    match draw.grid {
        MeshGrid::Threadgroups(groups) => encoder
            .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                mtl_size(groups),
                object,
                mesh,
            ),
        MeshGrid::Threads(threads) => encoder
            .drawMeshThreads_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                mtl_size(threads),
                object,
                mesh,
            ),
    }
    encoder.endEncoding();
    arm_fuel(fuel.as_mut(), &buffers);
    commit_and_wait(&command_buffer, "Metal mesh draw")?;
    check_fuel(fuel.as_ref())?;
    Ok(selected_output(
        case,
        &buffers,
        &argument_buffer_buffers,
        &textures,
        &texture_arrays,
        &argument_buffer_textures,
        MetalOutputResources {
            device_buffer_array_elements: &device_buffer_array_elements,
            colors: &render_targets,
            depth_stencil: None,
            fragment_imageblock: None,
        },
    )?)
}
