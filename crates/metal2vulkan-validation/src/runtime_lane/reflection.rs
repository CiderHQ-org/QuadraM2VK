//! Reflection through the runtime on the device the cfg selects, under the validation layer. The
//! functions are the owned reflection fixtures (`validation/fixtures/public/reflection/`), whose
//! answers Metal gave on an M3 Max and an M4 (`metal-answers.txt` there, which the product's
//! `reflection::tests` compare every answer with line for line).
//!
//! - Each `_with_reflection` creation form returns the state and, under `BINDING_INFO`,
//!   `BUFFER_TYPE_INFO` or both, the descriptor's reflection; under neither, none, as Metal leaves
//!   `reflection` nil. Spot checks hold Metal's answers: a false function constant leaves its
//!   argument out, a stage input or vertex descriptor's buffers come first, and a tile function's
//!   imageblock is a binding. The tile form is checked over the measured tile fixture, whose
//!   implicit imageblock is a `float4` over an `RGBA8Unorm` attachment, and over the engine
//!   fixture `k10ts_tile_grid`.
//! - The mesh form answers a mesh pipeline's object, mesh and fragment bindings, the object and
//!   mesh functions' payload among them, where the device makes mesh pipelines; a device without
//!   mesh shading refuses the state, as without reflection.
//! - `reflectionForFunctionWithName:` answers a raw module's function from its AIR and refuses a
//!   name the library lacks.
//! - A function states its `MTLFunction` properties (`functionType`, its patch, its attributes,
//!   its function constants); one made with constant values leaves out an attribute they gate
//!   off and states no constants, as Metal's does.
//! - An argument encoder made from a library's or a pipeline's buffer binding has Metal's length,
//!   constant offsets, nested encoder and placements (a pipeline's arrays at stride 0).
//! - A kernel reads every member kind the encoders wrote (the function's, a library binding's,
//!   the argument descriptors' and a nested buffer's): buffers, textures, samplers, constants,
//!   arrays of each, a struct array's element and a nested argument buffer's pointer and constant,
//!   or its texture.

use super::tile_states::descriptor as tile_descriptor;
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{ArgumentDescriptor, ArgumentLayoutError, SamplerDescriptor};
use metal2vulkan::error::{DescriptorError, Error as CompileError, Feature};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    AttributeFormat, BindingAccess, BindingType, DataType, FunctionType, IndirectCommandType,
    PatchType, PixelFormat, ResourceUsage, StepFunction, TextureUsage, VertexFormat,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource, TileRenderPipelineDescriptor,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};
use metal2vulkan::reflection::{Attribute, Binding};
use metal2vulkan::Budget;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ArgumentEncoder, ArgumentEncoderProblem, Buffer, Device, Error,
    IndirectCommandBufferDescriptor, Invalid, Library, Nil, PipelineOption, PipelineOptions,
    Problem, Region, ResourceOptions, SamplerStateOptions, Size, TextureDescriptor,
};

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// The library of reflection fixture `stem` on `device`.
fn library(device: &Device, stem: &str) -> Library {
    let bitcode = llvm_assemble(&fixtures_dir().join(format!("reflection/{stem}.ll"))).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The options that ask for reflection, and the one that does not.
fn options() -> [(PipelineOption, bool); 4] {
    [
        (PipelineOption::NONE, false),
        (PipelineOption::BINDING_INFO, true),
        (PipelineOption::BUFFER_TYPE_INFO, true),
        (
            PipelineOption::BINDING_INFO | PipelineOption::BUFFER_TYPE_INFO,
            true,
        ),
    ]
}

fn names(bindings: &[Binding]) -> Vec<&str> {
    bindings.iter().map(Binding::name).collect()
}

/// A compute pipeline's reflection: none without a reflection bit, the descriptor's with either
/// or both; the kernel's arguments without the one a false constant gates off; and a stage input
/// descriptor's buffers first. The function form answers as the descriptor form.
#[test]
fn compute_states_return_their_reflection() {
    on_device(|device| {
        let constants = FunctionConstantValues::defaults()
            .with_index(0, FunctionConstantValue::new(DataType::Bool, vec![0]))
            .with_index(
                1,
                FunctionConstantValue::new(DataType::Int, 3i32.to_le_bytes().to_vec()),
            );
        let function = library(device, "buffers")
            .new_function_with_name_constant_values("reflection_buffers", &constants)
            .unwrap();
        let descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        let expected = descriptor.reflection(&Budget::standard()).unwrap();
        assert_eq!(
            names(expected.bindings()),
            [
                "plain",
                "uniform",
                "ro",
                "f3",
                "nested",
                "unused_arg",
                "counter",
                "mat",
                "bytes",
                "written_only"
            ],
            "a false constant leaves `extra` out"
        );
        assert!(!expected.bindings()[5].is_used());
        for (option, reflects) in options() {
            let (_state, reflection) = device
                .new_compute_pipeline_state_with_reflection(
                    &descriptor,
                    &PipelineOptions::new(option),
                )
                .unwrap();
            assert_eq!(reflection.is_some(), reflects, "{option:?}");
            if let Some(r) = reflection {
                assert_eq!(r, expected, "{option:?}");
            }
            let (_state, reflection) = device
                .new_compute_pipeline_state_with_function_and_reflection(
                    &function,
                    &PipelineOptions::new(option),
                )
                .unwrap();
            assert_eq!(reflection.is_some(), reflects, "{option:?}");
            if let Some(r) = reflection {
                assert_eq!(r, expected, "{option:?}");
            }
        }

        let kernel = library(device, "stage_in")
            .new_function_with_name("reflection_stage_in")
            .unwrap();
        let mut stage_in =
            ComputePipelineDescriptor::new(StageSource::Function(kernel.function_descriptor()));
        let mut s = StageInputDescriptor::new();
        s.attributes.insert(
            0,
            StageInputAttributeDescriptor::new(AttributeFormat::Float4, 0, 1),
        );
        s.attributes.insert(
            1,
            StageInputAttributeDescriptor::new(AttributeFormat::UInt, 0, 2),
        );
        for (buffer, stride) in [(1, 16), (2, 4)] {
            let mut layout = StageInputLayoutDescriptor::new(Some(stride));
            layout.step_function = StepFunction::ThreadPositionInGridX;
            s.layouts.insert(buffer, layout);
        }
        stage_in.stage_input = Some(s);
        let (_state, reflection) = device
            .new_compute_pipeline_state_with_reflection(
                &stage_in,
                &PipelineOptions::new(PipelineOption::BINDING_INFO),
            )
            .unwrap();
        let reflection = reflection.unwrap();
        let bindings = reflection.bindings();
        assert_eq!(
            names(bindings),
            ["stageInBuffer.0", "stageInBuffer.1", "out"]
        );
        assert_eq!(
            bindings.iter().map(Binding::index).collect::<Vec<_>>(),
            [1, 2, 0]
        );
        assert!(!bindings[0].is_argument() && bindings[2].is_argument());
        assert_eq!(bindings[0].buffer().unwrap().buffer_data_size(), 16);
    });
}

/// A drawing pipeline's reflection: the vertex descriptor's buffers before the vertex function's
/// arguments, and the fragment function's; a tile pipeline's, its tile function's arguments with
/// its implicit imageblock. None without a reflection bit. The measured tile fixture's state is
/// refused, typed, where the compile layer does not lower its imageblock access; another tile
/// function's state carries its reflection.
#[test]
fn render_and_tile_states_return_their_reflection() {
    on_device(|device| {
        let render = library(device, "render");
        let stage = |name: &str| {
            StageSource::Function(
                render
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = RenderPipelineDescriptor::new(stage("reflection_vertex"));
        descriptor.fragment = Some(stage("reflection_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let mut vd = VertexDescriptor::new();
        for (index, format, offset, buffer) in [
            (0, VertexFormat::Float4, 0, 0),
            (1, VertexFormat::Float2, 16, 0),
            (2, VertexFormat::Half4, 0, 2),
        ] {
            vd.attributes.insert(
                index,
                VertexAttributeDescriptor::new(format, offset, buffer),
            );
        }
        vd.layouts
            .insert(0, VertexBufferLayoutDescriptor::new(Some(24)));
        vd.layouts
            .insert(2, VertexBufferLayoutDescriptor::new(Some(8)));
        descriptor.vertex_descriptor = Some(vd);
        let expected = descriptor.reflection(&Budget::standard()).unwrap();
        assert_eq!(
            names(expected.vertex_bindings()),
            ["vertexBuffer.0", "vertexBuffer.1", "mvp", "extra", "vt"]
        );
        assert_eq!(
            names(expected.fragment_bindings()),
            ["t", "s", "tint", "count"]
        );
        assert!(expected.tile_bindings().is_empty());
        for (option, reflects) in options() {
            let (_state, reflection) = device
                .new_render_pipeline_state_with_reflection(
                    &descriptor,
                    &PipelineOptions::new(option),
                )
                .unwrap();
            assert_eq!(reflection.is_some(), reflects, "{option:?}");
            if let Some(r) = reflection {
                assert_eq!(r, expected, "{option:?}");
            }
        }

        let tile_function = library(device, "tile")
            .new_function_with_name("reflection_tile")
            .unwrap();
        let mut tile = TileRenderPipelineDescriptor::new(StageSource::Function(
            tile_function.function_descriptor(),
        ));
        tile.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
        tile.threadgroup_size_matches_tile_size = true;
        let expected = tile.reflection(&Budget::standard()).unwrap();
        assert_eq!(names(expected.tile_bindings()), ["ib", "tg", "k"]);
        assert_eq!(
            expected.tile_bindings()[0].binding_type(),
            BindingType::Imageblock
        );
        assert!(expected.vertex_bindings().is_empty() && expected.fragment_bindings().is_empty());
        // Its one-member implicit imageblock is read and written as a `float4` over the
        // normalized attachment (`air.load.implicit_imageblock.v4f32`): the creation form returns
        // its descriptor's reflection.
        for (option, reflects) in options() {
            let (_state, reflection) = device
                .new_tile_render_pipeline_state_with_reflection(
                    &tile,
                    &PipelineOptions::new(option),
                )
                .unwrap();
            assert_eq!(reflection.is_some(), reflects, "{option:?}");
            if let Some(r) = reflection {
                assert_eq!(r, expected, "{option:?}");
            }
        }

        // A tile function the compile layer takes on every device (`tile_states` makes its
        // state): the creation form returns its descriptor's reflection.
        let grid = tile_descriptor(
            device,
            "k10ts_tile_grid",
            &[
                PixelFormat::R32Float,
                PixelFormat::R32Uint,
                PixelFormat::R32Uint,
            ],
        );
        let expected = grid.reflection(&Budget::standard()).unwrap();
        assert_eq!(names(expected.tile_bindings()), ["block", "out"]);
        assert_eq!(
            expected.tile_bindings()[0].binding_type(),
            BindingType::Imageblock
        );
        for (option, reflects) in options() {
            let (_state, reflection) = device
                .new_tile_render_pipeline_state_with_reflection(
                    &grid,
                    &PipelineOptions::new(option),
                )
                .unwrap();
            assert_eq!(reflection.is_some(), reflects, "{option:?}");
            if let Some(r) = reflection {
                assert_eq!(r, expected, "{option:?}");
            }
        }
    });
}

/// `reflectionForFunctionWithName:` answers a raw module's function from its AIR, every argument
/// listed (a library's answer lists the gated ones), and refuses a name the library lacks.
/// A mesh pipeline's reflection: none without a reflection bit, the descriptor's with either or
/// both; the object function's payload read-write and the mesh function's read-only, at Metal's
/// index `0xFFFFFFFF`, 32 bytes aligned 16; no vertex or tile bindings. A device without mesh
/// shading refuses the state `Unsupported{MeshShading}`, the reflection unasked.
#[test]
fn mesh_states_return_their_reflection() {
    on_device(|device| {
        let lib = library(device, "mesh_functions");
        let function = |name| {
            StageSource::Function(
                lib.new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let mut descriptor = MeshPipelineDescriptor::new(function("reflection_mesh"));
        descriptor.object = Some(function("reflection_object"));
        descriptor.fragment = Some(function("reflection_mesh_fragment"));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let expected = descriptor.reflection(&Budget::standard()).unwrap();
        assert_eq!(names(expected.object_bindings()), ["payload", "tint"]);
        assert_eq!(names(expected.mesh_bindings()), ["payload", "corners"]);
        assert!(expected.vertex_bindings().is_empty());
        assert!(expected.fragment_bindings().is_empty());
        assert!(expected.tile_bindings().is_empty());
        for (payload, access) in [
            (&expected.object_bindings()[0], BindingAccess::ReadWrite),
            (&expected.mesh_bindings()[0], BindingAccess::ReadOnly),
        ] {
            assert_eq!(payload.binding_type(), BindingType::ObjectPayload);
            assert_eq!(payload.access(), access);
            assert_eq!(payload.index(), u32::MAX);
            let p = payload.object_payload().unwrap();
            assert_eq!(
                (p.object_payload_alignment(), p.object_payload_data_size()),
                (16, 32)
            );
        }
        for (option, reflects) in options() {
            match device.new_mesh_render_pipeline_state_with_reflection(
                &descriptor,
                &PipelineOptions::new(option),
            ) {
                Ok((_state, reflection)) => {
                    assert_eq!(reflection.is_some(), reflects, "{option:?}");
                    if let Some(r) = reflection {
                        assert_eq!(r, expected, "{option:?}");
                    }
                }
                Err(Error::Unsupported(u)) if u.feature == Feature::MeshShading => {
                    eprintln!("refused: mesh render pipelines on {}: {u}", device.name());
                    return;
                }
                Err(other) => panic!("{option:?}: {other:?}"),
            }
        }
    });
}

#[test]
fn a_library_answers_its_functions_reflection() {
    on_device(|device| {
        let buffers = library(device, "buffers");
        let reflection = buffers
            .reflection_for_function_with_name("reflection_buffers")
            .unwrap()
            .expect("a raw module's function answers from its AIR");
        assert_eq!(reflection.bindings().len(), 11);
        assert!(names(reflection.bindings()).contains(&"extra"));
        match buffers.reflection_for_function_with_name("reflection_missing") {
            Err(Error::Compile(CompileError::InvalidDescriptor(
                DescriptorError::EntryNotFound { .. },
            ))) => {}
            other => panic!("{other:?}"),
        }
    });
}

fn attribute_names(attributes: Option<Vec<Attribute>>) -> Vec<String> {
    attributes
        .expect("attributes")
        .iter()
        .map(|a| a.name().to_owned())
        .collect()
}

/// A function's `MTLFunction` properties through the runtime, from a raw module's AIR: a vertex
/// function's type and attributes, a post-tessellation function's patch, a kernel's stage-input
/// attributes alone; the function constants it reads (an attribute's predicate's, required) and
/// none where its module declares none. A function made with constant values leaves out the
/// attribute they gate off and states no constants.
#[test]
fn a_function_states_its_properties() {
    on_device(|device| {
        let inputs = library(device, "vertex_inputs");
        let gated = inputs.new_function_with_name("reflection_gated").unwrap();
        assert_eq!(gated.function_type().unwrap(), FunctionType::Vertex);
        assert_eq!(gated.patch_type().unwrap(), PatchType::None);
        assert_eq!(gated.patch_control_point_count().unwrap(), None);
        assert_eq!(
            attribute_names(gated.vertex_attributes().unwrap()),
            ["pos", "uv"]
        );
        let constants = gated.function_constants_dictionary().unwrap();
        let read: Vec<_> = constants
            .values()
            .map(|c| (c.name(), c.data_type(), c.index(), c.is_required()))
            .collect();
        assert_eq!(read, [("reflection_use_uv", DataType::Bool, 0, true)]);
        let off = inputs
            .new_function_with_name_constant_values(
                "reflection_gated",
                &FunctionConstantValues::defaults()
                    .with_index(0, FunctionConstantValue::new(DataType::Bool, vec![0])),
            )
            .unwrap();
        assert_eq!(attribute_names(off.vertex_attributes().unwrap()), ["pos"]);
        assert_eq!(
            attribute_names(off.stage_input_attributes().unwrap()),
            ["pos"]
        );
        assert!(off.function_constants_dictionary().unwrap().is_empty());

        let tessellation = library(device, "tessellation")
            .new_function_with_name("reflection_tessellation")
            .unwrap();
        assert_eq!(tessellation.patch_type().unwrap(), PatchType::Quad);
        assert_eq!(tessellation.patch_control_point_count().unwrap(), Some(4));

        let kernel = library(device, "stage_in")
            .new_function_with_name("reflection_stage_in")
            .unwrap();
        assert_eq!(kernel.function_type().unwrap(), FunctionType::Kernel);
        assert_eq!(kernel.vertex_attributes().unwrap(), None);
        assert_eq!(
            attribute_names(kernel.stage_input_attributes().unwrap()),
            ["a", "b"]
        );
        assert!(kernel.function_constants_dictionary().unwrap().is_empty());
    });
}

/// The 8-byte little-endian word at each of `offsets` in a shared buffer.
fn words(buffer: &Buffer, offsets: &[usize]) -> Vec<u64> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer is shared and mapped, `length` bytes long, and nothing on the device
    // uses it while it is read.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) };
    offsets
        .iter()
        .map(|&at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()))
        .collect()
}

/// One case of [`a_setter_at_an_index_no_argument_has_writes_at_offset_0`]: an encoder of
/// `descriptors` at offset `at` of a buffer of 0xaa bytes, the setter `set` calls, and the 8-byte
/// words it leaves at the buffer's offsets (every other word keeps its 0xaa bytes).
struct Case<'a> {
    label: &'a str,
    descriptors: &'a [ArgumentDescriptor],
    at: u64,
    set: &'a dyn Fn(&ArgumentEncoder),
    written: &'a [(usize, u64)],
}

/// Every setter writes its word at the arguments' offset 0 for an index no argument has, whatever
/// argument lies there, a range's words in turn (the last stands), and no other byte changes, as
/// plain Metal writes them (an M4, whose descriptors these are: two pointers, a texture then a
/// pointer, and one sampler, the arguments at offsets 0 and 32 of a buffer of 0xaa bytes; Metal's
/// debug layer asserts). A buffer set as `None` writes 0 whatever the offset, and a sampler state
/// made without argument-buffer support writes its own ID, as plain Metal writes both. In a layout
/// of 4 bytes the word passes the encoded length, as Metal's did, and is refused where it would
/// pass the argument buffer's end.
#[test]
fn a_setter_at_an_index_no_argument_has_writes_at_offset_0() {
    on_device(|device| {
        let shared = ResourceOptions::shared();
        let target = device.new_buffer(64, shared).unwrap();
        let other = device.new_buffer(64, shared).unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        let texture = device.new_texture(&descriptor).unwrap();
        let second_texture = device.new_texture(&descriptor).unwrap();
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let supporting = device
            .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
            .unwrap();
        let without = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        assert!(without != supporting);
        let mut icb_descriptor = IndirectCommandBufferDescriptor::new();
        icb_descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        icb_descriptor.max_kernel_buffer_bind_count = 1;
        let commands = device
            .new_indirect_command_buffer(&icb_descriptor, 1, ResourceOptions::default())
            .unwrap();
        let address = |buffer: &Buffer| buffer.gpu_address().0;
        let pointers = [
            ArgumentDescriptor::new(DataType::Pointer, 0),
            ArgumentDescriptor::new(DataType::Pointer, 1),
        ];
        let texture_first = [
            ArgumentDescriptor::new(DataType::Texture, 0),
            ArgumentDescriptor::new(DataType::Pointer, 1),
        ];
        let one_sampler = [ArgumentDescriptor::new(DataType::Sampler, 0)];
        let one_uint = [ArgumentDescriptor::new(DataType::UInt, 0)];
        let cases: [Case; 16] = [
            Case {
                label: "ptr_index_7",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_buffer(Some(&target), 0, 7).unwrap(),
                written: &[(0, address(&target))],
            },
            Case {
                label: "ptr_index_100",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_buffer(Some(&target), 0, 100).unwrap(),
                written: &[(0, address(&target))],
            },
            Case {
                label: "offset32_index_7",
                descriptors: &pointers,
                at: 32,
                set: &|e| e.set_buffer(Some(&target), 0, 7).unwrap(),
                written: &[(32, address(&target))],
            },
            Case {
                label: "ptr_texture_index_5",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_texture(Some(&texture), 5).unwrap(),
                written: &[(0, texture.gpu_resource_id().0)],
            },
            Case {
                label: "tex_buffer_index_5",
                descriptors: &texture_first,
                at: 0,
                set: &|e| e.set_buffer(Some(&target), 0, 5).unwrap(),
                written: &[(0, address(&target))],
            },
            Case {
                label: "noarg_sampler_7",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_sampler_state(Some(&supporting), 7).unwrap(),
                written: &[(0, supporting.gpu_resource_id().0)],
            },
            Case {
                label: "noarg_sampler_5_texture_first",
                descriptors: &texture_first,
                at: 0,
                set: &|e| e.set_sampler_state(Some(&supporting), 5).unwrap(),
                written: &[(0, supporting.gpu_resource_id().0)],
            },
            Case {
                label: "noarg_samplers_range_0_samplerenc",
                descriptors: &one_sampler,
                at: 0,
                set: &|e| {
                    e.set_sampler_states(&[Some(&without), Some(&supporting), Some(&without)], 0)
                        .unwrap()
                },
                written: &[(0, without.gpu_resource_id().0)],
            },
            Case {
                label: "noarg_buffers_range_7",
                descriptors: &pointers,
                at: 0,
                set: &|e| {
                    e.set_buffers(&[Some(&target), Some(&other)], &[0, 16], 7)
                        .unwrap()
                },
                written: &[(0, address(&other) + 16)],
            },
            Case {
                label: "noarg_buffers_range_1",
                descriptors: &pointers,
                at: 0,
                set: &|e| {
                    e.set_buffers(&[Some(&target), Some(&other)], &[0, 16], 1)
                        .unwrap()
                },
                written: &[(0, address(&other) + 16), (8, address(&target))],
            },
            Case {
                label: "noarg_textures_range_7",
                descriptors: &pointers,
                at: 0,
                set: &|e| {
                    e.set_textures(&[Some(&texture), Some(&second_texture)], 7)
                        .unwrap()
                },
                written: &[(0, second_texture.gpu_resource_id().0)],
            },
            Case {
                label: "noarg_icb_7",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_indirect_command_buffer(Some(&commands), 7).unwrap(),
                written: &[(0, commands.gpu_resource_id().0)],
            },
            Case {
                label: "nil_offsets_range",
                descriptors: &pointers,
                at: 0,
                set: &|e| e.set_buffers(&[None, None], &[16, 8], 0).unwrap(),
                written: &[(0, 0), (8, 0)],
            },
            Case {
                label: "samp_noab_set",
                descriptors: &one_sampler,
                at: 0,
                set: &|e| e.set_sampler_state(Some(&without), 0).unwrap(),
                written: &[(0, without.gpu_resource_id().0)],
            },
            // A layout of 4 bytes: the 8 bytes pass its encoded length.
            Case {
                label: "small4_buffer_5",
                descriptors: &one_uint,
                at: 0,
                set: &|e| e.set_buffer(Some(&target), 0, 5).unwrap(),
                written: &[(0, address(&target))],
            },
            Case {
                label: "small4_sampler_5",
                descriptors: &one_uint,
                at: 0,
                set: &|e| e.set_sampler_state(Some(&supporting), 5).unwrap(),
                written: &[(0, supporting.gpu_resource_id().0)],
            },
        ];
        let all: Vec<usize> = (0..32).map(|word| word * 8).collect();
        for Case {
            label,
            descriptors,
            at,
            set,
            written,
        } in cases
        {
            let arguments = buffer_with(device, &[0xaa; 256]).unwrap();
            let mut encoder = device
                .new_argument_encoder_with_arguments(descriptors)
                .unwrap();
            encoder.set_argument_buffer(Some(&arguments), at).unwrap();
            set(&encoder);
            let expected: Vec<u64> = all
                .iter()
                .map(|&offset| {
                    written
                        .iter()
                        .find(|&&(place, _)| place == offset)
                        .map_or(0xaaaa_aaaa_aaaa_aaaa, |&(_, word)| word)
                })
                .collect();
            assert_eq!(words(&arguments, &all), expected, "{label}");
        }
        // Where the 8 bytes would pass the argument buffer's end, which holds memory Metal's
        // result depends on, the setter is refused and writes nothing.
        let arguments = buffer_with(device, &[0xaa; 256]).unwrap();
        let mut encoder = device
            .new_argument_encoder_with_arguments(&one_uint)
            .unwrap();
        encoder.set_argument_buffer(Some(&arguments), 252).unwrap();
        assert_eq!(
            encoder.set_buffer(Some(&target), 0, 5),
            Err(Error::Invalid(Invalid::new(
                "set_buffer",
                Problem::ArgumentEncoder(ArgumentEncoderProblem::NoArgument { index: 5 })
            )))
        );
        assert_eq!(words(&arguments, &all), vec![0xaaaa_aaaa_aaaa_aaaa; 32]);
    });
}

/// An encoder made from a reflected buffer binding (`newArgumentEncoderWithBufferBinding:`)
/// writes where Metal's does: one from a library's binding as the function's own encoder, in 224
/// bytes; one from a pipeline's binding, whose arrays reflect stride 0, in 208 bytes with both
/// elements of `ptrs` at element 0's place (the last written kept). Both answer the same constant
/// offsets and the 8-byte encoder of the nested argument buffer `sub`. A plain buffer's binding
/// makes Metal's encoder of no bytes, alignment 1, which API validation lets through; the
/// function's encoder at that buffer's index is refused, as validation stops it.
#[test]
fn a_binding_makes_its_argument_encoder() {
    const FUNCTION: &str = "reflection_argument_buffers";
    on_device(|device| {
        let lib = library(device, "argument_buffers");
        let function = lib.new_function_with_name(FUNCTION).unwrap();
        let library_bindings = lib
            .reflection_for_function_with_name(FUNCTION)
            .unwrap()
            .expect("a raw module's function answers from its AIR")
            .bindings()
            .to_vec();
        let pipeline_bindings =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()))
                .reflection(&Budget::standard())
                .unwrap()
                .bindings()
                .to_vec();
        let find = |bindings: &[Binding], name: &str| {
            bindings
                .iter()
                .find(|b| b.name() == name)
                .and_then(Binding::buffer)
                .cloned()
                .unwrap()
        };
        let shared = ResourceOptions::shared();
        // Set at 2 (`data`) and 18 and 19 (`ptrs`, an array of two pointers).
        let buffers = [2, 18, 19].map(|index| (index, device.new_buffer(64, shared).unwrap()));
        let [data, first, second] = buffers.each_ref().map(|(_, b)| b.gpu_address().0);
        // The words at 16 (`data`), 144 and 152 (`ptrs`) after the three are set.
        for (label, encoder, length, expected) in [
            (
                "function",
                function.new_argument_encoder_with_buffer_index(0),
                224,
                [data, first, second],
            ),
            (
                "library",
                device.new_argument_encoder_with_buffer_binding(&find(&library_bindings, "args")),
                224,
                [data, first, second],
            ),
            (
                "pipeline",
                device.new_argument_encoder_with_buffer_binding(&find(&pipeline_bindings, "args")),
                208,
                [data, second, 0],
            ),
        ] {
            let mut encoder = encoder.unwrap();
            assert_eq!(
                (encoder.encoded_length(), encoder.alignment()),
                (length, 16),
                "{label}"
            );
            let arguments = device.new_buffer(256, shared).unwrap();
            encoder.set_argument_buffer(Some(&arguments), 0).unwrap();
            let base = arguments.contents().unwrap().as_ptr() as usize;
            for (index, constant) in [(7, 56), (8, 64), (20, 160), (21, 164)] {
                let at = encoder.constant_data(index).unwrap().as_ptr() as usize;
                assert_eq!(at - base, constant, "{label} constant {index}");
            }
            for (index, buffer) in &buffers {
                encoder.set_buffer(Some(buffer), 0, *index).unwrap();
            }
            assert_eq!(words(&arguments, &[16, 144, 152]), expected, "{label}");
            let nested = encoder
                .new_argument_encoder_for_buffer_at_index(15)
                .unwrap();
            assert_eq!(
                (nested.encoded_length(), nested.alignment()),
                (8, 8),
                "{label}"
            );
            assert_eq!(
                encoder.new_argument_encoder_for_buffer_at_index(12).err(),
                Some(Error::Nil(Nil::new(
                    "new_argument_encoder_for_buffer_at_index",
                    Problem::ArgumentEncoder(ArgumentEncoderProblem::NoNestedArgumentBuffer {
                        index: 12
                    })
                ))),
                "{label}"
            );
        }
        for bindings in [&library_bindings, &pipeline_bindings] {
            let empty = device
                .new_argument_encoder_with_buffer_binding(&find(bindings, "out"))
                .unwrap();
            assert_eq!((empty.encoded_length(), empty.alignment()), (0, 1));
        }
        assert_eq!(
            function.new_argument_encoder_with_buffer_index(2).err(),
            Some(Error::Invalid(Invalid::new(
                "new_argument_encoder_with_buffer_index",
                Problem::ArgumentEncoder(ArgumentEncoderProblem::Layout(Box::new(
                    ArgumentLayoutError::NoArgumentBuffer { index: 2 }
                )))
            )))
        );
    });
}

/// The texel each resource of [`encoded_argument_buffers_reach_the_kernel`] holds: a power of two
/// per member the kernel reads, so its output, their sum, names any member the encoders misplace;
/// [`DECOY`] where a misplaced read would land.
const TEX: f32 = 1.0;
const DATA: f32 = 2.0;
const TEXS_2: f32 = 4.0;
const CONSTS_0: f32 = 8.0;
const COUNT: u32 = 16;
const COLOR: f32 = 32.0;
const DEPTH: f32 = 0.5;
const INNER_X: u32 = 128;
const SUB_P: f32 = 256.0;
const PAIR_A: f32 = 512.0;
const PTRS_1: f32 = 1024.0;
const VALS_1: u32 = 2048;
const PAIRS_1_A: f32 = 4096.0;
const SUB_N: u32 = 8192;
const SUB_T: f32 = 16384.0;
/// What an element or member the kernel does not read holds.
const DECOY: f32 = 1_000_000.0;
/// `imp`'s constant `n`, and the `half` texel of its texture `t` (3.0).
const IMP_N: u32 = 40;
const IMP_T: u16 = 0x4200;

/// A kernel reads what argument encoders wrote: the owned fixture `encoder_reads`
/// (`reflection_argument_buffers` with a nested argument buffer of a pointer and an inline constant
/// in place of a texture) sums a texture sampled through a sampler, a buffer, the third of an array
/// of read-only textures, a constant buffer, two inline constants, a depth texture, a struct
/// through a pointer, the nested argument buffer's pointer and constant, a struct member's texture
/// sampled through the second of an array of samplers, the second of an array of pointers, the
/// second of an array of constants and the second element of an array of structs (a texture and
/// a sampler), writes the sum to a write-only texture and its output, and from its second
/// argument buffer (implicit ids) writes a constant plus a `half` texel through a pointer. Each
/// read member holds its own power of two and each decoy element a million, so the sum names a
/// misplaced member. The arguments are written by the function's encoders, then by the library
/// binding's encoder and the argument descriptors' encoder; the nested buffer by its own encoder
/// both times, constants through `constant_data`; the second run's sampler state was made without
/// argument-buffer support. Both runs read Metal's sum: every member reaches the kernel where the
/// encoders put it.
#[test]
fn encoded_argument_buffers_reach_the_kernel() {
    on_device(|device| reach_the_kernel(device, Nested::PointerAndConstant));
}

/// A texture loaded from a nested argument buffer: the owned fixture `reflection_argument_buffers`,
/// which is `encoder_reads` with a texture in its nested argument buffer, samples `args.sub->t`
/// through `args.smp`, the nested buffer's texture written by the nested buffer's own encoder and
/// made resident with `use_resources`, as Metal requires of every texture an argument buffer names.
/// Both runs of [`encoded_argument_buffers_reach_the_kernel`] read Metal's sum with that texture's
/// texel in place of the pointer's and the constant's.
#[test]
fn a_nested_argument_buffer_s_texture_reaches_the_kernel() {
    on_device(|device| reach_the_kernel(device, Nested::Texture));
}

/// What the nested argument buffer `sub` (`[[id(15)]]`) of an encoded-arguments kernel holds.
#[derive(Clone, Copy, Debug)]
enum Nested {
    /// `encoder_reads`: a pointer `p` (`[[id(0)]]`) and an inline constant `n` (`[[id(1)]]`).
    PointerAndConstant,
    /// `reflection_argument_buffers`: a texture `t` (`[[id(0)]]`).
    Texture,
}

/// The run of [`encoded_argument_buffers_reach_the_kernel`] over the kernel whose nested argument
/// buffer holds `nested`.
fn reach_the_kernel(device: &Device, nested: Nested) {
    let (stem, entry) = match nested {
        Nested::PointerAndConstant => ("encoder_reads", "encoder_reads"),
        Nested::Texture => ("argument_buffers", "reflection_argument_buffers"),
    };
    let lib = library(device, stem);
    let function = lib.new_function_with_name(entry).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let library_bindings = lib
        .reflection_for_function_with_name(entry)
        .unwrap()
        .expect("a raw module's function answers from its AIR")
        .bindings()
        .to_vec();
    let args_binding = library_bindings
        .iter()
        .find(|b| b.name() == "args")
        .and_then(Binding::buffer)
        .cloned()
        .unwrap();
    let shared = ResourceOptions::shared();
    let floats = |value: f32, count: usize| -> Vec<u8> {
        std::iter::repeat_n(value.to_le_bytes(), count)
            .flatten()
            .collect()
    };
    let texture = |format: PixelFormat, usage: TextureUsage, texel: &[u8]| {
        let mut descriptor = TextureDescriptor::texture_2d(format, 1, 1, false);
        descriptor.usage = usage;
        let texture = device.new_texture(&descriptor).unwrap();
        let row = texel.len() as u64;
        texture
            .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, texel, row, row)
            .unwrap();
        texture
    };
    let color = |value: f32| {
        texture(
            PixelFormat::RGBA32Float,
            TextureUsage::SHADER_READ,
            &floats(value, 4),
        )
    };
    let mut options = SamplerStateOptions::default();
    options.support_argument_buffers = true;
    let supporting = device
        .new_sampler_state_with_options(&SamplerDescriptor::new(), options)
        .unwrap();
    // The second run's sampler state has no argument-buffer support: written and sampled through
    // as any other, as plain Metal does.
    let without = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let buffer = |bytes: &[u8]| buffer_with(device, bytes).unwrap();

    let tex = color(TEX);
    let texs = [color(DECOY), color(DECOY), color(TEXS_2)];
    let wtex = texture(
        PixelFormat::RGBA32Float,
        TextureUsage::SHADER_WRITE,
        &floats(0.0, 4),
    );
    let dtex = texture(
        PixelFormat::Depth32Float,
        TextureUsage::SHADER_READ,
        &DEPTH.to_le_bytes(),
    );
    let pair_a = color(PAIR_A);
    let pairs_a = [color(DECOY), color(PAIRS_1_A)];
    let imp_t = texture(
        PixelFormat::R16Float,
        TextureUsage::SHADER_READ,
        &IMP_T.to_le_bytes(),
    );
    let data = buffer(&floats(DATA, 1));
    let consts = buffer(&floats(CONSTS_0, 4));
    // `ReflectionArgInner { uint x; float2 y; }`: 16 bytes, `x` at 0.
    let mut inner_bytes = vec![0u8; 16];
    inner_bytes[..4].copy_from_slice(&INNER_X.to_le_bytes());
    let inner = buffer(&inner_bytes);
    let ptrs = [buffer(&floats(DECOY, 1)), buffer(&floats(PTRS_1, 1))];
    // The nested buffer's pointer's float, or its texture.
    let sub_p = buffer(&floats(SUB_P, 1));
    let sub_t = color(SUB_T);
    let nested_sum = match nested {
        Nested::PointerAndConstant => SUB_P + SUB_N as f32,
        Nested::Texture => SUB_T,
    };
    let expected = TEX
        + DATA
        + TEXS_2
        + CONSTS_0
        + COUNT as f32
        + COLOR
        + DEPTH
        + INNER_X as f32
        + nested_sum
        + PAIR_A
        + PTRS_1
        + VALS_1 as f32
        + PAIRS_1_A;
    let queue = device.new_command_queue().unwrap();

    for library_binding in [false, true] {
        let how = if library_binding {
            "the library binding's and the descriptors' encoders"
        } else {
            "the function's encoders"
        };
        let sampler = if library_binding {
            without.clone()
        } else {
            supporting.clone()
        };
        let (mut args_encoder, mut imp_encoder) = if library_binding {
            let descriptors = [
                ArgumentDescriptor::new(DataType::Pointer, 0),
                ArgumentDescriptor::new(DataType::Texture, 1),
                ArgumentDescriptor::new(DataType::Sampler, 2),
                ArgumentDescriptor::new(DataType::UInt, 3),
            ];
            (
                device
                    .new_argument_encoder_with_buffer_binding(&args_binding)
                    .unwrap(),
                device
                    .new_argument_encoder_with_arguments(&descriptors)
                    .unwrap(),
            )
        } else {
            (
                function.new_argument_encoder_with_buffer_index(0).unwrap(),
                function.new_argument_encoder_with_buffer_index(1).unwrap(),
            )
        };
        let args = device
            .new_buffer(args_encoder.encoded_length(), shared)
            .unwrap();
        let imp = device
            .new_buffer(imp_encoder.encoded_length(), shared)
            .unwrap();
        let out = buffer(&floats(0.0, 4));
        let imp_a = buffer(&0u32.to_le_bytes());
        // The nested argument buffer `sub` (id 15), by its own encoder.
        let mut sub_encoder = args_encoder
            .new_argument_encoder_for_buffer_at_index(15)
            .unwrap();
        let sub = device
            .new_buffer(sub_encoder.encoded_length(), shared)
            .unwrap();
        sub_encoder.set_argument_buffer(Some(&sub), 0).unwrap();
        match nested {
            Nested::PointerAndConstant => {
                sub_encoder.set_buffer(Some(&sub_p), 0, 0).unwrap();
                let at = sub_encoder.constant_data(1).unwrap();
                // SAFETY: the constant's bytes lie inside the shared nested buffer's mapping,
                // which no command uses yet.
                unsafe {
                    std::ptr::copy_nonoverlapping(SUB_N.to_le_bytes().as_ptr(), at.as_ptr(), 4)
                };
            }
            Nested::Texture => sub_encoder.set_texture(Some(&sub_t), 0).unwrap(),
        }

        args_encoder.set_argument_buffer(Some(&args), 0).unwrap();
        let e = &args_encoder;
        e.set_texture(Some(&tex), 0).unwrap();
        e.set_sampler_state(Some(&sampler), 1).unwrap();
        e.set_buffer(Some(&data), 0, 2).unwrap();
        e.set_textures(&[Some(&texs[0]), Some(&texs[1]), Some(&texs[2])], 3)
            .unwrap();
        e.set_buffer(Some(&consts), 0, 6).unwrap();
        let write = |index: u32, bytes: &[u8]| {
            let at = e.constant_data(index).unwrap();
            // SAFETY: the constant's bytes lie inside the shared argument buffer's mapping,
            // which no command uses yet.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.as_ptr(), bytes.len()) };
        };
        write(7, &COUNT.to_le_bytes());
        write(8, &floats(COLOR, 4));
        e.set_texture(Some(&wtex), 10).unwrap();
        e.set_texture(Some(&dtex), 11).unwrap();
        e.set_buffer(Some(&inner), 0, 12).unwrap();
        e.set_buffer(Some(&sub), 0, 15).unwrap();
        e.set_sampler_states(&[None, Some(&sampler)], 16).unwrap();
        e.set_buffers(&[Some(&ptrs[0]), Some(&ptrs[1])], &[0, 0], 18)
            .unwrap();
        write(20, &0u32.to_le_bytes());
        write(21, &VALS_1.to_le_bytes());
        e.set_texture(Some(&pair_a), 22).unwrap();
        e.set_sampler_state(Some(&sampler), 23).unwrap();
        e.set_texture(Some(&pairs_a[0]), 24).unwrap();
        e.set_sampler_state(Some(&sampler), 25).unwrap();
        e.set_texture(Some(&pairs_a[1]), 26).unwrap();
        e.set_sampler_state(Some(&sampler), 27).unwrap();

        imp_encoder.set_argument_buffer(Some(&imp), 0).unwrap();
        imp_encoder.set_buffer(Some(&imp_a), 0, 0).unwrap();
        imp_encoder.set_texture(Some(&imp_t), 1).unwrap();
        imp_encoder.set_sampler_state(Some(&sampler), 2).unwrap();
        let at = imp_encoder.constant_data(3).unwrap();
        // SAFETY: as above, for `imp`'s constant `n`.
        unsafe { std::ptr::copy_nonoverlapping(IMP_N.to_le_bytes().as_ptr(), at.as_ptr(), 4) };

        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(Some(&args), 0, 0).unwrap();
        encoder.set_buffer(Some(&imp), 0, 1).unwrap();
        encoder.set_buffer(Some(&out), 0, 2).unwrap();
        let read: Vec<_> = [&data, &consts, &inner, &sub, &sub_p, &ptrs[0], &ptrs[1]]
            .into_iter()
            .map(Buffer::as_resource)
            .chain(
                [&tex, &dtex, &pair_a, &imp_t, &sub_t]
                    .into_iter()
                    .chain(&texs)
                    .chain(&pairs_a)
                    .map(|t| t.as_resource()),
            )
            .collect();
        encoder.use_resources(&read, ResourceUsage::READ).unwrap();
        encoder
            .use_resources(
                &[wtex.as_resource(), imp_a.as_resource()],
                ResourceUsage::READ | ResourceUsage::WRITE,
            )
            .unwrap();
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threads(one, one).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(std::time::Duration::from_secs(60))
            .unwrap();
        assert_eq!(command_buffer.error(), None, "{how}");

        let sum = floats(expected, 4);
        assert_eq!(read_buffer(&out, 0, 16).unwrap(), sum, "{how}: out");
        let mut written = vec![0u8; 16];
        wtex.get_bytes(&mut written, 16, 16, Region::new_2d(0, 0, 1, 1), 0, 0)
            .unwrap();
        assert_eq!(written, sum, "{how}: wtex");
        assert_eq!(
            read_buffer(&imp_a, 0, 4).unwrap(),
            (IMP_N + 3).to_le_bytes(),
            "{how}: imp.a"
        );
    }
}
