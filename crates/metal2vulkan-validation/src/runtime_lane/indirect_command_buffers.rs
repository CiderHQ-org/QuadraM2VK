//! Indirect command buffers on the device the cfg selects, under the validation layer: creation,
//! CPU encoding of compute commands, resets and the typed read-back answer as Metal's measured ones
//! do (an M3 Max on macOS 26.7 and an M4 on 27.0.1), what Metal's validation stops or ends the
//! process for is refused typed, and pipeline states made with `supportIndirectCommandBuffers` take
//! dense IDs of their own space. The pipeline states are of the owned `icb_replay_template`
//! fixture's kernel. Render commands: their setters, read-back and refusals, and render states' IDs
//! in the space compute states take theirs from, through the owned `icb_render_replay_template`
//! fixture's vertex function.

use super::degrade_notices::{self as notices, counts};
use super::host_memory::on_device;
use super::{buffer_with, read_buffer, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::indirect_commands::COMMAND_BYTES;
use metal2vulkan::bind::ArgumentDescriptor;
use metal2vulkan::metal::{
    BlendFactor, ColorWriteMask, DataType, IndexType, IndirectCommandType, PixelFormat,
    PrimitiveType, ResourceUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, MeshPipelineDescriptor,
    RenderPipelineDescriptor, Stage, StageSource,
};
use metal2vulkan::reflection::LinkedUseGap;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    BlitCommandEncoder, ComputePipelineState, CpuCacheMode, Degradation, Device, DrawProblem,
    EncodedComputeCommand, EncodedRenderCommand, Error, GpuAddress, HazardTrackingMode,
    IndirectBufferBinding, IndirectCommand, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectCommandBufferProblem, IndirectDispatch, IndirectDraw, Invalid, MetalError,
    PipelineOptions, Problem, PurgeableState, Region, RenderPipelineState, ResourceId,
    ResourceOptions, RuntimeGap, Size, StorageMode,
};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

fn icb_problem(call: &'static str, problem: IndirectCommandBufferProblem) -> Error {
    invalid(call, Problem::IndirectCommandBuffer(problem))
}

/// A descriptor of both dispatch kinds, 4 kernel buffers and 2 threadgroup lengths, nothing
/// inherited (the probes' own).
fn compute_descriptor() -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
        .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
    descriptor.max_kernel_buffer_bind_count = 4;
    descriptor.max_kernel_threadgroup_memory_bind_count = 2;
    descriptor
}

fn icb(
    device: &Device,
    descriptor: &IndirectCommandBufferDescriptor,
    count: u64,
) -> IndirectCommandBuffer {
    device
        .new_indirect_command_buffer(descriptor, count, ResourceOptions::shared())
        .unwrap()
}

/// The template's state, made with `supportIndirectCommandBuffers` as `support` says.
fn template_state(device: &Device, support: bool) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("icb_replay_template.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("icb_replay_template")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = support;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A descriptor of draws, 2 vertex buffers and 1 fragment buffer, nothing inherited (the probes'
/// own).
fn render_descriptor() -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = 1;
    descriptor
}

/// The render template's state (rasterization disabled), made with
/// `supportIndirectCommandBuffers` as `support` says.
pub(super) fn render_template_state(device: &Device, support: bool) -> RenderPipelineState {
    let mut descriptor = render_template_descriptor(device);
    descriptor.support_indirect_command_buffers = support;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of [`render_template_state`]'s state, without `supportIndirectCommandBuffers`.
pub(super) fn render_template_descriptor(device: &Device) -> RenderPipelineDescriptor {
    let bitcode = llvm_assemble(&engine_dir().join("icb_render_replay_template.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("icb_render_replay_template")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor
}

/// The render command `index` of `icb` reads back as.
fn render_command(icb: &IndirectCommandBuffer, index: u64) -> EncodedRenderCommand {
    match icb.commands(index..index + 1).unwrap().remove(0) {
        IndirectCommand::Render(command) => command,
        other => panic!("not a render command: {other:?}"),
    }
}

/// The compute command `index` of `icb` reads back as.
fn command(icb: &IndirectCommandBuffer, index: u64) -> EncodedComputeCommand {
    match icb.commands(index..index + 1).unwrap().remove(0) {
        IndirectCommand::Compute(command) => command,
        other => panic!("not a compute command: {other:?}"),
    }
}

/// A fresh or emptied command: nothing set.
fn assert_unset(command: &EncodedComputeCommand) {
    assert_eq!(
        (command.imageblock_width, command.imageblock_height),
        (0, 0)
    );
    assert_reset(command);
}

/// A command after its `reset`: nothing set but the imageblock size, which the reset keeps.
fn assert_reset(command: &EncodedComputeCommand) {
    assert_eq!(command.dispatch, None);
    assert_eq!(command.command_type, IndirectCommandType::empty());
    assert!(!command.barrier);
    assert!(command.pipeline.is_none());
    assert!(command.kernel_buffers.iter().all(Option::is_none));
    assert!(command
        .threadgroup_memory_lengths
        .iter()
        .all(|&length| length == 0));
    assert_eq!(command.stage_in_region, Region::default());
}

/// What creation answers: the size asked for, 0 included; the options
/// asked for with hazard tracking resolved to tracked; no label, heap or aliasing; and the
/// descriptor's defaults. `allocatedSize` is Metal's (docs/API.md section 10).
#[test]
fn indirect_command_buffers_are_made_as_metal_makes_them() {
    let defaults = IndirectCommandBufferDescriptor::new();
    assert!(defaults.command_types.is_empty());
    assert!(!defaults.inherit_pipeline_state && !defaults.inherit_buffers);
    assert_eq!(defaults.max_kernel_buffer_bind_count, 0);
    assert_eq!(defaults.max_kernel_threadgroup_memory_bind_count, 31);
    assert!(!defaults.support_dynamic_attribute_stride);
    on_device(|device| {
        // Metal's `allocatedSize`, whatever the storage mode.
        for (count, allocated) in [(0, 16384), (1, 16384), (16, 16384), (1024, 802816)] {
            let made = icb(device, &compute_descriptor(), count);
            assert_eq!(made.size(), count);
            assert_eq!(made.allocated_size(), allocated, "{count} commands");
            assert_eq!(made.resource_options().raw(), 512);
            assert_eq!(made.hazard_tracking_mode(), HazardTrackingMode::Tracked);
            assert_eq!(made.label(), None);
            assert!(made.heap().is_none() && !made.is_aliasable());
            assert_eq!(made.heap_offset(), 0);
            if count > 0 {
                assert_unset(&command(&made, count - 1));
            }
        }
        for (options, raw) in [
            (ResourceOptions::private(), 544),
            (
                ResourceOptions::new(
                    StorageMode::Managed,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                ),
                528,
            ),
            (
                ResourceOptions::new(
                    StorageMode::Shared,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Untracked,
                ),
                256,
            ),
            (
                ResourceOptions::new(
                    StorageMode::Shared,
                    CpuCacheMode::WriteCombined,
                    HazardTrackingMode::Default,
                ),
                513,
            ),
        ] {
            let made = device
                .new_indirect_command_buffer(&compute_descriptor(), 4, options)
                .unwrap();
            assert_eq!(made.resource_options().raw(), raw);
        }
        let named = icb(device, &compute_descriptor(), 4);
        named.set_label("named");
        assert_eq!(named.label().as_deref(), Some("named"));
        // Each ID is a small integer of the buffers' own space: the buffers made above are gone, so
        // `named` took 1 and `other` takes 2.
        let other = icb(device, &compute_descriptor(), 4);
        assert_eq!(
            [&named, &other].map(IndirectCommandBuffer::gpu_resource_id),
            [ResourceId(1), ResourceId(2)]
        );
    });
}

/// An indirect command buffer's ID is the lowest free of the buffers' own space, from 1, given
/// back when the buffer goes, as Metal's `gpuResourceID` (an M3 Max on macOS 26.7 and an M4 on
/// 27.0.1, identical): four buffers take 1 to 4; with 1 and 3 gone (in either order) the next three
/// take 1, 3 and 5; with 2, 3 and 4 of five gone the next four take 2, 3, 4 and 6. Buffers of
/// compute and of render commands share the space, and a pipeline state made with
/// `supportIndirectCommandBuffers` takes none of it.
#[test]
fn indirect_command_buffers_take_the_lowest_free_id_of_their_own_space() {
    on_device(|device| {
        let make = || icb(device, &compute_descriptor(), 4);
        let ids = |buffers: &[&IndirectCommandBuffer]| -> Vec<u64> {
            buffers.iter().map(|b| b.gpu_resource_id().0).collect()
        };
        for first_gone in [true, false] {
            let (a, b, c, d) = (make(), make(), make(), make());
            assert_eq!(ids(&[&a, &b, &c, &d]), [1, 2, 3, 4]);
            if first_gone {
                drop(a);
                drop(c);
            } else {
                drop(c);
                drop(a);
            }
            let (e, f, g) = (make(), make(), make());
            assert_eq!(
                ids(&[&e, &f, &g]),
                [1, 3, 5],
                "first gone first: {first_gone}"
            );
        }
        let five: Vec<IndirectCommandBuffer> = (0..5).map(|_| make()).collect();
        assert_eq!(ids(&five.iter().collect::<Vec<_>>()), [1, 2, 3, 4, 5]);
        let mut five = five.into_iter();
        let (first, second, third, fourth, fifth) = (
            five.next().unwrap(),
            five.next().unwrap(),
            five.next().unwrap(),
            five.next().unwrap(),
            five.next().unwrap(),
        );
        drop(fourth);
        drop(second);
        drop(third);
        let next: Vec<IndirectCommandBuffer> = (0..4).map(|_| make()).collect();
        assert_eq!(ids(&next.iter().collect::<Vec<_>>()), [2, 3, 4, 6]);
        drop((first, fifth, next));
        let compute = make();
        let state = template_state(device, true);
        assert_eq!(state.gpu_resource_id(), ResourceId(1));
        let render = icb(device, &render_descriptor(), 4);
        assert_eq!(ids(&[&compute, &render]), [1, 2]);
        // `resourceIndex` is the state's `gpuResourceID`: 0 without
        // `supportIndirectCommandBuffers` (an M4: compute 1 and render 2 with
        // it, both 0 without).
        let render_state = render_template_state(device, true);
        assert_eq!(state.resource_index(), state.gpu_resource_id().0);
        assert_eq!(render_state.resource_index(), 2);
        assert_eq!(
            render_state.resource_index(),
            render_state.gpu_resource_id().0
        );
        assert_eq!(template_state(device, false).resource_index(), 0);
        assert_eq!(render_template_state(device, false).resource_index(), 0);
    });
}

/// What plain Metal ends the process for at creation, or makes no defined storage of, is refused
/// typed; what it makes where its validation stops it is made; render command types other than
/// draws of primitives, indexed draws and patch draws wait for the record format; records past one
/// buffer are refused naming the limit.
#[test]
fn creation_refuses_what_metal_refuses() {
    const CALL: &str = "new_indirect_command_buffer";
    on_device(|device| {
        let refused = |descriptor: &IndirectCommandBufferDescriptor, count, options| {
            device
                .new_indirect_command_buffer(descriptor, count, options)
                .unwrap_err()
        };
        let shared = ResourceOptions::shared();
        // No command types: made, but its commands, reset and read-back are refused (plain
        // Metal's process ends at a command's first setter and at a reset).
        let none = icb(device, &IndirectCommandBufferDescriptor::new(), 4);
        assert_eq!(none.size(), 4);
        let no_types = |call| icb_problem(call, IndirectCommandBufferProblem::NoCommandTypes);
        assert_eq!(
            none.indirect_compute_command_at(0).unwrap_err(),
            no_types("indirect_compute_command_at")
        );
        assert_eq!(
            none.indirect_render_command_at(0).unwrap_err(),
            no_types("indirect_render_command_at")
        );
        assert_eq!(
            none.reset_with_range(0..1).unwrap_err(),
            no_types("reset_with_range")
        );
        assert_eq!(none.commands(0..1).unwrap_err(), no_types("commands"));
        // Draws of any kind beside dispatches make a buffer of compute commands, as plain
        // Metal's.
        for draws in [
            IndirectCommandType::DRAW,
            IndirectCommandType::DRAW_INDEXED,
            IndirectCommandType::DRAW_PATCHES,
            IndirectCommandType::DRAW_MESH_THREADGROUPS,
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED),
        ] {
            let mut mixed = compute_descriptor();
            mixed.command_types = mixed.command_types.union(draws);
            let compute = icb(device, &mixed, 4);
            compute
                .indirect_compute_command_at(0)
                .unwrap()
                .set_threadgroup_memory_length(16, 1)
                .unwrap();
            assert_eq!(command(&compute, 0).threadgroup_memory_lengths, [0, 16]);
            assert_eq!(
                compute.indirect_render_command_at(0).unwrap_err(),
                icb_problem(
                    "indirect_render_command_at",
                    IndirectCommandBufferProblem::CommandKind
                )
            );
        }
        assert_eq!(icb(device, &render_descriptor(), 4).size(), 4);
        // Indexed, patch and indexed patch draws, and mesh draws, alone or beside draws of
        // primitives.
        for held in [
            IndirectCommandType::DRAW_INDEXED,
            IndirectCommandType::DRAW_PATCHES,
            IndirectCommandType::DRAW_INDEXED_PATCHES,
            IndirectCommandType::DRAW_MESH_THREADGROUPS,
            IndirectCommandType::DRAW_MESH_THREADS,
            IndirectCommandType::DRAW
                .union(IndirectCommandType::DRAW_INDEXED)
                .union(IndirectCommandType::DRAW_PATCHES)
                .union(IndirectCommandType::DRAW_INDEXED_PATCHES)
                .union(IndirectCommandType::DRAW_MESH_THREADGROUPS)
                .union(IndirectCommandType::DRAW_MESH_THREADS),
        ] {
            let mut draws = render_descriptor();
            draws.command_types = held;
            assert_eq!(icb(device, &draws, 4).size(), 4);
        }
        // Up to 65536 buffers of a stage, or threadgroup lengths, as plain Metal makes them; past
        // that its storage is not defined.
        let counts = |count: u32| {
            let mut buffers = compute_descriptor();
            buffers.max_kernel_buffer_bind_count = count;
            let mut lengths = compute_descriptor();
            lengths.max_kernel_threadgroup_memory_bind_count = count;
            let mut vertex = render_descriptor();
            vertex.max_vertex_buffer_bind_count = count;
            let mut fragment = render_descriptor();
            fragment.max_fragment_buffer_bind_count = count;
            [buffers, lengths, vertex, fragment]
        };
        for descriptor in counts(65536) {
            assert_eq!(icb(device, &descriptor, 2).size(), 2);
        }
        // A buffer set past what a record holds is kept beside the records and read back with
        // the record's slots, every slot the descriptor counts; a length past the record is still
        // dropped, so the read-back lists the lengths a record holds.
        let [buffers, lengths, ..] = counts(64);
        let wide = icb(device, &buffers, 1);
        let set = wide.indirect_compute_command_at(0).unwrap();
        let bound = device.new_buffer(256, shared).unwrap();
        set.set_kernel_buffer(Some(&bound), 0, 63).unwrap();
        set.set_kernel_buffer(Some(&bound), 0, 31).unwrap();
        let kernel_buffers = command(&wide, 0).kernel_buffers;
        assert_eq!(kernel_buffers.len(), 64);
        assert!(kernel_buffers[31].is_some());
        assert!(kernel_buffers[63].is_some());
        assert!(kernel_buffers[32..63].iter().all(Option::is_none));
        let long = icb(device, &lengths, 1);
        let set = long.indirect_compute_command_at(0).unwrap();
        set.set_threadgroup_memory_length(16, 63).unwrap();
        set.set_threadgroup_memory_length(16, 30).unwrap();
        let lengths = command(&long, 0).threadgroup_memory_lengths;
        assert_eq!(lengths.len(), 31);
        assert_eq!(lengths[30], 16);
        for descriptor in counts(65537) {
            assert_eq!(
                refused(&descriptor, 4, shared),
                icb_problem(
                    CALL,
                    IndirectCommandBufferProblem::BindCount {
                        count: 65537,
                        limit: 65536
                    }
                )
            );
        }
        let memoryless = ResourceOptions::new(
            StorageMode::Memoryless,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        assert_eq!(
            refused(&compute_descriptor(), 4, memoryless),
            invalid(CALL, Problem::StorageMode)
        );
        let max_buffer_length = device.properties().max_buffer_length;
        for count in [(max_buffer_length / COMMAND_BYTES as u64) + 1, u64::MAX] {
            assert_eq!(
                refused(&compute_descriptor(), count, shared),
                Error::NotImplemented(RuntimeGap::IndirectCommandBufferPastMaxBufferLength {
                    count,
                    max_buffer_length,
                })
            );
        }
    });
}

/// A buffer set on a command at an offset of 2^32 or more lies at the offset's low 32 bits, as
/// Metal keeps it (an M4: its getters and its GPU's reads, within a buffer past
/// 4 GiB too): a kernel buffer plain and with a stride, a vertex buffer plain and with a stride,
/// a fragment buffer; a patch draw's factor buffer and an indexed patch draw's control-point index
/// buffer. An executed command reads its parameters and writes its output there.
#[test]
fn a_command_keeps_a_buffer_offset_s_low_32_bits() {
    on_device(|device| {
        let params = [0u32, 0, 0, 0, 7, 100]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let params = buffer_with(device, &params).unwrap();
        let out = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let at = |buffer: &metal2vulkan_runtime::Buffer, offset: u64, stride| {
            Some(IndirectBufferBinding::new(
                GpuAddress(buffer.gpu_address().0 + offset),
                stride,
            ))
        };
        let mut descriptor = compute_descriptor();
        descriptor.support_dynamic_attribute_stride = true;
        let made = icb(device, &descriptor, 1);
        let kernel = made.indirect_compute_command_at(0).unwrap();
        let state = template_state(device, true);
        kernel.set_compute_pipeline_state(&state).unwrap();
        kernel
            .set_kernel_buffer(Some(&params), (1 << 40) + 16, 0)
            .unwrap();
        kernel
            .set_kernel_buffer_with_stride(Some(&out), 1 << 32, 4, 1)
            .unwrap();
        kernel.set_threadgroup_memory_length(16, 0).unwrap();
        kernel
            .concurrent_dispatch_threadgroups(Size::new(1, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        assert_eq!(
            command(&made, 0).kernel_buffers[..2],
            [at(&params, 16, None), at(&out, 0, Some(4))]
        );
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder.execute_commands_in_buffer(&made, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let written = read_buffer(&out, 0, 16).unwrap();
        let written = written
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        assert_eq!(
            written,
            [107, 114, 121, 128],
            "7 * (t + 1) + 100 per thread"
        );

        let mut descriptor = render_descriptor();
        descriptor.support_dynamic_attribute_stride = true;
        let made = icb(device, &descriptor, 1);
        let render = made.indirect_render_command_at(0).unwrap();
        render
            .set_vertex_buffer(Some(&params), (1 << 40) + 16, 0)
            .unwrap();
        render
            .set_vertex_buffer_with_stride(Some(&params), (1 << 33) + 20, 4, 1)
            .unwrap();
        render
            .set_fragment_buffer(Some(&params), (1 << 32) + 8, 0)
            .unwrap();
        let read = render_command(&made, 0);
        assert_eq!(
            read.vertex_buffers[..2],
            [at(&params, 16, None), at(&params, 20, Some(4))]
        );
        assert_eq!(read.fragment_buffers[0], at(&params, 8, None));

        // A patch draw's factor buffer and an indexed one's control-point index buffer, which no
        // validation bounds, at their offsets' low 32 bits too.
        let mut descriptor = render_descriptor();
        descriptor.command_types =
            IndirectCommandType::DRAW_PATCHES.union(IndirectCommandType::DRAW_INDEXED_PATCHES);
        let made = icb(device, &descriptor, 2);
        let address = |offset: u64| GpuAddress(params.gpu_address().0 + offset);
        made.indirect_render_command_at(0)
            .unwrap()
            .draw_patches(3, 0, 1, None, 0, 1, 0, &params, (1 << 32) + 8, 0)
            .unwrap();
        made.indirect_render_command_at(1)
            .unwrap()
            .draw_indexed_patches(
                3,
                0,
                1,
                None,
                0,
                Some(&params),
                (1 << 33) + 4,
                1,
                0,
                &params,
                1 << 40,
                0,
            )
            .unwrap();
        assert_eq!(
            render_command(&made, 0).draw,
            Some(IndirectDraw::Patches {
                control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer: None,
                instance_count: 1,
                base_instance: 0,
                factor_buffer: address(8),
                factor_instance_stride: 0,
            })
        );
        assert_eq!(
            render_command(&made, 1).draw,
            Some(IndirectDraw::IndexedPatches {
                control_points: 3,
                patch_start: 0,
                patch_count: 1,
                patch_index_buffer: None,
                control_point_index_buffer: Some(address(4)),
                instance_count: 1,
                base_instance: 0,
                factor_buffer: address(0),
                factor_instance_stride: 0,
            })
        );
    });
}

/// A command's `pipeline_id` is the ID of the state its record names, whether or not that state
/// lives, and none where it names none, as Metal's getter answers a released state's identifier
/// and 0 for none (measured by the consumer on an M3 Max): a command copied into another buffer,
/// which holds no state, names the state after the state and its source buffer are released,
/// while `pipeline` reads none; the copy's fresh command names none.
#[test]
fn a_command_names_its_state_s_id_after_the_state_is_released() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let copy = |source: &IndirectCommandBuffer, into: &IndirectCommandBuffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_indirect_command_buffer(source, 0..1, into, 0)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        };
        // Compute.
        let state = template_state(device, true);
        let id = state.gpu_resource_id();
        let source = icb(device, &compute_descriptor(), 1);
        let set = source.indirect_compute_command_at(0).unwrap();
        set.set_compute_pipeline_state(&state).unwrap();
        assert_eq!(command(&source, 0).pipeline_id, Some(id));
        // A command read back holds its state: each is read and let go at once.
        let identifier = state.unique_identifier();
        assert_eq!(
            command(&source, 0).pipeline_unique_identifier,
            Some(identifier)
        );
        let copied = icb(device, &compute_descriptor(), 2);
        copy(&source, &copied);
        assert_eq!(
            command(&copied, 0).pipeline_unique_identifier,
            Some(identifier),
            "a record the GPU wrote: the live state of its ID"
        );
        drop(state);
        assert_eq!(
            command(&source, 0).pipeline_unique_identifier,
            Some(identifier),
            "the state the CPU set, after its release"
        );
        drop((set, source));
        let read = command(&copied, 0);
        assert_eq!(
            read.pipeline_unique_identifier, None,
            "a record the GPU wrote, naming an ID no state holds"
        );
        assert!(
            read.pipeline.is_none(),
            "a released compute state reads as none"
        );
        assert_eq!(
            read.pipeline_id,
            Some(id),
            "the released compute state's ID"
        );
        assert_eq!(
            command(&copied, 1).pipeline_id,
            None,
            "a fresh compute command"
        );
        assert_eq!(command(&copied, 1).pipeline_unique_identifier, Some(0));
        // Render.
        let state = render_template_state(device, true);
        let id = state.gpu_resource_id();
        let source = icb(device, &render_descriptor(), 1);
        let set = source.indirect_render_command_at(0).unwrap();
        set.set_render_pipeline_state(&state).unwrap();
        assert_eq!(render_command(&source, 0).pipeline_id, Some(id));
        let identifier = state.unique_identifier();
        let copied = icb(device, &render_descriptor(), 2);
        copy(&source, &copied);
        drop(state);
        assert_eq!(
            render_command(&source, 0).pipeline_unique_identifier,
            Some(identifier),
            "the render state the CPU set, after its release"
        );
        drop((set, source));
        let read = render_command(&copied, 0);
        assert!(
            read.pipeline.is_none(),
            "a released render state reads as none"
        );
        assert_eq!(read.pipeline_id, Some(id), "the released render state's ID");
        assert_eq!(
            render_command(&copied, 1).pipeline_id,
            None,
            "a fresh render command"
        );
        assert_eq!(
            render_command(&copied, 1).pipeline_unique_identifier,
            Some(0)
        );
    });
}

/// A command names the state set on it without holding it, as Metal's does (measured by the
/// consumer on an M3 Max): the program's last release of the state gives its ID back while the
/// command still names it, so a state made after takes the ID, of the same kind or the other
/// (compute and render states share one space); with no command naming a state, the same.
#[test]
fn a_state_set_on_a_command_gives_its_id_back_at_the_program_s_last_release() {
    on_device(|device| {
        // Compute, then a compute state and a render state after.
        let compute = icb(device, &compute_descriptor(), 1);
        let command = compute.indirect_compute_command_at(0).unwrap();
        let state = template_state(device, true);
        let id = state.gpu_resource_id();
        command.set_compute_pipeline_state(&state).unwrap();
        drop(state);
        let next = template_state(device, true);
        assert_eq!(
            next.gpu_resource_id(),
            id,
            "a compute state after a compute one"
        );
        command.set_compute_pipeline_state(&next).unwrap();
        drop(next);
        let render = render_template_state(device, true);
        assert_eq!(
            render.gpu_resource_id(),
            id,
            "a render state after a compute one"
        );
        // Render, then a render state and a compute state after.
        let drawn = icb(device, &render_descriptor(), 1);
        let command = drawn.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&render).unwrap();
        drop(render);
        let next = render_template_state(device, true);
        assert_eq!(
            next.gpu_resource_id(),
            id,
            "a render state after a render one"
        );
        command.set_render_pipeline_state(&next).unwrap();
        drop(next);
        let state = template_state(device, true);
        assert_eq!(
            state.gpu_resource_id(),
            id,
            "a compute state after a render one"
        );
    });
}

/// A command's setters write what Metal's getters read back (`3` and `4`): a buffer's address plus
/// its offset, whatever the offset; threadgroup lengths as set; sizes and stage-in regions to their
/// low 32 bits; the pipeline state set, kept alive while the command names it; a nil buffer unsets;
/// reset and `resetWithRange:` leave fresh commands.
#[test]
fn compute_commands_read_back_what_was_set() {
    on_device(|device| {
        let state = template_state(device, true);
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let address = buffer.gpu_address().0;
        let made = icb(device, &compute_descriptor(), 4);
        let first = made.indirect_compute_command_at(0).unwrap();
        first.set_compute_pipeline_state(&state).unwrap();
        first.set_kernel_buffer(Some(&buffer), 3, 0).unwrap();
        first.set_kernel_buffer(Some(&buffer), 8192, 3).unwrap();
        first.set_threadgroup_memory_length(20, 0).unwrap();
        first.set_threadgroup_memory_length(65536, 1).unwrap();
        first
            .concurrent_dispatch_threadgroups(
                Size::new((1 << 32) + 3, 5, (1 << 33) + 1),
                Size::new((1 << 32) + 2, 1, 1),
            )
            .unwrap();
        first.set_barrier().unwrap();
        first
            .set_stage_in_region(Region::new_3d((1 << 32) + 5, 1, 2, 3, (1 << 32) + 7, 4))
            .unwrap();
        first.set_imageblock_size(16, 8).unwrap();
        let id = state.gpu_resource_id();
        let read = command(&made, 0);
        assert_eq!(
            read.pipeline
                .as_ref()
                .map(ComputePipelineState::gpu_resource_id),
            Some(id)
        );
        // The command names the state without holding it, as Metal's: released, it reads as
        // none, and its ID as the command's.
        drop((state, read));
        let read = command(&made, 0);
        assert_eq!(read.pipeline_id, Some(id));
        assert!(read.pipeline.is_none(), "a released state reads as none");
        assert_eq!(
            read.dispatch,
            Some(IndirectDispatch::Threadgroups {
                threadgroups_per_grid: Size::new(3, 5, 1),
                threads_per_threadgroup: Size::new(2, 1, 1),
            })
        );
        assert!(read.barrier);
        let bound = |at: u64| Some(IndirectBufferBinding::new(GpuAddress(address + at), None));
        assert_eq!(read.kernel_buffers, vec![bound(3), None, None, bound(8192)]);
        assert_eq!(read.threadgroup_memory_lengths, vec![20, 65536]);
        assert_eq!(read.stage_in_region, Region::new_3d(5, 1, 2, 3, 7, 4));
        assert_eq!((read.imageblock_width, read.imageblock_height), (16, 8));
        // The released state's ID is free: a new state takes it, and the command names it.
        let taken = template_state(device, true);
        assert_eq!(taken.gpu_resource_id(), id);
        assert_eq!(
            command(&made, 0)
                .pipeline
                .as_ref()
                .map(ComputePipelineState::gpu_resource_id),
            Some(id)
        );
        drop(taken);

        first.clear_barrier().unwrap();
        first.set_kernel_buffer(None, 0, 0).unwrap();
        first
            .concurrent_dispatch_threads(Size::new(8, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        let read = command(&made, 0);
        assert!(!read.barrier);
        // A nil buffer is set, at its offset's low 32 bits, 0 here.
        assert_eq!(
            read.kernel_buffers[0],
            Some(IndirectBufferBinding::new(GpuAddress(0), None))
        );
        assert_eq!(
            read.dispatch,
            Some(IndirectDispatch::Threads {
                threads_per_grid: Size::new(8, 1, 1),
                threads_per_threadgroup: Size::new(4, 1, 1),
            })
        );
        // The reset keeps the imageblock size, as plain Metal's reads back after `-reset` (a
        // `CONCURRENT_DISPATCH_THREADS` buffer's command set to 8 by 4, measured by the
        // integration consumer on an M3 without API validation).
        first.reset().unwrap();
        let read = command(&made, 0);
        assert_reset(&read);
        assert_eq!((read.imageblock_width, read.imageblock_height), (16, 8));
        let mut threads = compute_descriptor();
        threads.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        let threads = icb(device, &threads, 1);
        let only = threads.indirect_compute_command_at(0).unwrap();
        only.set_imageblock_size(8, 4).unwrap();
        only.reset().unwrap();
        let read = command(&threads, 0);
        assert_eq!((read.imageblock_width, read.imageblock_height), (8, 4));

        for index in 0..4 {
            made.indirect_compute_command_at(index)
                .unwrap()
                .concurrent_dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
                .unwrap();
        }
        made.reset_with_range(0..0).unwrap();
        made.reset_with_range(1..3).unwrap();
        made.reset_with_range(3..4).unwrap();
        let dispatched: Vec<bool> = (0..4)
            .map(|index| command(&made, index).dispatch.is_some())
            .collect();
        assert_eq!(dispatched, [true, false, false, false]);
    });
}

/// Dynamic attribute strides: a buffer set with a stride reads it back, one set without reads
/// the pipeline's own; a buffer without them drops a kernel buffer's stride.
#[test]
fn dynamic_strides_read_back_as_set() {
    on_device(|device| {
        let mut descriptor = compute_descriptor();
        descriptor.support_dynamic_attribute_stride = true;
        let made = icb(device, &descriptor, 2);
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let address = buffer.gpu_address();
        let first = made.indirect_compute_command_at(0).unwrap();
        first.set_kernel_buffer(Some(&buffer), 0, 0).unwrap();
        first
            .set_kernel_buffer_with_stride(Some(&buffer), 16, 12, 1)
            .unwrap();
        first
            .set_kernel_buffer_with_stride(Some(&buffer), 32, u64::MAX, 2)
            .unwrap();
        let read = command(&made, 0);
        assert_eq!(
            read.kernel_buffers,
            vec![
                Some(IndirectBufferBinding::new(address, None)),
                Some(IndirectBufferBinding::new(
                    GpuAddress(address.0 + 16),
                    Some(12)
                )),
                Some(IndirectBufferBinding::new(GpuAddress(address.0 + 32), None)),
                None,
            ]
        );
        // A buffer without dynamic strides drops the stride and keeps the buffer, as plain
        // Metal's reads back (all ones, the pipeline's own; the M3 Max and the M4; Metal's API
        // validation stops it).
        let fixed = icb(device, &compute_descriptor(), 1);
        let only = fixed.indirect_compute_command_at(0).unwrap();
        only.set_kernel_buffer_with_stride(Some(&buffer), 16, 12, 0)
            .unwrap();
        assert_eq!(
            command(&fixed, 0).kernel_buffers[0],
            Some(IndirectBufferBinding::new(GpuAddress(address.0 + 16), None))
        );
    });
}

/// What Metal's validation stops at a setter, or ends the process for, is refused typed, and an
/// imageblock side the record cannot hold is a gap. A dispatch of the kind a buffer's command types
/// lack, which Metal's validation stops but plain Metal runs, is set.
#[test]
fn setters_refuse_what_metal_refuses() {
    on_device(|device| {
        let supported = template_state(device, true);
        let unsupported = template_state(device, false);
        assert_eq!(unsupported.gpu_resource_id(), ResourceId(0));
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let made = icb(device, &compute_descriptor(), 2);
        assert_eq!(
            made.indirect_compute_command_at(2).unwrap_err(),
            icb_problem(
                "indirect_compute_command_at",
                IndirectCommandBufferProblem::CommandIndex { index: 2, size: 2 }
            )
        );
        // A `Private` buffer's commands are the CPU's, as plain Metal encodes and reads them back.
        let private = device
            .new_indirect_command_buffer(&compute_descriptor(), 2, ResourceOptions::private())
            .unwrap();
        private
            .indirect_compute_command_at(0)
            .unwrap()
            .set_threadgroup_memory_length(32, 0)
            .unwrap();
        assert_eq!(command(&private, 0).threadgroup_memory_lengths, [32, 0]);
        private.reset_with_range(0..2).unwrap();
        // The empty range at the size resets nothing, as plain Metal's.
        made.reset_with_range(2..2).unwrap();
        for range in [3..3, 1..3, 0..5] {
            let length = range.end - range.start;
            assert_eq!(
                made.reset_with_range(range.clone()).unwrap_err(),
                icb_problem(
                    "reset_with_range",
                    IndirectCommandBufferProblem::Range {
                        location: range.start,
                        length,
                        size: 2
                    }
                )
            );
        }
        let first = made.indirect_compute_command_at(0).unwrap();
        assert_eq!(
            first.set_compute_pipeline_state(&unsupported).unwrap_err(),
            icb_problem(
                "set_compute_pipeline_state",
                IndirectCommandBufferProblem::PipelineWithoutSupport
            )
        );
        assert_eq!(
            first.set_kernel_buffer(Some(&buffer), 0, 4).unwrap_err(),
            invalid(
                "set_kernel_buffer",
                Problem::IndexOutOfRange { index: 4, limit: 4 }
            )
        );
        assert_eq!(
            first.set_threadgroup_memory_length(16, 2).unwrap_err(),
            invalid(
                "set_threadgroup_memory_length",
                Problem::IndexOutOfRange { index: 2, limit: 2 }
            )
        );
        assert_eq!(
            first.set_imageblock_size(65536, 8).unwrap_err(),
            Error::NotImplemented(RuntimeGap::IndirectImageblockSize {
                width: 65536,
                height: 8
            })
        );
        first.set_imageblock_size(65535, 8).unwrap();

        // A dispatch of the kind a buffer's command types lack: every compute buffer's records
        // hold it, and plain Metal runs it as on its own, where its validation stops the call.
        let mut threads_only = compute_descriptor();
        threads_only.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        let threads_icb = icb(device, &threads_only, 1);
        let threads = threads_icb.indirect_compute_command_at(0).unwrap();
        threads
            .concurrent_dispatch_threadgroups(Size::new(2, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        assert!(command(&threads_icb, 0).dispatch.is_some());
        let mut groups_only = compute_descriptor();
        groups_only.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
        let groups_icb = icb(device, &groups_only, 1);
        let groups = groups_icb.indirect_compute_command_at(0).unwrap();
        groups
            .concurrent_dispatch_threads(Size::new(6, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        assert!(command(&groups_icb, 0).dispatch.is_some());

        let mut inheriting = compute_descriptor();
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let inherits = icb(device, &inheriting, 1)
            .indirect_compute_command_at(0)
            .unwrap();
        assert_eq!(
            inherits.set_compute_pipeline_state(&supported).unwrap_err(),
            icb_problem(
                "set_compute_pipeline_state",
                IndirectCommandBufferProblem::InheritedPipelineState
            )
        );
        // A buffer of a buffer that inherits its buffers is kept and read back, as plain Metal's.
        inherits.set_kernel_buffer(Some(&buffer), 0, 0).unwrap();
        inherits.set_threadgroup_memory_length(16, 0).unwrap();
    });
}

/// An emptied buffer's commands read as fresh ones, and its purgeable state answers as a
/// buffer's.
#[test]
fn an_emptied_buffer_reads_fresh_commands() {
    on_device(|device| {
        let made = icb(device, &compute_descriptor(), 2);
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let first = made.indirect_compute_command_at(0).unwrap();
        first.set_kernel_buffer(Some(&buffer), 0, 0).unwrap();
        first
            .concurrent_dispatch_threadgroups(Size::new(2, 1, 1), Size::new(4, 1, 1))
            .unwrap();
        let answers = [
            PurgeableState::KeepCurrent,
            PurgeableState::Volatile,
            PurgeableState::KeepCurrent,
            PurgeableState::Empty,
            PurgeableState::KeepCurrent,
            PurgeableState::NonVolatile,
        ]
        .map(|state| made.set_purgeable_state(state).unwrap());
        assert_eq!(
            answers,
            [
                PurgeableState::NonVolatile,
                PurgeableState::NonVolatile,
                PurgeableState::Volatile,
                PurgeableState::Volatile,
                PurgeableState::Empty,
                PurgeableState::Empty,
            ]
        );
        assert_unset(&command(&made, 0));
    });
}

/// Pipeline states made with `supportIndirectCommandBuffers` take the lowest free ID of a space
/// of their own, from 1, and give it back when they go; others answer 0.
#[test]
fn pipeline_states_that_support_indirect_commands_take_dense_ids() {
    on_device(|device| {
        let first = template_state(device, true);
        let second = template_state(device, true);
        let without = template_state(device, false);
        assert_eq!(
            [&first, &second, &without].map(ComputePipelineState::gpu_resource_id),
            [ResourceId(1), ResourceId(2), ResourceId(0)]
        );
        drop(first);
        assert_eq!(
            template_state(device, true).gpu_resource_id(),
            ResourceId(1)
        );
    });
}

/// `uniqueIdentifier` as Metal allocates it (measured by the consumer on an M3 Max; for buffers): a
/// state made with `supportIndirectCommandBuffers` answers 2^36 plus an offset, the last released
/// one's again for the next of its kind whatever ID it takes, render states apart from compute
/// ones; any other a nonzero value of its own past them, a new one for each state of one
/// descriptor; a buffer 2^36 plus the lowest gap from `0x18000` holding its `allocatedSize` and a
/// 16-KiB page, at 32 KiB, and `resourceIndex` its ID.
#[test]
fn unique_identifiers_follow_metal_s_allocations() {
    const BASE: u64 = 1 << 36;
    on_device(|device| {
        let [a, b] = [template_state(device, true), template_state(device, true)];
        let [plain, again] = [template_state(device, false), template_state(device, false)];
        let render = render_template_state(device, true);
        let ids = [&a, &b, &plain, &again].map(ComputePipelineState::unique_identifier);
        let render_id = render.unique_identifier();
        assert!(ids[0] >= BASE && ids[1] > ids[0], "{ids:x?}");
        assert!(
            render_id >= BASE && !ids.contains(&render_id),
            "{render_id:x}"
        );
        assert!(ids[2] > render_id && ids[3] > render_id, "{ids:x?}");
        assert_ne!(ids[2], ids[3], "two states of one descriptor");
        assert_eq!(a.unique_identifier(), ids[0], "stable");
        // Of two released, the next takes the first one's ID and the second one's identifier.
        let a_id = a.gpu_resource_id();
        drop((a, b));
        let next = template_state(device, true);
        assert_eq!(
            (next.gpu_resource_id(), next.unique_identifier()),
            (a_id, ids[1])
        );
        let render_plain = render_template_state(device, false).unique_identifier();
        assert!(render_plain > render_id && !ids.contains(&render_plain));
        // Buffers of one page.
        let descriptor = compute_descriptor();
        let one = |count| icb(device, &descriptor, count);
        let [first, second, third] = [one(1), one(1), one(1)];
        assert_eq!(first.allocated_size(), 0x4000);
        assert_eq!(
            [&first, &second, &third].map(IndirectCommandBuffer::unique_identifier),
            [BASE + 0x1_8000, BASE + 0x2_0000, BASE + 0x2_8000]
        );
        assert_eq!(second.resource_index(), second.gpu_resource_id().0);
        drop(second);
        assert_eq!(one(1).unique_identifier(), BASE + 0x2_0000, "the gap left");
    });
}

/// The kinds a program's read of an identifier not measured to be Metal's number takes
/// ([`identifiers_not_metal_s_are_announced_once`]).
pub(super) const UNMEASURED_IDENTIFIERS: [Degradation; 2] = [
    Degradation::UnmeasuredUniqueIdentifier,
    Degradation::PlainStateUniqueIdentifier,
];

/// The kind a threadgroup length set past what a record holds takes
/// ([`commands_read_back_their_type_and_slots_past_the_record`]).
pub(super) const PAST_RECORD: [Degradation; 1] = [Degradation::IndirectValuePastRecord];

/// `getCommandType` answers the type of the dispatch or draw a record holds, none fresh and after
/// a reset (on an M4: `0x20`, `0x40`, 1 and 2, then 0). A buffer slot past the 32 a record holds,
/// below the descriptor's count, reads back as set, as Metal's getters answer every slot below the
/// count (slots 0 to 63 of 64): a buffer at slot 40 and a nil buffer at 0x30 at slot 63 of a
/// descriptor of 64, carried by a GPU copy within the buffer, cleared by a GPU reset and by the
/// command's reset, and announcing nothing. A copy into a buffer of another count carries the slots
/// both hold and clears the destination's past the source's. A threadgroup length at index 40,
/// which Metal keeps in storage not measured past about index 38, announces
/// `indirect-value-past-record` once a device and reads back as fresh.
#[test]
fn commands_read_back_their_type_and_slots_past_the_record() {
    notices::on_device(|device, messages| {
        messages.take_notices();
        let past = |messages: &Messages| -> Vec<Degradation> {
            let notices = messages.take_notices();
            counts(&notices);
            notices
                .into_iter()
                .filter_map(|message| message.degradation)
                .map(|notice| notice.degradation)
                .filter(|kind| PAST_RECORD.contains(kind))
                .collect()
        };
        let queue = device.new_command_queue().unwrap();
        let blit = |work: &dyn Fn(&mut metal2vulkan_runtime::BlitCommandEncoder)| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            work(&mut blit);
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        };
        let mut descriptor = compute_descriptor();
        descriptor.max_kernel_buffer_bind_count = 64;
        descriptor.max_kernel_threadgroup_memory_bind_count = 64;
        let made = icb(device, &descriptor, 3);
        assert_eq!(command(&made, 0).command_type, IndirectCommandType::empty());
        let one = Size::new(1, 1, 1);
        let set = made.indirect_compute_command_at(0).unwrap();
        set.concurrent_dispatch_threadgroups(one, one).unwrap();
        assert_eq!(
            command(&made, 0).command_type,
            IndirectCommandType::CONCURRENT_DISPATCH
        );
        set.concurrent_dispatch_threads(one, one).unwrap();
        set.set_barrier().unwrap();
        assert_eq!(
            command(&made, 0).command_type,
            IndirectCommandType::CONCURRENT_DISPATCH_THREADS
        );
        assert_eq!(past(messages), []);
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let address = buffer.gpu_address().0;
        set.set_kernel_buffer(Some(&buffer), 0x10, 40).unwrap();
        set.set_kernel_buffer(None, 0x30, 63).unwrap();
        assert_eq!(past(messages), [], "a buffer past the record is kept");
        set.set_threadgroup_memory_length(0x200, 40).unwrap();
        assert_eq!(past(messages), PAST_RECORD);
        set.set_threadgroup_memory_length(0x300, 41).unwrap();
        assert_eq!(past(messages), [], "once a device");
        let slots = |read: &EncodedComputeCommand| {
            [39, 40, 63, 64].map(|at| read.kernel_buffer_address(at))
        };
        let kept = [Some(0), Some(address + 0x10), Some(0x30), None];
        let read = command(&made, 0);
        assert_eq!(read.kernel_buffers.len(), 64, "every slot below the count");
        assert_eq!(slots(&read), kept);
        assert_eq!(read.threadgroup_memory_lengths.len(), 31);
        blit(&|blit| {
            blit.copy_indirect_command_buffer(&made, 0..1, &made, 1)
                .unwrap()
        });
        assert_eq!(slots(&command(&made, 1)), kept, "carried by a GPU copy");
        blit(&|blit| blit.reset_commands_in_buffer(&made, 1..2).unwrap());
        let fresh = [Some(0), Some(0), Some(0), None];
        assert_eq!(slots(&command(&made, 1)), fresh, "cleared by a GPU reset");
        assert_eq!(slots(&command(&made, 0)), kept, "the source kept");

        // A copy between counts: slot 33 carried, slot 45 of the destination, past the source's
        // 40, cleared.
        let mut fewer = compute_descriptor();
        fewer.max_kernel_buffer_bind_count = 40;
        let source = icb(device, &fewer, 1);
        let from = source.indirect_compute_command_at(0).unwrap();
        from.set_kernel_buffer(Some(&buffer), 0x20, 33).unwrap();
        let mut more = compute_descriptor();
        more.max_kernel_buffer_bind_count = 48;
        let destination = icb(device, &more, 1);
        let into = destination.indirect_compute_command_at(0).unwrap();
        into.set_kernel_buffer(None, 0x70, 33).unwrap();
        into.set_kernel_buffer(None, 0x60, 45).unwrap();
        blit(&|blit| {
            blit.copy_indirect_command_buffer(&source, 0..1, &destination, 0)
                .unwrap()
        });
        let read = command(&destination, 0);
        assert_eq!(
            [33, 45, 47, 48].map(|at| read.kernel_buffer_address(at)),
            [Some(address + 0x20), Some(0), Some(0), None]
        );

        set.reset().unwrap();
        assert_eq!(command(&made, 0).command_type, IndirectCommandType::empty());
        assert_eq!(
            slots(&command(&made, 0)),
            fresh,
            "cleared by the command's reset"
        );
        let render = icb(device, &render_descriptor(), 2);
        let draw = render.indirect_render_command_at(0).unwrap();
        assert_eq!(
            render_command(&render, 0).command_type,
            IndirectCommandType::empty()
        );
        draw.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        assert_eq!(
            render_command(&render, 0).command_type,
            IndirectCommandType::DRAW
        );
        draw.reset().unwrap();
        assert_eq!(
            render_command(&render, 0).command_type,
            IndirectCommandType::empty()
        );
        assert_eq!(past(messages), []);
    });
}

/// Metal's buffer getters read a command's slots as its setters left them, and a GPU copy
/// carries them and a GPU reset clears them as Metal's do (on an M4): an address getter answers the
/// buffer's address plus the offset's low 32 bits, the offset's low 32 bits alone for a nil buffer,
/// 0 where none is set; a stride getter all ones without dynamic attribute strides, and with them 0
/// where none is set, all ones after the plain form (a nil buffer's too) and the stride after the
/// stride form (0 included); none at or past the descriptor's count.
#[test]
fn buffer_getters_answer_metal_s_values_through_gpu_copies_and_resets() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let blit = |icb: &IndirectCommandBuffer, reset: bool| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            if reset {
                blit.reset_commands_in_buffer(icb, 0..1).unwrap();
            } else {
                blit.copy_indirect_command_buffer(icb, 0..1, icb, 1)
                    .unwrap();
            }
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.error(), None);
        };
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let address = buffer.gpu_address().0;
        for dynamic in [false, true] {
            // Compute: slot 1 a buffer at 0x30, 2 nil at 2^32 + 0x30, 3 a buffer at 0x10 at
            // stride 48, 4 nil at 0x20 at stride 48, 5 a buffer at 0x10 at stride 0.
            let mut descriptor = compute_descriptor();
            descriptor.max_kernel_buffer_bind_count = 8;
            descriptor.support_dynamic_attribute_stride = dynamic;
            let made = icb(device, &descriptor, 2);
            let set = made.indirect_compute_command_at(0).unwrap();
            set.set_kernel_buffer(Some(&buffer), 0x30, 1).unwrap();
            set.set_kernel_buffer(None, (1 << 32) + 0x30, 2).unwrap();
            set.set_kernel_buffer_with_stride(Some(&buffer), 0x10, 48, 3)
                .unwrap();
            set.set_kernel_buffer_with_stride(None, 0x20, 48, 4)
                .unwrap();
            set.set_kernel_buffer_with_stride(Some(&buffer), 0x10, 0, 5)
                .unwrap();
            let answers = |read: &EncodedComputeCommand| -> Vec<(Option<u64>, Option<u64>)> {
                (0..9)
                    .map(|at| {
                        (
                            read.kernel_buffer_address(at),
                            read.kernel_attribute_stride(at),
                        )
                    })
                    .collect()
            };
            let all = u64::MAX;
            let fresh = if dynamic { 0 } else { all };
            let stride = |set: u64| if dynamic { set } else { all };
            let set_answers = vec![
                (Some(0), Some(fresh)),
                (Some(address + 0x30), Some(all)),
                (Some(0x30), Some(all)),
                (Some(address + 0x10), Some(stride(48))),
                (Some(0x20), Some(stride(48))),
                (Some(address + 0x10), Some(stride(0))),
                (Some(0), Some(fresh)),
                (Some(0), Some(fresh)),
                (None, None),
            ];
            assert_eq!(answers(&command(&made, 0)), set_answers, "{dynamic}");
            assert_eq!(
                command(&made, 0).kernel_buffers[2],
                Some(IndirectBufferBinding::new(GpuAddress(0x30), None)),
                "a nil buffer is set"
            );
            blit(&made, false);
            assert_eq!(answers(&command(&made, 1)), set_answers, "{dynamic} copied");
            blit(&made, true);
            let reset: Vec<_> = (0..9)
                .map(|at| (at < 8).then_some(0))
                .zip((0..9).map(|at| (at < 8).then_some(fresh)))
                .collect();
            assert_eq!(answers(&command(&made, 0)), reset, "{dynamic} reset");
            assert_eq!(answers(&command(&made, 1)), set_answers, "{dynamic} kept");

            // Render: vertex slot 1 nil at 0x40, slot 0 a buffer at 0x30 at stride 48;
            // fragment slot 1 nil at 0x50.
            let mut descriptor = render_descriptor();
            descriptor.support_dynamic_attribute_stride = dynamic;
            let made = icb(device, &descriptor, 2);
            let draw = made.indirect_render_command_at(0).unwrap();
            draw.set_vertex_buffer_with_stride(Some(&buffer), 0x30, 48, 0)
                .unwrap();
            draw.set_vertex_buffer(None, 0x40, 1).unwrap();
            draw.set_fragment_buffer(None, 0x50, 0).unwrap();
            let answers = |read: &EncodedRenderCommand| {
                (
                    [0, 1, 2].map(|at| read.vertex_buffer_address(at)),
                    [0, 1, 2].map(|at| read.vertex_attribute_stride(at)),
                    [0, 1].map(|at| read.fragment_buffer_address(at)),
                )
            };
            let set_answers = (
                [Some(address + 0x30), Some(0x40), None],
                [Some(stride(48)), Some(all), None],
                [Some(0x50), None],
            );
            assert_eq!(answers(&render_command(&made, 0)), set_answers, "{dynamic}");
            blit(&made, false);
            assert_eq!(
                answers(&render_command(&made, 1)),
                set_answers,
                "{dynamic} copied"
            );
            blit(&made, true);
            assert_eq!(
                answers(&render_command(&made, 0)),
                (
                    [Some(0), Some(0), None],
                    [Some(fresh), Some(fresh), None],
                    [Some(0), None]
                ),
                "{dynamic} reset"
            );
        }
        // Object and mesh slots: a nil buffer's offset, a buffer's address.
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_MESH_THREADGROUPS;
        descriptor.max_object_buffer_bind_count = 2;
        descriptor.max_mesh_buffer_bind_count = 2;
        let made = icb(device, &descriptor, 1);
        let draw = made.indirect_render_command_at(0).unwrap();
        draw.set_object_buffer(None, 0x30, 0).unwrap();
        draw.set_object_buffer(Some(&buffer), 0x30, 1).unwrap();
        draw.set_mesh_buffer(None, 0x40, 1).unwrap();
        let read = render_command(&made, 0);
        assert_eq!(
            [0, 1, 2].map(|at| read.object_buffer_address(at)),
            [Some(0x30), Some(address + 0x30), None]
        );
        assert_eq!(
            [0, 1, 2].map(|at| read.mesh_buffer_address(at)),
            [Some(0), Some(0x40), None]
        );
    });
}

/// The [`UNMEASURED_IDENTIFIERS`] notices delivered so far.
fn identifier_notices(messages: &Messages) -> Vec<Degradation> {
    let notices = messages.take_notices();
    counts(&notices);
    notices
        .into_iter()
        .filter_map(|message| message.degradation)
        .map(|notice| notice.degradation)
        .filter(|kind| UNMEASURED_IDENTIFIERS.contains(kind))
        .collect()
}

/// Which identifiers answer Metal's number: an indirect command buffer made before its device
/// opened a command encoder answers it quietly, as does a command naming no state (0); a buffer
/// made after one, and a state made with indirect command buffer support, announce
/// `unmeasured-unique-identifier` at their first answer, and a plain state
/// `plain-state-unique-identifier`, each once a device. A compute state with indirect command
/// buffer support starts one step into Metal's range, `2^36 + 0xc0`.
#[test]
fn identifiers_not_metal_s_are_announced_once() {
    const BASE: u64 = 1 << 36;
    let kind = Degradation::UnmeasuredUniqueIdentifier;
    notices::on_device(|device, messages| {
        messages.take_notices();
        let descriptor = compute_descriptor();
        let first = icb(device, &descriptor, 1);
        assert_eq!(first.unique_identifier(), BASE + 0x1_8000);
        assert_eq!(command(&first, 0).pipeline_unique_identifier, Some(0));
        assert_eq!(identifier_notices(messages), []);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .blit_command_encoder()
            .unwrap()
            .end_encoding()
            .unwrap();
        assert_eq!(first.unique_identifier(), BASE + 0x1_8000);
        assert_eq!(identifier_notices(messages), [], "made before the encoder");
        let later = icb(device, &descriptor, 1);
        assert_eq!(later.unique_identifier(), BASE + 0x2_0000, "the rule still");
        assert_eq!(identifier_notices(messages), [kind]);
        let state = template_state(device, true);
        assert_eq!(state.unique_identifier(), BASE + 0xc0);
        assert_eq!(identifier_notices(messages), [], "once a device");
        let plain = template_state(device, false).unique_identifier();
        assert!(plain > BASE + 0xc0, "{plain:#x}");
        assert_eq!(
            identifier_notices(messages),
            [Degradation::PlainStateUniqueIdentifier]
        );
        // The count is the process's: another state's is past it.
        assert!(template_state(device, false).unique_identifier() > plain);
        assert_eq!(identifier_notices(messages), []);
    });
    notices::on_device(|device, messages| {
        messages.take_notices();
        let state = template_state(device, true);
        let made = icb(device, &compute_descriptor(), 1);
        let set = made.indirect_compute_command_at(0).unwrap();
        set.set_compute_pipeline_state(&state).unwrap();
        assert_eq!(identifier_notices(messages), [], "a setter answers nothing");
        let named = command(&made, 0).pipeline_unique_identifier;
        assert_eq!(identifier_notices(messages), [kind]);
        assert_eq!(named, Some(state.unique_identifier()));
        assert_eq!(identifier_notices(messages), []);
    });
}

/// A render command's setters write what Metal's getters read back: its
/// pipeline state, kept alive while the command names it; each vertex and fragment buffer's
/// address plus its offset, whatever the offset; a draw's values (the record's low 32 bits); a
/// nil buffer unsets, and a reset leaves a fresh command.
#[test]
fn render_commands_read_back_what_was_set() {
    on_device(|device| {
        let state = render_template_state(device, true);
        let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let address = buffer.gpu_address().0;
        let made = icb(device, &render_descriptor(), 2);
        let fresh = render_command(&made, 1);
        assert!(fresh.draw.is_none() && !fresh.barrier && fresh.pipeline.is_none());
        assert_eq!(fresh.vertex_buffers, vec![None, None]);
        assert_eq!(fresh.fragment_buffers, vec![None]);
        let first = made.indirect_render_command_at(0).unwrap();
        first.set_render_pipeline_state(&state).unwrap();
        first.set_vertex_buffer(Some(&buffer), 16, 0).unwrap();
        first.set_vertex_buffer(Some(&buffer), 3, 1).unwrap();
        first.set_fragment_buffer(Some(&buffer), 8192, 0).unwrap();
        first
            .draw_primitives(
                PrimitiveType::TriangleStrip,
                (1 << 32) + 5,
                (1 << 32) + 3,
                (1 << 32) + 2,
                (1 << 32) + 7,
            )
            .unwrap();
        let id = state.gpu_resource_id();
        assert_eq!(
            render_command(&made, 0)
                .pipeline
                .as_ref()
                .map(RenderPipelineState::gpu_resource_id),
            Some(id)
        );
        // Named without being held, as Metal's: released, it reads as none.
        drop(state);
        let read = render_command(&made, 0);
        assert_eq!(read.pipeline_id, Some(id));
        assert!(read.pipeline.is_none(), "a released state reads as none");
        assert_eq!(
            read.draw,
            Some(IndirectDraw::Primitives {
                primitive_type: PrimitiveType::TriangleStrip,
                vertex_start: 5,
                vertex_count: 3,
                instance_count: 2,
                base_instance: 7,
            })
        );
        assert!(!read.barrier);
        let bound = |at: u64| Some(IndirectBufferBinding::new(GpuAddress(address + at), None));
        assert_eq!(read.vertex_buffers, vec![bound(16), bound(3)]);
        assert_eq!(read.fragment_buffers, vec![bound(8192)]);
        // The released state's ID is free: a new state takes it.
        let taken = render_template_state(device, true);
        assert_eq!(taken.gpu_resource_id(), id);
        drop(taken);

        first.set_vertex_buffer(None, 0, 0).unwrap();
        let read = render_command(&made, 0);
        // A nil buffer is set, at its offset's low 32 bits, 0 here.
        let nil = Some(IndirectBufferBinding::new(GpuAddress(0), None));
        assert_eq!(read.vertex_buffers, vec![nil, bound(3)]);
        first.reset().unwrap();
        let reset = render_command(&made, 0);
        assert!(reset.draw.is_none() && reset.pipeline.is_none());
        assert_eq!(reset.vertex_buffers, vec![None, None]);

        let mut dynamic = render_descriptor();
        dynamic.support_dynamic_attribute_stride = true;
        let strided = icb(device, &dynamic, 1);
        let command = strided.indirect_render_command_at(0).unwrap();
        command
            .set_vertex_buffer_with_stride(Some(&buffer), 32, 12, 0)
            .unwrap();
        command
            .set_vertex_buffer_with_stride(Some(&buffer), 64, u64::MAX, 1)
            .unwrap();
        assert_eq!(
            render_command(&strided, 0).vertex_buffers,
            vec![
                Some(IndirectBufferBinding::new(
                    GpuAddress(address + 32),
                    Some(12)
                )),
                Some(IndirectBufferBinding::new(GpuAddress(address + 64), None)),
            ]
        );
    });
}

/// What Metal's validation stops at a render command's setter, or ends the process for, is
/// refused typed: an index past the descriptor's counts, a state without
/// `supportIndirectCommandBuffers`, a state or buffer where the buffer inherits it, no instances,
/// and a barrier on a buffer without mesh draws; so is a compute command of a render buffer, while
/// a render command of a buffer of dispatches answers as plain Metal's. A compute encoder's
/// execution of render commands, which Metal's validation lets through, is a gap. A vertex buffer's
/// stride without dynamic strides is dropped and the buffer set, as plain Metal's.
#[test]
fn render_setters_refuse_what_metal_refuses() {
    on_device(|device| {
        let supported = render_template_state(device, true);
        let unsupported = render_template_state(device, false);
        assert_eq!(unsupported.gpu_resource_id(), ResourceId(0));
        let buffer = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let made = icb(device, &render_descriptor(), 2);
        let compute = icb(device, &compute_descriptor(), 2);
        assert_eq!(
            made.indirect_compute_command_at(0).unwrap_err(),
            icb_problem(
                "indirect_compute_command_at",
                IndirectCommandBufferProblem::CommandKind
            )
        );
        // A buffer of dispatches gives a render command, whose setters plain Metal ignores do
        // nothing, the others refused.
        let of_compute = compute.indirect_render_command_at(0).unwrap();
        of_compute
            .draw_primitives(PrimitiveType::Point, 0, 1, 1, 0)
            .unwrap();
        of_compute.set_vertex_buffer(Some(&buffer), 0, 0).unwrap();
        of_compute.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
        of_compute.set_barrier().unwrap();
        of_compute.clear_barrier().unwrap();
        let kind = |call| icb_problem(call, IndirectCommandBufferProblem::CommandKind);
        assert_eq!(
            of_compute
                .set_render_pipeline_state(&supported)
                .unwrap_err(),
            kind("set_render_pipeline_state")
        );
        assert_eq!(of_compute.reset().unwrap_err(), kind("reset"));
        assert_eq!(
            of_compute
                .draw_indexed_primitives(
                    PrimitiveType::Point,
                    1,
                    IndexType::UInt16,
                    &buffer,
                    0,
                    1,
                    0,
                    0
                )
                .unwrap_err(),
            kind("draw_indexed_primitives")
        );
        assert_unset(&command(&compute, 0));
        assert_eq!(
            made.indirect_render_command_at(2).unwrap_err(),
            icb_problem(
                "indirect_render_command_at",
                IndirectCommandBufferProblem::CommandIndex { index: 2, size: 2 }
            )
        );
        let first = made.indirect_render_command_at(0).unwrap();
        assert_eq!(
            first.set_render_pipeline_state(&unsupported).unwrap_err(),
            icb_problem(
                "set_render_pipeline_state",
                IndirectCommandBufferProblem::PipelineWithoutSupport
            )
        );
        assert_eq!(
            first.set_vertex_buffer(Some(&buffer), 0, 2).unwrap_err(),
            invalid(
                "set_vertex_buffer",
                Problem::IndexOutOfRange { index: 2, limit: 2 }
            )
        );
        assert_eq!(
            first.set_fragment_buffer(Some(&buffer), 0, 1).unwrap_err(),
            invalid(
                "set_fragment_buffer",
                Problem::IndexOutOfRange { index: 1, limit: 1 }
            )
        );
        // A buffer without dynamic strides drops a vertex buffer's stride and keeps the buffer
        // at its offset, as plain Metal's reads back and draws.
        first
            .set_vertex_buffer_with_stride(Some(&buffer), 16, 12, 0)
            .unwrap();
        assert_eq!(
            render_command(&made, 0).vertex_buffers[0],
            Some(IndirectBufferBinding::new(
                GpuAddress(buffer.gpu_address().0 + 16),
                None
            ))
        );
        let mut inheriting = render_descriptor();
        inheriting.inherit_pipeline_state = true;
        inheriting.inherit_buffers = true;
        let inherits = icb(device, &inheriting, 1)
            .indirect_render_command_at(0)
            .unwrap();
        assert_eq!(
            inherits.set_render_pipeline_state(&supported).unwrap_err(),
            icb_problem(
                "set_render_pipeline_state",
                IndirectCommandBufferProblem::InheritedPipelineState
            )
        );
        inherits.set_vertex_buffer(Some(&buffer), 0, 0).unwrap();
        inherits
            .draw_primitives(PrimitiveType::Point, 0, 3, 1, 0)
            .unwrap();
        // No instances (the count's low 32 bits): plain Metal runs nothing, where its validation
        // stops the call.
        first
            .draw_primitives(PrimitiveType::Point, 0, 3, 1 << 32, 0)
            .unwrap();
        first
            .draw_primitives(PrimitiveType::Point, 0, 0, 1, 0)
            .unwrap();
        // A barrier is kept, as plain Metal keeps it, and a cleared one reads back none.
        first.set_barrier().unwrap();
        assert!(render_command(&made, 0).barrier);
        first.clear_barrier().unwrap();
        assert!(!render_command(&made, 0).barrier);

        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(
            encoder.execute_commands_in_buffer(&made, 0..1).unwrap_err(),
            Error::NotImplemented(RuntimeGap::IndirectRenderCommands)
        );
        encoder.end_encoding().unwrap();
    });
}

/// Render states made with `supportIndirectCommandBuffers` take their IDs from the space compute
/// states take theirs from, as Metal's share one; others answer 0; and
/// an argument encoder writes a render or compute state's ID, 0 for a state without the support.
#[test]
fn render_states_share_the_pipeline_states_ids() {
    on_device(|device| {
        let compute = template_state(device, true);
        let render = render_template_state(device, true);
        let without = render_template_state(device, false);
        assert_eq!(
            [
                compute.gpu_resource_id(),
                render.gpu_resource_id(),
                without.gpu_resource_id()
            ],
            [ResourceId(1), ResourceId(2), ResourceId(0)]
        );
        drop(compute);
        let next = render_template_state(device, true);
        assert_eq!(next.gpu_resource_id(), ResourceId(1));

        let mut encoder = device
            .new_argument_encoder_with_arguments(&[ArgumentDescriptor::new(
                DataType::RenderPipeline,
                0,
            )])
            .unwrap();
        let arguments = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        encoder.set_argument_buffer(Some(&arguments), 0).unwrap();
        encoder.set_render_pipeline_state(Some(&render), 0).unwrap();
        let word = |buffer: &metal2vulkan_runtime::Buffer| {
            let contents = buffer.contents().unwrap();
            // SAFETY: the buffer is shared and mapped, 64 bytes long, and nothing writes it now.
            let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 8) };
            u64::from_le_bytes(bytes.try_into().unwrap())
        };
        assert_eq!(word(&arguments), 2);
        // A state made without the support writes its ID, 0, as plain Metal writes it (an M4;
        // Metal's debug layer asserts), a compute state's too.
        encoder
            .set_render_pipeline_state(Some(&without), 0)
            .unwrap();
        assert_eq!(word(&arguments), 0);
        encoder.set_render_pipeline_state(Some(&render), 0).unwrap();
        encoder.set_render_pipeline_state(None, 0).unwrap();
        assert_eq!(word(&arguments), 0);
        let mut compute_encoder = device
            .new_argument_encoder_with_arguments(&[ArgumentDescriptor::new(
                DataType::ComputePipeline,
                0,
            )])
            .unwrap();
        compute_encoder
            .set_argument_buffer(Some(&arguments), 0)
            .unwrap();
        let supporting = template_state(device, true);
        compute_encoder
            .set_compute_pipeline_state(Some(&supporting), 0)
            .unwrap();
        assert_eq!(word(&arguments), supporting.gpu_resource_id().0);
        compute_encoder
            .set_compute_pipeline_state(Some(&template_state(device, false)), 0)
            .unwrap();
        assert_eq!(word(&arguments), 0);
    });
}

/// A descriptor of indexed and patch draws, 2 vertex buffers and 1 fragment buffer.
fn indexed_patch_descriptor() -> IndirectCommandBufferDescriptor {
    let mut descriptor = render_descriptor();
    descriptor.command_types =
        IndirectCommandType::DRAW_INDEXED.union(IndirectCommandType::DRAW_PATCHES);
    descriptor
}

/// An indexed draw and a patch draw read back as set: each buffer as its address plus the offset, a
/// patch draw without a patch index buffer as none; a base vertex (signed), a patch count and an
/// instance count to their low 32 bits, as Metal keeps the first two; the control points as set,
/// though an execution draws with its pipeline's.
#[test]
fn indexed_and_patch_draws_read_back_what_was_set() {
    on_device(|device| {
        let indices = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let factors = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let made = icb(device, &indexed_patch_descriptor(), 3);
        let at = |buffer: &metal2vulkan_runtime::Buffer, offset| {
            GpuAddress(buffer.gpu_address().0 + offset)
        };
        made.indirect_render_command_at(0)
            .unwrap()
            .draw_indexed_primitives(
                PrimitiveType::LineStrip,
                6,
                IndexType::UInt16,
                &indices,
                8,
                (1 << 32) + 2,
                (1 << 32) - 3,
                (1 << 32) + 4,
            )
            .unwrap();
        assert_eq!(
            render_command(&made, 0).draw,
            Some(IndirectDraw::IndexedPrimitives {
                primitive_type: PrimitiveType::LineStrip,
                index_count: 6,
                index_type: IndexType::UInt16,
                index_buffer: at(&indices, 8),
                instance_count: 2,
                base_vertex: -3,
                base_instance: 4,
            })
        );
        let patches = made.indirect_render_command_at(1).unwrap();
        patches
            .draw_patches(
                0,
                1,
                (1 << 32) + 2,
                Some(&indices),
                16,
                3,
                5,
                &factors,
                32,
                48,
            )
            .unwrap();
        assert_eq!(
            render_command(&made, 1).draw,
            Some(IndirectDraw::Patches {
                control_points: 0,
                patch_start: 1,
                patch_count: 2,
                patch_index_buffer: Some(at(&indices, 16)),
                instance_count: 3,
                base_instance: 5,
                factor_buffer: at(&factors, 32),
                factor_instance_stride: 48,
            })
        );
        made.indirect_render_command_at(2)
            .unwrap()
            .draw_patches(4, 0, 1, None, 0, 1, 0, &factors, 0, 0)
            .unwrap();
        let Some(IndirectDraw::Patches {
            patch_index_buffer, ..
        }) = render_command(&made, 2).draw
        else {
            panic!("not a patch draw");
        };
        assert_eq!(patch_index_buffer, None);
    });
}

/// What Metal's validation stops at an indexed or patch draw's setter, or ends the process for,
/// is refused typed: a draw whose record the buffer's records cannot hold (an indexed or a patch
/// draw in a buffer of draws of primitives), no indices or instances, an index offset that is not a
/// multiple of the index size, indices past the index buffer (of the whole count), a patch index
/// offset that is not a multiple of 4; and as the encoder's draw refuses them, patch indices past
/// the buffer (factor offsets and strides that are not multiples of 4, odd ones too, and more than
/// 32 control points, are taken). A refused draw leaves the command as it was. A draw of primitives
/// into a buffer of indexed and patch draws, whose records hold it, is set, as plain Metal runs it,
/// where its validation stops it.
#[test]
fn indexed_and_patch_setters_refuse_what_metal_refuses() {
    on_device(|device| {
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        let draws = icb(device, &render_descriptor(), 1);
        let only_draws = draws.indirect_render_command_at(0).unwrap();
        let kind = |call, kind| icb_problem(call, IndirectCommandBufferProblem::CommandType(kind));
        assert_eq!(
            only_draws
                .draw_indexed_primitives(
                    PrimitiveType::Point,
                    3,
                    IndexType::UInt32,
                    &buffer,
                    0,
                    1,
                    0,
                    0
                )
                .unwrap_err(),
            kind("draw_indexed_primitives", IndirectCommandType::DRAW_INDEXED)
        );
        assert_eq!(
            only_draws
                .draw_patches(3, 0, 1, None, 0, 1, 0, &buffer, 0, 0)
                .unwrap_err(),
            kind("draw_patches", IndirectCommandType::DRAW_PATCHES)
        );
        let made = icb(device, &indexed_patch_descriptor(), 1);
        let command = made.indirect_render_command_at(0).unwrap();
        // A draw of primitives fits the records of a buffer of indexed and patch draws: plain
        // Metal runs it as on its own, where its validation stops the call.
        command
            .draw_primitives(PrimitiveType::Point, 0, 3, 1, 0)
            .unwrap();
        assert!(matches!(
            render_command(&made, 0).draw,
            Some(IndirectDraw::Primitives {
                vertex_count: 3,
                ..
            })
        ));
        command.reset().unwrap();
        let indexed = |count, offset, instances| {
            command.draw_indexed_primitives(
                PrimitiveType::Point,
                count,
                IndexType::UInt32,
                &buffer,
                offset,
                instances,
                0,
                0,
            )
        };
        let draw = |call, problem| invalid(call, Problem::Draw(problem));
        const INDEXED: &str = "draw_indexed_primitives";
        // No indices or instances, and an offset off the index size: plain Metal runs nothing,
        // or reads from the offset rounded down, where its validation stops the call.
        assert_eq!(indexed(0, 0, 1), Ok(()));
        assert_eq!(indexed(3, 0, 1 << 32), Ok(()));
        assert_eq!(indexed(3, 2, 1), Ok(()));
        assert_eq!(
            indexed(15, 8, 1).unwrap_err(),
            draw(
                INDEXED,
                DrawProblem::IndexBufferRange {
                    offset: 8,
                    count: 15,
                    length: 64
                }
            )
        );
        assert_eq!(
            indexed(15, 10, 1).unwrap_err(),
            draw(
                INDEXED,
                DrawProblem::IndexBufferRange {
                    offset: 8,
                    count: 15,
                    length: 64
                }
            )
        );
        assert_eq!(
            indexed((1 << 32) + 1, 0, 1).unwrap_err(),
            draw(
                INDEXED,
                DrawProblem::IndexBufferRange {
                    offset: 0,
                    count: (1 << 32) + 1,
                    length: 64
                }
            )
        );
        const PATCHES: &str = "draw_patches";
        let patches = |start, count, pib: Option<u64>, factor: u64, stride: u64| {
            command
                .draw_patches(
                    3,
                    start,
                    count,
                    pib.map(|_| &buffer),
                    pib.unwrap_or(0),
                    1,
                    0,
                    &buffer,
                    factor,
                    stride,
                )
                .unwrap_err()
        };
        // No instances, and a patch index offset off a multiple of 4: plain Metal runs nothing,
        // or reads from the offset rounded down.
        command
            .draw_patches(3, 0, 1, None, 0, 0, 0, &buffer, 0, 0)
            .unwrap();
        command
            .draw_patches(3, 0, 1, Some(&buffer), 2, 1, 0, &buffer, 0, 0)
            .unwrap();
        assert_eq!(
            patches(10, 7, Some(0), 0, 0),
            draw(
                PATCHES,
                DrawProblem::PatchIndexBufferRange {
                    offset: 0,
                    indices: 17,
                    length: 64
                }
            )
        );
        // Factor offsets and strides that are not multiples of 4, odd ones too, and more than 32
        // control points, which Metal's validation stops (a probe: "numberOfPatchControlPoints(33)
        // must be <= 32"), are taken, as plain Metal draws them (Metal probes;
        // `render_executions.rs` draws them).
        for (control_points, offset, stride) in [
            (32, 0, 0),
            (33, 0, 0),
            (3, 2, 6),
            (3, 1, 0),
            (3, 0, 3),
            (3, 7, 15),
        ] {
            command
                .draw_patches(control_points, 0, 1, None, 0, 1, 0, &buffer, offset, stride)
                .unwrap();
        }
        command.reset().unwrap();
        assert!(render_command(&made, 0).draw.is_none());
    });
}

/// An indexed patch draw reads back as set, as Metal's `drawIndexedPatchesArguments` and
/// `getTessellationFactorArguments` answered it (a probe, an M4): each buffer as its address plus
/// the offset, a patch start, a patch count, an instance count, a base instance and a factor stride
/// to their low 32 bits, the control points as set (70000 included, without validation), and a null
/// patch index buffer as none. Without a control-point index buffer it reads back as a plain patch
/// draw of the same values, as Metal records it (probe 9b's `getCommandType`, an M4).
#[test]
fn indexed_patch_draws_read_back_what_was_set() {
    on_device(|device| {
        let indices = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let points = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let factors = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        let mut descriptor = render_descriptor();
        descriptor.command_types = IndirectCommandType::DRAW_INDEXED_PATCHES;
        let made = icb(device, &descriptor, 2);
        let at = |buffer: &metal2vulkan_runtime::Buffer, offset| {
            GpuAddress(buffer.gpu_address().0 + offset)
        };
        made.indirect_render_command_at(0)
            .unwrap()
            .draw_indexed_patches(
                5,
                7,
                11,
                Some(&indices),
                12,
                Some(&points),
                24,
                13,
                17,
                &factors,
                36,
                44,
            )
            .unwrap();
        assert_eq!(
            render_command(&made, 0).draw,
            Some(IndirectDraw::IndexedPatches {
                control_points: 5,
                patch_start: 7,
                patch_count: 11,
                patch_index_buffer: Some(at(&indices, 12)),
                control_point_index_buffer: Some(at(&points, 24)),
                instance_count: 13,
                base_instance: 17,
                factor_buffer: at(&factors, 36),
                factor_instance_stride: 44,
            })
        );
        made.indirect_render_command_at(1)
            .unwrap()
            .draw_indexed_patches(
                3,
                (1 << 32) + 1,
                (1 << 32) + 2,
                None,
                0,
                None,
                0,
                (1 << 32) + 3,
                (1 << 32) + 4,
                &factors,
                0,
                (1 << 32) + 8,
            )
            .unwrap();
        assert_eq!(
            render_command(&made, 1).draw,
            Some(IndirectDraw::Patches {
                control_points: 3,
                patch_start: 1,
                patch_count: 2,
                patch_index_buffer: None,
                instance_count: 3,
                base_instance: 4,
                factor_buffer: at(&factors, 0),
                factor_instance_stride: 8,
            })
        );
    });
}

/// What Metal's validation stops at an indexed patch draw's setter is refused typed (probes): a
/// buffer of neither patch kind, whose records cannot hold it, and patch indices past the patch
/// index buffer. Factor offsets and strides that are not multiples of 4, odd ones too, which
/// Metal's validation lets through here, and more than 32 control points, which it stops, are taken
/// as the plain patch draw's are. The control-point index buffer, which Metal's setter does not
/// check (an offset of 1 or 2, indices past its end, none at all), is taken. A refused draw leaves
/// the command as it was. On a buffer of patch draws, whose records hold it, the draw is set, as
/// plain Metal runs it, where its validation stops it.
#[test]
fn the_indexed_patch_setter_refuses_what_metal_refuses() {
    on_device(|device| {
        let buffer = device.new_buffer(64, ResourceOptions::shared()).unwrap();
        const CALL: &str = "draw_indexed_patches";
        let draw_on = |types| {
            let mut descriptor = render_descriptor();
            descriptor.command_types = types;
            let made = icb(device, &descriptor, 1);
            let set = made
                .indirect_render_command_at(0)
                .unwrap()
                .draw_indexed_patches(3, 0, 1, None, 0, Some(&buffer), 0, 1, 0, &buffer, 0, 0);
            (made, set)
        };
        for types in [
            IndirectCommandType::DRAW,
            IndirectCommandType::DRAW_INDEXED,
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED),
        ] {
            assert_eq!(
                draw_on(types).1.unwrap_err(),
                icb_problem(
                    CALL,
                    IndirectCommandBufferProblem::CommandType(
                        IndirectCommandType::DRAW_INDEXED_PATCHES
                    )
                )
            );
        }
        let (patches, set) = draw_on(IndirectCommandType::DRAW_PATCHES);
        set.unwrap();
        assert!(matches!(
            render_command(&patches, 0).draw,
            Some(IndirectDraw::IndexedPatches { .. })
        ));
        let mut descriptor = render_descriptor();
        descriptor.command_types = IndirectCommandType::DRAW_INDEXED_PATCHES;
        let made = icb(device, &descriptor, 1);
        let command = made.indirect_render_command_at(0).unwrap();
        let indexed = |control_points,
                       start,
                       count,
                       pib: Option<u64>,
                       points: Option<u64>,
                       instances,
                       factor: u64,
                       stride: u64| {
            command.draw_indexed_patches(
                control_points,
                start,
                count,
                pib.map(|_| &buffer),
                pib.unwrap_or(0),
                points.map(|_| &buffer),
                points.unwrap_or(0),
                instances,
                0,
                &buffer,
                factor,
                stride,
            )
        };
        let draw = |problem| Err(invalid(CALL, Problem::Draw(problem)));
        assert_eq!(
            indexed(3, 10, 7, Some(0), Some(0), 1, 0, 0),
            draw(DrawProblem::PatchIndexBufferRange {
                offset: 0,
                indices: 17,
                length: 64
            })
        );
        assert!(render_command(&made, 0).draw.is_none());
        // Taken: a control-point index offset of 1 or 2, indices past the buffer, none at all;
        // and no instances, which runs nothing, and a patch index offset off a multiple of 4, read
        // from the offset rounded down, as plain Metal runs them; and 33 control points, factor
        // offsets of 2 and 1 and strides of 6 and 3.
        for points in [Some(1), Some(2), Some(64), None] {
            indexed(32, 100, 2, None, points, 1, 0, 0).unwrap();
        }
        indexed(33, 0, 1, None, Some(0), 1, 2, 6).unwrap();
        indexed(3, 0, 1, None, Some(0), 1, 1, 0).unwrap();
        indexed(3, 0, 1, None, Some(0), 1, 0, 3).unwrap();
        indexed(3, 0, 1, None, Some(0), 1 << 32, 0, 0).unwrap();
        indexed(3, 0, 1, Some(2), Some(0), 1, 0, 0).unwrap();
    });
}

/// Pipeline states that support indirect command buffers, of the owned fixture
/// `examples/icb_direct_arguments.ll`, whose functions use a direct texture or sampler argument
/// are refused as plain Metal answers them, nil with an `NSError` (an M3
/// Max under macOS 26.7 and an M4 under 27.0.1): `MetalError` naming the stage Metal names (the
/// fragment function before the vertex function, and before the mesh function, which goes before
/// the object function), with Metal's code 3 and description and the M3 Max's domain. A state
/// whose functions only declare one, or use none, is made, and so is every refused one without
/// `supportIndirectCommandBuffers`.
#[test]
fn states_whose_functions_use_a_direct_texture_or_sampler_are_refused_as_metal_refuses_them() {
    on_device(|device| {
        let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/icb_direct_arguments.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let render = |vertex: &str, fragment: &str, support: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let mesh = |object: Option<&str>, mesh: &str, fragment: &str, support: bool| {
            let mut descriptor = MeshPipelineDescriptor::new(function(mesh));
            descriptor.object = object.map(function);
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let compute = |kernel: &str, support: bool| {
            let mut descriptor = ComputePipelineDescriptor::new(function(kernel));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let refused = |call: &'static str, stage: Stage, description: &'static str| {
            Err(Error::MetalError(MetalError::new(
                call,
                Problem::IndirectCommandsDirectArgument { stage },
                "AGXMetalG15X_M1",
                3,
                description,
            )))
        };
        const RENDER: &str = "new_render_pipeline_state";
        const MESH: &str = "new_mesh_render_pipeline_state";
        const COMPUTE: &str = "new_compute_pipeline_state";
        const VERTEX: &str = "Vertex shader cannot be used with indirect command buffers";
        const FRAGMENT: &str = "Fragment shader cannot be used with indirect command buffers";
        const MESH_SHADER: &str = "Mesh shader cannot be used with indirect command buffers";
        const OBJECT: &str = "Object shader cannot be used with indirect command buffers";
        const KERNEL: &str = "Compute function cannot be used with indirect command buffers";
        for (vertex, fragment, answer) in [
            ("plain_vertex", "plain_fragment", Ok(())),
            ("plain_vertex", "unused_arguments_fragment", Ok(())),
            (
                "texture_vertex",
                "plain_fragment",
                refused(RENDER, Stage::Vertex, VERTEX),
            ),
            (
                "plain_vertex",
                "texture_fragment",
                refused(RENDER, Stage::Fragment, FRAGMENT),
            ),
            (
                "plain_vertex",
                "sampler_fragment",
                refused(RENDER, Stage::Fragment, FRAGMENT),
            ),
            (
                "texture_vertex",
                "texture_fragment",
                refused(RENDER, Stage::Fragment, FRAGMENT),
            ),
        ] {
            assert_eq!(
                render(vertex, fragment, true),
                answer,
                "{vertex} {fragment}"
            );
            assert_eq!(
                render(vertex, fragment, false),
                Ok(()),
                "{vertex} {fragment}"
            );
        }
        for (object, mesh_function, fragment, answer) in [
            (None, "plain_mesh", "plain_fragment", Ok(())),
            (Some("plain_object"), "plain_mesh", "plain_fragment", Ok(())),
            (
                None,
                "texture_mesh",
                "plain_fragment",
                refused(MESH, Stage::Mesh, MESH_SHADER),
            ),
            (
                Some("texture_object"),
                "plain_mesh",
                "plain_fragment",
                refused(MESH, Stage::Object, OBJECT),
            ),
            (
                Some("texture_object"),
                "texture_mesh",
                "plain_fragment",
                refused(MESH, Stage::Mesh, MESH_SHADER),
            ),
            (
                Some("texture_object"),
                "texture_mesh",
                "texture_fragment",
                refused(MESH, Stage::Fragment, FRAGMENT),
            ),
        ] {
            let at = format!("{object:?} {mesh_function} {fragment}");
            assert_eq!(mesh(object, mesh_function, fragment, true), answer, "{at}");
            assert_eq!(mesh(object, mesh_function, fragment, false), Ok(()), "{at}");
        }
        assert_eq!(compute("buffer_kernel", true), Ok(()));
        assert_eq!(
            compute("texture_kernel", true),
            refused(COMPUTE, Stage::Kernel, KERNEL)
        );
        assert_eq!(compute("texture_kernel", false), Ok(()));
    });
}

/// `IndirectCommandBuffer::gpu_written`: `false` for a fresh buffer, and after a read use of it,
/// an empty copy into it, an empty reset of it and a copy out of it; `true`, at the call, before
/// any commit, after a write use of it in a compute encoder, a copy of a command into it, and a
/// reset of a command of it, each on a fresh buffer.
#[test]
fn the_gpu_written_flag_is_set_by_a_write_use_or_a_blit_into_the_buffer() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let fresh = || icb(device, &compute_descriptor(), 2);
        let used = |buffer: &IndirectCommandBuffer, usage: ResourceUsage| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.use_resource(buffer.as_resource(), usage).unwrap();
            let written = buffer.gpu_written();
            encoder.end_encoding().unwrap();
            written
        };
        let blitted = |blit: &dyn Fn(&mut BlitCommandEncoder) -> Result<(), Error>,
                       buffer: &IndirectCommandBuffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            blit(&mut encoder).unwrap();
            let written = buffer.gpu_written();
            encoder.end_encoding().unwrap();
            written
        };
        // Nothing the GPU may write through.
        let buffer = fresh();
        let other = fresh();
        assert!(!buffer.gpu_written());
        assert!(!used(&buffer, ResourceUsage::READ));
        assert!(!blitted(
            &|blit| blit.copy_indirect_command_buffer(&other, 0..0, &buffer, 0),
            &buffer
        ));
        assert!(!blitted(
            &|blit| blit.reset_commands_in_buffer(&buffer, 1..1),
            &buffer
        ));
        assert!(!blitted(
            &|blit| blit.copy_indirect_command_buffer(&buffer, 0..1, &other, 0),
            &buffer
        ));
        // Each way it is set, at the call, for good.
        let buffer = fresh();
        assert!(used(&buffer, ResourceUsage::WRITE));
        assert!(buffer.gpu_written());
        let buffer = fresh();
        assert!(used(
            &buffer,
            ResourceUsage::READ.union(ResourceUsage::WRITE)
        ));
        let buffer = fresh();
        assert!(blitted(
            &|blit| blit.copy_indirect_command_buffer(&other, 0..1, &buffer, 1),
            &buffer
        ));
        let buffer = fresh();
        assert!(blitted(
            &|blit| blit.reset_commands_in_buffer(&buffer, 0..2),
            &buffer
        ));
        eprintln!(
            "CBF-MARK {}: the GPU-written flag was set by a write use and blits into the buffer \
             alone",
            device.name()
        );
    });
}

/// Pipeline states that support indirect command buffers are refused or made from their
/// functions' arguments' roles and use alone, not the types the reflection derives: the owned
/// fixture `examples/icb_direct_arguments.ll` with the kernels' output buffer and the fragment
/// functions' texture of types whose reflection Metal's answer has not been measured for (a
/// `quaternion` buffer, a `texture2d<quaternion, sample>`; the translation reads neither name)
/// answers as the measured fixture does
/// (`states_whose_functions_use_a_direct_texture_or_sampler_are_refused_as_metal_refuses_them`):
/// a compute, render and mesh state whose function uses a direct texture or sampler is refused
/// with Metal's error, one that uses none (`buffer_kernel`, `plain_fragment`) is made, and each is
/// made without `supportIndirectCommandBuffers`.
#[test]
fn states_whose_arguments_reflection_is_unmeasured_are_answered_from_their_roles() {
    on_device(|device| {
        let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/icb_direct_arguments.ll");
        let text = std::fs::read_to_string(&ll).unwrap();
        let mut unmeasured = text.clone();
        for (from, to) in [
            (
                r#"!"float4", !"air.arg_name", !"out"}"#,
                r#"!"quaternion", !"air.arg_name", !"out"}"#,
            ),
            (
                r#"!"texture2d<float, sample>", !"air.arg_name", !"source"}"#,
                r#"!"texture2d<quaternion, sample>", !"air.arg_name", !"source"}"#,
            ),
        ] {
            assert_eq!(unmeasured.matches(from).count(), 1, "{from}");
            unmeasured = unmeasured.replace(from, to);
        }
        let library = device
            .new_library_with_air(&llvm_assemble_text(&unmeasured).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let render = |fragment: &str, support: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function("plain_vertex"));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let mesh = |fragment: &str, support: bool| {
            let mut descriptor = MeshPipelineDescriptor::new(function("plain_mesh"));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let compute = |kernel: &str, support: bool| {
            let mut descriptor = ComputePipelineDescriptor::new(function(kernel));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let refused = |call: &'static str, stage: Stage, description: &'static str| {
            Err(Error::MetalError(MetalError::new(
                call,
                Problem::IndirectCommandsDirectArgument { stage },
                "AGXMetalG15X_M1",
                3,
                description,
            )))
        };
        const RENDER: &str = "new_render_pipeline_state";
        const MESH: &str = "new_mesh_render_pipeline_state";
        const COMPUTE: &str = "new_compute_pipeline_state";
        const FRAGMENT: &str = "Fragment shader cannot be used with indirect command buffers";
        const KERNEL: &str = "Compute function cannot be used with indirect command buffers";
        assert_eq!(compute("buffer_kernel", true), Ok(()));
        assert_eq!(
            compute("texture_kernel", true),
            refused(COMPUTE, Stage::Kernel, KERNEL)
        );
        for kernel in ["texture_kernel", "buffer_kernel"] {
            assert_eq!(compute(kernel, false), Ok(()), "{kernel}");
        }
        assert_eq!(render("plain_fragment", true), Ok(()));
        assert_eq!(mesh("plain_fragment", true), Ok(()));
        for fragment in ["texture_fragment", "sampler_fragment"] {
            assert_eq!(
                render(fragment, true),
                refused(RENDER, Stage::Fragment, FRAGMENT),
                "{fragment}"
            );
            assert_eq!(
                mesh(fragment, true),
                refused(MESH, Stage::Fragment, FRAGMENT),
                "{fragment}"
            );
            assert_eq!(render(fragment, false), Ok(()), "{fragment}");
            assert_eq!(mesh(fragment, false), Ok(()), "{fragment}");
        }
        eprintln!(
            "CBF-MARK {}: states whose arguments' reflection is unmeasured were refused and made \
             from their roles as Metal answers the measured fixture",
            device.name()
        );
    });
}

/// Render and mesh pipeline states of the owned fixture `examples/icb_linked_arguments.ll`,
/// whose vertex or mesh function uses a direct texture or sampler argument, refused or made as
/// Metal answers each once it links the pipeline, one row per row Metal answered (on an M4 under
/// macOS 27.0.1, whose probes compile the same bodies from source). With
/// `supportIndirectCommandBuffers`, refused where the value read reaches the position or the point
/// size, a varying the fragment function reads, a store, or a branch, switch or loop trip count
/// those depend on; made where it reaches only a varying no fragment input reads (the fragment
/// function declares it and never reads it, declares no stage input, or the pipeline has none),
/// through a select, a merge or a loop's body alike, and where the sample's result is discarded. An
/// object function's texture is refused whatever the mesh function reads of its payload. Each made
/// without `supportIndirectCommandBuffers`. And a state with rasterization disabled whose vertex
/// function returns a value is refused with Metal's own error with or without indirect command
/// buffers, before their refusal; a `void` one is made without them.
#[test]
fn linked_vertex_and_mesh_functions_are_refused_where_metal_keeps_their_texture() {
    /// Metal's answer to a state's creation.
    #[derive(Clone, Copy, Debug)]
    enum Answer {
        Made,
        /// Refused for indirect command buffers, naming the stage.
        Refused(Stage),
        /// Refused for a value its vertex function returns with rasterization disabled.
        Unrasterized,
    }
    use Answer::{Made, Refused, Unrasterized};
    on_device(|device| {
        let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/icb_linked_arguments.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let render = |vertex: &str, fragment: Option<&str>, raster: bool, support: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = fragment.map(function);
            descriptor.rasterization_enabled = raster;
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let mesh = |object: Option<&str>, mesh: &str, fragment: &str, support: bool| {
            let mut descriptor = MeshPipelineDescriptor::new(function(mesh));
            descriptor.object = object.map(function);
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = support;
            device
                .new_mesh_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        let expected = |call: &'static str, answer: Answer| {
            let (problem, description) = match answer {
                Made => return Ok(()),
                Refused(stage) => (
                    Problem::IndirectCommandsDirectArgument { stage },
                    match stage {
                        Stage::Vertex => {
                            "Vertex shader cannot be used with indirect command buffers"
                        }
                        Stage::Mesh => "Mesh shader cannot be used with indirect command buffers",
                        Stage::Object => {
                            "Object shader cannot be used with indirect command buffers"
                        }
                        _ => "Fragment shader cannot be used with indirect command buffers",
                    },
                ),
                Unrasterized => (
                    Problem::UnrasterizedVertexReturn,
                    "RasterizationEnabled is false but the vertex shader's return type is not void",
                ),
            };
            Err(Error::MetalError(MetalError::new(
                call,
                problem,
                "AGXMetalG15X_M1",
                3,
                description,
            )))
        };
        const RENDER: &str = "new_render_pipeline_state";
        const MESH: &str = "new_mesh_render_pipeline_state";
        let vertex = Refused(Stage::Vertex);
        // (vertex function, fragment function, Metal's answer with indirect command buffers),
        // rasterization enabled.
        for (vertex_function, fragment, answer) in [
            // A varying the fragment function reads, or does not.
            ("varying_vertex", Some("varying_fragment"), vertex),
            ("sampled_varying_vertex", Some("varying_fragment"), vertex),
            ("varying_vertex", Some("declaring_fragment"), Made),
            ("varying_vertex", Some("inputless_fragment"), Made),
            ("varying_vertex", None, Made),
            ("sampled_varying_vertex", Some("declaring_fragment"), Made),
            ("sampled_varying_vertex", None, Made),
            // The position.
            ("depth_vertex", Some("declaring_fragment"), vertex),
            ("depth_vertex", Some("inputless_fragment"), vertex),
            ("depth_vertex", None, vertex),
            ("depth_vertex", Some("varying_fragment"), vertex),
            ("position_only_vertex", Some("depth_fragment"), vertex),
            ("position_only_vertex", Some("inputless_fragment"), vertex),
            ("position_only_vertex", None, vertex),
            // A store; a result discarded; no texture.
            ("store_vertex", Some("declaring_fragment"), vertex),
            ("store_vertex", Some("varying_fragment"), vertex),
            ("store_vertex", None, vertex),
            ("discarded_vertex", Some("varying_fragment"), Made),
            ("discarded_vertex", None, Made),
            ("plain_vertex", Some("varying_fragment"), Made),
            // Control flow.
            ("branch_vertex", Some("varying_fragment"), vertex),
            ("branch_vertex", Some("declaring_fragment"), vertex),
            ("branch_vertex", None, vertex),
            ("guarded_store_vertex", Some("varying_fragment"), vertex),
            ("guarded_store_vertex", Some("declaring_fragment"), vertex),
            ("guarded_store_vertex", None, vertex),
            ("trip_count_vertex", Some("varying_fragment"), vertex),
            ("trip_count_vertex", Some("declaring_fragment"), vertex),
            ("trip_count_vertex", None, vertex),
            ("select_vertex", Some("varying_fragment"), vertex),
            ("select_vertex", Some("declaring_fragment"), Made),
            ("select_vertex", None, Made),
            // The point size; another varying; merges, switches, loop bodies.
            ("point_size_vertex", Some("point_fragment"), vertex),
            ("point_size_vertex", None, vertex),
            ("other_varying_vertex", Some("first_varying_fragment"), Made),
            ("other_varying_vertex", Some("varying_fragment"), Made),
            ("dead_branch_vertex", Some("varying_fragment"), vertex),
            ("dead_branch_vertex", Some("declaring_fragment"), Made),
            ("dead_branch_vertex", None, Made),
            ("merged_vertex", Some("varying_fragment"), vertex),
            ("merged_vertex", Some("declaring_fragment"), Made),
            ("merged_vertex", None, Made),
            ("switch_vertex", Some("varying_fragment"), vertex),
            ("switch_vertex", Some("declaring_fragment"), vertex),
            ("switch_vertex", None, vertex),
            ("loop_body_vertex", Some("varying_fragment"), vertex),
            ("loop_body_vertex", Some("declaring_fragment"), Made),
            ("loop_body_vertex", None, Made),
            ("storing_loop_vertex", Some("varying_fragment"), vertex),
            ("storing_loop_vertex", Some("declaring_fragment"), Made),
            ("storing_loop_vertex", None, Made),
        ] {
            let at = format!("{vertex_function} {fragment:?}");
            assert_eq!(
                render(vertex_function, fragment, true, true),
                expected(RENDER, answer),
                "{at}"
            );
            assert_eq!(
                render(vertex_function, fragment, true, false),
                Ok(()),
                "{at}"
            );
        }
        // (vertex function, Metal's answers with and without indirect command buffers),
        // rasterization disabled, no fragment function.
        for (vertex_function, with, without) in [
            ("depth_vertex", Unrasterized, Unrasterized),
            ("position_only_vertex", Unrasterized, Unrasterized),
            ("plain_vertex", Unrasterized, Unrasterized),
            ("position_value_vertex", Unrasterized, Unrasterized),
            ("void_store_vertex", vertex, Made),
        ] {
            assert_eq!(
                render(vertex_function, None, false, true),
                expected(RENDER, with),
                "{vertex_function}"
            );
            assert_eq!(
                render(vertex_function, None, false, false),
                expected(RENDER, without),
                "{vertex_function}"
            );
        }
        // (object function, mesh function, fragment function, Metal's answer with indirect
        // command buffers).
        for (object, mesh_function, fragment, answer) in [
            (
                None,
                "varying_mesh",
                "mesh_varying_fragment",
                Refused(Stage::Mesh),
            ),
            (None, "varying_mesh", "mesh_declaring_fragment", Made),
            (
                None,
                "depth_mesh",
                "mesh_declaring_fragment",
                Refused(Stage::Mesh),
            ),
            (None, "plain_mesh", "mesh_varying_fragment", Made),
            (
                Some("texture_object"),
                "payload_mesh",
                "mesh_varying_fragment",
                Refused(Stage::Object),
            ),
            (
                Some("texture_object"),
                "payload_mesh",
                "mesh_declaring_fragment",
                Refused(Stage::Object),
            ),
            (
                Some("texture_object"),
                "plain_mesh",
                "mesh_varying_fragment",
                Refused(Stage::Object),
            ),
            (
                Some("plain_object"),
                "payload_mesh",
                "mesh_varying_fragment",
                Made,
            ),
            (
                Some("plain_object"),
                "varying_mesh",
                "mesh_varying_fragment",
                Refused(Stage::Mesh),
            ),
            (
                Some("plain_object"),
                "varying_mesh",
                "mesh_declaring_fragment",
                Made,
            ),
        ] {
            let at = format!("{object:?} {mesh_function} {fragment}");
            assert_eq!(
                mesh(object, mesh_function, fragment, true),
                expected(MESH, answer),
                "{at}"
            );
            assert_eq!(mesh(object, mesh_function, fragment, false), Ok(()), "{at}");
        }
    });
}

/// Render pipeline states of the owned fixture `examples/icb_linked_arguments.ll` whose fragment
/// function uses a direct texture, with `supportIndirectCommandBuffers`, refused or made as Metal
/// answers each once it links the pipeline, one row per row Metal answered (on an M4 under macOS
/// 27.0.1): refused where the texel reaches a discard, a store, a depth, stencil or sample mask
/// output, or a color its attachment writes; made where it reaches only a color the pipeline has no
/// attachment for, writes no channel of, or a dual-source second color without blending. A color
/// under a partial write mask, which Metal keeps channel by channel and the analysis does not, is
/// answered `Error::Unproven` (`IndirectCommandsDirectArgumentUnproven`, `UnmeasuredColorWrite`),
/// whatever Metal answers, and so is a vertex function's texel the fragment function reads into
/// such a color: never Metal's error. Each state is made without `supportIndirectCommandBuffers`,
/// or refused by no rule of Metal's. (The rows with rasterization disabled are the compile layer's
/// test's: the runtime does not take a fragment function there yet.)
#[test]
fn fragment_functions_are_refused_where_metal_keeps_their_texture_and_unproven_uses_are_told_apart()
{
    /// The runtime's answer to a state's creation with indirect command buffers.
    #[derive(Clone, Copy, Debug)]
    enum Answer {
        Made,
        /// Refused for indirect command buffers, naming the stage.
        Refused(Stage),
        /// Unproven, naming the stage and the rule.
        Unproven(Stage, LinkedUseGap),
    }
    use Answer::{Made, Refused, Unproven};
    on_device(|device| {
        let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/icb_linked_arguments.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let rgba8 = || Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let rgb = ColorWriteMask::RED
            .union(ColorWriteMask::GREEN)
            .union(ColorWriteMask::BLUE);
        let gba = ColorWriteMask::GREEN
            .union(ColorWriteMask::BLUE)
            .union(ColorWriteMask::ALPHA);
        // (color 0, color 1, color 0's write mask, a depth attachment, color 0 blending a second
        // source)
        type Setup = (bool, bool, ColorWriteMask, bool, bool);
        let color: Setup = (true, false, ColorWriteMask::ALL, false, false);
        let no_color: Setup = (false, false, ColorWriteMask::ALL, false, false);
        let no_color_depth: Setup = (false, false, ColorWriteMask::ALL, true, false);
        let second_color: Setup = (false, true, ColorWriteMask::ALL, false, false);
        let mask = |m: ColorWriteMask| -> Setup { (true, false, m, false, false) };
        let depth: Setup = (true, false, ColorWriteMask::ALL, true, false);
        let second_source: Setup = (true, false, ColorWriteMask::ALL, false, true);
        let render = |vertex: &str, fragment: &str, setup: Setup, support: bool| {
            let (c0, c1, write_mask, depth, source1) = setup;
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.color_attachments[0] = if c0 { rgba8() } else { None };
            descriptor.color_attachments[1] = if c1 { rgba8() } else { None };
            if let Some(a) = &mut descriptor.color_attachments[0] {
                a.write_mask = write_mask;
                if source1 {
                    a.blending_enabled = true;
                    a.source_rgb_blend_factor = BlendFactor::Source1Color;
                    a.destination_rgb_blend_factor = BlendFactor::OneMinusSource1Color;
                }
            }
            if depth {
                descriptor.depth_format = Some(PixelFormat::Depth32Float);
            }
            descriptor.support_indirect_command_buffers = support;
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop)
        };
        const RENDER: &str = "new_render_pipeline_state";
        let expected = |answer: Answer| match answer {
            Made => Ok(()),
            Refused(stage) => Err(Error::MetalError(MetalError::new(
                RENDER,
                Problem::IndirectCommandsDirectArgument { stage },
                "AGXMetalG15X_M1",
                3,
                match stage {
                    Stage::Vertex => "Vertex shader cannot be used with indirect command buffers",
                    _ => "Fragment shader cannot be used with indirect command buffers",
                },
            ))),
            Unproven(stage, reason) => Err(Error::Unproven(metal2vulkan_runtime::Unproven::new(
                RENDER,
                Problem::IndirectCommandsDirectArgumentUnproven { stage, reason },
            ))),
        };
        let fragment = Refused(Stage::Fragment);
        let partial = |stage| Unproven(stage, LinkedUseGap::UnmeasuredColorWrite);
        // (vertex function, fragment function, the pipeline, the answer with indirect command
        // buffers; Metal's where it is not `Unproven`).
        for (vertex_function, fragment_function, setup, answer) in [
            ("plain_vertex", "texel_fragment", color, fragment),
            ("plain_vertex", "texel_fragment", no_color, Made),
            ("plain_vertex", "texel_fragment", no_color_depth, Made),
            ("plain_vertex", "texel_fragment", second_color, Made),
            (
                "plain_vertex",
                "texel_fragment",
                mask(ColorWriteMask::NONE),
                Made,
            ),
            ("plain_vertex", "discarding_fragment", color, fragment),
            ("plain_vertex", "discarding_fragment", no_color, fragment),
            (
                "plain_vertex",
                "discarding_fragment",
                mask(ColorWriteMask::NONE),
                fragment,
            ),
            ("plain_vertex", "storing_fragment", color, fragment),
            ("plain_vertex", "storing_fragment", no_color, fragment),
            (
                "plain_vertex",
                "storing_fragment",
                mask(ColorWriteMask::NONE),
                fragment,
            ),
            ("plain_vertex", "second_color_fragment", color, Made),
            ("plain_vertex", "second_color_fragment", no_color, Made),
            (
                "plain_vertex",
                "second_color_fragment",
                second_color,
                fragment,
            ),
            // Metal refuses, refuses, makes, refuses, makes, refuses the
            // partial masks' rows.
            (
                "plain_vertex",
                "texel_fragment",
                mask(rgb),
                partial(Stage::Fragment),
            ),
            ("plain_vertex", "alpha_fragment", color, fragment),
            (
                "plain_vertex",
                "alpha_fragment",
                mask(rgb),
                partial(Stage::Fragment),
            ),
            (
                "plain_vertex",
                "alpha_fragment",
                mask(ColorWriteMask::ALPHA),
                partial(Stage::Fragment),
            ),
            (
                "plain_vertex",
                "red_fragment",
                mask(gba),
                partial(Stage::Fragment),
            ),
            (
                "plain_vertex",
                "red_fragment",
                mask(ColorWriteMask::RED),
                partial(Stage::Fragment),
            ),
            ("plain_vertex", "depth_writing_fragment", depth, fragment),
            (
                "plain_vertex",
                "depth_writing_fragment",
                no_color_depth,
                fragment,
            ),
            ("plain_vertex", "mask_fragment", color, fragment),
            ("plain_vertex", "mask_fragment", no_color, fragment),
            ("plain_vertex", "stencil_fragment", color, fragment),
            ("plain_vertex", "dual_source_fragment", color, Made),
            (
                "plain_vertex",
                "dual_source_fragment",
                second_source,
                fragment,
            ),
            ("plain_vertex", "dual_source_fragment", no_color, Made),
            // A vertex function's texel the fragment function reads into its color: Metal's two
            // measured rules composed, and a partial write mask unproven.
            (
                "varying_vertex",
                "varying_fragment",
                color,
                Refused(Stage::Vertex),
            ),
            ("varying_vertex", "varying_fragment", no_color, Made),
            (
                "varying_vertex",
                "varying_fragment",
                mask(ColorWriteMask::NONE),
                Made,
            ),
            (
                "varying_vertex",
                "varying_fragment",
                mask(rgb),
                partial(Stage::Vertex),
            ),
        ] {
            let at = format!("{vertex_function} {fragment_function} {setup:?}");
            assert_eq!(
                render(vertex_function, fragment_function, setup, true),
                expected(answer),
                "{at}"
            );
            let without = render(vertex_function, fragment_function, setup, false);
            assert!(
                !matches!(without, Err(Error::MetalError(_) | Error::Unproven(_))),
                "{at}: {without:?}"
            );
        }
        // Color 0 of a format of fewer channels: a write mask holding none of them writes
        // nothing, made (the consumer's row on an M3 Max under macOS 26.6.2: `R8Unorm` under a
        // mask of alpha alone); one holding them writes the color in part, unproven.
        for (fragment_function, format, write_mask, answer) in [
            (
                "texel_fragment",
                PixelFormat::R8Unorm,
                ColorWriteMask::ALPHA,
                Made,
            ),
            (
                "alpha_fragment",
                PixelFormat::R8Unorm,
                ColorWriteMask::ALPHA,
                Made,
            ),
            (
                "texel_fragment",
                PixelFormat::RG8Unorm,
                ColorWriteMask::BLUE.union(ColorWriteMask::ALPHA),
                Made,
            ),
            (
                "texel_fragment",
                PixelFormat::R8Unorm,
                ColorWriteMask::ALL,
                partial(Stage::Fragment),
            ),
        ] {
            let at = format!("{fragment_function} {format:?} {write_mask:?}");
            for support in [true, false] {
                let mut descriptor = RenderPipelineDescriptor::new(function("plain_vertex"));
                descriptor.fragment = Some(function(fragment_function));
                let mut a = ColorAttachmentDescriptor::new(format);
                a.write_mask = write_mask;
                descriptor.color_attachments[0] = Some(a);
                descriptor.support_indirect_command_buffers = support;
                let made = device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .map(drop);
                match support {
                    true => assert_eq!(made, expected(answer), "{at}"),
                    false => assert!(made.is_ok(), "{at} without: {made:?}"),
                }
            }
        }
    });
}

/// Render pipeline states of the owned fixture `examples/icb_linked_arguments.ll` refused or made
/// as Metal answers them for what the fragment function does once linked, with rasterization on
/// and off (Metal links the fragment function either way), with and without
/// `supportIndirectCommandBuffers`, one row per row Metal answered (on an M4 under macOS 27.0.1): a
/// fragment function writing depth without a depth attachment is refused (before the indirect
/// command buffers' refusal); one whose stage input the linking keeps is written by no vertex
/// output is refused naming it (after that refusal); one reading nothing, or whose read reaches no
/// color attachment, is made; and a fragment function's texture under rasterization disabled is
/// refused for indirect command buffers where color 0 is attached.
#[test]
fn fragment_functions_are_linked_with_or_without_rasterization_as_metal_links_them() {
    /// Metal's answer to a state's creation.
    #[derive(Clone, Debug)]
    enum Answer {
        Made,
        /// Refused for a direct texture with indirect command buffers, naming the stage.
        Refused(Stage),
        /// Refused for the stage inputs no vertex output writes, naming them.
        Unwritten(&'static [&'static str]),
        /// Refused for a depth output without a depth attachment.
        Depth,
    }
    use Answer::{Depth, Made, Refused, Unwritten};
    on_device(|device| {
        let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/public/examples/icb_linked_arguments.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name: &str| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        // (rasterization, color 0, a depth attachment)
        let describe = |vertex: &str,
                        fragment: &str,
                        (raster, color, depth): (bool, bool, bool),
                        support: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function(vertex));
            descriptor.fragment = Some(function(fragment));
            descriptor.rasterization_enabled = raster;
            if color {
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            }
            if depth {
                descriptor.depth_format = Some(PixelFormat::Depth32Float);
            }
            descriptor.support_indirect_command_buffers = support;
            descriptor
        };
        let render = |vertex: &str, fragment: &str, setup: (bool, bool, bool), support: bool| {
            device
                .new_render_pipeline_state(
                    &describe(vertex, fragment, setup, support),
                    &PipelineOptions::default(),
                )
                .map(drop)
        };
        const RENDER: &str = "new_render_pipeline_state";
        let expected = |answer: &Answer| match answer {
            Made => Ok(()),
            Refused(stage) => Err(Error::MetalError(MetalError::new(
                RENDER,
                Problem::IndirectCommandsDirectArgument { stage: *stage },
                "AGXMetalG15X_M1",
                3,
                "Fragment shader cannot be used with indirect command buffers",
            ))),
            Unwritten(inputs) => Err(Error::MetalError(MetalError::with_description(
                RENDER,
                Problem::UnwrittenFragmentInputs {
                    inputs: inputs.iter().map(|i| i.to_string()).collect(),
                },
                "AGXMetalG15X_M1",
                3,
                format!(
                    "Fragment input(s) `{}` mismatching vertex shader output type(s) or not \
                     written by vertex shader",
                    inputs.join(",")
                ),
            ))),
            Depth => Err(Error::MetalError(MetalError::new(
                RENDER,
                Problem::DepthWrittenWithoutAttachment,
                "AGXMetalG15X_M1",
                3,
                "depthAttachmentPixelFormat is not valid and shader writes to depth",
            ))),
        };
        let off_color = (false, true, false);
        let off_none = (false, false, false);
        let on_color = (true, true, false);
        let on_none = (true, false, false);
        let on_depth = (true, true, true);
        let fragment = Refused(Stage::Fragment);
        // (vertex function, fragment function, the pipeline, Metal's answers with and without
        // indirect command buffers)
        for (vertex_function, fragment_function, setup, with, without) in [
            // Rasterization disabled.
            (
                "void_plain_vertex",
                "texel_fragment",
                off_color,
                fragment.clone(),
                Made,
            ),
            ("void_plain_vertex", "texel_fragment", off_none, Made, Made),
            (
                "void_plain_vertex",
                "varying_fragment",
                off_color,
                Unwritten(&["color"]),
                Unwritten(&["color"]),
            ),
            (
                "void_plain_vertex",
                "varying_fragment",
                off_none,
                Made,
                Made,
            ),
            (
                "void_plain_vertex",
                "declaring_fragment",
                off_color,
                Made,
                Made,
            ),
            (
                "void_plain_vertex",
                "inputless_fragment",
                off_color,
                Made,
                Made,
            ),
            (
                "void_plain_vertex",
                "inputless_fragment",
                off_none,
                Made,
                Made,
            ),
            // Rasterization on, a stage input no vertex output writes.
            (
                "position_value_vertex",
                "varying_fragment",
                on_color,
                Unwritten(&["color"]),
                Unwritten(&["color"]),
            ),
            // The input reaches only a color the pipeline has no attachment for, by its value or by
            // a branch whose arms make the color, and is read as zero; a branch on it whose arms
            // make a kept color keeps it.
            (
                "position_value_vertex",
                "varying_fragment",
                on_none,
                Made,
                Made,
            ),
            (
                "position_value_vertex",
                "branching_fragment",
                on_color,
                Unwritten(&["color"]),
                Unwritten(&["color"]),
            ),
            (
                "position_value_vertex",
                "branching_fragment",
                on_none,
                Made,
                Made,
            ),
            // Depth without a depth attachment.
            (
                "plain_vertex",
                "depth_writing_fragment",
                on_color,
                Depth,
                Depth,
            ),
            (
                "void_plain_vertex",
                "depth_writing_fragment",
                off_none,
                Depth,
                Depth,
            ),
            (
                "void_plain_vertex",
                "depth_writing_fragment",
                off_color,
                Depth,
                Depth,
            ),
            (
                "plain_vertex",
                "depth_writing_fragment",
                on_depth,
                fragment.clone(),
                Made,
            ),
        ] {
            let at = format!("{vertex_function} {fragment_function} {setup:?}");
            assert_eq!(
                render(vertex_function, fragment_function, setup, true),
                expected(&with),
                "{at}"
            );
            assert_eq!(
                render(vertex_function, fragment_function, setup, false),
                expected(&without),
                "{at}"
            );
        }
        // The path ran: the made rasterizing states whose input no vertex output writes compile
        // the fragment function with that input absent (read as zero, no interface variable);
        // a vertex function that writes it leaves none absent.
        let color = metal2vulkan::pipeline::VaryingKey::user(b"generated(5colorDv4_f)");
        for (vertex_function, fragment_function, setup, absent) in [
            ("position_value_vertex", "varying_fragment", on_none, true),
            ("position_value_vertex", "branching_fragment", on_none, true),
            ("plain_vertex", "varying_fragment", on_color, false),
            ("plain_vertex", "varying_fragment", on_none, false),
        ] {
            let at = format!("{vertex_function} {fragment_function} {setup:?}");
            assert_eq!(
                render(vertex_function, fragment_function, setup, true),
                Ok(()),
                "{at}"
            );
            let keys = device
                .compile_device()
                .render_pipeline_key(
                    &describe(vertex_function, fragment_function, setup, true),
                    &metal2vulkan::Budget::standard(),
                )
                .unwrap_or_else(|e| panic!("{at}: {e:?}"));
            let inputs = keys.fragment_inputs.expect("a fragment function's inputs");
            eprintln!("PATH {at}: absent {:?}", inputs.absent_varyings);
            assert_eq!(
                inputs.absent_varyings,
                match absent {
                    true => std::collections::BTreeSet::from([color.clone()]),
                    false => std::collections::BTreeSet::new(),
                },
                "{at}"
            );
        }
        // A post-tessellation vertex function's and a mesh function's unwritten
        // input, as a vertex function's: refused with color 0 (the mesh's text naming the mesh
        // shader), made without, the input absent.
        const MESH: &str = "new_mesh_render_pipeline_state";
        for attached in [true, false] {
            let at = format!("color 0 attached: {attached}");
            let mut descriptor = RenderPipelineDescriptor::new(function("post_tess_vertex"));
            descriptor.fragment = Some(function("varying_fragment"));
            let mut meshed = MeshPipelineDescriptor::new(function("position_mesh"));
            meshed.fragment = Some(function("mesh_varying_fragment"));
            if attached {
                descriptor.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
                meshed.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            }
            let tessellated = device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .map(drop);
            let mesh = device
                .new_mesh_render_pipeline_state(&meshed, &PipelineOptions::default())
                .map(drop);
            match attached {
                true => {
                    assert_eq!(tessellated, expected(&Unwritten(&["color"])), "{at}");
                    assert_eq!(
                        mesh,
                        Err(Error::MetalError(MetalError::with_description(
                            MESH,
                            Problem::UnwrittenFragmentInputs {
                                inputs: Box::new(["color".to_string()]),
                            },
                            "AGXMetalG15X_M1",
                            3,
                            "Fragment input(s) `color` mismatching mesh shader output type(s) or \
                             not written by mesh shader"
                                .to_string(),
                        ))),
                        "{at}"
                    );
                }
                false => {
                    assert_eq!(tessellated, Ok(()), "{at}");
                    assert_eq!(mesh, Ok(()), "{at}");
                    let keys = device
                        .compile_device()
                        .render_pipeline_key(&descriptor, &metal2vulkan::Budget::standard())
                        .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                    let inputs = keys.fragment_inputs.expect("a fragment function's inputs");
                    eprintln!("PATH tessellated {at}: absent {:?}", inputs.absent_varyings);
                    assert_eq!(
                        inputs.absent_varyings,
                        std::collections::BTreeSet::from([color.clone()]),
                        "{at}"
                    );
                }
            }
        }
        // A post-tessellation vertex function returning a value, rasterization
        // disabled, is refused as any vertex function is; a mesh state's fragment depth without a
        // depth attachment is refused first, before its indirect command buffers' refusal.
        let metal = |call, problem, description: &str| {
            Err(Error::MetalError(MetalError::with_description(
                call,
                problem,
                "AGXMetalG15X_M1",
                3,
                description.to_string(),
            )))
        };
        for icb in [false, true] {
            let at = format!("indirect command buffers: {icb}");
            let mut descriptor = RenderPipelineDescriptor::new(function("post_tess_vertex"));
            descriptor.rasterization_enabled = false;
            descriptor.color_attachments[0] =
                Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
            descriptor.support_indirect_command_buffers = icb;
            assert_eq!(
                device
                    .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                    .map(drop),
                metal(
                    RENDER,
                    Problem::UnrasterizedVertexReturn,
                    "RasterizationEnabled is false but the vertex shader's return type is not void"
                ),
                "{at}"
            );
            for depth in [None, Some(PixelFormat::Depth32Float)] {
                let mut meshed = MeshPipelineDescriptor::new(function("position_mesh"));
                meshed.fragment = Some(function("depth_writing_fragment"));
                meshed.color_attachments[0] =
                    Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
                meshed.depth_format = depth;
                meshed.support_indirect_command_buffers = icb;
                let made = device
                    .new_mesh_render_pipeline_state(&meshed, &PipelineOptions::default())
                    .map(drop);
                let want = match (depth, icb) {
                    (None, _) => metal(
                        MESH,
                        Problem::DepthWrittenWithoutAttachment,
                        "depthAttachmentPixelFormat is not valid and shader writes to depth",
                    ),
                    (Some(_), true) => metal(
                        MESH,
                        Problem::IndirectCommandsDirectArgument {
                            stage: Stage::Fragment,
                        },
                        "Fragment shader cannot be used with indirect command buffers",
                    ),
                    (Some(_), false) => Ok(()),
                };
                assert_eq!(made, want, "{at}, depth {depth:?}");
            }
        }
    });
}
