//! An indirect command buffer: two compute commands encoded on the CPU, each its own dispatch over
//! its own half of the arrays with its own scale, read back as typed records, and run by a compute
//! encoder's `executeCommandsInBuffer:withRange:`.
//!
//! ```sh
//! cargo run -p metal2vulkan-runtime --example indirect_command_buffer [-- path/to/library.metallib]
//! ```
//!
//! The kernel is `scale_and_offset` of the repository's owned fixture
//! (`validation/fixtures/public/examples/runtime_examples.metal`), or of the metallib the argument
//! names.

mod common;

use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    Device, IndirectCommand, IndirectCommandBufferDescriptor, PipelineOptions, ResourceOptions,
    ResourceRef, Size,
};

/// The floats each command scales.
const HALF: u64 = 256;

/// Each command's scale.
const SCALES: [f32; 2] = [2.0, -1.0];

fn main() -> common::Result<()> {
    let device = Device::system_default()?;
    println!("device: {}", device.name());
    let library = common::library(&device)?;
    let function = library.new_function_with_name("scale_and_offset")?;

    // A pipeline state an indirect command may name is made with
    // `supportIndirectCommandBuffers`, as Metal requires.
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    let state = device.new_compute_pipeline_state(&descriptor, &PipelineOptions::default())?;

    let input: Vec<u8> = (0..2 * HALF)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let input = device.new_buffer_with_bytes(&input, ResourceOptions::shared())?;
    let output = device.new_buffer(2 * HALF * 4, ResourceOptions::shared())?;
    let scales = [
        device.new_buffer_with_bytes(&SCALES[0].to_le_bytes(), ResourceOptions::shared())?,
        device.new_buffer_with_bytes(&SCALES[1].to_le_bytes(), ResourceOptions::shared())?,
    ];

    // `MTLIndirectCommandBufferDescriptor` and
    // `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`: two commands of
    // `dispatchThreads`, each with its own pipeline state and three kernel buffers.
    let mut icb_descriptor = IndirectCommandBufferDescriptor::new();
    icb_descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    icb_descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device.new_indirect_command_buffer(&icb_descriptor, 2, ResourceOptions::shared())?;

    // `indirectComputeCommandAtIndex:` and its setters: command h scales half h.
    for (h, scale) in scales.iter().enumerate() {
        let offset = h as u64 * HALF * 4;
        let command = icb.indirect_compute_command_at(h as u64)?;
        command.set_compute_pipeline_state(&state)?;
        command.set_kernel_buffer(Some(&input), offset, 0)?;
        command.set_kernel_buffer(Some(&output), offset, 1)?;
        command.set_kernel_buffer(Some(scale), 0, 2)?;
        command.concurrent_dispatch_threads(Size::new(HALF, 1, 1), Size::new(64, 1, 1))?;
    }

    // The typed read-back of the records (a runtime extension standing for Metal's private
    // getters): each command's dispatch, and each buffer as its Metal address plus its offset.
    for (h, command) in icb.commands(0..2)?.iter().enumerate() {
        if let IndirectCommand::Compute(command) = command {
            println!(
                "command {h}: {:?}, kernel buffers {:?}",
                command.dispatch, command.kernel_buffers
            );
        }
    }

    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    // The commands reach their buffers through the addresses their records hold, so the encoder
    // makes the buffers resident first, as Metal's `useResource:usage:` does; an address in no
    // resident resource reads zero and drops its writes.
    encoder.use_resource(ResourceRef::Buffer(&input), ResourceUsage::READ)?;
    encoder.use_resource(ResourceRef::Buffer(&output), ResourceUsage::WRITE)?;
    for scale in &scales {
        encoder.use_resource(ResourceRef::Buffer(scale), ResourceUsage::READ)?;
    }
    encoder.execute_commands_in_buffer(&icb, 0..2)?;
    encoder.end_encoding()?;
    common::run(&command_buffer)?;

    // Half h: output[h * HALF + i] = input[h * HALF + i] * SCALES[h] + i, its thread position i.
    let output: Vec<f32> = common::bytes(&output)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    for (at, &value) in output.iter().enumerate() {
        let (h, i) = (at / HALF as usize, at % HALF as usize);
        let expected = at as f32 * SCALES[h] + i as f32;
        if value != expected {
            return Err(format!("output[{at}] is {value}, not {expected}").into());
        }
    }
    println!(
        "output[0..3] = {:?}, output[{HALF}..{}] = {:?}: as expected",
        &output[..3],
        HALF + 3,
        &output[HALF as usize..HALF as usize + 3]
    );
    Ok(())
}
