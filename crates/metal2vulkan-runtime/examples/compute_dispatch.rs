//! A compute dispatch from a metallib, as a Metal program makes one: the system default device, a
//! library from bytes, a compute pipeline state of one of its functions, three buffers, one
//! command buffer, and the result read back on the CPU.
//!
//! ```sh
//! cargo run -p metal2vulkan-runtime --example compute_dispatch [-- path/to/library.metallib]
//! ```
//!
//! Without an argument it runs `scale_and_offset` of the repository's owned fixture
//! (`validation/fixtures/public/examples/runtime_examples.metal`), assembled from its committed
//! LLVM text with `llvm-as` from `PATH` and parsed by the translator; with one, the metallib it
//! names, which must define the same kernel. `new_library_with_data` takes a metallib alone, as
//! Metal does: AIR bitcode goes through the translator (`common::library`).

mod common;

use metal2vulkan_runtime::{Device, PipelineOptions, ResourceOptions, Size};

/// How many floats the kernel scales.
const COUNT: usize = 1000;

fn main() -> common::Result<()> {
    // `MTLCreateSystemDefaultDevice()`: the preferred Vulkan physical device (discrete, then
    // integrated, virtual, CPU) that the runtime can open, with the default options.
    let device = Device::system_default()?;
    println!("device: {}", device.name());

    // `newLibraryWithData:error:`, `newFunctionWithName:`, and
    // `newComputePipelineStateWithFunction:error:`. The translation to SPIR-V happens here, under
    // the 20-second budget; a refusal is a typed error naming what the device lacks.
    let library = common::library(&device)?;
    let function = library.new_function_with_name("scale_and_offset")?;
    let state =
        device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;
    println!(
        "pipeline: {} threads per threadgroup at most, SIMD width {}",
        state.max_total_threads_per_threadgroup(),
        state.thread_execution_width()
    );

    // `newBufferWithBytes:length:options:` and `newBufferWithLength:options:`, in `Shared`
    // storage, which the CPU reads and writes through `contents`.
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device.new_buffer_with_bytes(&input, ResourceOptions::shared())?;
    let output = device.new_buffer((COUNT * 4) as u64, ResourceOptions::shared())?;
    let scale = device.new_buffer_with_bytes(&2.0f32.to_le_bytes(), ResourceOptions::shared())?;

    // One command buffer with one compute encoder. Hazards between commands are tracked as Metal
    // tracks them: no barrier is written by hand.
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut encoder = command_buffer.compute_command_encoder()?;
    encoder.set_compute_pipeline_state(&state)?;
    encoder.set_buffer(Some(&input), 0, 0)?;
    encoder.set_buffer(Some(&output), 0, 1)?;
    encoder.set_buffer(Some(&scale), 0, 2)?;
    // `dispatchThreads:threadsPerThreadgroup:`: a grid of COUNT threads, the last threadgroup
    // partial, as Metal runs it.
    encoder.dispatch_threads(Size::new(COUNT as u64, 1, 1), Size::new(64, 1, 1))?;
    encoder.end_encoding()?;
    common::run(&command_buffer)?;

    // output[i] = input[i] * 2 + i = 3i.
    let output: Vec<f32> = common::bytes(&output)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    for (i, &value) in output.iter().enumerate() {
        if value != 3.0 * i as f32 {
            return Err(format!("output[{i}] is {value}, not {}", 3 * i).into());
        }
    }
    println!(
        "output[0..4] = {:?}, ..., output[{}] = {}: as expected",
        &output[..4],
        COUNT - 1,
        output[COUNT - 1]
    );
    Ok(())
}
