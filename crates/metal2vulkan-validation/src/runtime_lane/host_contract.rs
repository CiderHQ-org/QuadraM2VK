//! A program valid under Metal's contract over bytes the CPU and the GPU share, which no fast path
//! may break, on the device the cfg selects, under the validation layer: the shape of a probe on an
//! M3 Max under macOS 27.0.1, where Metal kept the CPU's bytes every time.
//!
//! A `Shared` buffer has a 16x16 `RGBA8Unorm` texture over its first bytes, in rows of 256 bytes
//! (a shadow on every device but MoltenVK). A kernel writes texel (0, 0) through the texture
//! (`invert_texels`). The CPU writes texel (5, 5), which no command touches, while the command
//! buffer runs. Metal keeps the CPU's bytes: the CPU and the GPU touching different bytes of a
//! `Shared` buffer at once is no race. A runtime that copied a shadow back over every texel the
//! commands could write would undo them.
//!
//! - [`a_write_through_a_texture_over_a_shared_buffer_copies_no_texel_back`] checks the copies: no
//!   texel goes back as a copy over such bytes, as the shadow is merged back
//!   (`WorkCounts::shadow_texels_copied_back`).
//! - [`a_cpu_write_beside_a_texture_s_write_while_the_command_buffer_runs_stays`] makes the CPU's
//!   write while the command buffer runs: behind the texture's write, the kernel `scale_and_offset`
//!   runs over a large buffer, repeated until the command buffer takes tens of milliseconds, and the
//!   CPU writes once its batch was submitted. The command buffer's GPU start and end show whether
//!   the write landed inside the GPU's run (`HOST-CONTRACT`).
//!
//! The deallocator's shape (it runs in its release, on the releasing thread) is
//! `freed_memory::a_deallocator_without_a_keep_alive_runs_in_its_release_on_its_thread`, and a
//! release with nothing waiting is the runtime's
//! `tests::freed_memory::a_release_with_nothing_waiting_frees_its_import_before_it_returns`.

use super::degrade_notices::on_device;
use super::kept_variants::kernel_state;
use super::WAIT;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use std::time::{Duration, Instant};

/// The texture: `SIDE` texels square, rows of `ROW` bytes.
const SIDE: u64 = 16;
const ROW: u64 = 256;
/// What every byte of the texture's rows holds before the command buffer.
const BEFORE: u8 = 0x11;
/// The kernel's texel: `invert_texels` of a zero texel.
const WRITTEN: [u8; 4] = [255, 255, 255, 0];
/// The texel the CPU writes, and its bytes.
const CPU_TEXEL: (u64, u64) = (5, 5);
const CPU_BYTES: [u8; 4] = [1, 2, 3, 4];

/// The elements `scale_and_offset` runs over in each repetition.
const ELEMENTS: u64 = 1 << 22;
/// The GPU time the command buffer is grown to, and the repetitions it may take at most.
const TARGET: Duration = Duration::from_millis(60);
const MOST_REPEATS: u64 = 4096;
/// The command buffers tried for a write inside the GPU's run.
const ATTEMPTS: usize = 4;

/// The `Shared` buffer, holding [`BEFORE`] in every byte of the texture's rows, and the texture
/// over it.
fn surface(device: &Device) -> (Buffer, Texture) {
    let buffer = device
        .new_buffer(ROW * SIDE, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: a new buffer's bytes, which no command uses yet.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), BEFORE, (ROW * SIDE) as usize) };
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    (buffer, texture)
}

/// A 1x1 `RGBA8Unorm` texture of a zero texel, which `invert_texels` reads.
fn zero_texel(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    texture
        .replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0)
        .unwrap();
    texture
}

/// Where the CPU's texel lies in the buffer.
fn cpu_at() -> usize {
    let (x, y) = CPU_TEXEL;
    (y * ROW + x * 4) as usize
}

/// Write the CPU's texel into `buffer`.
fn write_cpu_texel(buffer: &Buffer) {
    let contents = buffer.contents().unwrap();
    // SAFETY: a texel's bytes inside the buffer, which no command touches.
    unsafe {
        std::ptr::copy_nonoverlapping(CPU_BYTES.as_ptr(), contents.as_ptr().add(cpu_at()), 4);
    }
}

/// That `buffer` holds the kernel's texel at (0, 0), the CPU's at [`CPU_TEXEL`] and [`BEFORE`] in
/// every other byte of the texture's rows, what Metal leaves.
fn assert_metal_s_bytes(buffer: &Buffer, what: &str) {
    let contents = buffer.contents().unwrap();
    let span = (ROW * SIDE) as usize;
    // SAFETY: the command buffer completed; nothing writes the bytes now.
    let now = unsafe { std::slice::from_raw_parts(contents.as_ptr(), span) };
    let at = cpu_at();
    let mut expected = vec![BEFORE; span];
    expected[..4].copy_from_slice(&WRITTEN);
    expected[at..at + 4].copy_from_slice(&CPU_BYTES);
    let differing: Vec<usize> = (0..span).filter(|&i| now[i] != expected[i]).collect();
    assert!(
        differing.is_empty(),
        "{what}: bytes {:?} differ (texel (5, 5) holds {:?}, the CPU wrote {CPU_BYTES:?}; texel \
         (0, 0) holds {:?}, the kernel wrote {WRITTEN:?})",
        &differing[..differing.len().min(16)],
        &now[at..at + 4],
        &now[..4],
    );
}

/// A command buffer of `queue`'s whose compute encoder writes texel (0, 0) of `target` with
/// `invert`, then runs `scale` over [`ELEMENTS`] `repeats` times between `input` and `output`.
fn encoded(
    queue: &CommandQueue,
    (invert, scale): (&ComputePipelineState, &ComputePipelineState),
    (zero, target): (&Texture, &Texture),
    (input, output): (&Buffer, &Buffer),
    repeats: u64,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(invert).unwrap();
    encoder.set_texture(Some(zero), 0).unwrap();
    encoder.set_texture(Some(target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(1, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.set_compute_pipeline_state(scale).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_bytes(&2.0f32.to_ne_bytes(), 2).unwrap();
    for _ in 0..repeats {
        encoder
            .dispatch_threads(Size::new(ELEMENTS, 1, 1), Size::new(64, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Wait for `command_buffer`, checking it completed.
fn completed(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A kernel's write through a texture over a `Shared` buffer, the CPU's write to another texel
/// made before the commit: no texel goes back as a copy (the shadow is merged back), and the
/// buffer holds both writes.
#[test]
fn a_write_through_a_texture_over_a_shared_buffer_copies_no_texel_back() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let invert = kernel_state(device, "invert_texels");
        let scale = kernel_state(device, "scale_and_offset");
        let (buffer, texture) = surface(device);
        let zero = zero_texel(device);
        let input = device.new_buffer(4, ResourceOptions::private()).unwrap();
        let output = device.new_buffer(4, ResourceOptions::private()).unwrap();
        let before = device.work_counts().shadow_texels_copied_back;
        let command_buffer = encoded(
            &queue,
            (&invert, &scale),
            (&zero, &texture),
            (&input, &output),
            0,
        );
        write_cpu_texel(&buffer);
        command_buffer.commit().unwrap();
        completed(&command_buffer);
        let copied_back = device.work_counts().shadow_texels_copied_back - before;
        assert_eq!(copied_back, 0, "texels copied back over a Shared buffer");
        assert_metal_s_bytes(&buffer, "written before the commit");
    });
}

/// The CPU's write to a texel no command touches, made while the command buffer that writes
/// another texel through the texture runs, stays (the module docs). Each attempt grows the
/// command buffer toward [`TARGET`] of GPU time and writes a third of the way through the last
/// one's; the buffer holds Metal's bytes after every attempt, and the attempts stop at the first
/// write the GPU times place inside the GPU's run (printed).
#[test]
fn a_cpu_write_beside_a_texture_s_write_while_the_command_buffer_runs_stays() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let invert = kernel_state(device, "invert_texels");
        let scale = kernel_state(device, "scale_and_offset");
        let zero = zero_texel(device);
        let bytes = ELEMENTS * 4;
        let input = device
            .new_buffer(bytes, ResourceOptions::private())
            .unwrap();
        let output = device
            .new_buffer(bytes, ResourceOptions::private())
            .unwrap();
        let (mut repeats, mut ran) = (8, Duration::from_millis(5));
        let mut inside = false;
        for attempt in 1..=ATTEMPTS {
            let (buffer, texture) = surface(device);
            let command_buffer = encoded(
                &queue,
                (&invert, &scale),
                (&zero, &texture),
                (&input, &output),
                repeats,
            );
            command_buffer.commit().unwrap();
            let deadline = Instant::now() + WAIT;
            // Scheduled once the CPU finished scheduling it (`kernelEndTime`): submitted, or held
            // for its pass's kernel submission (`CompletionPolicy::Grouped`), which the sleep
            // below outlasts. Its kernel start is stored as scheduling begins, before its batch is
            // replayed and submitted.
            while command_buffer.kernel_end_time() == 0.0 {
                assert!(Instant::now() < deadline, "the batch was never submitted");
                std::thread::yield_now();
            }
            std::thread::sleep(ran / 3);
            let wrote = host_time();
            write_cpu_texel(&buffer);
            completed(&command_buffer);
            assert_metal_s_bytes(&buffer, &format!("attempt {attempt}"));
            let times = (
                command_buffer.gpu_start_time(),
                command_buffer.gpu_end_time(),
            );
            let (Ok(start), Ok(end)) = times else {
                println!(
                    "HOST-CONTRACT on {}: the device times no command buffer; the CPU's texel \
                     stayed",
                    device.name()
                );
                return;
            };
            inside = start < wrote && wrote < end;
            ran = Duration::from_secs_f64((end - start).max(1e-4));
            println!(
                "HOST-CONTRACT on {}: attempt {attempt}, {repeats} repeats, GPU run {:.2} ms, the \
                 CPU's write at {:+.2} ms from its start (inside {inside}); the CPU's texel stayed",
                device.name(),
                (end - start) * 1e3,
                (wrote - start) * 1e3,
            );
            if inside {
                break;
            }
            if ran < TARGET {
                let grow = (TARGET.as_secs_f64() / ran.as_secs_f64()).ceil() as u64;
                repeats = (repeats * grow.max(2)).min(MOST_REPEATS);
            }
        }
        if !inside {
            println!(
                "HOST-CONTRACT on {}: no write landed inside the GPU's run in {ATTEMPTS} attempts",
                device.name()
            );
        }
    });
}
