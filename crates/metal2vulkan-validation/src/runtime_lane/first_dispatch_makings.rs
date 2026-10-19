//! What a compute state's first dispatches wait for on the device the cfg selects, without the
//! validation layer: report only, in release builds, before and after a change to where a
//! dispatch's `VkPipeline` is made.
//!
//! Metal compiles a compute state inside `newComputePipelineState` and nowhere else: on an M3 Max
//! (macOS 27.0.1) a fresh 400-operation kernel's creation took 82 ms, and its first dispatches at
//! nine threadgroup sizes (non-uniform edge threadgroups among them) each went from commit to
//! scheduled in 0.04 to 0.09 ms, as a repeat did. A Vulkan pipeline bakes its local size, so the
//! runtime makes one per threadgroup size and edge region.
//!
//! Each round makes [`STATES`] compute states of kernels no cache holds (a fresh constant each,
//! [`kernel`]), waits a lead, then commits one command buffer dispatching each state at
//! [`SIZES`]'s threadgroup sizes, and right behind it on the same queue a command buffer of a
//! state made and dispatched before (the bystander). Printed as `FDM` lines: the creations' total
//! and slowest time, the dispatching command buffer's time from commit to scheduled and to
//! completed, the bystander's to completed, the `VkPipeline`s made (those the states' creations
//! made from their records, and of those the ones a dispatch took) and those dispatches handed to
//! the pool, and the slowest and total making times its execution record states. Each round
//! runs on a device over a fresh shader cache directory (`cold`), then on a second device over the
//! same directory and the same kernels (`warm`: the records the first device wrote). A timing run
//! names a fresh directory for the driver's own disk cache too, so `cold` is cold there.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::{buffer_with, loader, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::open_device;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBuffer, ComputePipelineState, Device, DeviceOptions, PipelineOptions,
    ShaderCacheOptions, Size, WorkCounts,
};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The states a round makes.
const STATES: usize = 24;
/// The threadgroup sizes each state's dispatches take, in order.
const SIZES: [[u64; 3]; 2] = [[8, 8, 1], [64, 1, 1]];
/// The threads a dispatch covers, a multiple of every size's threads.
const GRID: [u64; 3] = [256, 64, 1];
/// The operations of a kernel's chain.
const OPERATIONS: usize = 600;

/// A device over the shader cache in `cache`, with the runtime's defaults, recording executions,
/// without the validation layer.
fn timing_device(cache: &Path) -> Device {
    let mut options = DeviceOptions::default();
    options.shader_cache = ShaderCacheOptions::in_directory(cache);
    options.loader = loader();
    options.record_executions = true;
    open_device(TEST_DEVICE, options).unwrap()
}

/// LLVM's spelling of a `float` constant: the double of the float's value, in hex.
fn float(x: f32) -> String {
    format!("0x{:016X}", f64::from(x).to_bits())
}

/// The AIR of `fdm_kernel(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]])`:
/// a chain of [`OPERATIONS`] fused multiply-adds, sines and reciprocal square roots of `t` whose
/// constants carry `nonce`, stored to `out[t]`.
pub(super) fn kernel(nonce: u64) -> String {
    let mut s = String::from(
        "define void @fdm_kernel(ptr addrspace(1) %out, i32 %t) #0 {
  %a0 = tail call fast float @air.convert.f.f32.u.i32(i32 %t)
",
    );
    for i in 0..OPERATIONS {
        let c = 1.0 + ((nonce.wrapping_add(i as u64 * 2_654_435_761)) % 100_000) as f32 * 1e-7;
        let (a, b) = (format!("%a{i}"), format!("%a{}", i + 1));
        match i % 3 {
            0 => writeln!(
                s,
                "  {b} = tail call fast float @air.fma.f32(float {a}, float {}, float {})",
                float(c),
                float(c * 0.5)
            ),
            1 => writeln!(
                s,
                "  %s{i} = tail call fast float @air.fast_sin.f32(float {a})
  {b} = fadd fast float %s{i}, {a}"
            ),
            _ => writeln!(
                s,
                "  %q{i} = fmul fast float {a}, {a}
  %r{i} = fadd fast float %q{i}, {}
  %v{i} = tail call fast float @air.fast_rsqrt.f32(float %r{i})
  {b} = fmul fast float {a}, %v{i}",
                float(c)
            ),
        }
        .unwrap();
    }
    write!(
        s,
        "  %i = zext i32 %t to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %a{OPERATIONS}, ptr addrspace(1) %p, align 4
  ret void
}}

declare float @air.convert.f.f32.u.i32(i32)
declare float @air.fma.f32(float, float, float)
declare float @air.fast_sin.f32(float)
declare float @air.fast_rsqrt.f32(float)

attributes #0 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @fdm_kernel, !10, !11}}
!10 = !{{}}
!11 = !{{!12, !13}}
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}
!13 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
"
    )
    .unwrap();
    s
}

/// The compute state of [`kernel`]`(nonce)` on `device`, and the time its creation took (its
/// library's and function's making not counted).
fn state(device: &Device, bitcode: &[u8]) -> (ComputePipelineState, Duration) {
    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name("fdm_kernel").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let started = Instant::now();
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    (state, started.elapsed())
}

/// A command buffer on `queue` dispatching each of `states` at every size of [`SIZES`] over
/// [`GRID`], writing `out`.
fn dispatching(
    queue: &metal2vulkan_runtime::CommandQueue,
    states: &[ComputePipelineState],
    out: &metal2vulkan_runtime::Buffer,
) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    for state in states {
        encoder.set_compute_pipeline_state(state).unwrap();
        for [x, y, z] in SIZES {
            encoder
                .dispatch_threadgroups(
                    Size::new(GRID[0] / x, GRID[1] / y, GRID[2] / z),
                    Size::new(x, y, z),
                )
                .unwrap();
        }
    }
    encoder.end_encoding().unwrap();
    command_buffer
}

/// When a command buffer reached each point, from its commit.
#[derive(Default)]
struct Points {
    scheduled: Option<Instant>,
    completed: Option<Instant>,
}

/// Note `command_buffer`'s scheduled and completed instants in `points`.
fn noted(command_buffer: &CommandBuffer) -> Arc<Mutex<Points>> {
    let points = Arc::new(Mutex::new(Points::default()));
    let at = Arc::clone(&points);
    command_buffer
        .add_scheduled_handler(Box::new(move |_| {
            at.lock().unwrap().scheduled = Some(Instant::now());
        }))
        .unwrap();
    let at = Arc::clone(&points);
    command_buffer
        .add_completed_handler(Box::new(move |_| {
            at.lock().unwrap().completed = Some(Instant::now());
        }))
        .unwrap();
    points
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// One round on `device` (the module docs): `nonces`' kernels, dispatched `lead` after their
/// states' creations.
fn round(device: &Device, label: &str, bitcodes: &[Vec<u8>], lead: Duration) {
    let queue = device.new_command_queue().unwrap();
    let out = buffer_with(device, &vec![0; (GRID[0] * GRID[1] * 4) as usize]).unwrap();
    // The bystander: a state made and dispatched before the round.
    let bystander = state(device, &llvm_assemble_text(&kernel(1)).unwrap()).0;
    let warm = dispatching(&queue, std::slice::from_ref(&bystander), &out);
    warm.commit().unwrap();
    warm.wait_until_completed_for(WAIT).unwrap();
    let before: WorkCounts = device.work_counts();
    let mut created = Vec::with_capacity(bitcodes.len());
    let mut times = Vec::with_capacity(bitcodes.len());
    for bitcode in bitcodes {
        let (state, took) = state(device, bitcode);
        created.push(state);
        times.push(took);
    }
    std::thread::sleep(lead);
    let first = dispatching(&queue, &created, &out);
    let behind = dispatching(&queue, std::slice::from_ref(&bystander), &out);
    let (first_points, behind_points) = (noted(&first), noted(&behind));
    let committed = Instant::now();
    first.commit().unwrap();
    behind.commit().unwrap();
    behind.wait_until_completed_for(WAIT).unwrap();
    first.wait_until_completed_for(WAIT).unwrap();
    // The handlers ran before the waits returned or soon after.
    let until = Instant::now() + WAIT;
    while behind_points.lock().unwrap().completed.is_none()
        || first_points.lock().unwrap().completed.is_none()
    {
        assert!(Instant::now() < until, "the completed handlers never ran");
        std::thread::sleep(Duration::from_micros(100));
    }
    let after = device.work_counts();
    let makings: Vec<Duration> = first
        .executions()
        .iter()
        .flat_map(|execution| execution.pipeline_creation().to_vec())
        .collect();
    let since = |at: Option<Instant>| at.map_or(f64::NAN, |at| ms(at - committed));
    let (first_points, behind_points) =
        (first_points.lock().unwrap(), behind_points.lock().unwrap());
    eprintln!(
        "FDM {label} lead {:.0} ms on {}: {} creations total {:.2} ms, slowest {:.2} ms; \
         first CB commit->scheduled {:.2} ms, ->completed {:.2} ms; bystander ->completed {:.2} ms; \
         pipelines made {} ({} at creation, {} of them dispatched), dispatches pooled {}; makings \
         in the record {} (slowest {:.2} ms, total {:.2} ms)",
        ms(lead),
        device.name(),
        times.len(),
        ms(times.iter().sum()),
        ms(times.iter().copied().max().unwrap_or_default()),
        since(first_points.scheduled),
        since(first_points.completed),
        since(behind_points.completed),
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.creation_dispatch_pipelines_made - before.creation_dispatch_pipelines_made,
        after.creation_dispatch_pipelines_dispatched
            - before.creation_dispatch_pipelines_dispatched,
        after.dispatch_pipelines_pooled - before.dispatch_pipelines_pooled,
        makings.len(),
        ms(makings.iter().copied().max().unwrap_or_default()),
        ms(makings.iter().sum()),
    );
}

/// The rounds (the module docs), at a lead of 300 ms and one of 20 ms.
#[test]
#[ignore = "a measurement: run by name with --run-ignored all"]
fn first_dispatches_wait_for_their_makings() {
    let base = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    for (at, lead) in [300u64, 20].into_iter().enumerate() {
        let bitcodes: Vec<Vec<u8>> = (0..STATES as u64)
            .map(|i| llvm_assemble_text(&kernel(base + 1000 * at as u64 + 7 * i)).unwrap())
            .collect();
        let lead = Duration::from_millis(lead);
        let cache = ScratchCache::new().unwrap();
        {
            let device = timing_device(cache.path());
            round(&device, "cold", &bitcodes, lead);
        }
        {
            let device = timing_device(cache.path());
            round(&device, "warm", &bitcodes, lead);
        }
        let _ = std::fs::remove_dir_all(cache.path());
    }
}
