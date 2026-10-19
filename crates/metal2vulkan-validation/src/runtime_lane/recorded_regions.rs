//! A compute state makes, at its creation, the `VkPipeline`s its kernel's dispatches made in an
//! earlier process (`pipeline/region_record.rs`), so its first dispatches wait for no making, as
//! Metal's take the compile `newComputePipelineState` did; on the device the cfg selects, under
//! the validation layer.
//!
//! A first device over a fresh shader cache directory makes a state of a kernel no cache holds
//! and dispatches it at two threadgroup sizes and one grid that is not whole threadgroups: its
//! creation makes nothing, and its dispatches hand each region's making to the pool. A second
//! device over a fresh directory that holds only the first one's region records (none of the
//! driver's data, so its makings compile cold) makes the same state, which makes those keys at
//! its creation. Then a command buffer dispatches it as the first did and signals an event, and a
//! command buffer on another queue, waiting for the event, copies what it wrote: no dispatch
//! hands a making to the pool, each recorded key's first dispatch takes the creation's
//! `VkPipeline`, and the waiter copies the bytes the first device's dispatches wrote. The
//! signaller's batch waits for no making at its commit, so the waiter behind it goes in flight
//! once the signal is submitted: the case where a first run's waiter was submitted ahead of its
//! signaller's making.

use super::first_dispatch_makings::kernel;
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    EventRef, PipelineOptions, Size,
};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The threads each dispatch covers.
const THREADS: u64 = 256;

/// The magic a region record starts with (`region_record.rs`).
const RECORD_MAGIC: &[u8; 8] = b"M2VRGN1\0";

/// The state of `fdm_kernel` in `bitcode` on `device`.
fn state(device: &Device, bitcode: &[u8]) -> ComputePipelineState {
    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name("fdm_kernel").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A command buffer on `queue` dispatching `state` over [`THREADS`] threads at threadgroups of 8,
/// at threadgroups of 64, and as 100 threads in threadgroups of 64 (a whole region and an edge),
/// writing `out`.
fn dispatching(queue: &CommandQueue, state: &ComputePipelineState, out: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    for group in [8, 64] {
        encoder
            .dispatch_threadgroups(Size::new(THREADS / group, 1, 1), Size::new(group, 1, 1))
            .unwrap();
    }
    encoder
        .dispatch_threads(Size::new(100, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Copy each region record under `from` to the same place under `to` (the module docs).
fn copy_records(from: &Path, to: &Path) -> usize {
    let mut copied = 0;
    let mut dirs = vec![from.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            if bytes.starts_with(RECORD_MAGIC) {
                let place = to.join(path.strip_prefix(from).unwrap());
                std::fs::create_dir_all(place.parent().unwrap()).unwrap();
                std::fs::write(place, bytes).unwrap();
                copied += 1;
            }
        }
    }
    copied
}

/// The module docs' check.
#[test]
fn a_state_makes_the_regions_its_kernel_ran_before_at_its_creation() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64;
    let bitcode = llvm_assemble_text(&kernel(nonce)).unwrap();
    let length = THREADS * 4;
    let (first, cold) = (ScratchCache::new().unwrap(), ScratchCache::new().unwrap());
    let messages = Messages::default();
    let (written, pooled) = {
        let device = validated_device(TEST_DEVICE, &messages, first.path()).unwrap();
        let before = device.work_counts();
        let state = state(&device, &bitcode);
        let created = device.work_counts();
        assert_eq!(
            created.creation_dispatch_pipelines_made, before.creation_dispatch_pipelines_made,
            "a first creation makes no recorded key"
        );
        let queue = device.new_command_queue().unwrap();
        let out = buffer_with(&device, &vec![0; length as usize]).unwrap();
        let command_buffer = dispatching(&queue, &state, &out);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let after = device.work_counts();
        let pooled = after.dispatch_pipelines_pooled - created.dispatch_pipelines_pooled;
        assert!(
            pooled >= 3,
            "each threadgroup size's making is pooled: {pooled}"
        );
        (read_buffer(&out, 0, length).unwrap(), pooled)
    };
    assert!(
        copy_records(first.path(), cold.path()) >= 1,
        "no region record"
    );
    {
        let device = validated_device(TEST_DEVICE, &messages, cold.path()).unwrap();
        let name = device.name().to_owned();
        let before = device.work_counts();
        let started = Instant::now();
        let state = state(&device, &bitcode);
        let creation = started.elapsed();
        let created = device.work_counts();
        let made =
            created.creation_dispatch_pipelines_made - before.creation_dispatch_pipelines_made;
        assert_eq!(
            made, pooled,
            "{name}: the creation makes every recorded key"
        );
        let (signalling, waiting) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let event = device.new_event().unwrap();
        let out = buffer_with(&device, &vec![0; length as usize]).unwrap();
        let copy = buffer_with(&device, &vec![0; length as usize]).unwrap();
        let signaller = dispatching(&signalling, &state, &out);
        signaller
            .encode_signal_event(EventRef::from(&event), 1)
            .unwrap();
        let waiter = waiting.command_buffer().unwrap();
        waiter
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = waiter.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&out, 0, &copy, 0, length)
            .unwrap();
        blit.end_encoding().unwrap();
        signaller.commit().unwrap();
        waiter.commit().unwrap();
        waiter.wait_until_completed_for(WAIT).unwrap();
        signaller.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(waiter.status(), CommandBufferStatus::Completed);
        let after = device.work_counts();
        assert_eq!(
            after.dispatch_pipelines_pooled, created.dispatch_pipelines_pooled,
            "{name}: a dispatch of a recorded key hands no making to the pool"
        );
        assert_eq!(
            after.creation_dispatch_pipelines_dispatched
                - created.creation_dispatch_pipelines_dispatched,
            made,
            "{name}: each recorded key's first dispatch takes the creation's pipeline"
        );
        assert_eq!(
            read_buffer(&copy, 0, length).unwrap(),
            written,
            "{name}: the waiter copies what the first device's dispatches wrote"
        );
        println!(
            "RECORDED-REGIONS on {name}: {made} recorded keys made at a cold creation in {:.2} ms",
            creation.as_secs_f64() * 1e3
        );
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
