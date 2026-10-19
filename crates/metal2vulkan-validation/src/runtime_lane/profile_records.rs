//! A later process finds the records an earlier one wrote of every pipeline whose making reads the
//! same, though the two devices' compile profiles differ (the runtime's `shader_cache/driver.rs`,
//! "Not by the profile"), on the device the cfg selects, under the validation layer.
//!
//! A first device over a fresh shader cache directory makes a compute state of an integer kernel
//! no cache holds and dispatches it at two threadgroup sizes, which writes its region record
//! (`pipeline/region_record.rs`) and, where the driver keeps its data, a record of each region's
//! driver data. A second device over the same directory, whose profile differs, makes the same
//! state: its creation makes every key the first one's dispatches made, each from its driver
//! record where the first wrote one, and its dispatch hands no making to the pool and writes what
//! the first one's wrote. The two profiles differ:
//!
//! - **by a policy the kernel does not read**: the second blends a `half` output in the device's
//!   blend (`HalfBlendPolicy::Device`) where the first takes Metal's, which no kernel compiles;
//! - **by checks that settled after the first device's making**: the first device, over a fresh
//!   device-checks directory, leaves its checks of `Settle::Background` to its checks thread,
//!   whose answers its profile does not take (the test waits for them before its kernel's work);
//!   then the device goes, and the second, over the same directory, reads every answer, so its
//!   profile takes them where they differ from the tables'. Whether they differ on this device is
//!   printed (`PROFILE-RECORDS`); an integer kernel's pipelines read none of them.

use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DeviceChecksPolicy, HalfBlendPolicy};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, DeviceChecksOptions, DeviceOptions,
    PipelineOptions, Settled, Size,
};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The threads each dispatch covers.
const THREADS: u64 = 256;

/// The AIR of `pr_kernel(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]])`,
/// which stores `t * 2654435761 + nonce` to `out[t]`: integer work alone.
fn kernel(nonce: u32) -> String {
    format!(
        "define void @pr_kernel(ptr addrspace(1) %out, i32 %t) #0 {{
  %m = mul i32 %t, -1640531535
  %v = add i32 %m, {nonce}
  %i = zext i32 %t to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}}

attributes #0 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!20}}
!air.language_version = !{{!21}}
!0 = !{{ptr @pr_kernel, !10, !11}}
!10 = !{{}}
!11 = !{{!12, !13}}
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}
!13 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"t\"}}
!20 = !{{i32 2, i32 8, i32 0}}
!21 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
"
    )
}

/// The state of `pr_kernel` in `bitcode` on `device`.
fn state(device: &Device, bitcode: &[u8]) -> ComputePipelineState {
    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name("pr_kernel").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `state` on `device` over [`THREADS`] threads at threadgroups of 8 and of 64, waited
/// for; what it wrote.
fn dispatched(device: &Device, state: &ComputePipelineState) -> Vec<u8> {
    let length = THREADS * 4;
    let out = buffer_with(device, &vec![0; length as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    for group in [8, 64] {
        encoder
            .dispatch_threadgroups(Size::new(THREADS / group, 1, 1), Size::new(group, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, length).unwrap()
}

/// Wait until `device`'s checks thread, where it runs, settled its feature status, within a
/// minute.
fn settle(device: &Device) {
    let started = Instant::now();
    while matches!(device.feature_status().settled, Settled::Running { .. }) {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{}: the checks thread did not settle in a minute",
            device.name()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The module docs' check of two devices made with `first` and `second` over one shader cache
/// directory: the profile fingerprints of the two, equal or not.
fn second_finds_the_first_s_records(
    which: &str,
    first: impl Fn(&mut DeviceOptions),
    second: impl Fn(&mut DeviceOptions),
) -> bool {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let bitcode = llvm_assemble_text(&kernel(nonce)).unwrap();
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let (written, made, saved, profile) = {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), &first).unwrap();
        // Its checks thread's work done first, so the counts below are the kernel's alone: the
        // device's profile stays as it was made all the same.
        settle(&device);
        let before = device.work_counts();
        let state = state(&device, &bitcode);
        let written = dispatched(&device, &state);
        let after = device.work_counts();
        let made = after.dispatch_pipelines_pooled - before.dispatch_pipelines_pooled;
        assert!(
            made >= 2,
            "{which}: each threadgroup size's making is pooled"
        );
        let saved = device.shader_cache_statistics().driver_saves;
        let profile = device.compile_device().profile_fingerprint();
        (written, made, saved, profile)
    };
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), &second).unwrap();
    let name = device.name().to_owned();
    let differ = device.compile_device().profile_fingerprint() != profile;
    let before = (device.work_counts(), device.shader_cache_statistics());
    let state = state(&device, &bitcode);
    let created = (device.work_counts(), device.shader_cache_statistics());
    assert_eq!(
        created.0.creation_dispatch_pipelines_made - before.0.creation_dispatch_pipelines_made,
        made,
        "{name}, {which}: the creation makes every key the first device's dispatches made"
    );
    let loaded = created.1.driver_loads - before.1.driver_loads;
    if saved > 0 {
        assert_eq!(
            loaded, made,
            "{name}, {which}: each of the creation's makings loads the first device's record"
        );
    }
    assert_eq!(
        dispatched(&device, &state),
        written,
        "{name}, {which}: the second device's dispatches write the first one's bytes"
    );
    assert_eq!(
        device.work_counts().dispatch_pipelines_pooled,
        created.0.dispatch_pipelines_pooled,
        "{name}, {which}: no dispatch hands a making to the pool"
    );
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "{name}, {which}: the validation layer reported: {reported:#?}"
    );
    println!(
        "PROFILE-RECORDS on {name}, {which}: profiles {}, {made} keys made at the second \
         creation, {loaded} from driver records ({saved} written by the first)",
        if differ { "differ" } else { "equal" }
    );
    differ
}

/// The first profile difference of the module docs: a policy the kernel does not read.
#[test]
fn a_device_of_another_policy_finds_the_records_of_pipelines_it_makes_alike() {
    let differ = second_finds_the_first_s_records(
        "half blend policy",
        |_| {},
        |options| options.policy.half_blend = HalfBlendPolicy::Device,
    );
    assert!(differ, "the policies' profiles differ");
}

/// The second profile difference of the module docs: checks that settled after the first
/// device's making.
#[test]
fn a_device_after_one_whose_checks_settled_finds_its_records() {
    let checks = ScratchCache::new().unwrap();
    let trusted = |directory: &Path| {
        let directory = directory.to_owned();
        move |options: &mut DeviceOptions| {
            let mut kept = DeviceChecksOptions::default();
            kept.directory = Some(directory.clone());
            options.device_checks = kept;
            options.policy.device_checks = DeviceChecksPolicy::Trusted;
        }
    };
    second_finds_the_first_s_records(
        "checks settled",
        trusted(checks.path()),
        trusted(checks.path()),
    );
}
