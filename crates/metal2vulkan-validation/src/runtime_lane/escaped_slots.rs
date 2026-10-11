//! A buffer slot whose address the kernel stores and that it then writes only through a pointer
//! it loads back, on the device the cfg selects, under the validation layer.
//!
//! [`ESCAPED`]'s `escaped_write` stores the address of `out[t]` into `table[t]`, loads it back
//! and stores `k << 24 | t` through it, `k` read from `params`; `escaped_copy` copies `in[t]` to
//! `res[t]`. One serial compute encoder dispatches, for each `k` from 1 to [`PAIRS`], the write
//! with its own table and parameters, then the copy of `out` into the `k`th window of `res`, over
//! [`THREADS`] threads each. Metal runs a serial encoder's dispatches in order, so each window
//! reads `k << 24 | t`. Only the write dispatch's reported access of `out` orders the copy after
//! it, and the next write after the copy: the dispatches share no other memory, so a slot that
//! reported no access left them without a barrier. Through a device address the accesses are
//! outside the validation layer's synchronization checks, so only the values show it.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::error::Feature;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, Error, PipelineOptions, Size,
};
use std::time::Duration;

/// The longest the command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Threads per dispatch, and words of `out`.
const THREADS: u64 = 1 << 18;

/// Write and copy dispatches, one pair per `k`.
const PAIRS: u32 = 16;

/// The two kernels, synthetic AIR written for this test.
const ESCAPED: &str = r#"define void @escaped_write(ptr addrspace(1) noundef %out, ptr addrspace(1) noundef %table, ptr addrspace(1) noundef %params, i32 noundef %t) {
entry:
  %i = zext i32 %t to i64
  %entry_at = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %table, i64 %i
  %mine = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store ptr addrspace(1) %mine, ptr addrspace(1) %entry_at, align 8
  %p = load ptr addrspace(1), ptr addrspace(1) %entry_at, align 8
  %k = load i32, ptr addrspace(1) %params, align 4
  %high = shl i32 %k, 24
  %v = or i32 %high, %t
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define void @escaped_copy(ptr addrspace(1) noundef %in, ptr addrspace(1) noundef %res, i32 noundef %t) {
entry:
  %i = zext i32 %t to i64
  %from = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %i
  %v = load i32, ptr addrspace(1) %from, align 4
  %to = getelementptr inbounds i32, ptr addrspace(1) %res, i64 %i
  store i32 %v, ptr addrspace(1) %to, align 4
  ret void
}

!air.kernel = !{!0, !10}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @escaped_write, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint*", !"air.arg_name", !"table"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"params"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!10 = !{ptr @escaped_copy, !1, !11}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"res"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The state of `name` in [`ESCAPED`] on `device`, or the device's typed refusal of a pointer
/// loaded from memory (a device without device addresses).
fn state(device: &Device, name: &str) -> Result<ComputePipelineState, Error> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(ESCAPED).unwrap())
        .unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// Each copy reads what the write before it stored through the loaded pointer, in one encoder of
/// alternating dispatches that share only `out`.
#[test]
fn a_slot_written_through_a_loaded_pointer_is_ordered_with_its_readers() {
    on_device(|device, _| {
        let write = match state(device, "escaped_write") {
            Ok(state) => state,
            Err(Error::Unsupported(u)) if u.feature == Feature::DereferenceLoadedPointer => {
                println!(
                    "NO-DEVICE-ADDRESSES {}: the loaded pointer is refused, typed: {u:?}",
                    device.name()
                );
                return;
            }
            Err(error) => panic!("escaped_write: {error}"),
        };
        let copy = state(device, "escaped_copy").unwrap();
        let out = buffer_with(device, &vec![0xab; THREADS as usize * 4]).unwrap();
        let res = buffer_with(device, &vec![0xab; THREADS as usize * 4 * PAIRS as usize]).unwrap();
        let tables: Vec<_> = (0..PAIRS)
            .map(|_| buffer_with(device, &vec![0; THREADS as usize * 8]).unwrap())
            .collect();
        let params: Vec<_> = (1..=PAIRS)
            .map(|k| buffer_with(device, &k.to_le_bytes()).unwrap())
            .collect();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let grid = Size::new(THREADS, 1, 1);
        let group = Size::new(64, 1, 1);
        for (k, (table, param)) in tables.iter().zip(&params).enumerate() {
            encoder.set_compute_pipeline_state(&write).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
            encoder.set_buffer(Some(table), 0, 1).unwrap();
            encoder.set_buffer(Some(param), 0, 2).unwrap();
            encoder.dispatch_threads(grid, group).unwrap();
            encoder.set_compute_pipeline_state(&copy).unwrap();
            encoder.set_buffer(Some(&out), 0, 0).unwrap();
            encoder
                .set_buffer(Some(&res), k as u64 * THREADS * 4, 1)
                .unwrap();
            encoder.dispatch_threads(grid, group).unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        let got = words(&read_buffer(&res, 0, THREADS * 4 * u64::from(PAIRS)).unwrap());
        let mut wrong = 0usize;
        let mut first = Vec::new();
        for k in 1..=PAIRS {
            for t in 0..THREADS as u32 {
                let at = (k as usize - 1) * THREADS as usize + t as usize;
                let expected = (k << 24) | t;
                if got[at] != expected {
                    wrong += 1;
                    if first.len() < 8 {
                        first.push(format!("k {k} t {t}: {:#010x}", got[at]));
                    }
                }
            }
        }
        assert_eq!(
            wrong,
            0,
            "{}: {wrong} of {} words differ from Metal's order, first {first:?}",
            device.name(),
            THREADS * u64::from(PAIRS)
        );
    });
}
