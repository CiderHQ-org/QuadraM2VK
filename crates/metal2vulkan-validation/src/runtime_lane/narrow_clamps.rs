//! A device test of a narrow sum's signed clamp and truncation (wip-queue 0d) on the device the cfg
//! selects, under the validation layer.
//!
//! The RTX 3090's compiler (NVIDIA 615.71.09) drops the zero-extending mask after a signed clamp
//! of a sum or difference of two values sign-extended from `N` bits to exactly `N` bits' range,
//! so the result comes out sign-extended. Its narrow `add_sat` and `sub_sat`
//! no longer emit that shape, but source arithmetic still does: an `int` clamp of such a sum, then
//! a truncation. On the GA10x generation the profile carries
//! `DriverWorkaround::NarrowMaskAfterSignedClamp`, and the mask is `OpBitFieldUExtract`.
//!
//! The kernel is written here (an owned synthetic AIR module): each thread reads two words of
//! edge values, and writes four words: the 16-bit sum clamped by `llvm.smin` then `llvm.smax`,
//! the 16-bit difference clamped by `llvm.smax` then `llvm.smin`, and the 8-bit sum and
//! difference clamped by `air.clamp.s.i32`, each truncated to its width and zero-extended. Every
//! device answers each word zero-extended.

use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{PipelineOptions, ResourceOptions, Size};

/// The edge values, as words: each lane reads its low 16 or 8 bits.
const EDGES: [u32; 16] = [
    0x0000_0000,
    0x0000_0001,
    0x0000_7ffe,
    0x0000_7fff,
    0x0000_8000,
    0x0000_8001,
    0x0000_ffff,
    0x0000_fffe,
    0x0000_007f,
    0x0000_0080,
    0x0000_00ff,
    0x0000_0081,
    0x0000_4000,
    0x0000_c000,
    0x0000_0040,
    0x0000_ffc0,
];

/// The threads: every pair of edges.
const THREADS: u32 = 256;

/// The words each thread writes.
const WORDS: u32 = 4;

/// The owned synthetic kernel.
const KERNEL: &str = r#"; Owned synthetic kernel.
source_filename = "narrow_clamps.metal"

declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @air.clamp.s.i32(i32, i32, i32)

define void @narrow_clamps(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %gid) {
entry:
  %xi = and i32 %gid, 15
  %yi = lshr i32 %gid, 4
  %xw = zext i32 %xi to i64
  %yw = zext i32 %yi to i64
  %xp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %xw
  %yp = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %yw
  %x = load i32, ptr addrspace(1) %xp, align 4
  %y = load i32, ptr addrspace(1) %yp, align 4
  %x16 = trunc i32 %x to i16
  %y16 = trunc i32 %y to i16
  %a16 = sext i16 %x16 to i32
  %b16 = sext i16 %y16 to i32
  %s16 = add nsw i32 %a16, %b16
  %s16a = call i32 @llvm.smin.i32(i32 %s16, i32 32767)
  %s16b = call i32 @llvm.smax.i32(i32 %s16a, i32 -32768)
  %s16t = trunc i32 %s16b to i16
  %s16z = zext i16 %s16t to i32
  %d16 = sub nsw i32 %a16, %b16
  %d16a = call i32 @llvm.smax.i32(i32 %d16, i32 -32768)
  %d16b = call i32 @llvm.smin.i32(i32 %d16a, i32 32767)
  %d16t = trunc i32 %d16b to i16
  %d16z = zext i16 %d16t to i32
  %x8 = trunc i32 %x to i8
  %y8 = trunc i32 %y to i8
  %a8 = sext i8 %x8 to i32
  %b8 = sext i8 %y8 to i32
  %s8 = add nsw i32 %a8, %b8
  %s8c = call i32 @air.clamp.s.i32(i32 %s8, i32 -128, i32 127)
  %s8t = trunc i32 %s8c to i8
  %s8z = zext i8 %s8t to i32
  %d8 = sub nsw i32 %a8, %b8
  %d8c = call i32 @air.clamp.s.i32(i32 %d8, i32 -128, i32 127)
  %d8t = trunc i32 %d8c to i8
  %d8z = zext i8 %d8t to i32
  %o = mul i32 %gid, 4
  %o0 = zext i32 %o to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %o0
  store i32 %s16z, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 1
  store i32 %d16z, ptr addrspace(1) %p1, align 4
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 2
  store i32 %s8z, ptr addrspace(1) %p2, align 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 3
  store i32 %d8z, ptr addrspace(1) %p3, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @narrow_clamps, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;

/// `x op y` of the low `bits` bits of two words, sign-extended, clamped to `bits` bits' range,
/// and zero-extended.
fn clamped(x: u32, y: u32, bits: u32, sub: bool) -> u32 {
    let sh = 32 - bits;
    let (a, b) = (((x << sh) as i32) >> sh, ((y << sh) as i32) >> sh);
    let r = if sub { a - b } else { a + b };
    let limit = (1i32 << (bits - 1)) - 1;
    (r.clamp(-limit - 1, limit) as u32) & ((1u32 << bits) - 1)
}

/// Every thread's four words are its clamped sums and differences, zero-extended, on the device
/// as the lane opens it: on the RTX 3090 only through `OpBitFieldUExtract`, the mask the
/// profile's workaround writes.
#[test]
fn a_narrow_sum_clamped_signed_and_truncated_is_zero_extended() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let library = device
            .new_library_with_air(&llvm_assemble_text(KERNEL).unwrap())
            .unwrap();
        let function = library.new_function_with_name("narrow_clamps").unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let edges: Vec<u8> = EDGES.iter().flat_map(|w| w.to_le_bytes()).collect();
        let input = buffer_with(&device, &edges).unwrap();
        let output = device
            .new_buffer(u64::from(THREADS * WORDS * 4), ResourceOptions::shared())
            .unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&state).unwrap();
        encoder.set_buffer(Some(&input), 0, 0).unwrap();
        encoder.set_buffer(Some(&output), 0, 1).unwrap();
        encoder
            .dispatch_threads(Size::new(THREADS.into(), 1, 1), Size::new(64, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let bytes = read_buffer(&output, 0, u64::from(THREADS * WORDS * 4)).unwrap();
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let name = device.name().to_owned();
        let mut wrong = Vec::new();
        for gid in 0..THREADS {
            let (x, y) = (EDGES[(gid & 15) as usize], EDGES[(gid >> 4) as usize]);
            let want = [
                clamped(x, y, 16, false),
                clamped(x, y, 16, true),
                clamped(x, y, 8, false),
                clamped(x, y, 8, true),
            ];
            for (k, want) in want.into_iter().enumerate() {
                let got = words[(gid * WORDS) as usize + k];
                if got != want {
                    wrong.push(format!(
                        "thread {gid} word {k}: {got:#010x}, want {want:#010x}"
                    ));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "{name}: {} of {} words differ:\n{}",
            wrong.len(),
            THREADS * WORDS,
            wrong.join("\n")
        );
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}
