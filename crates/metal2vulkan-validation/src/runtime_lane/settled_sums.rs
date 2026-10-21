//! The sums the translation leaves undecorated, on the device the cfg selects, under the validation
//! layer: a two-leaf `fast` sum that gives a device nothing to fuse or regroup it with is left as
//! written, without `NoContraction` (`lower::arith::reassoc`, its module docs' **Alone**).
//!
//! Each kernel below has a settled twin whose one more use of a sum, a comparison or a product,
//! keeps that sum `NoContraction` (the translator's `tests::engine_settled_sums` checks which is
//! which in the SPIR-V). On the device each form and its twin must give the same words over 4096
//! inputs: `stored`'s sums (of a buffer's two lanes, stored to device memory), `converted`'s
//! integers (a difference converted to `int`), and `sampled`'s texels, sampled at sums of a
//! buffer's lanes and a constant buffer's, whole, lane by lane and with a negated lane
//! (`(d.x, -d.y) + uv`), and summed in turn. `stored`'s
//! sums must be the correctly rounded ones and `converted`'s integers the truncated differences
//! too.

use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, SamplerMinMagFilter};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, Device, PipelineOptions, Region, SamplerState, Size, Texture,
    TextureDescriptor,
};

const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @stored(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s, ptr addrspace(1) %po, align 4
  ret void
}

define void @stored_compared(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s, ptr addrspace(1) %po, align 4
  %lt = fcmp fast olt float %s, 0.000000e+00
  %z = zext i1 %lt to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %z, ptr addrspace(1) %pz, align 4
  ret void
}

define void @converted(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fsub fast float %a, %b
  %i = fptosi float %s to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %i, ptr addrspace(1) %pz, align 4
  ret void
}

define void @converted_multiplied(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fsub fast float %a, %b
  %i = fptosi float %s to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %i, ptr addrspace(1) %pz, align 4
  %p = fmul fast float %s, %a
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %p, ptr addrspace(1) %po, align 4
  ret void
}

define void @sampled(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, ptr addrspace(2) %k, ptr addrspace(1) %tex, ptr addrspace(2) %smp, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %uv = load <2 x float>, ptr addrspace(1) %pi, align 8
  %d = load <2 x float>, ptr addrspace(2) %k, align 8
  %c = fadd fast <2 x float> %uv, %d
  %a = extractelement <2 x float> %uv, i64 0
  %b = extractelement <2 x float> %d, i64 1
  %x = fadd fast float %a, %b
  %y = fsub fast float %a, %b
  %c0 = insertelement <2 x float> poison, float %x, i64 0
  %c1 = insertelement <2 x float> %c0, float %y, i64 1
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c1, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0
  %dx = extractelement <2 x float> %d, i64 0
  %dy = extractelement <2 x float> %d, i64 1
  %ny = fneg fast float %dy
  %n0 = insertelement <2 x float> poison, float %dx, i64 0
  %dn = insertelement <2 x float> %n0, float %ny, i64 1
  %c2 = fadd fast <2 x float> %dn, %uv
  %r2 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c2, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v2 = extractvalue { <4 x float>, i8 } %r2, 0
  %w = fadd fast <4 x float> %v0, %v1
  %w2 = fadd fast <4 x float> %v2, %v1
  %g2 = shl i32 %g, 1
  %po = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g2
  store <4 x float> %w, ptr addrspace(1) %po, align 16
  %g21 = or i32 %g2, 1
  %po2 = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g21
  store <4 x float> %w2, ptr addrspace(1) %po2, align 16
  ret void
}

define void @sampled_compared(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, ptr addrspace(2) %k, ptr addrspace(1) %tex, ptr addrspace(2) %smp, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %uv = load <2 x float>, ptr addrspace(1) %pi, align 8
  %d = load <2 x float>, ptr addrspace(2) %k, align 8
  %c = fadd fast <2 x float> %uv, %d
  %a = extractelement <2 x float> %uv, i64 0
  %b = extractelement <2 x float> %d, i64 1
  %x = fadd fast float %a, %b
  %y = fsub fast float %a, %b
  %c0 = insertelement <2 x float> poison, float %x, i64 0
  %c1 = insertelement <2 x float> %c0, float %y, i64 1
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c1, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0
  %dx = extractelement <2 x float> %d, i64 0
  %dy = extractelement <2 x float> %d, i64 1
  %ny = fneg fast float %dy
  %n0 = insertelement <2 x float> poison, float %dx, i64 0
  %dn = insertelement <2 x float> %n0, float %ny, i64 1
  %c2 = fadd fast <2 x float> %dn, %uv
  %r2 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c2, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v2 = extractvalue { <4 x float>, i8 } %r2, 0
  %w = fadd fast <4 x float> %v0, %v1
  %w2 = fadd fast <4 x float> %v2, %v1
  %g2 = shl i32 %g, 1
  %po = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g2
  store <4 x float> %w, ptr addrspace(1) %po, align 16
  %g21 = or i32 %g2, 1
  %po2 = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g21
  store <4 x float> %w2, ptr addrspace(1) %po2, align 16
  %cx = extractelement <2 x float> %c, i64 0
  %lt = fcmp fast olt float %cx, 5.000000e-01
  %z = zext i1 %lt to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %z, ptr addrspace(1) %pz, align 4
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!20, !21, !22, !23, !24, !25}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"aux"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!9 = !{!4, !5, !6, !7}
!11 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"k"}
!12 = !{i32 4, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!13 = !{i32 5, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!14 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!15 = !{!4, !16, !6, !11, !12, !13, !14}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!20 = !{ptr @stored, !3, !9}
!21 = !{ptr @stored_compared, !3, !9}
!22 = !{ptr @converted, !3, !9}
!23 = !{ptr @converted_multiplied, !3, !9}
!24 = !{ptr @sampled, !3, !15}
!25 = !{ptr @sampled_compared, !3, !15}
"#;

/// The threads, one input each, in threadgroups of 64.
const THREADS: usize = 4096;

/// The texture's side.
const SIDE: u64 = 16;

/// A fixed xorshift generator.
struct Words(u32);

impl Words {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// The pairs `stored` and `converted` read: ties, signed zeros and differences near an integer,
/// then normal operands of magnitude below 2^24 with random fractions.
fn pairs() -> Vec<[f32; 2]> {
    let ulp = 2f32.powi(-23);
    let mut out = vec![
        [1.0, ulp / 2.0],
        [1.0 + ulp, ulp / 2.0],
        [1.0, 3.0 * ulp / 2.0],
        [-1.0, -ulp / 2.0],
        [-0.0, -0.0],
        [0.0, -0.0],
        [1.5, -1.5],
        [16_777_216.0, -0.5],
        [4_194_304.5, 0.25],
        [-1_048_576.0 - 0.25, 0.125],
        [1.0e30, 1.0e30],
        [1.0e-37, 1.0e-37],
    ];
    let mut w = Words(0x9E37_79B9);
    while out.len() < THREADS {
        let mut normal = || {
            let x = w.next();
            let exponent = 127 - 20 + (x >> 24) % 44;
            f32::from_bits((x & 0x8000_0000) | (exponent << 23) | (w.next() & 0x7F_FFFF))
        };
        out.push([normal(), normal()]);
    }
    out
}

/// The coordinates `sampled` reads: 24 random bits in `[0, 1)` each.
fn coordinates() -> Vec<[f32; 2]> {
    let mut w = Words(0x2545_F491);
    let mut unit = || (w.next() >> 8) as f32 / 16_777_216.0;
    (0..THREADS).map(|_| [unit(), unit()]).collect()
}

/// An `RGBA8Unorm` texture of random bytes.
fn texture(device: &Device) -> Texture {
    let descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    let texture = device.new_texture(&descriptor).unwrap();
    let mut w = Words(0x0BAD_5EED);
    let bytes: Vec<u8> = (0..4 * SIDE * SIDE)
        .map(|_| (w.next() >> 24) as u8)
        .collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
            &bytes,
            4 * SIDE,
            4 * SIDE * SIDE,
        )
        .unwrap();
    texture
}

/// A linear sampler clamping to the edge.
fn sampler(device: &Device) -> SamplerState {
    let mut descriptor = SamplerDescriptor::new();
    descriptor.min_filter = SamplerMinMagFilter::Linear;
    descriptor.mag_filter = SamplerMinMagFilter::Linear;
    descriptor.s_address_mode = SamplerAddressMode::ClampToEdge;
    descriptor.t_address_mode = SamplerAddressMode::ClampToEdge;
    device.new_sampler_state(&descriptor).unwrap()
}

/// The words of `kernel`'s `out` (`out_words` a thread) and `aux` buffers on `device`, over
/// `input`.
fn run(
    device: &Device,
    library: &metal2vulkan_runtime::Library,
    kernel: &str,
    input: &[[f32; 2]],
    out_words: usize,
) -> (Vec<u32>, Vec<u32>) {
    let function = library.new_function_with_name(kernel).unwrap();
    let pipeline = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let bytes: Vec<u8> = input
        .iter()
        .flat_map(|[a, b]| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
        .collect();
    let input = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0xEE; 4 * out_words * THREADS]).unwrap();
    let aux = buffer_with(device, &vec![0xEE; 4 * THREADS]).unwrap();
    // `d`: 0.1 and 0.0123 as `float`s, which round the sums.
    let k: Vec<u8> = [0.1f32, 0.0123]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let k = buffer_with(device, &k).unwrap();
    let (texture, sampler) = (texture(device), sampler(device));
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    for (index, buffer) in [&input, &out, &aux, &k].into_iter().enumerate() {
        let index = u32::try_from(index).unwrap();
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    encoder.set_texture(Some(&texture), 0).unwrap();
    encoder.set_sampler_state(Some(&sampler), 0).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(THREADS as u64 / 64, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let words = |b, n: usize| -> Vec<u32> {
        read_buffer(b, 0, 4 * (n * THREADS) as u64)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    };
    (words(&out, out_words), words(&aux, 1))
}

/// The inputs whose word differs between `a` and `b`, with the two words, at most 24.
fn differing(a: &[u32], b: &[u32]) -> Vec<(usize, u32, u32)> {
    (a.iter().zip(b).enumerate())
        .filter(|(_, (x, y))| x != y)
        .map(|(g, (x, y))| (g, *x, *y))
        .take(24)
        .collect()
}

/// Each form gives its settled twin's words on the device, and `stored`'s sums and `converted`'s
/// integers are the correctly rounded sums and the truncated differences (module docs).
#[test]
fn a_sum_left_as_written_gives_its_settled_twin_s_words() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let name = device.name().to_owned();
        let bitcode = llvm_assemble_text(MODULE).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let (pairs, coordinates) = (pairs(), coordinates());
        let mut failures = Vec::new();
        let (sums, _) = run(&device, &library, "stored", &pairs, 1);
        let (twin_sums, _) = run(&device, &library, "stored_compared", &pairs, 1);
        let (_, ints) = run(&device, &library, "converted", &pairs, 1);
        let (_, twin_ints) = run(&device, &library, "converted_multiplied", &pairs, 1);
        let (texels, _) = run(&device, &library, "sampled", &coordinates, 8);
        let (twin_texels, _) = run(&device, &library, "sampled_compared", &coordinates, 8);
        for (what, a, b) in [
            ("stored", &sums, &twin_sums),
            ("converted", &ints, &twin_ints),
            ("sampled", &texels, &twin_texels),
        ] {
            let differ = differing(a, b);
            eprintln!(
                "settled sums {what} on {name}: {} words differ from the settled twin's",
                differ.len()
            );
            if !differ.is_empty() {
                failures.push(format!("{what} on {name}: (word, form, twin) {differ:x?}"));
            }
        }
        let rounded: Vec<u32> = pairs.iter().map(|[a, b]| (a + b).to_bits()).collect();
        let truncated: Vec<u32> = pairs.iter().map(|[a, b]| (a - b) as i32 as u32).collect();
        for (what, got, want) in [
            ("stored against the rounded sums", &sums, &rounded),
            (
                "converted against the truncated differences",
                &ints,
                &truncated,
            ),
        ] {
            let differ = differing(got, want);
            if !differ.is_empty() {
                failures.push(format!(
                    "{what} on {name}: (input, device, host) {differ:x?}"
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    let reported = messages.take();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}
