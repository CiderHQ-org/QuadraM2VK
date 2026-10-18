//! A `Depth16Unorm` texture's texels read and sampled with a `float` result, on the device the cfg
//! selects, under the validation layer, against plain Metal's: an Apple M4 under macOS 27.0.1
//! reads and samples (nearest) every 16-bit code `c` as `c / 65535` correctly rounded to `float`
//! (our own probe: all 65,536 codes, both accesses, the outputs' SHA-256 below),
//! where a device that multiplies by the reciprocal answers 512 codes one ulp off (lavapipe, the
//! format scout's `Depth16Unorm` read cell). The kernels are the format scout's (`format_scout.ll`):
//! a 16x16 texture of 256 codes at a time, each code read and sampled at its texel's centre.
//!
//! The same texture through an argument buffer ([`HEAP`]'s kernels, synthetic AIR written for
//! this lane) reads and samples alike: Metal reads a texture an argument buffer holds as the same
//! texture at a slot, and the module converts a heap depth texture's texel by the class its
//! handle carries.

use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::hash::sha256_bytes;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, SamplerMinMagFilter, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble, llvm_assemble_text};
use metal2vulkan_runtime::{
    BlitOption, CommandBufferStatus, ComputePipelineState, Device, Library, Origin,
    PipelineOptions, ResourceOptions, Size, TextureDescriptor,
};

/// The side of the scout's kernels' texture.
const SIDE: u64 = 16;

/// The SHA-256 of Metal's 65,536 `float`s, code by code, read and sampled alike.
const METAL: &str = "a940e05b402805a0f114a2009566daa556ac9cc732c04127d1cfaf7d98c13b0d";

/// Synthetic AIR written for this lane: `read_dh` reads, and `sample_dh` samples through sampler
/// slot 0 at `(g + (0.25, 0.75)) / 16`, the `depth2d<float>` its argument buffer (buffer 0, `{
/// depth2d<float> d; }`) holds at texel `g`, storing the depth to buffer 1's `float` `16 g.y +
/// g.x`: the scout's `scout_read_d` and `scout_sample_d` through an argument buffer.
const HEAP: &str = r#"; Owned synthetic kernels.
source_filename = "depth_texel_reads.metal"

%struct.R = type { ptr addrspace(1) }
%struct.S = type { ptr addrspace(1) }

define void @read_dh(ptr addrspace(2) %m, ptr addrspace(1) noalias %out, <2 x i32> %g) {
entry:
  %pt = getelementptr inbounds %struct.R, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %rs = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 1, <2 x i32> %g, <2 x i32> zeroinitializer, i32 0, i32 1)
  %x = extractvalue { float, i8 } %r, 0
  %gy = extractelement <2 x i32> %g, i64 1
  %row = shl i32 %gy, 4
  %gx = extractelement <2 x i32> %g, i64 0
  %i = add i32 %row, %gx
  %j = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %j
  store float %x, ptr addrspace(1) %p, align 4
  ret void
}

define void @sample_dh(ptr addrspace(2) %m, ptr addrspace(2) %s, ptr addrspace(1) noalias %out, <2 x i32> %g) {
entry:
  %pt = getelementptr inbounds %struct.S, ptr addrspace(2) %m, i64 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(2) %pt, align 8
  %f = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %g)
  %c = fadd <2 x float> %f, <float 2.500000e-01, float 7.500000e-01>
  %uv = fmul <2 x float> %c, <float 6.250000e-02, float 6.250000e-02>
  %r = tail call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %s, i32 1, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %x = extractvalue { float, i8 } %r, 0
  %gy = extractelement <2 x i32> %g, i64 1
  %row = shl i32 %gy, 4
  %gx = extractelement <2 x i32> %g, i64 0
  %i = add i32 %row, %gx
  %j = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %j
  store float %x, ptr addrspace(1) %p, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i32>, <2 x i32>, i32, i32)
declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)

!air.kernel = !{!0, !10}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_dh, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"R", !"air.arg_name", !"m"}
!4 = !{i32 0, i32 8, i32 0, !"depth2d<float, read>", !"d", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d<float, read>", !"air.arg_name", !"d"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!10 = !{ptr @sample_dh, !1, !11}
!11 = !{!12, !15, !16, !17}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"S", !"air.arg_name", !"m"}
!13 = !{i32 0, i32 8, i32 0, !"depth2d<float, sample>", !"d", !"air.indirect_argument", !14}
!14 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"d"}
!15 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!17 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"g"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Kernel `name` of `library`'s.
fn kernel(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

#[test]
fn a_depth16_texel_reads_and_samples_as_metal_s_code_over_65535() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        eprintln!("device {}", device.name());
        let scout = llvm_assemble(&engine_dir().join("format_scout.ll")).unwrap();
        let scout = device.new_library_with_air(&scout).unwrap();
        let heap = device
            .new_library_with_air(&llvm_assemble_text(HEAP).unwrap())
            .unwrap();
        // (the access, its pipeline, whether it samples, whether through an argument buffer)
        let accesses = [
            (
                "read",
                kernel(&device, &scout, "scout_read_d"),
                false,
                false,
            ),
            (
                "nearest sample",
                kernel(&device, &scout, "scout_sample_d"),
                true,
                false,
            ),
            (
                "argument buffer's read",
                kernel(&device, &heap, "read_dh"),
                false,
                true,
            ),
            (
                "argument buffer's nearest sample",
                kernel(&device, &heap, "sample_dh"),
                true,
                true,
            ),
        ];
        let mut nearest = SamplerDescriptor::new();
        nearest.min_filter = SamplerMinMagFilter::Nearest;
        nearest.mag_filter = SamplerMinMagFilter::Nearest;
        let nearest = device.new_sampler_state(&nearest).unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth16Unorm, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::private();
        let t = device.new_texture(&d).unwrap();
        let argument = buffer_with(&device, &t.gpu_resource_id().0.to_le_bytes()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let n = SIDE * SIDE;
        let mut outputs = vec![Vec::new(); accesses.len()];
        for page in 0..65536 / n {
            let codes: Vec<u8> = (0..n)
                .flat_map(|i| ((page * n + i) as u16).to_le_bytes())
                .collect();
            let src = buffer_with(&device, &codes).unwrap();
            let out: Vec<_> = accesses
                .iter()
                .map(|_| buffer_with(&device, &vec![0xab; (n * 4) as usize]).unwrap())
                .collect();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &src,
                0,
                SIDE * 2,
                n * 2,
                Size::new(SIDE, SIDE, 1),
                &t,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            for ((_, state, sampled, held), out) in accesses.iter().zip(&out) {
                let mut e = command_buffer.compute_command_encoder().unwrap();
                e.set_compute_pipeline_state(state).unwrap();
                match held {
                    true => {
                        e.use_resource(t.as_resource(), ResourceUsage::READ)
                            .unwrap();
                        e.set_buffer(Some(&argument), 0, 0).unwrap();
                        e.set_buffer(Some(out), 0, 1).unwrap();
                    }
                    false => {
                        e.set_texture(Some(&t), 0).unwrap();
                        e.set_buffer(Some(out), 0, 0).unwrap();
                    }
                }
                if *sampled {
                    e.set_sampler_state(Some(&nearest), 0).unwrap();
                }
                e.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))
                    .unwrap();
                e.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            for (got, out) in outputs.iter_mut().zip(&out) {
                got.extend(read_buffer(out, 0, n * 4).unwrap());
            }
        }
        for ((what, ..), got) in accesses.iter().zip(&outputs) {
            let off: Vec<(u32, u32, u32)> = got
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .filter_map(|(c, v)| {
                    let v = u32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                    let metal = (c as f32 / 65535.0).to_bits();
                    (v != metal).then_some((c as u32, v, metal))
                })
                .collect();
            assert!(
                off.is_empty(),
                "{} on {}: {} of 65536 codes not c / 65535 correctly rounded, the first \
                 (code, device, Metal): {:x?}",
                what,
                device.name(),
                off.len(),
                &off[..off.len().min(8)]
            );
            assert_eq!(sha256_bytes(got), METAL, "{what}: Metal's digest");
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
