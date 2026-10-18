//! A render state's texture slots and how its functions access them
//! (`RenderPipelineState::texture_access`), on the device the cfg selects, under the validation
//! layer. The state answers what Metal's `reflection:` out-parameter of
//! `newRenderPipelineStateWithDescriptor:options:reflection:error:` states for the same
//! functions. The answers were measured on an M4 with macOS 27.0.1, plain and under Metal's
//! validation, alike under the `BindingInfo`, `BufferTypeInfo` and combined options:
//! - the vertex function's `access::read` texture 0 is read-only;
//! - in the fragment function, `access::read` texture 0 is read-only, `access::write` 1
//!   write-only, `access::read_write` 2 read-write, and the two `access::write` elements of the
//!   array at 3 write-only (one binding, array length 2);
//! - the fragment function's `access::write` texture 5, which it never references, is listed
//!   write-only and unused;
//! - its sampled texture 6 is read-only.
//!
//! The functions' AIR is what Metal's compiler emitted for the probe's source, in opaque-pointer
//! form.

use super::render_state::{quiet, validated};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{BindingAccess, PixelFormat, RenderStages};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{PipelineOption, PipelineOptions};

/// The probe's vertex function `vtx` (`texture2d<float, access::read> vt [[texture(0)]]`) and
/// fragment function `frag`: `r` read at 0, `w` written at 1, `rw` read and written at 2, the
/// array `wa` of two written at 3, `unused` (write) at 5, and `s` sampled at 6.
const AIR: &str = r#"define <{ <4 x float> }> @vtx(i32 noundef %0, ptr addrspace(1) nocapture readonly %1) local_unnamed_addr #0 {
  %3 = and i32 %0, 1
  %4 = tail call fast float @air.convert.f.f32.u.i32(i32 %3)
  %5 = lshr i32 %0, 1
  %6 = tail call fast float @air.convert.f.f32.u.i32(i32 %5)
  %7 = tail call ptr addrspace(2) @air.get_read_sampler()
  %8 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) nocapture readonly %1, ptr addrspace(2) %7, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %9 = extractvalue { <4 x float>, i8 } %8, 0
  %10 = insertelement <4 x float> <float poison, float poison, float poison, float 1.000000e+00>, float %4, i64 0
  %11 = insertelement <4 x float> %10, float %6, i64 1
  %12 = shufflevector <4 x float> %11, <4 x float> %9, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  %13 = insertvalue <{ <4 x float> }> undef, <4 x float> %12, 0
  ret <{ <4 x float> }> %13
}

declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr

define <4 x float> @frag(<4 x float> %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr nocapture readonly byval([2 x ptr addrspace(1)]) %4, ptr addrspace(1) nocapture readnone %5, ptr addrspace(1) %6, ptr addrspace(2) nocapture readonly %7) local_unnamed_addr #1 {
  %9 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %10 = tail call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %9)
  %11 = tail call ptr addrspace(2) @air.get_read_sampler()
  %12 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) nocapture readonly %1, ptr addrspace(2) %11, <2 x i32> %10, <2 x i32> zeroinitializer, i32 0, i32 1)
  %13 = extractvalue { <4 x float>, i8 } %12, 0
  %14 = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) nocapture readonly %3, ptr addrspace(2) %11, <2 x i32> %10, <2 x i32> zeroinitializer, i32 0, i32 3)
  %15 = extractvalue { <4 x float>, i8 } %14, 0
  %16 = fadd fast <4 x float> %15, %13
  %17 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) nocapture readonly %6, ptr addrspace(2) nocapture readonly %7, <2 x float> %9, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %18 = extractvalue { <4 x float>, i8 } %17, 0
  %19 = fadd fast <4 x float> %16, %18
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) nocapture %2, <2 x i32> %10, <4 x float> %19, i32 0, i32 2)
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) nocapture %3, <2 x i32> %10, <4 x float> %19, i32 0, i32 3)
  %20 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %4, i64 0, i64 0
  %21 = load ptr addrspace(1), ptr %20, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) nocapture %21, <2 x i32> %10, <4 x float> %19, i32 0, i32 2)
  %22 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %4, i64 0, i64 1
  %23 = load ptr addrspace(1), ptr %22, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) nocapture %23, <2 x i32> %10, <4 x float> %19, i32 0, i32 2)
  ret <4 x float> %19
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>) local_unnamed_addr
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) nocapture readonly, ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32) local_unnamed_addr
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) nocapture readonly, ptr addrspace(2) nocapture readonly, <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #2
declare void @air.write_texture_2d.v4f32(ptr addrspace(1) nocapture, <2 x i32>, <4 x float>, i32, i32) local_unnamed_addr

attributes #0 = { mustprogress nofree nounwind willreturn }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent }

!air.vertex = !{!9}
!air.fragment = !{!15}
!air.compile_options = !{!27, !28, !29}
!air.version = !{!31}
!air.language_version = !{!32}

!9 = !{ptr @vtx, !10, !12}
!10 = !{!11}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!12 = !{!13, !14}
!13 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!14 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"vt"}
!15 = !{ptr @frag, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!18 = !{!19, !20, !21, !22, !23, !24, !25, !26}
!19 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!20 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"r"}
!21 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"w"}
!22 = !{i32 3, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"rw"}
!23 = !{i32 4, !"air.texture", !"air.location_index", i32 3, i32 2, !"air.write", !"air.arg_type_name", !"array<texture2d<float, write>, 2>", !"air.arg_name", !"wa"}
!24 = !{i32 5, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"unused", !"air.arg_unused"}
!25 = !{i32 6, !"air.texture", !"air.location_index", i32 6, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"s"}
!26 = !{i32 7, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!27 = !{!"air.compile.denorms_disable"}
!28 = !{!"air.compile.fast_math_enable"}
!29 = !{!"air.compile.framebuffer_fetch_enable"}
!31 = !{i32 2, i32 8, i32 0}
!32 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A render state answers each texture slot's access as Metal's reflection states it (the
/// module docs): per stage, an array's at each element's slot, `None` at a slot no texture
/// argument covers and at one whose argument the function never references, and the same as
/// the `reflection:` out-parameter's bindings. Asked of several stages it joins their answers.
#[test]
fn a_render_state_answers_its_texture_slots_accesses_as_metal_s_reflection() {
    use BindingAccess::{ReadOnly, ReadWrite, WriteOnly};
    let (messages, _cache, probe) = validated();
    let name = probe.device.name().to_owned();
    let library = probe
        .device
        .new_library_with_air(&llvm_assemble_text(AIR).unwrap())
        .unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("vtx"));
    descriptor.fragment = Some(function("frag"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let options = PipelineOptions::new(PipelineOption::BINDING_INFO);
    let (state, reflection) = probe
        .device
        .new_render_pipeline_state_with_reflection(&descriptor, &options)
        .unwrap();
    let (vertex, fragment) = (RenderStages::VERTEX, RenderStages::FRAGMENT);
    let access = |stages, index| state.texture_access(stages, index).unwrap();
    let want = [
        (vertex, 0, Some(ReadOnly)),
        (vertex, 1, None),
        (fragment, 0, Some(ReadOnly)),
        (fragment, 1, Some(WriteOnly)),
        (fragment, 2, Some(ReadWrite)),
        (fragment, 3, Some(WriteOnly)),
        (fragment, 4, Some(WriteOnly)),
        (fragment, 5, None),
        (fragment, 6, Some(ReadOnly)),
        (fragment, 7, None),
        (RenderStages::TILE, 0, None),
        (RenderStages::MESH, 1, None),
        (vertex.union(fragment), 0, Some(ReadOnly)),
        (vertex.union(fragment), 1, Some(WriteOnly)),
        (RenderStages::empty(), 0, None),
    ];
    for (stages, index, answer) in want {
        assert_eq!(access(stages, index), answer, "{name}: {stages:?} {index}");
    }
    // The out-parameter's bindings answer alike: each used texture at each slot it covers.
    let reflection = reflection.expect("asked for");
    for (stage, bindings) in [
        (vertex, reflection.vertex_bindings()),
        (fragment, reflection.fragment_bindings()),
    ] {
        for binding in bindings {
            let Some(texture) = binding.texture() else {
                continue;
            };
            let first = u64::from(binding.index());
            for index in first..first + u64::from(texture.array_length()) {
                let answer = binding.is_used().then_some(binding.access());
                assert_eq!(access(stage, index), answer, "{name}: {stage:?} {index}");
            }
        }
    }
    quiet(probe, &messages);
}
