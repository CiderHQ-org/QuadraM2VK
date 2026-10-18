//! Numbered samplers (the runtime's `resource_ids/numbered.rs`) on the device the cfg selects,
//! under the validation layer: a pipeline state's creation takes plain sampler IDs for its
//! functions' samplers, and the device's own for a render pipeline with a fragment function, held
//! by the state, its function objects and their library, as Metal's are. Each test replays a row of
//! an M4 probe (the M3 Max answered TTG2's the same) on our own AIR: sampler states marked
//! `label=ID` between steps that make and drop the objects, each state kept to the test's end
//! (plain allocation hands each the lowest free ID).

use super::{validated_device, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    ComputePipelineState, Device, Function, Library, PipelineOptions, RenderPipelineState,
    SamplerState, SamplerStateOptions,
};
use std::any::Any;

/// The module's functions: the vertex function `v` (a position of its vertex ID), the fragment
/// functions `f_read` (reads texture 0 at its position) and `f_const` (a constant color), and
/// the kernels `k_read` (reads texture 0) and `k_sample` (samples texture 0 through the
/// constexpr sampler `s`, linear and clamped to its edge). Hand-written AIR.
const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@s = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0], align 8

define <{ <4 x float> }> @v(i32 %vid) {
  %x = and i32 %vid, 1
  %fx = uitofp i32 %x to float
  %y = lshr i32 %vid, 1
  %fy = uitofp i32 %y to float
  %p0 = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %fx, i64 0
  %p1 = insertelement <4 x float> %p0, float %fy, i64 1
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %p1, 0
  ret <{ <4 x float> }> %o
}

define <4 x float> @f_read(<4 x float> %pos, ptr addrspace(1) %t) {
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %c = call <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float> %xy)
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %c, <2 x i32> zeroinitializer, i32 0, i32 1)
  %texel = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %texel
}

define <4 x float> @f_const(<4 x float> %pos) {
  ret <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>
}

define void @k_read(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %texel = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %texel, ptr addrspace(1) %out, align 16
  ret void
}

define void @k_sample(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @s, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %texel = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %texel, ptr addrspace(1) %out, align 16
  ret void
}

declare <2 x i32> @air.convert.u.v2i32.f.v2f32(<2 x float>)
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{!9}
!air.fragment = !{!15, !21}
!air.kernel = !{!30, !36}
!air.sampler_states = !{!40}
!air.version = !{!41}
!air.language_version = !{!42}

!1 = !{}
!9 = !{ptr @v, !10, !12}
!10 = !{!11}
!11 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!12 = !{!13}
!13 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!15 = !{ptr @f_read, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!18 = !{!19, !20}
!19 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!20 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!21 = !{ptr @f_const, !16, !22}
!22 = !{!19}
!30 = !{ptr @k_read, !1, !31}
!31 = !{!32, !34, !35}
!32 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!34 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!35 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!36 = !{ptr @k_sample, !1, !37}
!37 = !{!38, !34, !35}
!38 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!40 = !{!"air.sampler_state", ptr addrspace(2) @s}
!41 = !{i32 2, i32 8, i32 0}
!42 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// One process of a probe row: a validated device of the lane, its marks and their answers.
struct Row {
    device: Device,
    marks: Vec<SamplerState>,
    answers: Vec<String>,
    lod: f32,
}

impl Row {
    fn new(device: Device) -> Row {
        Row {
            device,
            marks: Vec::new(),
            answers: Vec::new(),
            lod: 0.0,
        }
    }

    /// A sampler state no other equals, kept: its ID, answered as `label=ID`.
    fn mark(&mut self, label: &str) {
        let mut descriptor = SamplerDescriptor::new();
        self.lod += 1.0;
        descriptor.lod_min_clamp = self.lod;
        let mut options = SamplerStateOptions::default();
        options.support_argument_buffers = true;
        let state = self
            .device
            .new_sampler_state_with_options(&descriptor, options)
            .unwrap();
        self.answers
            .push(format!("{label}={}", state.gpu_resource_id().0));
        self.marks.push(state);
    }

    /// A library of the module, parsed anew.
    fn library(&self) -> Library {
        self.library_of(MODULE)
    }

    /// A library of `text`, parsed anew.
    fn library_of(&self, text: &str) -> Library {
        self.device
            .new_library_with_air(&llvm_assemble_text(text).unwrap())
            .unwrap()
    }

    /// The render pipeline state of vertex `v` and fragment `f`.
    fn render(&self, v: &Function, f: &Function) -> RenderPipelineState {
        let mut descriptor =
            RenderPipelineDescriptor::new(StageSource::Function(v.function_descriptor()));
        descriptor.fragment = Some(StageSource::Function(f.function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The compute pipeline state of `k`.
    fn compute(&self, k: &Function) -> ComputePipelineState {
        self.device
            .new_compute_pipeline_state_with_function(k, &PipelineOptions::default())
            .unwrap()
    }

    fn answers(&self) -> String {
        self.answers.join(" ")
    }
}

/// Run `row` on a fresh validated device of the lane, then fail on any validation message.
fn on_device(row: impl FnOnce(&mut Row)) -> String {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let answers = {
        let mut r = Row::new(validated_device(TEST_DEVICE, &messages, cache.path()).unwrap());
        row(&mut r);
        r.answers()
    };
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
    answers
}

/// Our probe's `nq_q8_own_v_fread` and `nq_q8_own_f_fread`: a render pipeline state of `v`
/// and `f_read` takes the device's sampler ID (2) and a read's (3); with the library, the state
/// and one function object gone, the fragment function's object holds both and the vertex
/// function's neither.
#[test]
fn a_render_state_numbers_the_device_s_sampler_and_its_fragment_function_s_read() {
    for (kept, expected) in [
        ('v', "a=1 b=4 c=2 d=3 e=5 f=6"),
        ('f', "a=1 b=4 c=5 d=6 e=2 f=3"),
    ] {
        let got = on_device(|row| {
            row.mark("a");
            let library = row.library();
            let v = library.new_function_with_name("v").unwrap();
            let f = library.new_function_with_name("f_read").unwrap();
            let state = row.render(&v, &f);
            row.mark("b");
            let (keep, gone) = if kept == 'v' { (v, f) } else { (f, v) };
            drop((library, state, gone));
            row.mark("c");
            row.mark("d");
            drop(keep);
            row.mark("e");
            row.mark("f");
        });
        assert_eq!(got, expected, "{kept} kept");
    }
}

/// Our probe's `nq_q8_k1_holds`: a compute state of `k_read` takes a read's sampler ID (2),
/// and no device sampler; its kernel's function object holds it once the library and the state
/// are gone.
#[test]
fn a_compute_state_numbers_its_kernel_s_read_held_by_its_function_object() {
    let got = on_device(|row| {
        row.mark("a");
        let library = row.library();
        let k = library.new_function_with_name("k_read").unwrap();
        let state = row.compute(&k);
        row.mark("b");
        drop((library, state));
        row.mark("c");
        row.mark("d");
        drop(k);
        row.mark("e");
        row.mark("f");
    });
    assert_eq!(got, "a=1 b=3 c=4 d=5 e=2 f=6");
}

/// Our probe's `nq_q5_queue_after_render`: a command queue made after a render state of `v`
/// and `f_const` keeps the device's sampler ID the state took (2) for good.
#[test]
fn a_queue_keeps_the_device_s_sampler_a_render_state_numbered() {
    let got = on_device(|row| {
        row.mark("a");
        let library = row.library();
        let state = {
            let v = library.new_function_with_name("v").unwrap();
            let f = library.new_function_with_name("f_const").unwrap();
            row.render(&v, &f)
        };
        row.mark("b");
        let queue = row.device.new_command_queue().unwrap();
        row.mark("c");
        drop((library, state));
        row.mark("d");
        row.mark("e");
        drop(queue);
    });
    assert_eq!(got, "a=1 b=3 c=4 d=5 e=6");
}

/// A constexpr sampler's ID is the device's for its state (the probes' orders): the compute states
/// of `k_sample` and of the same kernel of a library of other content (renamed `k_other`) share one
/// (1) while either lives, and it comes back after both.
#[test]
fn equal_constexpr_samplers_share_one_id_device_wide() {
    let other = MODULE.replace("@k_sample", "@k_other");
    let got = on_device(|row| {
        let mut held: Vec<Box<dyn Any>> = Vec::new();
        for (text, name) in [(MODULE, "k_sample"), (other.as_str(), "k_other")] {
            let library = row.library_of(text);
            let state = {
                let k = library.new_function_with_name(name).unwrap();
                row.compute(&k)
            };
            held.push(Box::new((library, state)));
            row.mark("x");
        }
        drop(held.remove(0));
        row.mark("y");
        drop(held);
        row.mark("z");
    });
    assert_eq!(got, "x=2 x=3 y=4 z=1");
}

/// `Device::sampler_ids_live` on the device, by the rule the tests above replay: a sampler state
/// counts one; the compute states of `k_sample` and of the same kernel of a library of other
/// content number one constexpr sampler between them, counted once; a command queue takes the
/// device's sampler; a render state of `v` and `f_read` of a third content then numbers a read
/// alone (a library of the first content would share its holds with the first state's library,
/// which would hold the read past the state). Each goes with its last holder, at its drop, the
/// device's sampler never. `Device::sampler_id_pool_size`
/// stays 16 until the 15th ID beside the device's, which doubles it, and stays 32 once they go.
#[test]
fn the_live_sampler_id_count_follows_the_numbering() {
    let other = MODULE.replace("@k_sample", "@k_other");
    on_device(|row| {
        let live = |row: &Row| row.device.sampler_ids_live();
        assert_eq!(live(row), 0);
        row.mark("a");
        assert_eq!(live(row), 1);
        let mut held: Vec<Box<dyn Any>> = Vec::new();
        for (text, name) in [(MODULE, "k_sample"), (other.as_str(), "k_other")] {
            let library = row.library_of(text);
            let state = {
                let k = library.new_function_with_name(name).unwrap();
                row.compute(&k)
            };
            held.push(Box::new((library, state)));
            assert_eq!(live(row), 2, "{name}");
        }
        let queue = row.device.new_command_queue().unwrap();
        assert_eq!(live(row), 3);
        let library = row.library_of(&MODULE.replace("@k_read", "@k_third"));
        let state = {
            let v = library.new_function_with_name("v").unwrap();
            let f = library.new_function_with_name("f_read").unwrap();
            row.render(&v, &f)
        };
        assert_eq!(live(row), 4);
        drop((library, state));
        assert_eq!(live(row), 3);
        drop(held.remove(0));
        assert_eq!(live(row), 3);
        drop(held);
        assert_eq!(live(row), 2);
        drop(queue);
        assert_eq!(live(row), 2);
        row.marks.clear();
        assert_eq!(live(row), 1);
        let pool = |row: &Row| row.device.sampler_id_pool_size();
        assert_eq!(pool(row), 16);
        for _ in 0..14 {
            row.mark("s");
        }
        assert_eq!((live(row), pool(row)), (15, 16));
        row.mark("s");
        assert_eq!((live(row), pool(row)), (16, 32));
        row.marks.clear();
        assert_eq!((live(row), pool(row)), (1, 32));
    });
}
