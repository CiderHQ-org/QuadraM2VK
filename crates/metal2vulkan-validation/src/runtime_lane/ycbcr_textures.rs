//! Device tests of `YCBCR8_420_2P` textures over a CoreVideo `420v` or `420f` surface's two
//! planes (`runtime/src/ycbcr.rs`), on the device the cfg selects, under the validation layer. Each
//! surface is a Shared buffer standing for the IOSurface, laid out as CoreVideo lays a 40x24 one
//! out: the luma plane at byte 0, rows 64 bytes apart, and the interleaved chroma plane after it,
//! rows 64 bytes apart.
//!
//! - A kernel's `read()` of every texel is the conversion Metal's reads were measured to follow
//!   (`metal2vulkan::bind::ycbcr_texel`, after the 9/3/3/1 chroma filter), for both ranges and
//!   both matrices, and for one surface Metal's own reads: the same digest. A
//!   nearest sample at each texel's centre reads the same texel, unannounced; a linear one there
//!   too, and between the texels it is Metal's filter of them, unannounced (the lane's devices
//!   sample as Metal does). Under native sampling it is Metal's filter under
//!   `YCbCrFilterPolicy::Exact`, through one pipeline more at the draw, and under the default the
//!   device's filter of the same texels (on MoltenVK Metal's own sampler's, announced), within
//!   [`DEVICE_FILTER_BOUND`] of Metal's, through the pipeline the state's creation made. A function
//!   that samples handles and reaches the texture is refused, typed.
//! - The texture follows writes of its planes' bytes through other textures over them, as a
//!   decoder writes the planes as `R8Unorm` and `RG8Unorm` textures: in an earlier command
//!   buffer, an earlier encoder of the same one, an earlier dispatch of the same encoder, and a
//!   blit; and the CPU's writes between command buffers.
//! - A kernel's writes to it are dropped, as Metal drops them; `get_bytes` reads the luma rows.
//!   A 4:4:4 format's writes are stored into both planes as Metal stores them, every other byte
//!   kept; those of the 4:4:4 formats whose writes are not known (the `_sRGB` twins, the packed
//!   format) and over plane 1 are refused, typed.
//! - Plain creation and a buffer's texture of the format answer Metal's nil (`Error::Nil`,
//!   `NoTextureOfFormat`) and the heap query 0 bytes at 16; a surface without its facts, plane 1
//!   and a matrix other than BT.601, BT.709 and BT.2020 are refused, typed. Every other two-plane
//!   YCbCr format answers as Metal does without a surface, and is refused over one.

use super::argument_linear_samples::{self, LINEAR_EDGE};
use super::blits::SYSTEM_MEMORY;
use super::degrade_notices::{assert_once, counts, on_device, on_device_with};
use super::depth_compares::{pipeline_of, run_held, Held, Through};
use super::host_memory::{self, host_page, Mapping};
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{
    ycbcr_read, ycbcr_texel, ycbcr_write, SamplerDescriptor, YCbCrCodes, YCbCrMatrix, YCbCrOutput,
    YCbCrRange,
};
use metal2vulkan::device::{SamplingPolicy, VkProperty, YCbCrFilterPolicy};
use metal2vulkan::error::{BindError, Feature, RuntimeDriverDefect};
use metal2vulkan::metal::format::{ycbcr_format, ChromaSubsampling, YCbCrReads};
use metal2vulkan::metal::{
    PixelFormat, PrimitiveType, ResourceUsage, SamplerAddressMode, SamplerBorderColor,
    SamplerMinMagFilter, SamplerMipFilter, TextureSwizzle, TextureSwizzleChannels, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandQueue, ComputePipelineState, Deallocator,
    Degradation, Device, DeviceOptions, Error, Invalid, IoSurfacePlane, IoSurfaceYCbCr, Library,
    LoadAction, MappedSampling, Nil, Origin, PipelineOptions, Problem, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, RuntimeGap, SamplerState, Size,
    StoreAction, SurfaceYCbCrMatrix, Texture, TextureDescriptor, TextureProblem,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The surface's size, and its planes' rows and offsets.
const W: u64 = 40;
const H: u64 = 24;
const ROW: u64 = 64;
const CHROMA_AT: u64 = ROW * H;
const CW: u64 = W / 2;
const CH: u64 = H / 2;
const LENGTH: u64 = 4096;
const VIDEO: u32 = u32::from_be_bytes(*b"420v");
const FULL: u32 = u32::from_be_bytes(*b"420f");

/// Synthetic AIR written for this lane: `read_texels` reads texel `gid` of texture 0 into
/// element `gid.y * width + gid.x` of buffer 0; `sample_texels` samples texture 0 through
/// sampler 0 at that texel's centre, and `sample_between` at [`BETWEEN`] past it; `fill` writes
/// buffer 0's `float4` to every texel of texture 0. `sample_outside`, dispatched over twice the
/// texture's size, samples point `gid` at the centre of texel `gid - size / 4`, a quarter of
/// the points inside the texture and the rest outside it. `write_masked` writes buffer 0's
/// `float4` element `gid.y * width + gid.x` to texel `gid` of texture 0 where buffer 1's `uint`
/// element there is not 0, and `write_masked_half` the same of `half4`s.
pub(super) const KERNELS: &str = r#"; Owned synthetic kernels.
source_filename = "ycbcr_textures.metal"

define void @read_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_texels(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_between(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 9.062500e-01, float 7.187500e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @read_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %c = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %n
  %r = tail call { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @sample_outside(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %q = fmul <2 x float> %n, <float 2.500000e-01, float 2.500000e-01>
  %c0 = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %c = fsub <2 x float> %c0, %q
  %m = fmul <2 x float> %n, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %m
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

; ---- Gathers of red and blue at texel corners ----
define void @gather_red(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_blue(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_red_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @gather_blue_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, ptr addrspace(2) %smp, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n0 = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %n = fsub <2 x float> %n0, <float 1.000000e+00, float 1.000000e+00>
  %uv = fdiv <2 x float> %g, %n
  %r = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %smp, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %h = extractvalue { <4 x half>, i8 } %r, 0
  %v = fpext <4 x half> %h to <4 x float>
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}

define void @fill(ptr addrspace(1) %t, ptr addrspace(1) %value, <2 x i32> %gid) {
entry:
  %v = load <4 x float>, ptr addrspace(1) %value, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

define void @write_masked(ptr addrspace(1) %t, ptr addrspace(1) noalias %values, ptr addrspace(1) noalias %mask, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %mp = getelementptr inbounds i32, ptr addrspace(1) %mask, i64 %j
  %m = load i32, ptr addrspace(1) %mp, align 4
  %on = icmp ne i32 %m, 0
  br i1 %on, label %write, label %done

write:
  %vp = getelementptr inbounds <4 x float>, ptr addrspace(1) %values, i64 %j
  %v = load <4 x float>, ptr addrspace(1) %vp, align 16
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %t, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  br label %done

done:
  ret void
}

define void @write_masked_half(ptr addrspace(1) %t, ptr addrspace(1) noalias %values, ptr addrspace(1) noalias %mask, <2 x i32> %gid, <2 x i32> %size) {
entry:
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %mp = getelementptr inbounds i32, ptr addrspace(1) %mask, i64 %j
  %m = load i32, ptr addrspace(1) %mp, align 4
  %on = icmp ne i32 %m, 0
  br i1 %on, label %write, label %done

write:
  %vp = getelementptr inbounds <4 x half>, ptr addrspace(1) %values, i64 %j
  %v = load <4 x half>, ptr addrspace(1) %vp, align 8
  tail call void @air.write_texture_2d.v4f16(ptr addrspace(1) %t, <2 x i32> %gid, <4 x half> %v, i32 0, i32 2)
  br label %done

done:
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare void @air.write_texture_2d.v4f16(ptr addrspace(1), <2 x i32>, <4 x half>, i32, i32)
declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0, !10, !20, !40, !50, !60, !70, !80, !81, !82, !83, !90, !96}
!air.version = !{!30}
!air.language_version = !{!31}
!0 = !{ptr @read_texels, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!10 = !{ptr @sample_texels, !1, !11}
!11 = !{!12, !4, !13, !14, !15}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!13 = !{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!14 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!15 = !{i32 4, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!20 = !{ptr @fill, !1, !21}
!21 = !{!22, !23, !24}
!22 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"t"}
!23 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"value"}
!24 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!40 = !{ptr @sample_between, !1, !11}
!50 = !{ptr @read_half, !1, !51}
!51 = !{!52, !4, !5, !6}
!52 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"t"}
!60 = !{ptr @sample_half, !1, !61}
!61 = !{!62, !4, !13, !14, !15}
!62 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"t"}
!70 = !{ptr @sample_outside, !1, !11}
!80 = !{ptr @gather_red, !1, !11}
!81 = !{ptr @gather_blue, !1, !11}
!82 = !{ptr @gather_red_half, !1, !61}
!83 = !{ptr @gather_blue_half, !1, !61}
!90 = !{ptr @write_masked, !1, !91}
!91 = !{!22, !92, !93, !94, !95}
!92 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"values"}
!93 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"mask"}
!94 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!95 = !{i32 4, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}
!96 = !{ptr @write_masked_half, !1, !97}
!97 = !{!98, !99, !93, !94, !95}
!98 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"t"}
!99 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"values"}
!30 = !{i32 2, i32 8, i32 0}
!31 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The lane's kernels on one device.
struct Kernels {
    read: ComputePipelineState,
    sample: ComputePipelineState,
    between: ComputePipelineState,
    fill: ComputePipelineState,
    /// A `half` `read()` and a `half` sample at the texels' centres, widened to `float`.
    read_half: ComputePipelineState,
    sample_half: ComputePipelineState,
    outside: ComputePipelineState,
    /// Gathers of red and of blue at the texel corners, `float` and `half` widened to `float`.
    gathers: [[ComputePipelineState; 2]; 2],
    /// Writes of `float4` and of `half4` values to the texels a mask names.
    write_masked: [ComputePipelineState; 2],
}

impl Kernels {
    fn new(device: &Device) -> Kernels {
        let library = device
            .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
            .unwrap();
        let state = |name: &str| {
            let function = library.new_function_with_name(name).unwrap();
            device
                .new_compute_pipeline_state(
                    &ComputePipelineDescriptor::new(StageSource::Function(
                        function.function_descriptor(),
                    )),
                    &PipelineOptions::default(),
                )
                .unwrap()
        };
        Kernels {
            read: state("read_texels"),
            sample: state("sample_texels"),
            between: state("sample_between"),
            fill: state("fill"),
            read_half: state("read_half"),
            sample_half: state("sample_half"),
            outside: state("sample_outside"),
            gathers: [
                [state("gather_red"), state("gather_red_half")],
                [state("gather_blue"), state("gather_blue_half")],
            ],
            write_masked: [state("write_masked"), state("write_masked_half")],
        }
    }
}

/// Our probes' generator of random bytes (`ycc1.swift`, `ycc2.swift`).
struct Random(u64);

impl Random {
    fn next(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u8
    }
}

/// The bytes of a surface whose planes our probes' generator at `seed` fills (luma, then the
/// chroma pairs, each packed), laid out as the module docs say, every other byte `0xa5`.
fn surface_bytes(seed: u64) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; LENGTH as usize];
    for y in 0..H {
        for x in 0..W {
            bytes[(y * ROW + x) as usize] = random.next();
        }
    }
    for y in 0..CH {
        for x in 0..2 * CW {
            bytes[(CHROMA_AT + y * ROW + x) as usize] = random.next();
        }
    }
    bytes
}

/// Each texel's red, green and blue `v` of `v / 2048` the conversion gives the surface `bytes`.
fn modelled(bytes: &[u8], range: YCbCrRange, matrix: YCbCrMatrix) -> Vec<[u16; 3]> {
    let codes = ycbcr_format(PixelFormat::YCBCR84202P)
        .and_then(YCbCrCodes::of)
        .unwrap();
    modelled_as(bytes, codes, (range, matrix))
}

/// Each texel `codes` (8-bit) reads of the surface `bytes` under `conversion`
/// (`metal2vulkan::bind::ycbcr_read`).
fn modelled_as(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    let taps = |q: u64, n: u64| -> [(u64, u32); 2] {
        let c = q / 2;
        if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, CH) {
                for (cx, wx) in taps(x, CW) {
                    let at = (CHROMA_AT + cy * ROW + 2 * cx) as usize;
                    sums[0] += wx * wy * u32::from(bytes[at]);
                    sums[1] += wx * wy * u32::from(bytes[at + 1]);
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            let luma = u16::from(bytes[(y * ROW + x) as usize]);
            out.push(ycbcr_read(codes, conversion, [luma, cb, cr]));
        }
    }
    out
}

/// Metal's `read()` of the surface of seed 5 as a `420v` one without a matrix (an Apple M4 under
/// macOS 27.0.1): the FNV-1a digest of each texel's red, green and blue `v` as little-endian 16-bit
/// words.
const METAL_READ_DIGEST: u32 = 0x26e2_da15;

fn digest(texels: &[[u16; 3]]) -> u32 {
    texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|v| v.to_le_bytes()))
        .fold(2_166_136_261u32, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(16_777_619)
        })
}

/// The surface's facts for OSType `pixel_format` and `matrix`.
fn facts(pixel_format: u32, matrix: Option<SurfaceYCbCrMatrix>) -> IoSurfaceYCbCr {
    let facts = IoSurfaceYCbCr::new(pixel_format, 0, [W, H], CHROMA_AT, ROW);
    match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    }
}

/// The `YCBCR8_420_2P` texture of `surface`'s plane 0 with `facts`, of `usage`.
fn ycbcr_texture(
    surface: &Buffer,
    facts: Option<IoSurfaceYCbCr>,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    eight_bit_texture(surface, PixelFormat::YCBCR84202P, facts, usage)
}

/// The texture of `pf` (an 8-bit two-plane format) of `surface`'s plane 0 with `facts`, of
/// `usage`.
fn eight_bit_texture(
    surface: &Buffer,
    pf: PixelFormat,
    facts: Option<IoSurfaceYCbCr>,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = usage;
    d.resource_options = ResourceOptions::shared();
    let plane = IoSurfacePlane::new(0, ROW, true);
    let plane = match facts {
        Some(facts) => plane.with_ycbcr(facts),
        None => plane,
    };
    surface.new_texture_with_iosurface_plane(&d, plane)
}

/// A texture of `pf` and `size` over `surface`'s plane at `offset`, rows [`ROW`] apart, of
/// shader read and write usage.
fn plane_texture(surface: &Buffer, pf: PixelFormat, size: [u64; 2], offset: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, size[0], size[1], false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    surface
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(offset, ROW, true))
        .unwrap()
}

/// The `v` of `v / 2048` of each channel of `texels` (`float4`s), checked to be on the grid,
/// alpha checked to be 1.
fn grid(texels: &[u8]) -> Vec<[u16; 3]> {
    grid_of(texels, 2048)
}

/// The `v` of `v / d` of each channel of `texels` (`float4`s), checked to be on the grid `d`,
/// alpha checked to be 1.
fn grid_of(texels: &[u8], d: u32) -> Vec<[u16; 3]> {
    let d = d as f32;
    texels
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "alpha 1");
            std::array::from_fn(|i| {
                let v = c(i) * d;
                assert!(v == v.round() && (0.0..=d).contains(&v), "{v} off the grid");
                v as u16
            })
        })
        .collect()
}

/// Dispatch `state` over the texture's texels with `texture`, `out` and `sampler` bound, in a
/// new compute encoder of `command_buffer`.
fn dispatch(
    command_buffer: &CommandBuffer,
    state: &ComputePipelineState,
    texture: &Texture,
    out: &Buffer,
    sampler: Option<&SamplerState>,
    size: [u64; 2],
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    if let Some(sampler) = sampler {
        encoder.set_sampler_state(Some(sampler), 0).unwrap();
    }
    encoder
        .dispatch_threads(Size::new(size[0], size[1], 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
}

/// What `state` (reading or sampling texture 0 into buffer 0) reads of `texture`, through
/// `sampler`, in one command buffer of its own.
fn texels_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let out = device
        .new_buffer(16 * W * H, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(&command_buffer, state, texture, &out, sampler, [W, H]);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, 16 * W * H).unwrap()
}

/// A sampler of `filter` for both min and mag.
fn sampler(device: &Device, filter: SamplerMinMagFilter) -> SamplerState {
    let mut d = SamplerDescriptor::new();
    d.min_filter = filter;
    d.mag_filter = filter;
    device.new_sampler_state(&d).unwrap()
}

/// Linear samplers through each mip filter: a YCbCr texture has one level, which Metal's mip
/// filter leaves as it is.
fn linear_samplers(device: &Device) -> Vec<(SamplerMipFilter, SamplerState)> {
    [
        SamplerMipFilter::NotMipmapped,
        SamplerMipFilter::Nearest,
        SamplerMipFilter::Linear,
    ]
    .into_iter()
    .map(|mip| {
        let mut d = SamplerDescriptor::new();
        d.min_filter = SamplerMinMagFilter::Linear;
        d.mag_filter = SamplerMinMagFilter::Linear;
        d.mip_filter = mip;
        (mip, device.new_sampler_state(&d).unwrap())
    })
    .collect()
}

/// A kernel's `read()` and a nearest sample of every texel are the conversion of the surface's
/// planes, Metal's for the measured surface, for both ranges and each matrix (BT.601, BT.709 and
/// BT.2020); unannounced.
#[test]
fn ycbcr_textures_read_the_texels_metal_converts() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let conversions = [
            (VIDEO, None, YCbCrRange::Video, YCbCrMatrix::Bt601),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu601),
                YCbCrRange::Full,
                YCbCrMatrix::Bt601,
            ),
            (
                VIDEO,
                Some(SurfaceYCbCrMatrix::Itu709),
                YCbCrRange::Video,
                YCbCrMatrix::Bt709,
            ),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu709),
                YCbCrRange::Full,
                YCbCrMatrix::Bt709,
            ),
            (
                VIDEO,
                Some(SurfaceYCbCrMatrix::Itu2020),
                YCbCrRange::Video,
                YCbCrMatrix::Bt2020,
            ),
            (
                FULL,
                Some(SurfaceYCbCrMatrix::Itu2020),
                YCbCrRange::Full,
                YCbCrMatrix::Bt2020,
            ),
        ];
        for (seed, (ostype, matrix, range, model_matrix)) in (5..).zip(conversions) {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = ycbcr_texture(
                &surface,
                Some(facts(ostype, matrix)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &texture, None));
            assert_eq!(
                read,
                modelled(&bytes, range, model_matrix),
                "{range:?} {matrix:?}"
            );
            if seed == 5 {
                assert_eq!(digest(&read), METAL_READ_DIGEST, "Metal's reads");
            }
            let sampled = grid(&texels_of(
                device,
                &kernels.sample,
                &texture,
                Some(&nearest),
            ));
            assert_eq!(sampled, read, "a nearest sample is the texel");
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::FloatSampleFilter),
            None,
            "{notices:?}"
        );
    });
}

/// The fractions, in 1/256 steps, at which `sample_between` samples past each texel's centre:
/// `(0.90625, 0.71875)` texels past its corner.
const BETWEEN: [u32; 2] = [104, 56];

/// Metal's step of a linear sample of a `YCBCR8_420_2P` texture at fractions `[a, b]` past texel
/// `(x, y)`, from the texels' steps `read`: the taps `(x, y)` to `(x + 1, y + 1)` clamped to the
/// edge, an axis whose two taps address one texel weighed `(256, 0)`, each term `k · w` truncated
/// by `2^7`, then `(T + 2^8) >> 9`.
fn metal_step(read: &[[u16; 3]], (x, y): (u64, u64), [a, b]: [u32; 2], c: usize) -> u32 {
    let (x1, y1) = ((x + 1).min(W - 1), (y + 1).min(H - 1));
    let a = if x1 == x { 0 } else { a };
    let b = if y1 == y { 0 } else { b };
    let k = |x: u64, y: u64| u32::from(read[(y * W + x) as usize][c]);
    let taps = [(k(x, y), 256 - a, 256 - b), (k(x1, y), a, 256 - b)];
    let taps = taps
        .into_iter()
        .chain([(k(x, y1), 256 - a, b), (k(x1, y1), a, b)]);
    let t: u32 = taps.map(|(k, wx, wy)| (k * wx * wy) >> 7).sum();
    (t + 256) >> 9
}

/// The `half` bits of step `k / 2048` (`k` within `0..=2048`), which a `half` holds exactly.
fn half_of_step(k: u16) -> u16 {
    if k == 0 {
        return 0;
    }
    let lead = 15 - k.leading_zeros() as u16;
    // `k / 2048` is `2^(lead - 11)` times `k / 2^lead` in `[1, 2)`.
    let significand = match lead {
        11 => k >> 1,
        _ => k << (10 - lead),
    } & 0x3ff;
    ((lead + 4) << 10) | significand
}

/// A linear sample is Metal's filter of the exact texels, unannounced: at each texel's centre it
/// is the texel, and between texels Metal's step, on every device (the emulation of Metal's filter;
/// under MoltenVK, the slot's YCbCr variant).
#[test]
fn a_linear_sample_of_a_ycbcr_texture_is_metals() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let bytes = surface_bytes(9);
        let surface = buffer_with(device, &bytes).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let sampled = grid(&texels_of(device, &kernels.sample, &texture, Some(&linear)));
        assert_eq!(
            sampled, read,
            "a linear sample at a texel's centre is the texel"
        );
        let between = grid(&texels_of(
            device,
            &kernels.between,
            &texture,
            Some(&linear),
        ));
        for (i, got) in (0u64..).zip(&between) {
            let at = (i % W, i / W);
            let want: [u16; 3] = std::array::from_fn(|c| metal_step(&read, at, BETWEEN, c) as u16);
            assert_eq!(*got, want, "texel {at:?}");
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// A linear sample between the texels through a constexpr sampler and through a runtime sampler
/// slot of each mip filter, each with a `float` and a `half` result, is Metal's step (every value
/// on the grid `2048` is a `half`), and announces no filter notice: the path the consumer
/// measured off the grid, whose slot the texture's alpha-one swizzle had left on the device's
/// filter, and a runtime sampler's mip filter, which Metal's sample of the texture's one level does
/// not read.
#[test]
fn a_linear_sample_is_metals_through_each_sampler_and_result() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let at = |p: u64, f: u32, n: u64| (p as f32 + 0.5 + f as f32 / 256.0) / n as f32;
        let jobs: Vec<[f32; 4]> = (0..W * H)
            .map(|i| [at(i % W, BETWEEN[0], W), at(i / W, BETWEEN[1], H), 0.0, 0.0])
            .collect();
        let samplers = linear_samplers(device);
        let runtime = samplers
            .iter()
            .map(|(mip, s)| (Through::Runtime, Some((*mip, s))));
        for (through, runtime) in [(Through::Constexpr(LINEAR_EDGE), None)]
            .into_iter()
            .chain(runtime)
        {
            for half in [false, true] {
                let source = argument_linear_samples::kernel(false, through, false, half);
                let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
                let held = (&texture, Held::Slot, runtime.map(|(_, s)| s));
                let got = run_held(device, held, &state, &jobs, 1);
                let mip = runtime.map(|(mip, _)| mip);
                for (i, texel) in (0..W * H).zip(got.as_chunks::<4>().0.iter()) {
                    let at = (i % W, i / W);
                    let want: [f32; 4] = std::array::from_fn(|c| match c {
                        3 => 1.0,
                        c => metal_step(&read, at, BETWEEN, c) as f32 / 2048.0,
                    });
                    assert_eq!(
                        *texel, want,
                        "texel {at:?} through {through:?} {mip:?}, half {half}"
                    );
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// Synthetic AIR written for this lane: vertex `corner` places vertex `vid` at buffer 0's
/// `float4` `vid`; fragment `between_{runtime,constexpr}_{float,half}` samples texture 0 through
/// sampler slot 0, or a constexpr sampler of [`LINEAR_EDGE`]'s words, at [`BETWEEN`] past the
/// pixel's texel, with a `float` or a `half` result, and writes it (widened) to color 0.
fn fragments() -> String {
    let mut functions = String::new();
    let mut entries = Vec::new();
    let mut metadata = String::new();
    for (n, (constexpr, half)) in [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .enumerate()
    {
        let name = format!(
            "between_{}_{}",
            if constexpr { "constexpr" } else { "runtime" },
            if half { "half" } else { "float" }
        );
        let (element, ty) = if half {
            ("half", "v4f16")
        } else {
            ("float", "v4f32")
        };
        let (param, operand, args) = match constexpr {
            true => (
                "",
                "@__air_sampler_state",
                format!("!101, !{}", 110 + 10 * n),
            ),
            false => (
                ", ptr addrspace(2) %s",
                "%s",
                format!("!101, !{}, !103", 110 + 10 * n),
            ),
        };
        let widen = match half {
            true => "  %h = extractvalue { <4 x half>, i8 } %r, 0\n  %v = fpext <4 x half> %h to <4 x float>\n",
            false => "  %v = extractvalue { <4 x float>, i8 } %r, 0\n",
        };
        functions.push_str(&format!(
            r#"define <4 x float> @{name}(<4 x float> %pos, ptr addrspace(1) %t{param}) {{
  %xy = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %c = fadd <2 x float> %xy, <float 4.062500e-01, float 2.187500e-01>
  %uv = fdiv <2 x float> %c, <float {W}.0, float {H}.0>
  %r = call {{ <4 x {element}>, i8 }} @air.sample_texture_2d.{ty}(ptr addrspace(1) %t, ptr addrspace(2) {operand}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
{widen}  ret <4 x float> %v
}}

"#
        ));
        let id = 110 + 10 * n;
        entries.push(format!("!{}", id + 1));
        metadata.push_str(&format!(
            "!{} = !{{ptr @{name}, !102, !{}}}\n!{id} = !{{i32 1, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<{element}, sample>\", !\"air.arg_name\", !\"t\"}}\n!{} = !{{{args}}}\n",
            id + 1,
            id + 2,
            id + 2
        ));
    }
    format!(
        r#"; Owned synthetic shaders.
@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {LINEAR_EDGE}, i64 0], align 8

define <{{ <4 x float> }}> @corner(i32 %vid, ptr addrspace(2) %p) {{
  %i = zext i32 %vid to i64
  %q = getelementptr inbounds <4 x float>, ptr addrspace(2) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(2) %q, align 16
  %o = insertvalue <{{ <4 x float> }}> undef, <4 x float> %v, 0
  ret <{{ <4 x float> }}> %o
}}

{functions}declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare {{ <4 x half>, i8 }} @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.vertex = !{{!90}}
!air.fragment = !{{{fragments}}}
!air.sampler_states = !{{!99}}
!air.version = !{{!97}}
!air.language_version = !{{!98}}
!90 = !{{ptr @corner, !91, !92}}
!91 = !{{!93}}
!92 = !{{!94, !95}}
!93 = !{{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!94 = !{{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}}
!95 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}}
!97 = !{{i32 2, i32 8, i32 0}}
!98 = !{{!"Metal", i32 4, i32 0, i32 0}}
!99 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!100 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!101 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!102 = !{{!100}}
!103 = !{{i32 2, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}}
{metadata}"#,
        fragments = entries.join(", ")
    )
}

/// What a draw of `fragment` (over the whole [`W`]x[`H`] `RGBA32Float` target) samples of
/// `texture`, through `sampler` at sampler slot 0 where given: each pixel's `float4`.
fn drawn_between(
    device: &Device,
    library: &Library,
    fragment: &str,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<[f32; 4]> {
    drawn_after(device, library, fragment, (texture, sampler), |_, _| {})
}

/// [`drawn_between`]'s draw, in a command buffer of a new queue in which `before` first encodes
/// its own work (given the queue, for work of another command buffer committed first).
fn drawn_after(
    device: &Device,
    library: &Library,
    fragment: &str,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    before: impl FnOnce(&CommandQueue, &CommandBuffer),
) -> Vec<[f32; 4]> {
    let pipeline = between_state(device, library, fragment);
    drawn_with(device, &pipeline, (texture, sampler), before)
}

/// The render pipeline state of vertex `corner` and `fragment` into one `RGBA32Float` color.
fn between_state(device: &Device, library: &Library, fragment: &str) -> RenderPipelineState {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("corner"));
    descriptor.fragment = Some(function(fragment));
    descriptor.color_attachments[0] =
        Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// [`drawn_after`]'s draw through `pipeline`.
fn drawn_with(
    device: &Device,
    pipeline: &RenderPipelineState,
    (texture, sampler): (&Texture, Option<&SamplerState>),
    before: impl FnOnce(&CommandQueue, &CommandBuffer),
) -> Vec<[f32; 4]> {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, W, H, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    let corners: Vec<u8> = [
        [-1.0f32, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ]
    .iter()
    .flat_map(|&[x, y]| [x, y, 0.0, 1.0])
    .flat_map(|v| v.to_le_bytes())
    .collect();
    let corners = buffer_with(device, &corners).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    before(&queue, &command_buffer);
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(pipeline).unwrap();
    encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    if let Some(sampler) = sampler {
        encoder
            .set_fragment_sampler_state(Some(sampler), 0)
            .unwrap();
    }
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let mut bytes = vec![0u8; (16 * W * H) as usize];
    target
        .get_bytes(&mut bytes, 16 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
        .unwrap();
    bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            std::array::from_fn(|i| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap()))
        })
        .collect()
}

/// A fragment function's linear sample between the texels, through a runtime sampler slot of
/// each mip filter and through a constexpr sampler, each with a `float` and a `half` result, is
/// Metal's step, and announces no filter notice: the fragment twin of
/// `a_linear_sample_is_metals_through_each_sampler_and_result`. The pixel `(x, y)`'s centre
/// plus `(0.40625, 0.21875)` is the kernel's `(x, y) + (0.90625, 0.71875)`.
#[test]
fn a_fragments_linear_sample_is_metals_through_each_sampler_and_result() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        let samplers = linear_samplers(device);
        let runtime = ["between_runtime_float", "between_runtime_half"]
            .into_iter()
            .flat_map(|f| samplers.iter().map(move |(mip, s)| (f, Some((*mip, s)))));
        let constexpr = ["between_constexpr_float", "between_constexpr_half"].map(|f| (f, None));
        for (fragment, runtime) in runtime.chain(constexpr) {
            let got = drawn_between(
                device,
                &library,
                fragment,
                &texture,
                runtime.map(|(_, s)| s),
            );
            let mip = runtime.map(|(mip, _)| mip);
            for (i, texel) in (0..W * H).zip(&got) {
                let at = (i % W, i / W);
                let want: [f32; 4] = std::array::from_fn(|c| match c {
                    3 => 1.0,
                    c => metal_step(&read, at, BETWEEN, c) as f32 / 2048.0,
                });
                assert_eq!(*texel, want, "pixel {at:?} of {fragment} {mip:?}");
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// How far the device's own linear filter of a two-plane YCbCr texture's texels may land from
/// Metal's filter of the format at coordinates on the 1/256 grid, in codes of `1/255`: the bound
/// `PrecisionNote::DeviceSampling` states for such a texture there (docs/FAST-PATHS.md), 0.498 on
/// NVIDIA and 0.125 on RADV, ANV and lavapipe; through Metal's own sampler one step of the grid
/// `1/2048`, 0.125 (`ycbcr-filter-not-exact`).
const DEVICE_FILTER_BOUND: f32 = 0.5;

/// Under native sampling (`SamplingPolicy::NativeDevice`) and the default
/// (`YCbCrFilterPolicy::Device`), a fragment function's linear sample of a `YCBCR8_420_2P` texture
/// through a runtime sampler, with a `float` and a `half` result, takes the pipeline a draw binding
/// any other texture takes: the draw binding it makes no pipeline after the one that bound an
/// `RGBA16Float` texture of the same texels, and its samples are that texture's bit for bit (the
/// device's filter of them), within [`DEVICE_FILTER_BOUND`] of Metal's step. On a device that
/// samples natively the module's `device-sampling` is announced and no filter notice; on one that
/// samples through Metal's own sampler `ycbcr-filter-not-exact`, once, naming the slot and the
/// format, and no `device-sampling`.
#[test]
fn a_ycbcr_slot_takes_the_devices_filter_in_the_pipeline_its_creation_made() {
    ycbcr_draws_under(YCbCrFilterPolicy::Device);
}

/// [`a_ycbcr_slot_takes_the_devices_filter_in_the_pipeline_its_creation_made`] under Metal's
/// filter of the format (`YCbCrFilterPolicy::Exact`): on every device the draw binding the
/// texture takes the slot's YCbCr variant, one pipeline more, and its samples are Metal's step
/// exactly, announced by no filter notice.
#[test]
fn a_ycbcr_slot_takes_metals_filter_under_the_exact_switch() {
    ycbcr_draws_under(YCbCrFilterPolicy::Exact);
}

/// The draws of [`a_ycbcr_slot_takes_the_devices_filter_in_the_pipeline_its_creation_made`] on a
/// device that samples natively, under `filter`.
fn ycbcr_draws_under(filter: YCbCrFilterPolicy) {
    let native = |options: &mut DeviceOptions| {
        options.policy.sampling = SamplingPolicy::NativeDevice;
        options.policy.ycbcr_filter = filter;
    };
    on_device_with(native, |device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        let d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
        let twin = device.new_texture(&d).unwrap();
        let bytes: Vec<u8> = read
            .iter()
            .flat_map(|t| [t[0], t[1], t[2], 2048].map(half_of_step))
            .flat_map(u16::to_le_bytes)
            .collect();
        twin.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, 8 * W, 0)
            .unwrap();
        let metal = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        let exact = filter == YCbCrFilterPolicy::Exact;
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        for fragment in ["between_runtime_float", "between_runtime_half"] {
            let pipeline = between_state(device, &library, fragment);
            let plain = drawn_with(device, &pipeline, (&twin, Some(&linear)), |_, _| {});
            let before = device.work_counts().variant_pipelines_made;
            let got = drawn_with(device, &pipeline, (&texture, Some(&linear)), |_, _| {});
            let made = device.work_counts().variant_pipelines_made - before;
            assert_eq!(
                made,
                u64::from(exact),
                "{fragment}: pipelines the YCbCr draw made"
            );
            let mut worst = 0f32;
            for (i, texel) in (0..W * H).zip(&got) {
                let at = (i % W, i / W);
                for (c, &value) in texel.iter().enumerate().take(3) {
                    let want = metal_step(&read, at, BETWEEN, c) as f32 / 2048.0;
                    match exact {
                        true => assert_eq!(value, want, "pixel {at:?} of {fragment}"),
                        false => worst = worst.max((value - want).abs() * 255.0),
                    }
                }
                assert_eq!(texel[3], 1.0, "pixel {at:?} of {fragment}");
            }
            if !exact {
                println!(
                    "YCBCR-DEVICE-FILTER {} {fragment}: largest difference from Metal's step \
                     {worst} codes",
                    device.name()
                );
                assert!(
                    got == plain,
                    "{fragment}: the device's filter of the same texels"
                );
                assert!(worst <= DEVICE_FILTER_BOUND, "{fragment}: {worst} codes");
            }
        }
        let notices = messages.take_notices();
        let announced = counts(&notices);
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
        ] {
            assert_eq!(announced.get(&kind), None, "{announced:?}");
        }
        let ycbcr = announced.get(&Degradation::YCbCrFilterNotExact);
        match metal && !exact {
            true => {
                assert_eq!(ycbcr, Some(&1), "{announced:?}");
                let notice = notices
                    .iter()
                    .filter_map(|m| m.degradation.as_ref())
                    .find(|n| n.degradation == Degradation::YCbCrFilterNotExact)
                    .unwrap();
                let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
                assert!(notice.reason.contains(&named), "{notice:?}");
            }
            false => assert_eq!(ycbcr, None, "{announced:?}"),
        }
        assert_eq!(
            announced.contains_key(&Degradation::DeviceSampling),
            !metal,
            "{announced:?}"
        );
    });
}

/// The kind a filtered sample of a YCbCr texture that does not take Metal's exact filter takes
/// ([`a_ycbcr_sample_the_exact_filter_does_not_reach_is_announced`] on a device that samples
/// through Metal's own sampler, [`a_ycbcr_sample_through_an_unmeasured_filter_is_announced`] on
/// every device).
pub(super) const YCBCR_FILTERS: [Degradation; 1] = [Degradation::YCbCrFilterNotExact];

/// A linear sample of a YCbCr texture made with a swizzle that moves its channels, `(B, G, R,
/// A)`. On a device that samples through Metal's own sampler (`VkProperty::MetalSampler`) the
/// view composes the swizzle and the class drops the field the emulation of Metal's filter reads
/// (`TexelFormat::swizzled`): the sample takes the device's filter, Metal's own of the converted
/// texels, the class measured by no family keeping the device's sample under the slot's YCbCr
/// variant, bit for bit an `RGBA16Float` texture's of the same texels under the same swizzle, and
/// announces [`YCBCR_FILTERS`] once, naming the slot and the format, in place of the generic filter
/// notes, which it does not announce. Elsewhere the class is a swizzled packed one
/// (`TexelFormat::packed_exact`): the slot's view takes no swizzle, the emulation filters the
/// unswizzled texels exactly and the function swizzles the result, so the sample is Metal's step of
/// each channel the swizzle reads (Metal's swizzle permutes the texel it filters), announced by
/// nothing.
#[test]
fn a_ycbcr_sample_the_exact_filter_does_not_reach_is_announced() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let swizzle = TextureSwizzleChannels::new(
            TextureSwizzle::Blue,
            TextureSwizzle::Green,
            TextureSwizzle::Red,
            TextureSwizzle::Alpha,
        );
        let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        d.swizzle = swizzle;
        let plane = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts(VIDEO, None));
        let texture = surface.new_texture_with_iosurface_plane(&d, plane).unwrap();
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let sampled = texels_of(device, &kernels.between, &texture, Some(&linear));
        let metal = device
            .compile_device()
            .profile()
            .enabled()
            .properties
            .contains(&VkProperty::MetalSampler);
        if metal {
            let plain = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &plain, None));
            let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            d.swizzle = swizzle;
            let twin = device.new_texture(&d).unwrap();
            let bytes: Vec<u8> = read
                .iter()
                .flat_map(|t| [t[0], t[1], t[2], 2048].map(half_of_step))
                .flat_map(u16::to_le_bytes)
                .collect();
            twin.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &bytes, 8 * W, 0)
                .unwrap();
            let want = texels_of(device, &kernels.between, &twin, Some(&linear));
            assert!(
                sampled == want,
                "the device's filter of the converted texels"
            );
        } else {
            let plain = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let read = grid(&texels_of(device, &kernels.read, &plain, None));
            // The swizzle's source channel of each color channel: blue, green, red.
            let from = [2, 1, 0];
            let (mut compared, mut differ) = (0usize, Vec::new());
            for (i, texel) in (0u64..).zip(sampled.as_chunks::<16>().0.iter()) {
                let at = (i % W, i / W);
                let got =
                    |c: usize| f32::from_le_bytes(texel[4 * c..4 * c + 4].try_into().unwrap());
                let want: [f32; 4] = std::array::from_fn(|c| match c {
                    3 => 1.0,
                    c => metal_step(&read, at, BETWEEN, from[c]) as f32 / 2048.0,
                });
                for (c, want) in want.into_iter().enumerate() {
                    compared += 1;
                    if got(c).to_bits() != want.to_bits() {
                        differ.push(format!("texel {at:?} channel {c}: {} for {want}", got(c)));
                    }
                }
            }
            println!(
                "YCBCR-SWIZZLED {}: {compared} values compared with Metal's step swizzled, {} \
                 differ {:?}",
                device.name(),
                differ.len(),
                &differ[..differ.len().min(16)]
            );
            assert!(differ.is_empty(), "Metal's step, swizzled: {differ:?}");
        }
        let notices = messages.take_notices();
        let counts = counts(&notices);
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::HalfSampleFilter,
        ] {
            assert_eq!(counts.get(&kind), None, "{counts:?}");
        }
        if !metal {
            assert_eq!(counts.get(&YCBCR_FILTERS[0]), None, "{counts:?}");
            return;
        }
        assert_eq!(counts.get(&YCBCR_FILTERS[0]), Some(&1), "{counts:?}");
        let notice = notices
            .iter()
            .filter_map(|m| m.degradation.as_ref())
            .find(|n| n.degradation == YCBCR_FILTERS[0])
            .unwrap();
        let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
        assert!(notice.reason.contains(&named), "{notice:?}");
    });
}

/// A filtered sample of a YCbCr texture through a sampler Metal's exact filter is not measured
/// for, which the device filters, is announced, never silent: through a runtime sampler slot an
/// anisotropic one (16) and one whose min and mag filters differ announce [`YCBCR_FILTERS`] once,
/// naming the slot and the format, and no generic filter notice; through a constexpr sampler an
/// anisotropic one, one with a linear mip filter and one whose filters differ announce the
/// generic `float-sample-filter` once (the constexpr note holds whatever the slot holds, so it
/// names no format). Each case runs on a device of its own.
#[test]
fn a_ycbcr_sample_through_an_unmeasured_filter_is_announced() {
    let anisotropic = |d: &mut SamplerDescriptor| d.max_anisotropy = 16;
    let min_nearest = |d: &mut SamplerDescriptor| d.min_filter = SamplerMinMagFilter::Nearest;
    let mag_nearest = |d: &mut SamplerDescriptor| d.mag_filter = SamplerMinMagFilter::Nearest;
    type Adjust = fn(&mut SamplerDescriptor);
    let runtime: [(&str, Adjust); 3] = [
        ("anisotropic", anisotropic),
        ("min nearest", min_nearest),
        ("mag nearest", mag_nearest),
    ];
    // The constexpr words: `LINEAR_EDGE` with a max anisotropy of 16 (bits 20 to 23 hold it
    // less 1), a linear mip filter (bits 13 and 14), a nearest min filter (bits 11 and 12), or a
    // nearest mag filter (bits 9 and 10).
    let constexpr: [(&str, u64); 4] = [
        ("anisotropic", LINEAR_EDGE | 15 << 20),
        ("linear mip filter", LINEAR_EDGE | 2 << 13),
        ("min nearest", LINEAR_EDGE & !(3 << 11)),
        ("mag nearest", LINEAR_EDGE & !(3 << 9)),
    ];
    let cases = runtime
        .into_iter()
        .map(|(name, adjust)| (name, Through::Runtime, Some(adjust)))
        .chain(
            constexpr
                .into_iter()
                .map(|(name, words)| (name, Through::Constexpr(words), None)),
        );
    for (name, through, adjust) in cases {
        on_device(|device, messages| {
            let surface = buffer_with(device, &surface_bytes(9)).unwrap();
            let texture = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let sampler = adjust.map(|adjust| {
                let mut d = SamplerDescriptor::new();
                d.min_filter = SamplerMinMagFilter::Linear;
                d.mag_filter = SamplerMinMagFilter::Linear;
                adjust(&mut d);
                device.new_sampler_state(&d).unwrap()
            });
            let at = |p: u64, f: u32, n: u64| (p as f32 + 0.5 + f as f32 / 256.0) / n as f32;
            let jobs: Vec<[f32; 4]> = (0..W * H)
                .map(|i| [at(i % W, BETWEEN[0], W), at(i / W, BETWEEN[1], H), 0.0, 0.0])
                .collect();
            let source = argument_linear_samples::kernel(false, through, false, false);
            let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
            run_held(
                device,
                (&texture, Held::Slot, sampler.as_ref()),
                &state,
                &jobs,
                1,
            );
            let notices = messages.take_notices();
            let counts = counts(&notices);
            let case = format!("{name} through {through:?}: {counts:?}");
            match through {
                Through::Runtime => {
                    assert_eq!(counts.get(&YCBCR_FILTERS[0]), Some(&1), "{case}");
                    for kind in [
                        Degradation::FloatSampleFilter,
                        Degradation::HalfSampleFilter,
                    ] {
                        assert_eq!(counts.get(&kind), None, "{case}");
                    }
                    let notice = notices
                        .iter()
                        .filter_map(|m| m.degradation.as_ref())
                        .find(|n| n.degradation == YCBCR_FILTERS[0])
                        .unwrap();
                    let named = format!("texture 0 bound as {:?}", PixelFormat::YCBCR84202P);
                    assert!(notice.reason.contains(&named), "{case}: {notice:?}");
                }
                _ => {
                    assert_eq!(
                        counts.get(&Degradation::FloatSampleFilter),
                        Some(&1),
                        "{case}"
                    );
                }
            }
        });
    }
}

/// A function that samples textures through handles (an argument buffer's) reads its sampler
/// from memory, where no slot's specialization can give it Metal's exact filter of a YCbCr
/// texture it reaches, nor the texture's address-mode map: a command whose function samples
/// handles and reaches a YCbCr texture, held by the argument buffer and made resident, is
/// refused, typed, naming the texture's map (`MappedTextureSampling { through: Handles }`), whether
/// the sample's sampler is a constexpr one, a runtime slot's or the argument buffer's, and nothing
/// is recorded.
#[test]
fn a_ycbcr_texture_sampled_through_handles_is_refused() {
    on_device(|device, _| {
        let surface = buffer_with(device, &surface_bytes(9)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        assert!(!texture.address_mode_map().is_identity());
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let input = buffer_with(device, &[0u8; 16 * 8]).unwrap();
        let out = device
            .new_buffer(16 * 8, ResourceOptions::shared())
            .unwrap();
        for through in [
            Through::Constexpr(LINEAR_EDGE),
            Through::Runtime,
            Through::Argument,
        ] {
            let source = argument_linear_samples::kernel(true, through, false, false);
            let state = pipeline_of(device, &llvm_assemble_text(&source).unwrap(), "k");
            let held = match through {
                Through::Argument => linear.gpu_resource_id().0,
                _ => 0,
            };
            let ids = [texture.gpu_resource_id().0, held];
            let words: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let argument = buffer_with(device, &words).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&state).unwrap();
            encoder
                .use_resource(texture.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder.set_buffer(Some(&input), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder.set_buffer(Some(&argument), 0, 2).unwrap();
            if let Through::Runtime = through {
                encoder.set_sampler_state(Some(&linear), 0).unwrap();
            }
            let dispatched = encoder.dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                dispatched,
                Err(Error::NotImplemented(RuntimeGap::MappedTextureSampling {
                    map: texture.address_mode_map(),
                    through: MappedSampling::Handles,
                })),
                "{through:?}"
            );
            assert!(command_buffer.executions().is_empty(), "{through:?}");
        }
    });
}

/// The bytes of a surface whose luma bytes are all `y` and chroma pairs all `(cb, cr)`.
fn uniform(y: u8, cb: u8, cr: u8) -> Vec<u8> {
    let mut bytes = vec![0xa5; LENGTH as usize];
    for row in 0..H {
        let at = (row * ROW) as usize;
        bytes[at..at + W as usize].fill(y);
    }
    for row in 0..CH {
        for x in 0..CW {
            let at = (CHROMA_AT + row * ROW + 2 * x) as usize;
            bytes[at] = cb;
            bytes[at + 1] = cr;
        }
    }
    bytes
}

/// How a round's planes are written before the texture is read.
#[derive(Clone, Copy, Debug)]
enum Writer {
    /// Kernels writing the `R8Unorm` and `RG8Unorm` textures over the planes, in a command
    /// buffer committed before the reader's.
    EarlierCommandBuffer,
    /// The same kernels in encoders before the reader's, in its command buffer.
    EarlierEncoder,
    /// The same kernels as dispatches before the reader's, in its encoder.
    EarlierDispatch,
    /// Blits from a buffer into the two textures, before the reader's encoder.
    Blit,
    /// The CPU, between command buffers.
    Cpu,
}

/// The texture follows each way its planes' bytes are written (the module docs): every round
/// writes new codes, and the texture then reads their conversion everywhere.
#[test]
fn ycbcr_textures_follow_writes_of_their_planes() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let surface = buffer_with(device, &uniform(0, 0, 0)).unwrap();
        let texture = ycbcr_texture(
            &surface,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let luma = plane_texture(&surface, PixelFormat::R8Unorm, [W, H], 0);
        let chroma = plane_texture(&surface, PixelFormat::RG8Unorm, [CW, CH], CHROMA_AT);
        let queue = device.new_command_queue().unwrap();
        let out = device
            .new_buffer(16 * W * H, ResourceOptions::shared())
            .unwrap();
        let writers = [
            Writer::EarlierCommandBuffer,
            Writer::EarlierEncoder,
            Writer::EarlierDispatch,
            Writer::Blit,
            Writer::Cpu,
        ];
        // Before any write: the zeroed planes' texels.
        let first = grid(&texels_of(device, &kernels.read, &texture, None));
        let zero = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, 0, 0, 0);
        assert!(first.iter().all(|t| *t == zero));
        for (round, writer) in writers.into_iter().enumerate() {
            let k = round as u8;
            let (y, cb, cr) = (40 + 37 * k, 200 - 23 * k, 60 + 31 * k);
            let code = |c: u8| f32::from(c) / 255.0;
            let fill = |texture: &Texture, value: [f32; 4]| {
                let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
                (texture.clone(), buffer_with(device, &bytes).unwrap())
            };
            let fills = [
                (fill(&luma, [code(y), 0.0, 0.0, 1.0]), [W, H]),
                (fill(&chroma, [code(cb), code(cr), 0.0, 1.0]), [CW, CH]),
            ];
            let command_buffer = queue.command_buffer().unwrap();
            let write_kernels = |command_buffer: &CommandBuffer| {
                for ((texture, value), size) in &fills {
                    dispatch(command_buffer, &kernels.fill, texture, value, None, *size);
                }
            };
            match writer {
                Writer::EarlierCommandBuffer => {
                    let earlier = queue.command_buffer().unwrap();
                    write_kernels(&earlier);
                    earlier.commit().unwrap();
                }
                Writer::EarlierEncoder => write_kernels(&command_buffer),
                Writer::EarlierDispatch => {}
                Writer::Blit => {
                    let source = buffer_with(device, &uniform(y, cb, cr)).unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    for (texture, offset, size) in
                        [(&luma, 0, [W, H]), (&chroma, CHROMA_AT, [CW, CH])]
                    {
                        blit.copy_from_buffer_to_texture(
                            &source,
                            offset,
                            ROW,
                            0,
                            Size::new(size[0], size[1], 1),
                            texture,
                            0,
                            0,
                            Origin::new(0, 0, 0),
                            BlitOption::None,
                        )
                        .unwrap();
                    }
                    blit.end_encoding().unwrap();
                }
                Writer::Cpu => {
                    let bytes = uniform(y, cb, cr);
                    let contents = surface.contents().unwrap();
                    // SAFETY: a Shared buffer of `LENGTH` bytes no command uses now (every
                    // round waits for its command buffer).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            contents.as_ptr(),
                            LENGTH as usize,
                        )
                    };
                }
            }
            {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                if matches!(writer, Writer::EarlierDispatch) {
                    for ((texture, value), size) in &fills {
                        encoder.set_compute_pipeline_state(&kernels.fill).unwrap();
                        encoder.set_texture(Some(texture), 0).unwrap();
                        encoder.set_buffer(Some(value), 0, 0).unwrap();
                        encoder
                            .dispatch_threads(Size::new(size[0], size[1], 1), Size::new(8, 8, 1))
                            .unwrap();
                    }
                }
                encoder.set_compute_pipeline_state(&kernels.read).unwrap();
                encoder.set_texture(Some(&texture), 0).unwrap();
                encoder.set_buffer(Some(&out), 0, 0).unwrap();
                encoder
                    .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
                    .unwrap();
                encoder.end_encoding().unwrap();
            }
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let want = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, y, cb, cr);
            let got = grid(&read_buffer(&out, 0, 16 * W * H).unwrap());
            let stale = got.iter().filter(|t| **t != want).count();
            assert_eq!(stale, 0, "{writer:?}: {stale} texels not the new codes'");
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::FloatSampleFilter),
            None,
            "{notices:?}"
        );
    });
}

/// A draw samples the planes as written before it (the render pass twin of
/// [`ycbcr_textures_follow_writes_of_their_planes`]; `shadow.rs`'s render passes): a fragment
/// function's linear sample of a `YCBCR8_420_2P` texture over a Shared buffer, and over a no-copy
/// buffer of the caller's memory, reads the conversion of codes written by kernels in an earlier
/// command buffer and in an earlier encoder of its own, by a blit, and by the CPU before the
/// commit. The planes hold one code each, so every sample is that texel.
#[test]
fn a_draw_samples_the_planes_as_written_before_it() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&fragments()).unwrap())
            .unwrap();
        for no_copy in [false, true] {
            let zeroed = uniform(0, 0, 0);
            let surface = match no_copy {
                false => buffer_with(device, &zeroed).unwrap(),
                true => {
                    let len = host_page().max(LENGTH as usize);
                    let mapping = Mapping::anonymous(len);
                    mapping.write(0, &zeroed);
                    let pointer = mapping.pointer();
                    let release: Deallocator = Box::new(move |_, _| drop(mapping));
                    // SAFETY: the mapping stays mapped until the deallocator, which owns it, runs.
                    let made = unsafe {
                        host_memory::no_copy(
                            device,
                            pointer,
                            len,
                            ResourceOptions::default(),
                            Some(release),
                        )
                    };
                    match made {
                        Ok(buffer) => buffer,
                        Err(error) => {
                            assert!(host_memory::not_importable(&error), "{error:?}");
                            continue;
                        }
                    }
                }
            };
            let texture = ycbcr_texture(
                &surface,
                Some(facts(VIDEO, None)),
                TextureUsage::SHADER_READ,
            )
            .unwrap();
            let luma = plane_texture(&surface, PixelFormat::R8Unorm, [W, H], 0);
            let chroma = plane_texture(&surface, PixelFormat::RG8Unorm, [CW, CH], CHROMA_AT);
            for (round, writer) in [
                Writer::EarlierCommandBuffer,
                Writer::EarlierEncoder,
                Writer::Blit,
                Writer::Cpu,
            ]
            .into_iter()
            .enumerate()
            {
                let k = round as u8 + 4 * u8::from(no_copy);
                let (y, cb, cr) = (40 + 29 * k, 200 - 17 * k, 60 + 23 * k);
                let code = |c: u8| f32::from(c) / 255.0;
                let fill = |texture: &Texture, value: [f32; 4]| {
                    let bytes: Vec<u8> = value.iter().flat_map(|v| v.to_le_bytes()).collect();
                    (texture.clone(), buffer_with(device, &bytes).unwrap())
                };
                let fills = [
                    (fill(&luma, [code(y), 0.0, 0.0, 1.0]), [W, H]),
                    (fill(&chroma, [code(cb), code(cr), 0.0, 1.0]), [CW, CH]),
                ];
                let write_kernels = |command_buffer: &CommandBuffer| {
                    for ((texture, value), size) in &fills {
                        dispatch(command_buffer, &kernels.fill, texture, value, None, *size);
                    }
                };
                let source = buffer_with(device, &uniform(y, cb, cr)).unwrap();
                let before = |queue: &CommandQueue, command_buffer: &CommandBuffer| match writer {
                    Writer::EarlierCommandBuffer => {
                        let earlier = queue.command_buffer().unwrap();
                        write_kernels(&earlier);
                        earlier.commit().unwrap();
                    }
                    Writer::EarlierEncoder => write_kernels(command_buffer),
                    Writer::Blit => {
                        let mut blit = command_buffer.blit_command_encoder().unwrap();
                        for (texture, offset, size) in
                            [(&luma, 0, [W, H]), (&chroma, CHROMA_AT, [CW, CH])]
                        {
                            blit.copy_from_buffer_to_texture(
                                &source,
                                offset,
                                ROW,
                                0,
                                Size::new(size[0], size[1], 1),
                                texture,
                                0,
                                0,
                                Origin::new(0, 0, 0),
                                BlitOption::None,
                            )
                            .unwrap();
                        }
                        blit.end_encoding().unwrap();
                    }
                    Writer::Cpu => {
                        let bytes = uniform(y, cb, cr);
                        let contents = surface.contents().unwrap();
                        // SAFETY: the surface's first `LENGTH` bytes, which no command uses now
                        // (every round waits for its command buffer).
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr(),
                                contents.as_ptr(),
                                LENGTH as usize,
                            )
                        };
                    }
                    Writer::EarlierDispatch => unreachable!("a draw's round"),
                };
                let got = drawn_after(
                    device,
                    &library,
                    "between_constexpr_float",
                    (&texture, None),
                    before,
                );
                let want = ycbcr_texel(YCbCrRange::Video, YCbCrMatrix::Bt601, y, cb, cr);
                let want: [f32; 4] = std::array::from_fn(|c| match c {
                    3 => 1.0,
                    c => f32::from(want[c]) / 2048.0,
                });
                let stale = got.iter().filter(|t| **t != want).count();
                assert_eq!(
                    stale, 0,
                    "{writer:?}, no-copy {no_copy}: {stale} pixels not the new codes' ({:?})",
                    got[0]
                );
            }
        }
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&Degradation::YCbCrFilterNotExact),
            None,
            "{notices:?}"
        );
    });
}

/// A kernel's writes to the texture are dropped, as Metal drops them: both planes keep their
/// bytes. `get_bytes` reads the luma rows.
#[test]
fn ycbcr_texture_writes_are_dropped_and_its_bytes_are_luma() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let bytes = surface_bytes(11);
        let surface = buffer_with(device, &bytes).unwrap();
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let texture = ycbcr_texture(&surface, Some(facts(VIDEO, None)), usage).unwrap();
        let value = buffer_with(device, &[0x3f; 16]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        dispatch(
            &command_buffer,
            &kernels.fill,
            &texture,
            &value,
            None,
            [W, H],
        );
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(read_buffer(&surface, 0, LENGTH).unwrap(), bytes);
        let mut luma = vec![0; (W * H) as usize];
        texture
            .get_bytes(&mut luma, W, 0, Region::new_2d(0, 0, W, H), 0, 0)
            .unwrap();
        let want: Vec<u8> = (0..H)
            .flat_map(|y| bytes[(y * ROW) as usize..(y * ROW + W) as usize].to_vec())
            .collect();
        assert_eq!(luma, want);
    });
}

/// The texture of `pf` over plane 0 of `surface`, as [`family_texture`] makes it, of shader read
/// and write usage.
fn written_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMAQ_AT, ROWQ);
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROWQ, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// A 32-bit word of `random`.
fn random_word(random: &mut Random) -> u32 {
    u32::from_le_bytes(std::array::from_fn(|_| random.next()))
}

/// A `float` channel a test writes, of `random`: on the grid of 1/2048, between two of its steps,
/// any multiple of 2^-24 in `[0, 1)`, one out of `[0, 1]`, or a NaN, an infinity, a zero, a
/// subnormal, 1 or a value beside it, or a huge one.
fn written_channel(random: &mut Random) -> f32 {
    const SPECIALS: [f32; 12] = [
        f32::NAN,
        f32::NEG_INFINITY,
        f32::INFINITY,
        -0.0,
        0.0,
        1.0,
        1.000_000_1,
        0.999_999_94,
        1.0e-45,
        1.0e30,
        -1.0e30,
        0.999_755_86,
    ];
    let unit = |random: &mut Random| (random_word(random) % (1 << 24)) as f32 / (1 << 24) as f32;
    match random.next() % 8 {
        0..=2 => (random_word(random) % 2049) as f32 / 2048.0,
        3 => ((random_word(random) % 2048) as f32 + 0.5) / 2048.0,
        4 => unit(random),
        5 => unit(random) * 4.0 - 1.5,
        6 => SPECIALS[random_word(random) as usize % SPECIALS.len()],
        _ => (random_word(random) % 1024) as f32 / 1023.0,
    }
}

/// A `half` channel's bits a test writes, of `random`: one of `[0, 1]`, any bits (NaNs,
/// infinities, negative values, values past 1 and subnormals among them), or a NaN, an
/// infinity, a zero, a subnormal, or 1 or a value beside it.
fn written_half(random: &mut Random) -> u16 {
    const SPECIALS: [u16; 9] = [
        0x7e00, 0xfe00, 0x7c00, 0xfc00, 0x8000, 0x0001, 0x3bff, 0x3c00, 0x3c01,
    ];
    let word = random_word(random);
    match random.next() % 6 {
        0..=3 => (word % 0x3c01) as u16,
        4 => word as u16,
        _ => SPECIALS[word as usize % SPECIALS.len()],
    }
}

/// The value of `half` bits `h`, widened exactly.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f32::from(h & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 if fraction == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

/// `bytes` (a surface of `codes` laid out as [`family_bytes`] lays it) after writes of `texels`
/// (each texel's `[r, g, b]`, row by row, `None` where unwritten) as Metal stores them over
/// `conversion` (`metal2vulkan::bind::ycbcr_write`): each written texel's luma code at its place
/// in its row of the luma plane, and its Cb and Cr codes at its pair in its row of the chroma
/// plane, a code a byte or the high bits of a little-endian 16-bit word; every other byte as it
/// was.
fn written_bytes(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
    texels: &[Option<[f32; 3]>],
) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let unit = u64::from(codes.bytes);
    let mut put = |at: u64, code: u16| {
        let at = at as usize;
        match codes.bytes {
            1 => out[at] = code as u8,
            _ => out[at..at + 2].copy_from_slice(&(code << (16 - codes.bits)).to_le_bytes()),
        }
    };
    for (i, texel) in texels.iter().enumerate() {
        let Some(texel) = texel else {
            continue;
        };
        let (x, y) = (i as u64 % W, i as u64 / W);
        let [luma, cb, cr] = ycbcr_write(codes, conversion, *texel).expect("a stored layout");
        put(y * ROWQ + x * unit, luma);
        let pair = CHROMAQ_AT + y * ROWQ + 2 * x * unit;
        put(pair, cb);
        put(pair + unit, cr);
    }
    out
}

/// The first byte where `after` and `want` differ, and both there, for a failure's message.
fn first_difference(after: &[u8], want: &[u8]) -> Option<(usize, u8, u8)> {
    after
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (a, w))| a != w)
        .map(|(i, (&a, &w))| (i, a, w))
}

/// Dispatch a masked write of `values` to `texture` (`half4`s where `half`, else `float4`s)
/// over the texels `mask` names, and then, in the same encoder, a `read()` of every texel into
/// `out`, in one command buffer of `queue`.
fn write_then_read(
    kernels: &Kernels,
    queue: &CommandQueue,
    texture: &Texture,
    (values, mask, half): (&Buffer, &Buffer, bool),
    out: &Buffer,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .set_compute_pipeline_state(&kernels.write_masked[usize::from(half)])
        .unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(values), 0, 0).unwrap();
    encoder.set_buffer(Some(mask), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.set_compute_pipeline_state(&kernels.read).unwrap();
    encoder.set_buffer(Some(out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
}

/// Metal stores a function's writes to `YCBCR8_444_2P`, `YCBCR10_444_2P`, `RGB8_444_2P` and
/// `RGB10_444_2P` into both planes of the surface (on an Apple M4, on an Apple M3 Max, byte for
/// byte alike): each written texel's luma code and its Cb and Cr pair, the codes
/// `metal2vulkan::bind::ycbcr_write` states, and every other byte kept. Here a dispatch writes
/// `float` and `half` values (on the code grid and off it, out of `[0, 1]`, NaNs, infinities) to
/// about two texels in three, over both ranges and each matrix: the surface's bytes are those codes
/// and otherwise as they were, the rows' padding and the bytes past the planes among them. A
/// `read()` of the texture in the same encoder after the writes, and one in a later command buffer,
/// read the planes as written. A dispatch writing no texel keeps every byte. Nothing is announced
/// past the lane's pipeline states' own notes and, on a device without unified memory, the first
/// `Shared` buffer's `shared-in-system-memory`.
#[test]
fn a_four_four_four_texture_s_writes_store_metal_s_codes() {
    use PixelFormat as P;
    use YCbCrMatrix::{Bt2020, Bt601, Bt709};
    use YCbCrRange::{Full, Video};
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        // The lane's pipeline states announce their own notes as they are made (a `half` read's
        // NaN encoding, `read_half`): the writes are to announce nothing past them.
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (b709, b2020) = (
            Some(SurfaceYCbCrMatrix::Itu709),
            Some(SurfaceYCbCrMatrix::Itu2020),
        );
        let mut seed = 640;
        for (pf, ostype, matrix, conversion) in [
            (P::YCBCR84442P, "444v", None, (Video, Bt601)),
            (P::YCBCR84442P, "444f", b709, (Full, Bt709)),
            (P::YCBCR84442P, "444v", b2020, (Video, Bt2020)),
            (P::YCBCR104442P, "x444", b709, (Video, Bt709)),
            (P::YCBCR104442P, "xf44", None, (Full, Bt601)),
            (P::YCBCR104442P, "xf44", b2020, (Full, Bt2020)),
            (P::RGB84442P, "444f", None, (Full, Bt601)),
            (P::RGB104442P, "x444", b709, (Video, Bt709)),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            for half in [false, true] {
                seed += 1;
                let case = format!("{pf:?} over {ostype}, {matrix:?}, half {half}");
                let mut random = Random(seed);
                let bytes = family_bytes(seed, codes);
                let surface = buffer_with(device, &bytes).unwrap();
                let texture = written_texture(&surface, pf, ostype, matrix).unwrap();
                let (mut values, mut mask, mut texels) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..W * H {
                    let written = !random.next().is_multiple_of(3);
                    let texel = if half {
                        let h: [u16; 4] = std::array::from_fn(|_| written_half(&mut random));
                        values.extend(h.iter().flat_map(|h| h.to_le_bytes()));
                        [h[0], h[1], h[2]].map(half_value)
                    } else {
                        let v: [f32; 4] = std::array::from_fn(|_| written_channel(&mut random));
                        values.extend(v.iter().flat_map(|v| v.to_le_bytes()));
                        [v[0], v[1], v[2]]
                    };
                    mask.extend(u32::from(written).to_le_bytes());
                    texels.push(written.then_some(texel));
                }
                let values = buffer_with(device, &values).unwrap();
                let mask = buffer_with(device, &mask).unwrap();
                let out = device
                    .new_buffer(16 * W * H, ResourceOptions::shared())
                    .unwrap();
                write_then_read(&kernels, &queue, &texture, (&values, &mask, half), &out);
                let want = written_bytes(&bytes, codes, conversion, &texels);
                let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
                assert!(
                    after == want,
                    "{case}: the surface's bytes after the writes, first difference (at, got, \
                     want) {:?}",
                    first_difference(&after, &want)
                );
                // The converted formats' reads, of the planes as written.
                if codes.output == YCbCrOutput::Converted {
                    let read = family_modelled(&want, codes, conversion);
                    let same = read_buffer(&out, 0, 16 * W * H).unwrap();
                    assert_eq!(grid(&same), read, "{case}: read() in the same encoder");
                    let later = texels_of(device, &kernels.read, &texture, None);
                    assert_eq!(
                        grid(&later),
                        read,
                        "{case}: read() in a later command buffer"
                    );
                }
            }
        }
        // A dispatch writing no texel keeps every byte.
        for (seed, (pf, ostype)) in (680..).zip([
            (P::YCBCR84442P, "444f"),
            (P::YCBCR104442P, "x444"),
            (P::RGB84442P, "444v"),
            (P::RGB104442P, "xf44"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = written_texture(&surface, pf, ostype, None).unwrap();
            let values = buffer_with(device, &vec![0x3f; (16 * W * H) as usize]).unwrap();
            let mask = buffer_with(device, &vec![0; (4 * W * H) as usize]).unwrap();
            let out = device
                .new_buffer(16 * W * H, ResourceOptions::shared())
                .unwrap();
            write_then_read(&kernels, &queue, &texture, (&values, &mask, false), &out);
            let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
            assert!(
                after == bytes,
                "{pf:?} over {ostype}: a dispatch writing no texel changed byte (at, got, want) \
                 {:?}",
                first_difference(&after, &bytes)
            );
        }
        // Nothing but, on a device without unified memory, the first `Shared` buffer the case made
        // (`blits::SYSTEM_MEMORY`, announced as that buffer is made), as
        // `blits::shared_buffers_follow_the_memory_model` expects it.
        let expected: &[Degradation] = match device.properties().has_unified_memory {
            true => &[],
            false => &SYSTEM_MEMORY,
        };
        let notices = assert_once(&messages.take_notices(), expected);
        assert!(
            notices.keys().all(|kind| expected.contains(kind)),
            "{notices:?}"
        );
    });
}

/// The 4:4:4 writes whose codes are not known stay refused, typed (`Feature::Format`), and leave
/// the surface's bytes as they were: the `_sRGB` twins' (no arithmetic found matches Metal's
/// codes), the packed format's (the M3 drops them where the M4 stores them), and a texture's over
/// plane 1 of a 4:4:4 surface (not measured). A 4:2:2 texture over its own surface still drops
/// them, as Metal does.
#[test]
fn four_four_four_writes_of_unknown_codes_are_refused_typed() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let value = buffer_with(device, &[0x3f; 16]).unwrap();
        let queue = device.new_command_queue().unwrap();
        for (seed, pf, ostype, plane, dropped) in [
            (141, PixelFormat::YCBCR84442PSRGB, "444v", 0, false),
            (142, PixelFormat::YCBCR104442PSRGB, "xf44", 0, false),
            (143, PixelFormat::YCBCR104442PPACKED, "p444", 0, false),
            (144, PixelFormat::YCBCR84442P, "444f", 1, false),
            (145, PixelFormat::RGB104442P, "x444", 1, false),
            (146, PixelFormat::YCBCR84222P, "422v", 0, true),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
            let offset = [0, CHROMAQ_AT][plane];
            let facts = IoSurfaceYCbCr::new(
                u32::from_be_bytes(code),
                plane as u64,
                [W, H],
                CHROMAQ_AT,
                ROWQ,
            );
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
            d.resource_options = ResourceOptions::shared();
            let at = IoSurfacePlane::new(offset, ROWQ, true).with_ycbcr(facts);
            let texture = surface.new_texture_with_iosurface_plane(&d, at).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&kernels.fill).unwrap();
            encoder.set_texture(Some(&texture), 0).unwrap();
            encoder.set_buffer(Some(&value), 0, 0).unwrap();
            let result = encoder.dispatch_threads(Size::new(W, H, 1), Size::new(8, 8, 1));
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let case = format!("{pf:?} over plane {plane} of {ostype}");
            match (dropped, result) {
                (true, Ok(())) => {}
                (false, Err(Error::Unsupported(u) | Error::Bind(BindError::Unsupported(u)))) => {
                    assert_eq!(u.feature, Feature::Format, "{case}: {u:?}")
                }
                (_, other) => panic!("{case}: {other:?}"),
            }
            let after = read_buffer(&surface, 0, bytes.len() as u64).unwrap();
            assert!(after == bytes, "{case}: the surface's bytes changed");
        }
    });
}

/// Metal makes no `YCBCR8_420_2P` texture from a descriptor alone; the runtime refuses, typed,
/// what it does not hold over a surface.
#[test]
fn ycbcr_textures_are_made_only_over_their_surfaces() {
    on_device(|device, _| {
        let mut d = TextureDescriptor::texture_2d(PixelFormat::YCBCR84202P, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        let nil = |call| {
            Error::Nil(Nil::new(
                call,
                Problem::Texture(TextureProblem::NoTextureOfFormat),
            ))
        };
        assert_eq!(device.new_texture(&d).err(), Some(nil("new_texture")));
        let size = device.heap_texture_size_and_align(&d).unwrap();
        assert_eq!((size.size, size.align), (0, 16));
        let surface = buffer_with(device, &surface_bytes(3)).unwrap();
        let unsupported = |result: Result<Texture, Error>| {
            assert!(
                matches!(result, Err(Error::Unsupported(_))),
                "{:?}",
                result.err()
            );
        };
        assert_eq!(
            surface.new_texture(&d, 0, ROW).err(),
            Some(nil("new_texture"))
        );
        unsupported(ycbcr_texture(&surface, None, TextureUsage::SHADER_READ));
        let plane_one = IoSurfaceYCbCr::new(VIDEO, 1, [W, H], CHROMA_AT, ROW);
        unsupported(ycbcr_texture(
            &surface,
            Some(plane_one),
            TextureUsage::SHADER_READ,
        ));
        let other_matrix = facts(VIDEO, Some(SurfaceYCbCrMatrix::Other));
        unsupported(ycbcr_texture(
            &surface,
            Some(other_matrix),
            TextureUsage::SHADER_READ,
        ));
        let other = facts(u32::from_be_bytes(*b"x420"), None);
        unsupported(ycbcr_texture(
            &surface,
            Some(other),
            TextureUsage::SHADER_READ,
        ));
    });
}

/// Every two-plane YCbCr format answers as Metal does on an M4 without a surface: Metal's nil from
/// a descriptor alone, a 2D multisample texture or array stopped, and the heap query 0 bytes at 16,
/// or 128 for a 2D array; over a `420v` surface the runtime refuses, typed, every one but
/// `YCBCR8_420_2P`, its sRGB twin and `RGB8_420_2P` (the 10-bit ones it makes over 10-bit surfaces,
/// `ten_bit_textures_read_what_metal_reads`).
#[test]
fn two_plane_ycbcr_formats_answer_as_metal_does() {
    on_device(|device, _| {
        let surface = buffer_with(device, &surface_bytes(3)).unwrap();
        let family = PixelFormat::ALL
            .iter()
            .copied()
            .filter(|&pf| ycbcr_format(pf).is_some());
        let mut seen = 0;
        for pf in family {
            seen += 1;
            let refused = |d: &TextureDescriptor, want: Error| {
                let made = device.new_texture(d).map(|_| ()).err();
                assert_eq!(made, Some(want), "{pf:?} {:?}", d.texture_type);
            };
            let stopped = Error::Invalid(Invalid::new(
                "new_texture",
                Problem::Texture(TextureProblem::TypeForFormat),
            ));
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            refused(
                &d,
                Error::Nil(Nil::new(
                    "new_texture",
                    Problem::Texture(TextureProblem::NoTextureOfFormat),
                )),
            );
            let size = device.heap_texture_size_and_align(&d).unwrap();
            assert_eq!((size.size, size.align), (0, 16), "{pf:?}");
            let mut array = d;
            array.texture_type = TextureType::Type2DArray;
            array.array_length = 2;
            let size = device.heap_texture_size_and_align(&array).unwrap();
            assert_eq!((size.size, size.align), (0, 128), "{pf:?}");
            let mut multisample = d;
            multisample.texture_type = TextureType::Type2DMultisample;
            multisample.sample_count = 4;
            refused(&multisample, stopped.clone());
            // A 2D multisample array too.
            multisample.texture_type = TextureType::Type2DMultisampleArray;
            multisample.array_length = 2;
            refused(&multisample, stopped);
            d.resource_options = ResourceOptions::shared();
            let plane = IoSurfacePlane::new(0, ROW, true).with_ycbcr(facts(VIDEO, None));
            let made = surface.new_texture_with_iosurface_plane(&d, plane);
            if matches!(
                pf,
                PixelFormat::YCBCR84202P | PixelFormat::YCBCR84202PSRGB | PixelFormat::RGB84202P
            ) {
                assert!(made.is_ok(), "{pf:?}: {:?}", made.err());
            } else {
                assert!(
                    matches!(made, Err(Error::Unsupported(_))),
                    "{pf:?}: {:?}",
                    made.err()
                );
            }
        }
        assert_eq!(seen, 69);
    });
}

/// A 10-bit surface's rows: each code a little-endian 16-bit word, rows wider than the planes'
/// words (`2 W` bytes of luma, `4 CW` of chroma).
const ROW10: u64 = 128;
/// The 10-bit surface's chroma plane's first byte.
const CHROMA10_AT: u64 = ROW10 * H;
/// The 10-bit surface's buffer.
const LENGTH10: u64 = 8192;

/// The bytes of a 10-bit surface whose planes' words our generator at `seed` fills, laid out by
/// [`ROW10`] and [`CHROMA10_AT`], every other byte `0xa5`. Metal reads a word's high 10 bits.
fn ten_bit_bytes(seed: u64) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; LENGTH10 as usize];
    for y in 0..H {
        for x in 0..2 * W {
            bytes[(y * ROW10 + x) as usize] = random.next();
        }
    }
    for y in 0..CH {
        for x in 0..4 * CW {
            bytes[(CHROMA10_AT + y * ROW10 + x) as usize] = random.next();
        }
    }
    bytes
}

/// Each texel `codes` reads of the 10-bit surface `bytes` under `conversion`
/// (`metal2vulkan::bind::ycbcr_read`, the chroma filtered as [`modelled`] filters it).
fn ten_bit_modelled(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    let code = |at: u64| u16::from_le_bytes([bytes[at as usize], bytes[at as usize + 1]]) >> 6;
    let taps = |q: u64, n: u64| -> [(u64, u32); 2] {
        let c = q / 2;
        if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, CH) {
                for (cx, wx) in taps(x, CW) {
                    let at = CHROMA10_AT + cy * ROW10 + 4 * cx;
                    sums[0] += wx * wy * u32::from(code(at));
                    sums[1] += wx * wy * u32::from(code(at + 2));
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            out.push(ycbcr_read(
                codes,
                conversion,
                [code(y * ROW10 + 2 * x), cb, cr],
            ));
        }
    }
    out
}

/// The texture of `pf` over plane `plane` of `surface`, a 10-bit surface of OSType `ostype` and
/// `matrix`.
fn ten_bit_texture(
    surface: &Buffer,
    pf: PixelFormat,
    (ostype, plane): (&[u8; 4], u64),
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let facts = IoSurfaceYCbCr::new(
        u32::from_be_bytes(*ostype),
        plane,
        [W, H],
        CHROMA10_AT,
        ROW10,
    );
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROW10, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// The codes of each channel of `texels` (`float4`s), each checked to be `code / max` correctly
/// rounded, alpha checked to be 1: an `RGB10_420_2P` (`max` 1023) or `RGB8_420_2P` (255) read's
/// red Cr, green Y, blue Cb.
fn code_grid(texels: &[u8], max: u16) -> Vec<[u16; 3]> {
    let max = f32::from(max);
    texels
        .as_chunks::<16>()
        .0
        .iter()
        .map(|t| {
            let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
            assert_eq!(c(3), 1.0, "alpha 1");
            std::array::from_fn(|i| {
                let code = (c(i) * max).round();
                assert!((0.0..=max).contains(&code), "{} off the codes", c(i));
                assert_eq!(c(i), code / max, "{} not code / {max}", c(i));
                code as u16
            })
        })
        .collect()
}

/// `YCBCR10_420_2P` and `RGB10_420_2P` textures over 10-bit surfaces read what Metal reads
/// (`ycbcr_read` states it): `read()` and a linear sample at each texel's centre, over `x420`,
/// `xf20`, `x422` and `xf44` surfaces without a matrix, with BT.709 and with BT.2020; and
/// `YCBCR10_420_2P`'s linear sample between the texels is Metal's step on the
/// grid `2048`, as `YCBCR8_420_2P`'s. No filter notice.
#[test]
fn ten_bit_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut seed = 20;
        for pf in [PixelFormat::YCBCR104202P, PixelFormat::RGB104202P] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            for (ostype, range) in [
                (b"x420", YCbCrRange::Video),
                (b"xf20", YCbCrRange::Full),
                (b"x422", YCbCrRange::Video),
                (b"xf44", YCbCrRange::Full),
            ] {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                ] {
                    seed += 1;
                    let bytes = ten_bit_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = ten_bit_texture(&surface, pf, (ostype, 0), matrix).unwrap();
                    let want = ten_bit_modelled(&bytes, codes, (range, modelled));
                    let case = format!("{pf:?} over {ostype:?}, {matrix:?}");
                    let of = |texels: &[u8]| match codes.output {
                        YCbCrOutput::Codes => code_grid(texels, 1023),
                        _ => grid(texels),
                    };
                    let read = of(&texels_of(device, &kernels.read, &texture, None));
                    assert_eq!(read, want, "{case}: read()");
                    let sampled = texels_of(device, &kernels.sample, &texture, Some(&linear));
                    assert_eq!(of(&sampled), want, "{case}: a linear sample at the centres");
                    if codes.output == YCbCrOutput::Converted {
                        let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                        for (i, got) in (0u64..).zip(grid(&between)) {
                            let at = (i % W, i / W);
                            let step: [u16; 3] =
                                std::array::from_fn(|c| metal_step(&read, at, BETWEEN, c) as u16);
                            assert_eq!(got, step, "{case}: between, texel {at:?}");
                        }
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// A 10-bit texture's copies move its luma words, two bytes a texel, as Metal's do: `get_bytes`
/// reads the luma rows' words, and `replace_region` writes them, keeping the chroma plane, the
/// texture then reading the new codes' conversion.
#[test]
fn ten_bit_textures_copy_their_luma_words() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let bytes = ten_bit_bytes(41);
        let surface = buffer_with(device, &bytes).unwrap();
        let pf = PixelFormat::YCBCR104202P;
        let texture = ten_bit_texture(&surface, pf, (b"x420", 0), None).unwrap();
        let mut luma = vec![0u8; (2 * W * H) as usize];
        texture
            .get_bytes(&mut luma, 2 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
            .unwrap();
        let rows: Vec<u8> = (0..H)
            .flat_map(|y| bytes[(y * ROW10) as usize..(y * ROW10 + 2 * W) as usize].to_vec())
            .collect();
        assert_eq!(luma, rows, "get_bytes");
        let mut random = Random(43);
        let new: Vec<u8> = (0..2 * W * H).map(|_| random.next()).collect();
        texture
            .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &new, 2 * W, 0)
            .unwrap();
        let mut want = bytes.clone();
        for y in 0..H {
            let at = (y * ROW10) as usize;
            want[at..at + 2 * W as usize]
                .copy_from_slice(&new[(2 * y * W) as usize..(2 * (y + 1) * W) as usize]);
        }
        assert_eq!(
            read_buffer(&surface, 0, LENGTH10).unwrap(),
            want,
            "replace_region"
        );
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let modelled = ten_bit_modelled(&want, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
        let read = grid(&texels_of(device, &kernels.read, &texture, None));
        assert_eq!(read, modelled, "read() after replace_region");
    });
}

/// The runtime refuses, typed, naming the format, what it does not hold of the 10-bit formats
/// over a surface: an OSType whose codes are not the format's (`420v` under `YCBCR10_420_2P`,
/// `x420` under `YCBCR8_420_2P`), plane 1, and a matrix other than BT.601, BT.709 and BT.2020.
#[test]
fn ten_bit_textures_refuse_what_is_not_held() {
    on_device(|device, _| {
        let surface = buffer_with(device, &ten_bit_bytes(5)).unwrap();
        let refused = |made: Result<Texture, Error>, pf: PixelFormat, what: &str| {
            let error = made.err();
            let named = format!("{error:?}");
            assert!(
                matches!(error, Some(Error::Unsupported(_))) && named.contains(pf.name()),
                "{pf:?} {what}: {named}"
            );
        };
        for pf in [PixelFormat::YCBCR104202P, PixelFormat::RGB104202P] {
            let made = ten_bit_texture(&surface, pf, (b"420v", 0), None);
            refused(made, pf, "over 420v");
            let made = ten_bit_texture(&surface, pf, (b"x420", 1), None);
            refused(made, pf, "of plane 1");
            let other = Some(SurfaceYCbCrMatrix::Other);
            refused(
                ten_bit_texture(&surface, pf, (b"x420", 0), other),
                pf,
                "another matrix",
            );
        }
        let pf = PixelFormat::YCBCR84202P;
        refused(
            ten_bit_texture(&surface, pf, (b"x420", 0), None),
            pf,
            "over x420",
        );
    });
}

/// The `f32` of the `half` nearest `step / 8192` with ties away from zero, as Metal rounds a
/// `half` read of a step of `1 / 8192`. Every nonzero step is a normal `half`'s
/// magnitude, so the `half`s about it lie `2^(e - 10)` apart for its binade `e`.
fn half_away(step: u16) -> f32 {
    if step == 0 {
        return 0.0;
    }
    let v = f64::from(step) / 8192.0;
    let ulp = 2f64.powi(v.log2().floor() as i32 - 10);
    ((v / ulp + 0.5).floor() * ulp) as f32
}

/// The `float4`s `texels` widened from `half`s, against Metal's `half` read of the steps
/// `want`: each channel [`half_away`] of its step, alpha 1.
fn assert_half_reads(texels: &[u8], want: &[[u16; 3]], case: &str) {
    for (i, (t, w)) in texels.as_chunks::<16>().0.iter().zip(want).enumerate() {
        let c = |i: usize| f32::from_le_bytes(t[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!(c(3), 1.0, "{case}: texel {i}'s alpha");
        let got: [f32; 3] = std::array::from_fn(c);
        assert_eq!(got, w.map(half_away), "{case}: texel {i} ({w:?})");
    }
}

/// The `_sRGB` twins over their surfaces read what Metal reads (`ycbcr_read` states it):
/// `YCBCR8_420_2P_sRGB` over `420v` and `420f`, and `YCBCR10_420_2P_sRGB` over `x420`, `xf20`,
/// `x422` and `xf44`, without a matrix, with BT.709 and with BT.2020. Each texel's `float` `read()`
/// and a linear sample at its centre are its step of `1/8192`; a `half` read and a `half` nearest
/// sample that step rounded with ties away from zero; and a linear sample between the texels
/// Metal's step on the grid `8192`. No filter notice.
#[test]
fn srgb_twins_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let mut seed = 70;
        let surfaces: [(PixelFormat, &[u8; 4], YCbCrRange); 6] = [
            (PixelFormat::YCBCR84202PSRGB, b"420v", YCbCrRange::Video),
            (PixelFormat::YCBCR84202PSRGB, b"420f", YCbCrRange::Full),
            (PixelFormat::YCBCR104202PSRGB, b"x420", YCbCrRange::Video),
            (PixelFormat::YCBCR104202PSRGB, b"xf20", YCbCrRange::Full),
            (PixelFormat::YCBCR104202PSRGB, b"x422", YCbCrRange::Video),
            (PixelFormat::YCBCR104202PSRGB, b"xf44", YCbCrRange::Full),
        ];
        for (pf, ostype, range) in surfaces {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert_eq!(codes.output, YCbCrOutput::Srgb);
            for (matrix, modelled) in [
                (None, YCbCrMatrix::Bt601),
                (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
            ] {
                seed += 1;
                let (want, texture) = if codes.bits == 8 {
                    let bytes = surface_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let facts = facts(u32::from_be_bytes(*ostype), matrix);
                    let made =
                        eight_bit_texture(&surface, pf, Some(facts), TextureUsage::SHADER_READ);
                    (modelled_as(&bytes, codes, (range, modelled)), made.unwrap())
                } else {
                    let bytes = ten_bit_bytes(seed);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let made = ten_bit_texture(&surface, pf, (ostype, 0), matrix);
                    (
                        ten_bit_modelled(&bytes, codes, (range, modelled)),
                        made.unwrap(),
                    )
                };
                let case = format!("{pf:?} over {ostype:?}, {matrix:?}");
                let read = grid_of(&texels_of(device, &kernels.read, &texture, None), 8192);
                assert_eq!(read, want, "{case}: read()");
                let sampled = texels_of(device, &kernels.sample, &texture, Some(&linear));
                assert_eq!(
                    grid_of(&sampled, 8192),
                    want,
                    "{case}: a linear sample at the centres"
                );
                let half = texels_of(device, &kernels.read_half, &texture, None);
                assert_half_reads(&half, &want, &format!("{case}: a half read()"));
                let half = texels_of(device, &kernels.sample_half, &texture, Some(&nearest));
                assert_half_reads(&half, &want, &format!("{case}: a half nearest sample"));
                let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                for (i, got) in (0u64..).zip(grid_of(&between, 8192)) {
                    let at = (i % W, i / W);
                    let step: [u16; 3] =
                        std::array::from_fn(|c| metal_step(&read, at, BETWEEN, c) as u16);
                    assert_eq!(got, step, "{case}: between, texel {at:?}");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// A converted texture's view as its `_sRGB` twin reads what a texture of the twin over the same
/// planes reads, as Metal's view does (made with `ShaderRead` alone):
/// `YCBCR8_420_2P` over `420f` with BT.709 as `YCBCR8_420_2P_sRGB`, and `YCBCR10_420_2P` over
/// `x420` as `YCBCR10_420_2P_sRGB`; the texture itself still reads its own conversion, and after
/// `replace_region` writes the luma plane through it, the view reads the new codes' twin. The other
/// order and a view of another layout are refused.
#[test]
fn ycbcr_views_as_srgb_twins_read_the_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
        // 8 bits.
        let bytes = surface_bytes(81);
        let surface = buffer_with(device, &bytes).unwrap();
        let facts = facts(FULL, Some(SurfaceYCbCrMatrix::Itu709));
        let usage = TextureUsage::SHADER_READ;
        let base =
            eight_bit_texture(&surface, PixelFormat::YCBCR84202P, Some(facts), usage).unwrap();
        let view = base.new_texture_view(PixelFormat::YCBCR84202PSRGB).unwrap();
        let twin =
            eight_bit_texture(&surface, PixelFormat::YCBCR84202PSRGB, Some(facts), usage).unwrap();
        let srgb = codes(PixelFormat::YCBCR84202PSRGB);
        let want = modelled_as(&bytes, srgb, bt709);
        let read = |t: &Texture, d| grid_of(&texels_of(device, &kernels.read, t, None), d);
        assert_eq!(read(&view, 8192), want, "the view");
        assert_eq!(read(&twin, 8192), want, "the twin");
        assert_eq!(
            read(&base, 2048),
            modelled(&bytes, bt709.0, bt709.1),
            "the texture"
        );
        let mut random = Random(83);
        let luma: Vec<u8> = (0..W * H).map(|_| random.next()).collect();
        base.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &luma, W, 0)
            .unwrap();
        let written = read_buffer(&surface, 0, LENGTH).unwrap();
        assert_eq!(
            read(&view, 8192),
            modelled_as(&written, srgb, bt709),
            "the view, written"
        );
        // 10 bits.
        let bytes = ten_bit_bytes(85);
        let surface = buffer_with(device, &bytes).unwrap();
        let base =
            ten_bit_texture(&surface, PixelFormat::YCBCR104202P, (b"x420", 0), None).unwrap();
        let view = base
            .new_texture_view(PixelFormat::YCBCR104202PSRGB)
            .unwrap();
        let video = (YCbCrRange::Video, YCbCrMatrix::Bt601);
        let want = ten_bit_modelled(&bytes, codes(PixelFormat::YCBCR104202PSRGB), video);
        assert_eq!(read(&view, 8192), want, "the 10-bit view");
        // Refused: the twin as the converted format (not measured), and another layout.
        let twin =
            ten_bit_texture(&surface, PixelFormat::YCBCR104202PSRGB, (b"x420", 0), None).unwrap();
        let error = twin.new_texture_view(PixelFormat::YCBCR104202P).err();
        let named = format!("{error:?}");
        assert!(
            matches!(error, Some(Error::Unsupported(_)))
                && named.contains(PixelFormat::YCBCR104202P.name())
                && !named.contains(PixelFormat::YCBCR104202PSRGB.name()),
            "the other order: {named}"
        );
        assert!(
            base.new_texture_view(PixelFormat::YCBCR84202PSRGB).is_err(),
            "another layout's twin"
        );
    });
}

/// What `sample_outside` reads of `texture` through `sampler` (or `state`'s own constexpr
/// sampler, given none): a `float4` a point, `2 W` points a row, `2 H` rows.
fn outside_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: Option<&SamplerState>,
) -> Vec<u8> {
    let out = device
        .new_buffer(64 * W * H, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(
        &command_buffer,
        state,
        texture,
        &out,
        sampler,
        [2 * W, 2 * H],
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, 64 * W * H).unwrap()
}

/// The bytes of the `float4` `v`.
fn float4_bytes(v: [f32; 4]) -> Vec<u8> {
    v.iter().flat_map(|c| c.to_le_bytes()).collect()
}

/// Samples outside a two-plane YCbCr texture read as Metal reads them (the M4 and the M3 Max alike;
/// for 4:2:2 and 4:4:4), nearest and linear, at texel centres from `-W/2` to `3W/2`: under
/// clamp-to-edge, mirror-clamp-to-edge, repeat and mirrored repeat each point of a 4:2:0 or 4:2:2
/// texture reads its texel clamped to the texture, of a 4:4:4 one its texel by that mode; under
/// clamp-to-zero and clamp-to-border-color (transparent black, opaque black, opaque white) the
/// converted formats and their sRGB twins read the border outside, its alpha 1, and `RGB10_420_2P`
/// and `RGB8_420_2P` (the M4), read as codes, its texel clamped to the texture again, under every
/// border color. The packed formats read as their unpacked twins.
#[test]
fn ycbcr_samples_outside_read_metals_modes() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let usage = TextureUsage::SHADER_READ;
        let eight = buffer_with(device, &surface_bytes(91)).unwrap();
        let ten = buffer_with(device, &ten_bit_bytes(92)).unwrap();
        let ten_full = buffer_with(device, &ten_bit_bytes(93)).unwrap();
        let family: Vec<(PixelFormat, &str, Buffer)> = (94..)
            .zip([
                (PixelFormat::YCBCR84222P, "422v"),
                (PixelFormat::YCBCR84442P, "444v"),
                (PixelFormat::YCBCR104222P, "x422"),
                (PixelFormat::YCBCR104442P, "x444"),
                (PixelFormat::YCBCR104442PSRGB, "x444"),
                (PixelFormat::RGB84222P, "422v"),
                (PixelFormat::RGB84442P, "444v"),
                (PixelFormat::RGB104222P, "x422"),
                (PixelFormat::RGB104442P, "x444"),
                (PixelFormat::YCBCR104202PPACKED, "p420"),
                (PixelFormat::YCBCR104222PPACKEDSRGB, "pf22"),
                (PixelFormat::YCBCR104442PPACKED, "p444"),
                (PixelFormat::YCBCR104442PPACKEDSRGB, "pf44"),
            ])
            .map(|(seed, (pf, ostype))| {
                let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
                let surface = buffer_with(device, &family_bytes(seed, codes)).unwrap();
                (pf, ostype, surface)
            })
            .collect();
        let mut textures: Vec<(PixelFormat, &str, Texture)> = Vec::new();
        for pf in [
            PixelFormat::YCBCR84202P,
            PixelFormat::YCBCR84202PSRGB,
            PixelFormat::RGB84202P,
        ] {
            let surface_facts = Some(facts(VIDEO, None));
            let texture = eight_bit_texture(&eight, pf, surface_facts, usage).unwrap();
            textures.push((pf, "420v", texture));
        }
        for (pf, surface, ostype) in [
            (PixelFormat::YCBCR104202P, &ten, b"x420"),
            (PixelFormat::YCBCR104202PSRGB, &ten, b"x420"),
            (PixelFormat::RGB104202P, &ten, b"x420"),
            (PixelFormat::RGB104202P, &ten_full, b"xf20"),
        ] {
            let texture = ten_bit_texture(surface, pf, (ostype, 0), None).unwrap();
            textures.push((pf, std::str::from_utf8(ostype).unwrap(), texture));
        }
        for &(pf, ostype, ref surface) in &family {
            let texture = family_texture(surface, pf, ostype, None).unwrap();
            textures.push((pf, ostype, texture));
        }
        let border = SamplerAddressMode::ClampToBorderColor;
        let clear = SamplerBorderColor::TransparentBlack;
        let modes = [
            (SamplerAddressMode::ClampToEdge, clear),
            (SamplerAddressMode::MirrorClampToEdge, clear),
            (SamplerAddressMode::Repeat, clear),
            (SamplerAddressMode::MirrorRepeat, clear),
            (SamplerAddressMode::ClampToZero, clear),
            (border, clear),
            (border, SamplerBorderColor::OpaqueBlack),
            (border, SamplerBorderColor::OpaqueWhite),
        ];
        // Every state made before any command, so that each command's bind shape holds them all.
        let samplers: Vec<_> = [SamplerMinMagFilter::Nearest, SamplerMinMagFilter::Linear]
            .into_iter()
            .flat_map(|filter| modes.map(|mode| (filter, mode)))
            .map(|(filter, (mode, color))| {
                let mut d = SamplerDescriptor::new();
                d.min_filter = filter;
                d.mag_filter = filter;
                d.s_address_mode = mode;
                d.t_address_mode = mode;
                d.border_color = color;
                ((filter, mode, color), device.new_sampler_state(&d).unwrap())
            })
            .collect();
        for (pf, ostype, texture) in &textures {
            let pf = *pf;
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let read = texels_of(device, &kernels.read, texture, None);
            let texel = |x: u64, y: u64| read[(16 * (y * W + x)) as usize..][..16].to_vec();
            for ((filter, mode, color), state) in &samplers {
                let reads_border = codes.output != YCbCrOutput::Codes
                    && [SamplerAddressMode::ClampToZero, border].contains(mode);
                let border_texel = if *color == SamplerBorderColor::OpaqueWhite {
                    [1.0; 4]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                };
                let got = outside_of(device, &kernels.outside, texture, Some(state));
                for (i, point) in (0u64..).zip(got.as_chunks::<16>().0.iter()) {
                    let at = (i % (2 * W)) as i64 - (W / 2) as i64;
                    let row = (i / (2 * W)) as i64 - (H / 2) as i64;
                    let inside = (0..W as i64).contains(&at) && (0..H as i64).contains(&row);
                    let want = if reads_border && !inside {
                        float4_bytes(border_texel)
                    } else {
                        let wraps = codes.subsampling == ChromaSubsampling::Chroma444;
                        let mode = if wraps {
                            *mode
                        } else {
                            SamplerAddressMode::ClampToEdge
                        };
                        texel(addressed(at, W, mode), addressed(row, H, mode))
                    };
                    assert_eq!(
                        point,
                        &want[..],
                        "{pf:?} {ostype:?} {filter:?} {mode:?} {color:?}: texel ({at}, {row})"
                    );
                }
            }
        }
    });
}

/// A constexpr sampler's words: nearest filters, no mip filter, normalized coordinates that
/// repeat on every axis.
const NEAREST_REPEAT: u64 = 0x007b_ff00_0008_0000 | 2 | 2 << 3 | 2 << 6;

/// Synthetic AIR written for this lane: `sample_outside_repeat` is `sample_outside` through a
/// constexpr sampler of [`NEAREST_REPEAT`]'s words in place of sampler slot 0.
fn constexpr_outside() -> String {
    format!(
        r#"; Owned synthetic kernel.
source_filename = "ycbcr_constexpr.metal"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {NEAREST_REPEAT}, i64 0], align 8

define void @sample_outside_repeat(ptr addrspace(1) %t, ptr addrspace(1) noalias %out, <2 x i32> %gid, <2 x i32> %size) {{
entry:
  %g = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %gid)
  %n = tail call <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %size)
  %q = fmul <2 x float> %n, <float 2.500000e-01, float 2.500000e-01>
  %c0 = fadd <2 x float> %g, <float 5.000000e-01, float 5.000000e-01>
  %c = fsub <2 x float> %c0, %q
  %m = fmul <2 x float> %n, <float 5.000000e-01, float 5.000000e-01>
  %uv = fdiv <2 x float> %c, %m
  %r = tail call {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %w = extractelement <2 x i32> %size, i64 0
  %row = mul i32 %y, %w
  %i = add i32 %row, %x
  %j = zext i32 %i to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %j
  store <4 x float> %v, ptr addrspace(1) %p, align 16
  ret void
}}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare {{ <4 x float>, i8 }} @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{{!0}}
!air.sampler_states = !{{!9}}
!air.version = !{{!30}}
!air.language_version = !{{!31}}
!0 = !{{ptr @sample_outside_repeat, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
!6 = !{{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"size"}}
!9 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!30 = !{{i32 2, i32 8, i32 0}}
!31 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// A constexpr sampler takes a two-plane YCbCr texture's address-mode map through one pipeline
/// variant of its slot, and every other texture keeps the state's pipeline:
/// `sample_outside_repeat`, through a constexpr nearest repeat sampler, makes nothing when
/// dispatched again over an `RGBA8Unorm` texture, and no variant (no shader module) over a
/// `YCBCR10_444_2P` one, whose map keeps repeat (a 4:4:4 texture repeats); over a `YCBCR10_420_2P`
/// one, whose map samples repeat as clamp-to-edge, it makes exactly one variant (one shader
/// module), once, which a `YCBCR8_420_2P` one, whose map agrees on repeat, shares. Over each it
/// reads, word for word, what the runtime repeat sampler at slot 0 reads (`sample_outside`), as
/// Metal samples a constexpr sampler as a runtime one of its modes: the 4:2:0 textures their
/// texels clamped to the edge. A dispatch again of a made variant, after the texture's shadow
/// copy, records in a `VkCommandBuffer` of its own exactly where the driver loses baked samplers
/// after a copy (`RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`: lavapipe 26.0.8 sampled
/// it as Repeat in place).
#[test]
fn a_constexpr_sampler_takes_a_ycbcr_map_through_one_variant() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let library = device
            .new_library_with_air(&llvm_assemble_text(&constexpr_outside()).unwrap())
            .unwrap();
        let constexpr = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    library
                        .new_function_with_name("sample_outside_repeat")
                        .unwrap()
                        .function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let mut d = SamplerDescriptor::new();
        d.s_address_mode = SamplerAddressMode::Repeat;
        d.t_address_mode = SamplerAddressMode::Repeat;
        let repeat = device.new_sampler_state(&d).unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        let rgba = device.new_texture(&d).unwrap();
        let mut random = Random(95);
        let texels: Vec<u8> = (0..4 * W * H).map(|_| random.next()).collect();
        rgba.replace_region(Region::new_2d(0, 0, W, H), 0, 0, &texels, 4 * W, 0)
            .unwrap();
        let ten = buffer_with(device, &ten_bit_bytes(96)).unwrap();
        let ten_420 = ten_bit_texture(&ten, PixelFormat::YCBCR104202P, (b"x420", 0), None).unwrap();
        let eight = buffer_with(device, &surface_bytes(97)).unwrap();
        let eight_420 = eight_bit_texture(
            &eight,
            PixelFormat::YCBCR84202P,
            Some(facts(VIDEO, None)),
            TextureUsage::SHADER_READ,
        )
        .unwrap();
        let codes = ycbcr_format(PixelFormat::YCBCR104442P)
            .and_then(YCbCrCodes::of)
            .unwrap();
        let four = buffer_with(device, &family_bytes(98, codes)).unwrap();
        let ten_444 = family_texture(&four, PixelFormat::YCBCR104442P, "x444", None).unwrap();
        // Each texture read through the runtime sampler first: what the constexpr sampler reads,
        // and any work of the texture's own done before the counts are taken.
        let textures = [&rgba, &ten_420, &eight_420, &ten_444];
        let want: Vec<Vec<u8>> = textures
            .iter()
            .map(|texture| outside_of(device, &kernels.outside, texture, Some(&repeat)))
            .collect();
        let read = texels_of(device, &kernels.read, &ten_420, None);
        let texel = |x: u64, y: u64| read[(16 * (y * W + x)) as usize..][..16].to_vec();
        for (i, point) in (0u64..).zip(want[1].as_chunks::<16>().0.iter()) {
            let at = (i % (2 * W)) as i64 - (W / 2) as i64;
            let row = (i / (2 * W)) as i64 - (H / 2) as i64;
            let edge = SamplerAddressMode::ClampToEdge;
            assert_eq!(
                point,
                &texel(addressed(at, W, edge), addressed(row, H, edge))[..],
                "YCBCR10_420_2P repeat: texel ({at}, {row})"
            );
        }
        // What one dispatch of the constexpr kernel over `texture` reads, the shader modules its
        // new variant's stages took, and the `VkPipeline`s it made. A new variant takes its
        // module, made or the device's shared one of the same SPIR-V (on the M4, the YCBCR10_420_2P
        // variant made none of its own). A variant kept takes none.
        let dispatched = |texture: &Texture| {
            let before = device.work_counts();
            let got = outside_of(device, &constexpr, texture, None);
            let after = device.work_counts();
            (
                got,
                (after.shader_modules_made + after.shader_modules_shared)
                    - (before.shader_modules_made + before.shader_modules_shared),
                after.variant_pipelines_made - before.variant_pipelines_made,
                after.baked_sampler_dispatches_apart - before.baked_sampler_dispatches_apart,
            )
        };
        // Each variant bakes its sampler: a dispatch whose pipeline is made records apart where
        // the driver loses baked samplers after a copy, the texture's shadow copy among them.
        let apart = u64::from(
            device
                .properties()
                .driver_defects
                .contains(&RuntimeDriverDefect::ImmutableSamplersLostAfterCopies),
        );
        let (got, _, _, _) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm");
        let (got, modules, pipelines, recorded_apart) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm, again");
        assert_eq!((modules, pipelines), (0, 0), "RGBA8Unorm, again");
        assert_eq!(recorded_apart, apart, "RGBA8Unorm, again: recorded apart");
        // A texture of another texel class may specialize the variant's module anew (a
        // `VkPipeline`, the filter class), never compile another.
        let (got, modules, _, _) = dispatched(&ten_444);
        assert_eq!(got, want[3], "YCBCR10_444_2P");
        assert_eq!(
            modules, 0,
            "YCBCR10_444_2P keeps the RGBA8Unorm texture's variant"
        );
        let (got, modules, pipelines, _) = dispatched(&ten_420);
        assert_eq!(got, want[1], "YCBCR10_420_2P");
        assert_eq!(modules, 1, "YCBCR10_420_2P makes one variant");
        assert!(
            pipelines >= 1,
            "YCBCR10_420_2P makes its variant's pipeline"
        );
        let (got, modules, pipelines, recorded_apart) = dispatched(&ten_420);
        assert_eq!(got, want[1], "YCBCR10_420_2P, again");
        assert_eq!((modules, pipelines), (0, 0), "YCBCR10_420_2P, again");
        assert_eq!(
            recorded_apart, apart,
            "YCBCR10_420_2P, again: recorded apart"
        );
        let (got, modules, _, _) = dispatched(&eight_420);
        assert_eq!(got, want[2], "YCBCR8_420_2P");
        assert_eq!(modules, 0, "YCBCR8_420_2P shares YCBCR10_420_2P's variant");
        let (got, modules, _, _) = dispatched(&rgba);
        assert_eq!(got, want[0], "RGBA8Unorm, after");
        assert_eq!(modules, 0, "RGBA8Unorm, after");
    });
}

/// `RGB8_420_2P` textures over `420v` and `420f` surfaces read their codes as Metal does
/// (`ycbcr_read` states it): red Cr, green Y, blue Cb, each `code / 255`, alpha 1, after the
/// 9/3/3/1 chroma filter, whatever the matrix: `read()`, and a nearest and a linear sample at each
/// texel's centre. No filter notice.
#[test]
fn eight_bit_codes_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let pf = PixelFormat::RGB84202P;
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let usage = TextureUsage::SHADER_READ;
        for (seed, ostype, matrix) in [
            (41, VIDEO, None),
            (42, FULL, Some(SurfaceYCbCrMatrix::Itu601)),
            (43, VIDEO, Some(SurfaceYCbCrMatrix::Itu709)),
            (44, FULL, Some(SurfaceYCbCrMatrix::Itu709)),
        ] {
            let bytes = surface_bytes(seed);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture =
                eight_bit_texture(&surface, pf, Some(facts(ostype, matrix)), usage).unwrap();
            // The codes, which no range or matrix changes.
            let want = modelled_as(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            let case = format!("over {:?}, {matrix:?}", ostype.to_be_bytes());
            let read = code_grid(&texels_of(device, &kernels.read, &texture, None), 255);
            assert_eq!(read, want, "{case}: read()");
            for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                assert_eq!(code_grid(&sampled, 255), want, "{case}: a {filter} sample");
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
    });
}

/// A 4:2:2 or 4:4:4 surface's rows, wide enough for a 4:4:4 10-bit chroma row (`4 W` bytes).
const ROWQ: u64 = 256;
/// The 4:2:2 or 4:4:4 surface's chroma plane's first byte.
const CHROMAQ_AT: u64 = ROWQ * H;

/// The texel `p` of an axis of `n` texels reads under `mode` (a point at its centre): clamped
/// to the texture, repeated, mirrored once and clamped, or mirrored and repeated.
fn addressed(p: i64, n: u64, mode: SamplerAddressMode) -> u64 {
    let n = n as i64;
    let at = if mode == SamplerAddressMode::Repeat {
        p.rem_euclid(n)
    } else if mode == SamplerAddressMode::MirrorClampToEdge {
        (if p < 0 { -1 - p } else { p }).min(n - 1)
    } else if mode == SamplerAddressMode::MirrorRepeat {
        let q = p.rem_euclid(2 * n);
        if q < n {
            q
        } else {
            2 * n - 1 - q
        }
    } else {
        p.clamp(0, n - 1)
    };
    at as u64
}

/// The bytes of a surface of `codes` whose planes' codes our generator at `seed` fills, laid out
/// by [`ROWQ`] and [`CHROMAQ_AT`] (a code a byte or a 16-bit word's bytes, or three packed codes
/// a 32-bit word's, every bit of a row's units random), every other byte `0xa5`.
fn family_bytes(seed: u64, codes: YCbCrCodes) -> Vec<u8> {
    let mut random = Random(seed);
    let mut bytes = vec![0xa5; (2 * ROWQ * H) as usize];
    let [cw, ch] = chroma_size(codes);
    for y in 0..H {
        for x in 0..row_bytes(codes, W) {
            bytes[(y * ROWQ + x) as usize] = random.next();
        }
    }
    for y in 0..ch {
        for x in 0..row_bytes(codes, 2 * cw) {
            bytes[(CHROMAQ_AT + y * ROWQ + x) as usize] = random.next();
        }
    }
    bytes
}

/// The bytes of a row of `n` codes of `codes`: `n` units, or `⌈n / 3⌉` packed.
fn row_bytes(codes: YCbCrCodes, n: u64) -> u64 {
    codes.units(n) * u64::from(codes.bytes)
}

/// Code `index` of `codes` of the row of `bytes` whose first byte is at `row`: a byte, a 16-bit
/// word's high bits, or a packed word's code at bits 0, 10 or 20.
fn code_of(bytes: &[u8], codes: YCbCrCodes, row: u64, index: u64) -> u16 {
    let (at, shift) = match codes.packed {
        true => (row + 4 * (index / 3), 10 * (index % 3)),
        false => (
            row + u64::from(codes.bytes) * index,
            u64::from(8 * codes.bytes - codes.bits),
        ),
    };
    let at = at as usize;
    let unit = &bytes[at..at + codes.bytes as usize];
    let word = unit
        .iter()
        .rev()
        .fold(0u32, |word, &b| (word << 8) | u32::from(b));
    ((word >> shift) & ((1 << codes.bits) - 1)) as u16
}

/// The chroma plane's pairs across and down of a `W x H` texture of `codes`.
fn chroma_size(codes: YCbCrCodes) -> [u64; 2] {
    let halved = codes.subsampling.halved();
    [(W, halved[0]), (H, halved[1])].map(|(n, halved)| if halved { n.div_ceil(2) } else { n })
}

/// Each texel `codes` reads of the surface `bytes` (laid out as [`family_bytes`] lays them)
/// under `conversion` (`metal2vulkan::bind::ycbcr_read`): along a halved axis the two nearest
/// chroma pairs weighted 3 and 1, along a direct one the texel's own pair weighted 4, the sum `S`
/// of the 16 weights rounded `(S + 8) >> 4`.
fn family_modelled(
    bytes: &[u8],
    codes: YCbCrCodes,
    conversion: (YCbCrRange, YCbCrMatrix),
) -> Vec<[u16; 3]> {
    let code = |row: u64, index: u64| code_of(bytes, codes, row, index);
    let [cw, ch] = chroma_size(codes);
    let halved = codes.subsampling.halved();
    let taps = |q: u64, n: u64, halved: bool| -> [(u64, u32); 2] {
        let c = q / 2;
        if !halved {
            [(q, 4), (q, 0)]
        } else if q % 2 == 1 {
            [(c, 3), ((c + 1).min(n - 1), 1)]
        } else {
            [(c.saturating_sub(1), 1), (c, 3)]
        }
    };
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let mut sums = [0u32; 2];
            for (cy, wy) in taps(y, ch, halved[1]) {
                for (cx, wx) in taps(x, cw, halved[0]) {
                    let row = CHROMAQ_AT + cy * ROWQ;
                    sums[0] += wx * wy * u32::from(code(row, 2 * cx));
                    sums[1] += wx * wy * u32::from(code(row, 2 * cx + 1));
                }
            }
            let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
            out.push(ycbcr_read(codes, conversion, [code(y * ROWQ, x), cb, cr]));
        }
    }
    out
}

/// The texture of `pf` over plane 0 of `surface`, a surface of OSType `ostype` and `matrix` laid
/// out by [`ROWQ`] and [`CHROMAQ_AT`], of shader read usage.
fn family_texture(
    surface: &Buffer,
    pf: PixelFormat,
    ostype: &str,
    matrix: Option<SurfaceYCbCrMatrix>,
) -> Result<Texture, Error> {
    let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
    let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), 0, [W, H], CHROMAQ_AT, ROWQ);
    let facts = match matrix {
        Some(matrix) => facts.with_matrix(matrix),
        None => facts,
    };
    let at = IoSurfacePlane::new(0, ROWQ, true).with_ycbcr(facts);
    surface.new_texture_with_iosurface_plane(&d, at)
}

/// The 4:2:2 and 4:4:4 formats over their own surfaces read what Metal reads (`ycbcr_read` states
/// it): `YCBCR8_422_2P`, `YCBCR8_444_2P`, `YCBCR10_422_2P`, `YCBCR10_444_2P` and their `_sRGB`
/// twins over `422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44`, without a matrix,
/// with BT.709, with BT.2020 and with SMPTE 240M (read as BT.709): `read()`, a nearest and a linear
/// sample at each texel's centre, and a `half` read (the twins' steps rounded with ties away from
/// zero). No filter notice. A format over a surface whose chroma plane holds its own reads its
/// pairs at the surface's rows; over one whose plane is smaller it is refused.
#[test]
fn four_two_two_and_four_four_four_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut seed = 120;
        for (pf, ostypes) in [
            (PixelFormat::YCBCR84222P, ["422v", "422f"]),
            (PixelFormat::YCBCR84442P, ["444v", "444f"]),
            (PixelFormat::YCBCR104222P, ["x422", "xf22"]),
            (PixelFormat::YCBCR104442P, ["x444", "xf44"]),
            (PixelFormat::YCBCR84222PSRGB, ["422v", "422f"]),
            (PixelFormat::YCBCR84442PSRGB, ["444v", "444f"]),
            (PixelFormat::YCBCR104222PSRGB, ["x422", "xf22"]),
            (PixelFormat::YCBCR104442PSRGB, ["x444", "xf44"]),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            for (ostype, range) in ostypes
                .into_iter()
                .zip([YCbCrRange::Video, YCbCrRange::Full])
            {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                    (Some(SurfaceYCbCrMatrix::Smpte240M), YCbCrMatrix::Bt709),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    let want = family_modelled(&bytes, codes, (range, modelled));
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let of = |texels: &[u8]| grid_of(texels, grid);
                    let read = of(&texels_of(device, &kernels.read, &texture, None));
                    assert_eq!(read, want, "{case}: read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        assert_eq!(of(&sampled), want, "{case}: a {filter} sample");
                    }
                    let half = texels_of(device, &kernels.read_half, &texture, None);
                    match codes.output {
                        YCbCrOutput::Srgb => {
                            assert_half_reads(&half, &want, &format!("{case}: a half read()"))
                        }
                        _ => assert_eq!(of(&half), want, "{case}: a half read()"),
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
        // Over a surface whose chroma plane holds the format's (for 4:2:2 over
        // 4:4:4): the format's pairs, at the surface's rows.
        for (seed, (pf, ostype, range)) in (160..).zip([
            (PixelFormat::YCBCR84222P, "444v", YCbCrRange::Video),
            (PixelFormat::YCBCR104222PSRGB, "xf44", YCbCrRange::Full),
            (PixelFormat::YCBCR84202P, "422f", YCbCrRange::Full),
            (PixelFormat::YCBCR84202PSRGB, "444v", YCbCrRange::Video),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (range, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?} over {ostype}");
        }
        // Over a surface whose chroma plane is smaller than the format's (Metal reads past it):
        // refused.
        for (pf, ostype) in [
            (PixelFormat::YCBCR84222P, "420v"),
            (PixelFormat::YCBCR84442P, "422v"),
            (PixelFormat::YCBCR104222P, "x420"),
            (PixelFormat::YCBCR104442P, "x422"),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let surface = buffer_with(device, &family_bytes(1, codes)).unwrap();
            let made = family_texture(&surface, pf, ostype, None);
            let named = format!("{:?}", made.as_ref().err());
            assert!(
                matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                "{pf:?} over {ostype}: {named}"
            );
        }
    });
}

/// A 4:2:2 or 4:4:4 texture's view as its `_sRGB` twin reads what a texture of the twin over the
/// same planes reads, as Metal's view does (made with `ShaderRead` alone), and the
/// texture itself its own conversion.
#[test]
fn four_two_two_and_four_four_four_views_read_their_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, twin, ostype)) in (140..).zip([
            (
                PixelFormat::YCBCR84222P,
                PixelFormat::YCBCR84222PSRGB,
                "422f",
            ),
            (
                PixelFormat::YCBCR84442P,
                PixelFormat::YCBCR84442PSRGB,
                "444f",
            ),
            (
                PixelFormat::YCBCR104222P,
                PixelFormat::YCBCR104222PSRGB,
                "xf22",
            ),
            (
                PixelFormat::YCBCR104442P,
                PixelFormat::YCBCR104442PSRGB,
                "xf44",
            ),
        ]) {
            let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes(pf));
            let surface = buffer_with(device, &bytes).unwrap();
            let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
            let matrix = Some(SurfaceYCbCrMatrix::Itu709);
            let base = family_texture(&surface, pf, ostype, matrix).unwrap();
            let view = base.new_texture_view(twin).unwrap();
            let direct = family_texture(&surface, twin, ostype, matrix).unwrap();
            let read = |t: &Texture, d| grid_of(&texels_of(device, &kernels.read, t, None), d);
            let want = family_modelled(&bytes, codes(twin), bt709);
            assert_eq!(read(&view, 8192), want, "{pf:?}'s view");
            assert_eq!(read(&direct, 8192), want, "{twin:?}");
            assert_eq!(
                read(&base, 2048),
                family_modelled(&bytes, codes(pf), bt709),
                "{pf:?}"
            );
        }
    });
}

/// A 4:2:2 or 4:4:4 texture's copies move its luma codes, a code's bytes a texel, as Metal's do:
/// `get_bytes` reads the luma rows, and `replace_region` writes them, keeping
/// the chroma plane, the texture then reading the new codes' conversion.
#[test]
fn four_two_two_and_four_four_four_textures_copy_their_luma_codes() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, ostype)) in (180..).step_by(2).zip([
            (PixelFormat::YCBCR84222P, "422v"),
            (PixelFormat::YCBCR104442P, "x444"),
            (PixelFormat::YCBCR104442PSRGB, "x444"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let row = u64::from(codes.bytes) * W;
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let mut luma = vec![0u8; (row * H) as usize];
            texture
                .get_bytes(&mut luma, row, 0, Region::new_2d(0, 0, W, H), 0, 0)
                .unwrap();
            let rows: Vec<u8> = (0..H)
                .flat_map(|y| bytes[(y * ROWQ) as usize..(y * ROWQ + row) as usize].to_vec())
                .collect();
            assert_eq!(luma, rows, "{pf:?}: get_bytes");
            let mut random = Random(seed + 1);
            let new: Vec<u8> = (0..row * H).map(|_| random.next()).collect();
            texture
                .replace_region(Region::new_2d(0, 0, W, H), 0, 0, &new, row, 0)
                .unwrap();
            let mut want = bytes.clone();
            for y in 0..H {
                let at = (y * ROWQ) as usize;
                want[at..at + row as usize]
                    .copy_from_slice(&new[(y * row) as usize..((y + 1) * row) as usize]);
            }
            let length = want.len() as u64;
            let written = read_buffer(&surface, 0, length).unwrap();
            assert_eq!(written, want, "{pf:?}: replace_region");
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let modelled = family_modelled(&want, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            assert_eq!(read, modelled, "{pf:?}: read() after replace_region");
        }
    });
}

/// What the gather kernel `state` reads of `texture` through `sampler` at the `(W + 1) x (H + 1)`
/// texel corners, a `float4` each, in one command buffer of its own.
fn gathered_of(
    device: &Device,
    state: &ComputePipelineState,
    texture: &Texture,
    sampler: &SamplerState,
) -> Vec<u8> {
    let length = 16 * (W + 1) * (H + 1);
    let out = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    dispatch(
        &command_buffer,
        state,
        texture,
        &out,
        Some(sampler),
        [W + 1, H + 1],
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&out, 0, length).unwrap()
}

/// Whether the step `step / 8192` lies halfway between two `half`s, where nearest even and ties
/// away part.
fn is_half_tie(step: u16) -> bool {
    let v = f64::from(step) / 8192.0;
    step != 0 && (v / 2f64.powi(v.log2().floor() as i32 - 10)).fract() == 0.5
}

/// A `half` gather of a two-plane YCbCr texture rounds each lane with ties away from zero, as
/// Metal's does: red and blue of `YCBCR8_420_2P_sRGB` over `420v` without a matrix, with BT.709 and
/// with BT.2020, `YCBCR10_420_2P_sRGB` over `x420`, the 4:2:2 and 4:4:4 twins, the packed 4:2:0 and
/// 4:4:4 twins, and `YCBCR8_420_2P`, at every texel corner under clamp-to-edge, clamp-to-zero and
/// clamp-to-border-color: each lane the `float` gather's step of `1/8192` rounded so, ties among
/// them.
#[test]
fn ycbcr_half_gathers_round_ties_away() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let usage = TextureUsage::SHADER_READ;
        let samplers = [
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToZero,
            SamplerAddressMode::ClampToBorderColor,
        ]
        .map(|mode| {
            let mut d = SamplerDescriptor::new();
            d.s_address_mode = mode;
            d.t_address_mode = mode;
            (mode, device.new_sampler_state(&d).unwrap())
        });
        let mut textures: Vec<(PixelFormat, String, Texture)> = Vec::new();
        for (seed, (pf, matrix)) in (200..).zip([
            (PixelFormat::YCBCR84202PSRGB, None),
            (
                PixelFormat::YCBCR84202PSRGB,
                Some(SurfaceYCbCrMatrix::Itu709),
            ),
            (
                PixelFormat::YCBCR84202PSRGB,
                Some(SurfaceYCbCrMatrix::Itu2020),
            ),
            (PixelFormat::YCBCR84202P, Some(SurfaceYCbCrMatrix::Itu709)),
        ]) {
            let surface = buffer_with(device, &surface_bytes(seed)).unwrap();
            let texture = eight_bit_texture(&surface, pf, Some(facts(VIDEO, matrix)), usage);
            textures.push((pf, format!("420v {matrix:?}"), texture.unwrap()));
        }
        let ten = buffer_with(device, &ten_bit_bytes(210)).unwrap();
        let pf = PixelFormat::YCBCR104202PSRGB;
        let matrix = Some(SurfaceYCbCrMatrix::Itu709);
        let texture = ten_bit_texture(&ten, pf, (b"x420", 0), matrix).unwrap();
        textures.push((pf, "x420".into(), texture));
        for (seed, (pf, ostype)) in (220..).zip([
            (PixelFormat::YCBCR84222PSRGB, "422v"),
            (PixelFormat::YCBCR84442PSRGB, "444f"),
            (PixelFormat::YCBCR104222PSRGB, "x422"),
            (PixelFormat::YCBCR104442PSRGB, "xf44"),
            (PixelFormat::YCBCR104202PPACKEDSRGB, "p420"),
            (PixelFormat::YCBCR104442PPACKEDSRGB, "pf44"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let surface = buffer_with(device, &family_bytes(seed, codes)).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            textures.push((pf, ostype.into(), texture));
        }
        let mut ties = 0;
        for (pf, over, texture) in &textures {
            for (mode, state) in &samplers {
                for (channel, [float, half]) in ["red", "blue"].iter().zip(&kernels.gathers) {
                    let floats = gathered_of(device, float, texture, state);
                    let halves = gathered_of(device, half, texture, state);
                    let lanes = floats
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(halves.as_chunks::<4>().0.iter());
                    for (i, (f, h)) in lanes.enumerate() {
                        let [f, h] = [f, h].map(|b| f32::from_le_bytes(*b));
                        let case = format!("{pf:?} over {over}, {mode:?}, {channel} lane {i}");
                        let step = f * 8192.0;
                        assert!(
                            step.fract() == 0.0 && (0.0..=8192.0).contains(&step),
                            "{case}: {f}"
                        );
                        let step = step as u16;
                        assert_eq!(h, half_away(step), "{case}: step {step}");
                        ties += usize::from(is_half_tie(step));
                    }
                }
            }
        }
        assert!(ties > 0, "no lane lay halfway between two halves");
    });
}

/// The 4:2:2 and 4:4:4 code twins read their codes as Metal does (every
/// texel of 24 surfaces; `ycbcr_read` states it): `RGB8_422_2P`, `RGB8_444_2P`, `RGB10_422_2P`
/// and `RGB10_444_2P` over their own surfaces (`422v`, `422f`, `444v`, `444f`, `x422`, `xf22`,
/// `x444`, `xf44`), and the 4:2:2 ones over a 4:4:4 surface (whose plane theirs fits in), without
/// a matrix, with BT.709 and with BT.2020: red Cr, green Y, blue Cb, each `code / (2^b - 1)`,
/// alpha 1, after the format's chroma filter, whatever the matrix: `read()`, and a nearest and a
/// linear sample at each texel's centre. No filter notice.
#[test]
fn four_two_two_and_four_four_four_codes_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut seed = 160;
        for (pf, ostypes) in [
            (PixelFormat::RGB84222P, &["422v", "422f", "444v"][..]),
            (PixelFormat::RGB84442P, &["444v", "444f"]),
            (PixelFormat::RGB104222P, &["x422", "xf22", "x444"]),
            (PixelFormat::RGB104442P, &["x444", "xf44"]),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert_eq!(codes.output, YCbCrOutput::Codes, "{pf:?}");
            let max = (1u16 << codes.bits) - 1;
            for &ostype in ostypes {
                for matrix in [
                    None,
                    Some(SurfaceYCbCrMatrix::Itu709),
                    Some(SurfaceYCbCrMatrix::Itu2020),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    // The codes, which no range or matrix changes.
                    let want =
                        family_modelled(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let read = code_grid(&texels_of(device, &kernels.read, &texture, None), max);
                    assert_eq!(read, want, "{case}: read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        assert_eq!(code_grid(&sampled, max), want, "{case}: a {filter} sample");
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
        // Over a surface whose chroma plane theirs does not fit in (Metal reads past it): refused.
        for (pf, ostype) in [
            (PixelFormat::RGB84222P, "420v"),
            (PixelFormat::RGB84442P, "422v"),
            (PixelFormat::RGB104222P, "x420"),
            (PixelFormat::RGB104442P, "x422"),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let surface = buffer_with(device, &family_bytes(1, codes)).unwrap();
            let made = family_texture(&surface, pf, ostype, None);
            let named = format!("{:?}", made.as_ref().err());
            assert!(
                matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                "{pf:?} over {ostype}: {named}"
            );
        }
    });
}

/// A texture over plane 1 of a 4:2:2 or 4:4:4 surface reads plane 1's codes as its luma plane,
/// at plane 1's rows, with chroma codes 0, as Metal does (every one of the 18
/// formats over plane 1 of `422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44`,
/// 0 texels off that model): each format over each such surface of its bits, without a matrix
/// and with BT.709, reads (`read()`, a nearest sample, and the converted ones' `half` read) what
/// it reads over plane 0 of a surface whose luma rows hold plane 1's codes and whose chroma
/// plane holds 0; `get_bytes` moves plane 1's codes. Over plane 1 of a 4:2:0 surface (whose rows
/// Metal reads past the plane), over plane 2, and over plane 1 at other bytes than the facts
/// name, it is refused, naming the format.
#[test]
fn ycbcr_textures_over_plane_1_read_its_codes_as_luma() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let over_plane_1 = |surface: &Buffer,
                            pf: PixelFormat,
                            ostype: &str,
                            plane: u64,
                            at: u64,
                            matrix: Option<SurfaceYCbCrMatrix>| {
            let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::shared();
            let code: [u8; 4] = ostype.as_bytes().try_into().unwrap();
            let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(code), plane, [W, H], at, ROWQ);
            let facts = match matrix {
                Some(matrix) => facts.with_matrix(matrix),
                None => facts,
            };
            let plane = IoSurfacePlane::new(CHROMAQ_AT, ROWQ, true).with_ycbcr(facts);
            surface.new_texture_with_iosurface_plane(&d, plane)
        };
        let mut seed = 300;
        for pf in [
            PixelFormat::YCBCR84202P,
            PixelFormat::YCBCR84222P,
            PixelFormat::YCBCR84442P,
            PixelFormat::YCBCR84202PSRGB,
            PixelFormat::YCBCR84222PSRGB,
            PixelFormat::YCBCR84442PSRGB,
            PixelFormat::RGB84202P,
            PixelFormat::RGB84222P,
            PixelFormat::RGB84442P,
            PixelFormat::YCBCR104202P,
            PixelFormat::YCBCR104222P,
            PixelFormat::YCBCR104442P,
            PixelFormat::YCBCR104202PSRGB,
            PixelFormat::YCBCR104222PSRGB,
            PixelFormat::YCBCR104442PSRGB,
            PixelFormat::RGB104202P,
            PixelFormat::RGB104222P,
            PixelFormat::RGB104442P,
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let unit = u64::from(codes.bytes);
            let ostypes = match codes.bits {
                8 => ["422v", "422f", "444v", "444f"],
                _ => ["x422", "xf22", "x444", "xf44"],
            };
            for (ostype, range) in ostypes.into_iter().zip([
                YCbCrRange::Video,
                YCbCrRange::Full,
                YCbCrRange::Video,
                YCbCrRange::Full,
            ]) {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                ] {
                    seed += 1;
                    let mut random = Random(seed);
                    let bytes: Vec<u8> = (0..2 * ROWQ * H).map(|_| random.next()).collect();
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture =
                        over_plane_1(&surface, pf, ostype, 1, CHROMAQ_AT, matrix).unwrap();
                    // Plane 1's codes as the luma rows, the chroma plane 0.
                    let mut moved = vec![0; bytes.len()];
                    let row = (unit * W) as usize;
                    for y in 0..H as usize {
                        let from = CHROMAQ_AT as usize + y * ROWQ as usize;
                        let to = y * ROWQ as usize;
                        moved[to..to + row].copy_from_slice(&bytes[from..from + row]);
                    }
                    let case = format!("{pf:?} over plane 1 of {ostype}, {matrix:?}");
                    let (want, grid) = match codes.output {
                        YCbCrOutput::Codes => (
                            family_modelled(&moved, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601)),
                            None,
                        ),
                        YCbCrOutput::Srgb => (
                            family_modelled(&moved, codes, (range, modelled)),
                            Some(8192),
                        ),
                        _ => (
                            family_modelled(&moved, codes, (range, modelled)),
                            Some(2048),
                        ),
                    };
                    let of = |texels: &[u8]| match grid {
                        Some(d) => grid_of(texels, d),
                        None => code_grid(texels, (1u16 << codes.bits) - 1),
                    };
                    let read = texels_of(device, &kernels.read, &texture, None);
                    assert_eq!(of(&read), want, "{case}: read()");
                    let sampled = texels_of(device, &kernels.sample, &texture, Some(&nearest));
                    assert_eq!(of(&sampled), want, "{case}: a nearest sample");
                    if grid.is_some() {
                        let half = texels_of(device, &kernels.read_half, &texture, None);
                        match codes.output {
                            YCbCrOutput::Srgb => {
                                assert_half_reads(&half, &want, &format!("{case}: a half read()"))
                            }
                            _ => assert_eq!(of(&half), want, "{case}: a half read()"),
                        }
                    }
                    let mut luma = vec![0u8; row * H as usize];
                    texture
                        .get_bytes(&mut luma, row as u64, 0, Region::new_2d(0, 0, W, H), 0, 0)
                        .unwrap();
                    let codes_of_plane_1: Vec<u8> = (0..H as usize)
                        .flat_map(|y| moved[y * ROWQ as usize..][..row].to_vec())
                        .collect();
                    assert!(luma == codes_of_plane_1, "{case}: get_bytes");
                }
            }
        }
        // Refused: plane 1 of a 4:2:0 surface, plane 2, plane 1 at other bytes than the facts'.
        let surface = buffer_with(device, &vec![0; (2 * ROWQ * H) as usize]).unwrap();
        for (pf, ostype, plane, at) in [
            (PixelFormat::YCBCR84202P, "420v", 1, CHROMAQ_AT),
            (PixelFormat::YCBCR104222P, "x420", 1, CHROMAQ_AT),
            (PixelFormat::YCBCR84222P, "422v", 2, CHROMAQ_AT),
            (PixelFormat::YCBCR84442P, "444v", 1, CHROMAQ_AT + ROWQ),
        ] {
            let made = over_plane_1(&surface, pf, ostype, plane, at, None);
            let named = format!("{:?}", made.as_ref().err());
            assert!(
                matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                "{pf:?} over plane {plane} of {ostype} at {at}: {named}"
            );
        }
    });
}

/// The packed formats, each with its own surfaces of video and of full range.
const PACKED: [(PixelFormat, [&str; 2]); 6] = [
    (PixelFormat::YCBCR104202PPACKED, ["p420", "pf20"]),
    (PixelFormat::YCBCR104222PPACKED, ["p422", "pf22"]),
    (PixelFormat::YCBCR104442PPACKED, ["p444", "pf44"]),
    (PixelFormat::YCBCR104202PPACKEDSRGB, ["p420", "pf20"]),
    (PixelFormat::YCBCR104222PPACKEDSRGB, ["p422", "pf22"]),
    (PixelFormat::YCBCR104442PPACKEDSRGB, ["p444", "pf44"]),
];

/// The packed formats over their own surfaces read what Metal reads (`ycbcr_read` states it):
/// `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED`, `YCBCR10_444_2P_PACKED` and their `_sRGB`
/// twins over `p420`, `pf20`, `p422`, `pf22`, `p444` and `pf44`, three codes a 32-bit word, without
/// a matrix, with BT.709 and with BT.2020: `read()`, a nearest and a linear sample at each texel's
/// centre, a `half` read (the twins' steps rounded with ties away from zero), and a linear sample
/// between texels Metal's step (every case of the six formats). No filter notice. A 4:2:0 format
/// over a packed 4:4:4 surface reads its pairs at the surface's rows; a format over a surface whose
/// plane is smaller, a packed one over an unpacked surface, an unpacked one over a packed surface,
/// and one over plane 1 of a packed surface are refused.
#[test]
fn packed_textures_read_what_metal_reads() {
    on_device(|device, messages| {
        let kernels = Kernels::new(device);
        let nearest = sampler(device, SamplerMinMagFilter::Nearest);
        let linear = sampler(device, SamplerMinMagFilter::Linear);
        let mut seed = 300;
        for (pf, ostypes) in PACKED {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            assert!(codes.packed, "{pf:?}");
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            for (ostype, range) in ostypes
                .into_iter()
                .zip([YCbCrRange::Video, YCbCrRange::Full])
            {
                for (matrix, modelled) in [
                    (None, YCbCrMatrix::Bt601),
                    (Some(SurfaceYCbCrMatrix::Itu709), YCbCrMatrix::Bt709),
                    (Some(SurfaceYCbCrMatrix::Itu2020), YCbCrMatrix::Bt2020),
                ] {
                    seed += 1;
                    let bytes = family_bytes(seed, codes);
                    let surface = buffer_with(device, &bytes).unwrap();
                    let texture = family_texture(&surface, pf, ostype, matrix).unwrap();
                    let want = family_modelled(&bytes, codes, (range, modelled));
                    let case = format!("{pf:?} over {ostype}, {matrix:?}");
                    let of = |texels: &[u8]| grid_of(texels, grid);
                    let read = of(&texels_of(device, &kernels.read, &texture, None));
                    assert_eq!(read, want, "{case}: read()");
                    for (filter, state) in [("nearest", &nearest), ("linear", &linear)] {
                        let sampled = texels_of(device, &kernels.sample, &texture, Some(state));
                        assert_eq!(of(&sampled), want, "{case}: a {filter} sample");
                    }
                    let half = texels_of(device, &kernels.read_half, &texture, None);
                    match codes.output {
                        YCbCrOutput::Srgb => {
                            assert_half_reads(&half, &want, &format!("{case}: a half read()"))
                        }
                        _ => assert_eq!(of(&half), want, "{case}: a half read()"),
                    }
                    let between = texels_of(device, &kernels.between, &texture, Some(&linear));
                    for (i, got) in (0u64..).zip(of(&between)) {
                        let at = (i % W, i / W);
                        let step: [u16; 3] =
                            std::array::from_fn(|c| metal_step(&read, at, BETWEEN, c) as u16);
                        assert_eq!(got, step, "{case}: a linear sample between, texel {at:?}");
                    }
                }
            }
        }
        let notices = counts(&messages.take_notices());
        for kind in [
            Degradation::FloatSampleFilter,
            Degradation::YCbCrFilterNotExact,
        ] {
            assert_eq!(notices.get(&kind), None, "{notices:?}");
        }
        // Over a surface whose chroma plane holds the format's (4:2:0 over 4:2:2
        // and 4:4:4, 4:2:2 over 4:4:4): the format's pairs, at the surface's rows.
        for (seed, (pf, ostype, range)) in (340..).zip([
            (PixelFormat::YCBCR104202PPACKED, "p444", YCbCrRange::Video),
            (
                PixelFormat::YCBCR104202PPACKEDSRGB,
                "p422",
                YCbCrRange::Video,
            ),
            (PixelFormat::YCBCR104222PPACKED, "p444", YCbCrRange::Video),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (range, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?} over {ostype}");
        }
        // Over a surface whose chroma plane is smaller than the format's (Metal reads past it),
        // or whose codes are held otherwise (not measured): refused.
        for (pf, ostype) in [
            (PixelFormat::YCBCR104222PPACKED, "p420"),
            (PixelFormat::YCBCR104442PPACKEDSRGB, "pf22"),
            (PixelFormat::YCBCR104202PPACKED, "x420"),
            (PixelFormat::YCBCR104442PPACKED, "xf44"),
            (PixelFormat::YCBCR104202P, "p420"),
            (PixelFormat::YCBCR104442PSRGB, "pf44"),
        ] {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let surface = buffer_with(device, &family_bytes(1, codes)).unwrap();
            let made = family_texture(&surface, pf, ostype, None);
            let named = format!("{:?}", made.as_ref().err());
            assert!(
                matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                "{pf:?} over {ostype}: {named}"
            );
        }
        // Over plane 1 of a packed surface (not measured; only the unpacked ones' are read):
        // refused.
        let pf = PixelFormat::YCBCR104442PPACKED;
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
        let surface = buffer_with(device, &family_bytes(2, codes)).unwrap();
        let mut d = TextureDescriptor::texture_2d(pf, W, H, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let facts = IoSurfaceYCbCr::new(u32::from_be_bytes(*b"p444"), 1, [W, H], CHROMAQ_AT, ROWQ);
        let plane = IoSurfacePlane::new(CHROMAQ_AT, ROWQ, true).with_ycbcr(facts);
        let made = surface.new_texture_with_iosurface_plane(&d, plane);
        let named = format!("{:?}", made.as_ref().err());
        assert!(
            matches!(made, Err(Error::Unsupported(_))) && named.contains(pf.name()),
            "{pf:?} over plane 1 of p444: {named}"
        );
    });
}

/// A packed texture's view as its `_sRGB` twin reads what a texture of the twin over the same
/// planes reads, as Metal's view does (for 4:2:0 and 4:2:2, made with `ShaderRead`
/// alone; 4:4:4 by the rule every measured pair follows), and the texture itself its own
/// conversion.
#[test]
fn packed_views_read_their_twins_texels() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        for (seed, (pf, ostype)) in (350..).zip([
            (PixelFormat::YCBCR104202PPACKED, "pf20"),
            (PixelFormat::YCBCR104222PPACKED, "pf22"),
            (PixelFormat::YCBCR104442PPACKED, "pf44"),
        ]) {
            let twin = PACKED
                .iter()
                .map(|&(twin, _)| twin)
                .find(|&twin| {
                    let [a, b] = [pf, twin].map(|pf| ycbcr_format(pf).unwrap());
                    (a.subsampling, b.reads) == (b.subsampling, YCbCrReads::Srgb)
                })
                .unwrap();
            let codes = |pf: PixelFormat| ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes(pf));
            let surface = buffer_with(device, &bytes).unwrap();
            let bt709 = (YCbCrRange::Full, YCbCrMatrix::Bt709);
            let matrix = Some(SurfaceYCbCrMatrix::Itu709);
            let base = family_texture(&surface, pf, ostype, matrix).unwrap();
            let view = base.new_texture_view(twin).unwrap();
            let direct = family_texture(&surface, twin, ostype, matrix).unwrap();
            let read = |t: &Texture, d| grid_of(&texels_of(device, &kernels.read, t, None), d);
            let want = family_modelled(&bytes, codes(twin), bt709);
            assert_eq!(read(&view, 8192), want, "{pf:?}'s view");
            assert_eq!(read(&direct, 8192), want, "{twin:?}");
            assert_eq!(
                read(&base, 2048),
                family_modelled(&bytes, codes(pf), bt709),
                "{pf:?}"
            );
        }
    });
}

/// A packed texture's CPU calls move nothing, as Metal's (48 `getBytes` and 48
/// `replaceRegion` calls): `get_bytes` leaves the bytes it is given and `replace_region` the
/// planes, at a row's words, twice them, four bytes a texel and 256 bytes a row, the texture
/// reading the planes' conversion still. A blit with a buffer either way, whose bytes Metal moves
/// by no rule found, is refused, naming the format.
#[test]
fn packed_textures_copy_nothing_and_refuse_blits() {
    on_device(|device, _| {
        let kernels = Kernels::new(device);
        let queue = device.new_command_queue().unwrap();
        for (seed, (pf, ostype)) in (360..).zip([
            (PixelFormat::YCBCR104202PPACKED, "p420"),
            (PixelFormat::YCBCR104222PPACKEDSRGB, "p422"),
            (PixelFormat::YCBCR104442PPACKED, "p444"),
        ]) {
            let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).unwrap();
            let bytes = family_bytes(seed, codes);
            let surface = buffer_with(device, &bytes).unwrap();
            let texture = family_texture(&surface, pf, ostype, None).unwrap();
            let words = row_bytes(codes, W);
            let region = Region::new_2d(0, 0, W, H);
            for row in [words, 2 * words, 4 * W, 256] {
                let mut got = vec![0x5a; (row * H) as usize];
                texture.get_bytes(&mut got, row, 0, region, 0, 0).unwrap();
                assert!(
                    got.iter().all(|&b| b == 0x5a),
                    "{pf:?}: get_bytes at {row} wrote"
                );
                let new: Vec<u8> = (0..row * H).map(|i| (i * 7) as u8).collect();
                texture.replace_region(region, 0, 0, &new, row, 0).unwrap();
                let length = bytes.len() as u64;
                let planes = read_buffer(&surface, 0, length).unwrap();
                assert_eq!(planes, bytes, "{pf:?}: replace_region at {row}");
            }
            let grid = match codes.output {
                YCbCrOutput::Srgb => 8192,
                _ => 2048,
            };
            let read = grid_of(&texels_of(device, &kernels.read, &texture, None), grid);
            let want = family_modelled(&bytes, codes, (YCbCrRange::Video, YCbCrMatrix::Bt601));
            assert_eq!(read, want, "{pf:?}: read() after the calls");
            let other = device
                .new_buffer(4 * W * H, ResourceOptions::shared())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            let size = Size::new(W, H, 1);
            let out = blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                size,
                &other,
                0,
                4 * W,
                0,
                BlitOption::None,
            );
            let into = blit.copy_from_buffer_to_texture(
                &other,
                0,
                4 * W,
                0,
                size,
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            );
            for (way, refused) in [("out", out), ("in", into)] {
                let named = format!("{refused:?}");
                assert!(
                    matches!(refused, Err(Error::Unsupported(_))) && named.contains(pf.name()),
                    "{pf:?}: a blit {way}: {named}"
                );
            }
            blit.end_encoding().unwrap();
        }
    });
}
