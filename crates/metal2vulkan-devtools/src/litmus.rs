//! The store-buffering litmus kernel, the AIR twin of our MSL probe, which the product's
//! unit test compiles and validation's lane runs on devices.
//!
//! Kernel `sb(device atomic_uint *x [[buffer(0)]], device atomic_uint *y [[buffer(1)]], device
//! uint *r [[buffer(2)]])`, run as threadgroups of 256 threads: threadgroup `2k` thread `l`
//! stores 1 to `x[i]` (relaxed, device scope), runs the fence, and loads `y[i]` into `r[2i]`, for
//! `i = 256k + l`; threadgroup `2k + 1` thread `l` stores 1 to `y[i]`, runs the fence, and loads
//! `x[i]` into `r[2i + 1]`, for `i = 256k + (37 l mod 256)`, so each pair's two threads sit at
//! different lanes of neighbouring threadgroups. A pair that reads 0 twice saw both stores
//! buffered past the fences, which a sequentially consistent fence forbids.

/// The store-buffering kernel `sb` with `fence` between each thread's store and load (empty for
/// none).
pub fn store_buffering(fence: &str) -> String {
    let side = |me: &str, other: &str, slot: u32, index: &str| {
        format!(
            "  {index}
  %e{me} = zext i32 %i{me} to i64
  %p{me} = getelementptr inbounds i32, ptr addrspace(1) %{me}, i64 %e{me}
  tail call void @air.atomic.global.store.i32(ptr addrspace(1) %p{me}, i32 1, i32 0, i32 2, i1 true)
{fence}
  %q{me} = getelementptr inbounds i32, ptr addrspace(1) %{other}, i64 %e{me}
  %v{me} = tail call i32 @air.atomic.global.load.i32(ptr addrspace(1) %q{me}, i32 0, i32 2, i1 true)
  %d{me} = shl i32 %i{me}, 1
  %s{me} = add i32 %d{me}, {slot}
  %f{me} = zext i32 %s{me} to i64
  %r{me} = getelementptr inbounds i32, ptr addrspace(1) %r, i64 %f{me}
  store i32 %v{me}, ptr addrspace(1) %r{me}, align 4
  ret void
"
        )
    };
    let a = side("x", "y", 0, "%ix = add i32 %base, %l");
    let b = side(
        "y",
        "x",
        1,
        "%lm = mul i32 %l, 37\n  %lp = and i32 %lm, 255\n  %iy = add i32 %base, %lp",
    );
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @sb(ptr addrspace(1) noundef \"air-buffer-no-alias\" %x, ptr addrspace(1) noundef \"air-buffer-no-alias\" %y, ptr addrspace(1) noundef \"air-buffer-no-alias\" %r, i32 noundef %l, i32 noundef %tg) {{
entry:
  %odd = and i32 %tg, 1
  %pair = lshr i32 %tg, 1
  %base = shl i32 %pair, 8
  %isodd = icmp ne i32 %odd, 0
  br i1 %isodd, label %second, label %first
first:
{a}second:
{b}}}

declare void @air.atomic.fence(i32, i32, i32)
declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)

!air.kernel = !{{!0}}
!air.version = !{{!8}}
!air.language_version = !{{!9}}
!0 = !{{ptr @sb, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6, !7}}
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"x\"}}
!4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"y\"}}
!5 = !{{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"r\"}}
!6 = !{{i32 3, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"l\"}}
!7 = !{{i32 4, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tg\"}}
!8 = !{{i32 2, i32 8, i32 0}}
!9 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
"
    )
}
