// Owned fixture: reads of an argument a function constant removes, behind a
// run-time flag. With `has_source` false or undefined, `source` (and `smp`) are not declared in the
// specialized function (MSL 5.8.1.2), so Metal gives a read of one no value. A run whose `flag`
// is 0 never reaches the read and has Metal's answer: each kernel writes values of its own. A run
// whose `flag` is nonzero reaches it: the translation takes zero for the read (Metal's answer
// under its debug layer) and marks the command buffer's execution status, and the command buffer
// completes. With `has_source` true they are reads.
#include <metal_stdlib>
using namespace metal;

constant bool has_source [[function_constant(0)]];

// Thread i writes 7 + i, or, where `flag` is nonzero, the width of `source`.
kernel void rfs_absent_query(
    texture2d<float, access::read> source [[texture(0), function_constant(has_source)]],
    device uint *out [[buffer(0)]],
    constant uint &flag [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  uint v = 7u + tid;
  if (flag != 0u) {
    v = source.get_width();
  }
  out[tid] = v;
}

// Thread i writes (i, 0.5, 0.25, 1), or, where `flag` is nonzero, texel (i, 0) of `source`.
kernel void rfs_absent_read(
    texture2d<float, access::read> source [[texture(0), function_constant(has_source)]],
    device float4 *out [[buffer(0)]],
    constant uint &flag [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  float4 c = float4(float(tid), 0.5f, 0.25f, 1.0f);
  if (flag != 0u) {
    c = source.read(uint2(tid, 0));
  }
  out[tid] = c;
}

// Thread i writes texel (i, 0) of `tex`, or, where `flag` is nonzero, a sample of `tex` through
// `smp`, a sampler the constant removes.
kernel void rfs_absent_sampler(
    texture2d<float, access::sample> tex [[texture(0)]],
    sampler smp [[sampler(0), function_constant(has_source)]],
    device float4 *out [[buffer(0)]],
    constant uint &flag [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  float4 c = tex.read(uint2(tid, 0));
  if (flag != 0u) {
    c = tex.sample(smp, float2(0.5f));
  }
  out[tid] = c;
}
