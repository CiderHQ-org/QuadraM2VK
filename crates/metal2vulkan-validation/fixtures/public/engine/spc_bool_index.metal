// A `bool` element index. MSL's `v[b]` with a `bool` `b` compiles to
// `extractelement` and `insertelement` with an `i1` index; the engine widens it to the `u32`
// 0 or 1 before `OpVectorExtractDynamic` and `OpVectorInsertDynamic`, which take an integer
// index (it was an internal builder error before).
//
// A case runs four threads over `in` and `sel`: `out[tid]` is `(v[b], v'.x, v'.y, 0)`, where `v'`
// is `v` with element `b` set to 7, and `b` is `sel[tid] != 0`.
#include <metal_stdlib>
using namespace metal;

kernel void spc_bool_index(device const float2 *in [[buffer(0)]],
                           device const uint *sel [[buffer(1)]],
                           device float4 *out [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
  float2 v = in[tid];
  bool b = sel[tid] != 0;
  float x = v[b];
  v[b] = 7.0f;
  out[tid] = float4(x, v.x, v.y, 0.0f);
}
