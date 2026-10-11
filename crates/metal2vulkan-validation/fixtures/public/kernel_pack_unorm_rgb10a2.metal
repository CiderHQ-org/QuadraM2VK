// `air.pack.unorm.rgb10a2.v4f32` (6 corpus sources) and `air.pack.unorm.rgb10a2.v4f16` (6). Neither
// had any lowering at all before this fixture -- the translator answered "unhandled pack intrinsic"
// and fell back -- and neither had an authored case.
//
// The pack is not the same arithmetic as the other normalized packs. Metal rounds the EXACT real
// product `x * 1023`, not the `float` product: the inputs in the first fourteen threads were chosen
// by search precisely so the two models disagree, and the float-product model gets eight of them
// wrong. The half inputs in the last sixteen threads are chosen the same way against a model that
// computes the product in half precision. Between them the two halves of this case pin the rounding
// model itself, not merely that a pack happened.
//
// The remaining threads cover the edges the models share: components below zero and above one (the
// pack saturates), NaN (Metal answers zero), the exactly-representable 0.5, and the two-bit alpha
// lane, which uses the same rounding against a multiplier of 3.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_pack_unorm_rgb10a2(
    device uint *out [[buffer(0)]],
    device const float4 *wide [[buffer(1)]],
    device const half4 *narrow [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid * 2 + 0] = pack_float_to_unorm10a2(wide[tid]);
    out[tid * 2 + 1] = pack_half_to_unorm10a2(narrow[tid]);
}
