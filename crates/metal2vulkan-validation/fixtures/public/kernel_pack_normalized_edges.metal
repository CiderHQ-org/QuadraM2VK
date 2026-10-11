// Every normalized pack family asked for NaN, both infinities and out-of-range magnitudes, on
// inputs the compiler cannot see.
//
// The seven families here already have cases for their in-range arithmetic. What none of them has
// is an EDGE, and the translator does not answer the edge the same way twice:
// `lower_pack_rgb10a2` selects an explicit zero for a NaN component "because Metal answers 0 for a
// NaN component and SPIR-V leaves FClamp of a NaN undefined", and `lower_pack_rgb565` -- the other
// hand-rolled normalized pack, written to the same shape -- clamps without that select. Two
// derivations of one fact, so one of them is wrong; this case asks Metal which.
//
// **The inputs come from a buffer, and they have to.** Metal compiles this corpus with
// `no-nans-fp-math`, so a NaN literal in the source is licence for the frontend to fold the call
// away and the probe would measure the optimizer instead of the runtime.
//
// Every component is one of NaN, -NaN, +/-inf, +/-0, +/-1, +/-2, +/-1e30 or +/-1e-30, so after the
// clamp every product with its field maximum is an exact integer or is 1e-28 away from one. No lane
// depends on a rounding rule, on a tie direction, or on the order of a multiply -- which is the
// point: the only thing this case can disagree about is what the edge itself does. The rows mix the
// edges rather than repeating them, so a lowering that handled NaN per-vector instead of
// per-component (t4 and t5 pair a NaN with an ordinary value) or that confused the alpha field with
// a colour one (t0 and t6 differ only in lanes 0 and 1) still lands somewhere visible.
//
// `snorm` clamps to [-1, 1] and `unorm` to [0, 1], so the same -1 input is a saturated field in one
// family and a zero in the other, and the same +2 input saturates both. The sRGB family transfers
// only its three colour lanes and leaves alpha linear, so t7 -- all ones -- is the one row where
// every family must answer all-ones, and any row where sRGB matches plain unorm on the colour lanes
// is a row whose colours are saturated at an endpoint the transfer fixes.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_pack_normalized_edges(
    device const float *in [[buffer(0)]],
    device uint *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    float4 f = float4(in[tid * 4 + 0], in[tid * 4 + 1], in[tid * 4 + 2], in[tid * 4 + 3]);
    uint b = tid * 7;
    out[b + 0] = pack_float_to_unorm4x8(f);
    out[b + 1] = pack_float_to_snorm4x8(f);
    out[b + 2] = pack_float_to_unorm2x16(f.xy);
    out[b + 3] = pack_float_to_snorm2x16(f.xy);
    out[b + 4] = pack_float_to_unorm10a2(f);
    out[b + 5] = uint(pack_float_to_unorm565(f.xyz));
    out[b + 6] = pack_float_to_srgb_unorm4x8(f);
}
