#include <metal_stdlib>
using namespace metal;

// One buffer SLOT declared at two widths, each live under a mutually exclusive function constant --
// Metal's own way of typing a single descriptor by specialization. Reading element `gid` through the
// selected view strides by 2 bytes or by 4 over the SAME bytes, so the two specializations disagree
// on every lane and neither can be answered without honouring the function constant.
constant bool kWeightsAreHalf [[function_constant(0)]];
constant bool kWeightsAreFloat = !kWeightsAreHalf;

kernel void read_the_fc_selected_view_of_one_slot(
    device const half  *wh [[buffer(0), function_constant(kWeightsAreHalf)]],
    device const float *wf [[buffer(0), function_constant(kWeightsAreFloat)]],
    device float *out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = kWeightsAreHalf ? float(wh[gid]) : wf[gid];
}
