// Owned fixture: 8-bit control-point and patch attributes of a
// post-tessellation vertex function. A rasterization-disabled function of three control points per
// triangle patch; every invocation of patch p writes what it fetched for the patch, each value
// widened to a word, to out[8p .. 8p + 8] (each store is the same at every domain point of the
// patch, so the result does not depend on how often the function runs):
//   0-2: control points 0, 1, and 2's uchar4 `color`, its four bytes as one word;
//   3:   control point 0's char2 `offset.x`, sign-extended;
//   4:   control point 2's char2 `offset.y`, sign-extended;
//   5:   the patch's uchar `tag`, zero-extended;
//   6:   the patch's char3 `bias.z`, sign-extended;
//   7:   control point k's uchar4 `color`, k read from a buffer.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    uchar4 color [[attribute(0)]];
    char2 offset [[attribute(1)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> cps;
    uchar tag [[attribute(2)]];
    char3 bias [[attribute(3)]];
};

[[patch(triangle, 3)]]
vertex void men2_eight_bit_points(PatchIn in [[stage_in]],
                                  float3 bc [[position_in_patch]],
                                  uint patch [[patch_id]],
                                  constant uint &k [[buffer(4)]],
                                  device uint *out [[buffer(5)]]) {
    device uint *o = out + patch * 8u;
    o[0] = as_type<uint>(in.cps[0].color);
    o[1] = as_type<uint>(in.cps[1].color);
    o[2] = as_type<uint>(in.cps[2].color);
    o[3] = as_type<uint>(int(in.cps[0].offset.x));
    o[4] = as_type<uint>(int(in.cps[2].offset.y));
    o[5] = uint(in.tag);
    o[6] = as_type<uint>(int(in.bias.z));
    o[7] = as_type<uint>(in.cps[k].color);
}
