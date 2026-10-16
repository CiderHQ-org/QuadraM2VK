// Owned fixture: reads through an index past a texture handle array held in
// an argument buffer, the handle array index containment form. The argument buffer `ts` holds the
// two-texture array `tex` and, after it, the two-texture array `spare`, so an index of 2 or 3 into
// `tex` names a handle inside the argument buffer (one of `spare`'s). Eight threads; thread t reads
// texel (0, 0) of tex[idx[t]] into out[t], and texel (0, 0) of spare[t & 1] into out[8 + t].
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Textures {
    array<texture2d<uint, access::read>, 2> tex;
    array<texture2d<uint, access::read>, 2> spare;
};

kernel void ob2_handle_array(device const uint *idx [[buffer(0)]],
                             constant Textures &ts [[buffer(1)]],
                             device uint *out [[buffer(2)]],
                             uint t [[thread_position_in_grid]]) {
    out[t] = ts.tex[idx[t]].read(uint2(0, 0)).x;
    out[8 + t] = ts.spare[t & 1].read(uint2(0, 0)).x;
}
