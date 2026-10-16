// Owned fixture: a texture chosen between two adjacent members of an
// argument buffer (`cond ? m.second : m.first`). Metal's front end selects between the two
// members' addresses, loads the chosen member's resource id as a 64-bit integer, and turns the
// integer back into the texture (`inttoptr`): the corpus shape this fixture stands for. Thread t
// reads texel (t >> 1, 0) of `second` for an odd t and of `first` for an even t, and writes its
// red channel scaled by `scale` plus the chosen texture's width.
// Loop-free by construction.
#include <metal_stdlib>
using namespace metal;

struct Material {
    float scale;
    texture2d<half> first;
    texture2d<half> second;
};

kernel void selected_member_texture(constant Material &m [[buffer(0)]],
                                    device float *out [[buffer(1)]],
                                    uint t [[thread_position_in_grid]]) {
    texture2d<half> tex = (t & 1u) ? m.second : m.first;
    out[t] = float(tex.read(uint2(t >> 1, 0u)).x) * m.scale + float(tex.get_width());
}
