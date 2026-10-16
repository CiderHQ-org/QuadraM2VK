// Owned fixture: a write through a storage texture selected between two
// parameters (`texture.write.switch`). Thread g writes (g, 1, 2, 3) at texel (g >> 1, 0) of a
// when g is odd, else of b: texel k of a is (2k + 1, 1, 2, 3), of b (2k, 1, 2, 3). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_select_write(texture2d<float, access::write> a [[texture(0)]],
                            texture2d<float, access::write> b [[texture(1)]],
                            uint gid [[thread_position_in_grid]]) {
    texture2d<float, access::write> w = (gid & 1) ? a : b;
    w.write(float4(float(gid), 1.0, 2.0, 3.0), uint2(gid >> 1, 0));
}
