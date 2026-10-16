// Owned fixture: a texture selected between two parameters, read, queried,
// and tested for null through the handle (`texture.is_null.switch`), as `handles`'s `select_read`,
// alone in its module so a case can name it. Thread g uses a when (g & 1) == pick, else b:
// out[g] = t(g, 0) + (width, height, is_null, 0). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a4_select_is_null(texture2d<float> a [[texture(0)]],
                              texture2d<float> b [[texture(1)]],
                              device float4 *out [[buffer(0)]],
                              constant uint &pick [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    texture2d<float> t = (gid & 1) == pick ? a : b;
    out[gid] = t.read(uint2(gid, 0))
        + float4(t.get_width(), t.get_height(), is_null_texture(t) ? 1.0 : 0.0, 0.0);
}
