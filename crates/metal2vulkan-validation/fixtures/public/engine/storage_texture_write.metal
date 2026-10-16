// Owned fixture: the storage-texture pair. MSL cannot name a pixel format, so the
// bgra8/rgba8 pair is this one kernel bound once to a bgra8Unorm and once to an rgba8Unorm
// texture. On a profile where bgra8 lacks STORAGE_IMAGE the two pipelines must get distinct keys
// and distinct OpTypeImage/swizzle code. Channel values are distinct, so a swizzle error is
// observable. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void storage_texture_write(texture2d<float, access::write> dst [[texture(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    dst.write(float4(0.25, 0.5, 0.75, 1.0), gid);
}
