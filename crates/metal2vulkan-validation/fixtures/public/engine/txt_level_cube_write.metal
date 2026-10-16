// Owned fixture: a write to a storage cube at a level read from a buffer.
// The cube is the binder's 2D-array view (layer = face), one per level. Thread (x, y, face)
// writes (x, y, face, 1) at texel (x, y) of that face of level `lod`. A case passes level 0.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void txt_level_cube_write(texturecube<float, access::write> c [[texture(0)]],
                                 constant uint &lod [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    c.write(float4(float(gid.x), float(gid.y), float(gid.z), 1.0f), gid.xy, gid.z, lod);
}
