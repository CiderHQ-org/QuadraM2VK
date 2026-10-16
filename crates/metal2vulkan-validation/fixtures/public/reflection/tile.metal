// Owned fixture: a tile function, whose pipeline reflection Metal
// answers as tile bindings: its implicit imageblock (MTLBindingTypeImageblock), threadgroup
// memory and a buffer. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionTileData {
    float4 c [[color(0)]];
};

kernel void reflection_tile(imageblock<ReflectionTileData, imageblock_layout_implicit> ib,
                            threadgroup float *tg [[threadgroup(0)]],
                            constant float &k [[buffer(0)]],
                            ushort2 lid [[thread_position_in_threadgroup]]) {
    ReflectionTileData d = ib.read(lid);
    d.c *= k;
    tg[0] = d.c.x;
    ib.write(d, lid);
}
