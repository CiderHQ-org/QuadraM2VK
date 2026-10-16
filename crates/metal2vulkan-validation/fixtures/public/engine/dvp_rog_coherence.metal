// Owned fixture: every access inside a raster order group's
// critical section is coherent at device scope. Each function needs the Vulkan memory model, so
// its buffer, device-address, and storage-image accesses carry the per-access operands: a
// coherent access selects the model, and so does a device address, which no decoration covers.
// Loop-free; nothing here runs on a GPU.
#include <metal_stdlib>
using namespace metal;

// A read_write texture and a plain buffer in a group, with a coherent store selecting the model.
fragment float4 rog_texture_coherent(float4 pos [[position]],
                                     texture2d<float, access::read_write> t [[texture(0), raster_order_group(0)]],
                                     device float *acc [[buffer(0), raster_order_group(0)]],
                                     coherent(device) device uint *log [[buffer(1)]]) {
    uint2 p = uint2(pos.xy);
    uint i = p.y * 8u + p.x;
    float4 v = t.read(p) + float4(0.25f);
    t.write(v, p);
    acc[i] = acc[i] + 1.0f;
    log[i] = 1u;
    return v;
}

struct Targets {
    device uint *acc [[id(0), raster_order_group(0)]];
};

// A read_write texture in a group and a device pointer held in an argument buffer: the pointer's
// accesses reach a device address.
fragment float4 rog_texture_pointer(float4 pos [[position]],
                                    texture2d<float, access::read_write> t [[texture(0), raster_order_group(0)]],
                                    constant Targets &targets [[buffer(0)]]) {
    uint2 p = uint2(pos.xy);
    uint i = p.y * 8u + p.x;
    float4 v = t.read(p) + float4(0.25f);
    t.write(v, p);
    targets.acc[i] = targets.acc[i] * 3u + 1u;
    return v;
}

// Framebuffer fetch with a coherent access: on a device that orders fetch only through the pixel
// interlock, the attachment's storage image is read and written under the Vulkan model.
fragment float4 fetch_coherent(float4 pos [[position]],
                               float4 dst [[color(0)]],
                               coherent(device) device uint *log [[buffer(0)]]) {
    uint i = uint(pos.y) * 8u + uint(pos.x);
    log[i] = log[i] + 1u;
    return dst * 0.5f + float4(0.25f);
}
