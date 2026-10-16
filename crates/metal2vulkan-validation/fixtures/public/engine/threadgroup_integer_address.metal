// Owned fixture: an integer threadgroup address names a byte of Metal's
// threadgroup layout, which places a kernel's static threadgroup objects in the reverse of their
// order in the module, each at the next offset aligned to its alignment, from 0. The arrays are
// declared (so in the module) colors, bases, centers and first used in the other order, as in
// the corpus shape this fixture stands for: centers (2048 bytes) lies at 0, bases (4096) at 2048,
// colors (2048) at 6144. Measured on an M3 Max and an M4.
// With a threadgroup of 256 threads, thread t fills element t of each array with t; after a
// barrier thread 0 subtracts 1 through the address 1024 (word 0 of centers[128], so the float
// 128.0, 0x43000000, becomes 0x42ffffff) and stores 0xC0FFEE through the address 2064 (word 0 of
// bases[1]); after a second barrier thread t writes its elements' words:
//     out[2t .. 2t+2)          centers[t]
//     out[512 + 4t .. 4t+4)    bases[t]
//     out[1536 + 2t .. 2t+2)   colors[t]
// Loop-free by construction.
#include <metal_stdlib>
using namespace metal;

kernel void threadgroup_integer_address(device uint *out [[buffer(0)]],
                                        uint t [[thread_index_in_threadgroup]]) {
    threadgroup half4 colors[256];
    threadgroup float4 bases[256];
    threadgroup float2 centers[256];
    centers[t] = float2(float(t));
    bases[t] = float4(float(t));
    colors[t] = half4(half(t));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
        atomic_fetch_sub_explicit(reinterpret_cast<threadgroup atomic_uint *>(1024ul), 1u,
                                  memory_order_relaxed);
        *reinterpret_cast<threadgroup uint *>(2064ul) = 0xC0FFEEu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 c = as_type<uint2>(centers[t]);
    out[2u * t] = c.x;
    out[2u * t + 1u] = c.y;
    uint4 b = as_type<uint4>(bases[t]);
    out[512u + 4u * t] = b.x;
    out[512u + 4u * t + 1u] = b.y;
    out[512u + 4u * t + 2u] = b.z;
    out[512u + 4u * t + 3u] = b.w;
    uint2 k = as_type<uint2>(colors[t]);
    out[1536u + 2u * t] = k.x;
    out[1536u + 2u * t + 1u] = k.y;
}
