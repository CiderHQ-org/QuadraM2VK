// Three LOCAL scratch views that reinterpret while they cross the pointer, one per direction of the
// same rule: an equal-width reinterpret is decided by the WIDTH, never by the element type name or
// the lane count, and the load and store arms have to accept the same set or a module cannot read
// back what it just wrote.
//
//   (1) a `float4` written through a `uint[4]` scratch  -- vector object, scalar-ARRAY pointee whose
//       element type is incompatible but the same width. The load direction already reinterpreted
//       lane-for-lane; the store direction demanded compatible elements.
//   (2) a `float[4]` scratch read as `uint4`            -- scalar-SLOT pointee, vector result whose
//       element type is incompatible but the same width. The store direction already reinterpreted
//       lane-for-lane; the load direction demanded compatible elements.
//   (3) a struct whose leading member is `half4`, read as `float2` -- equal TOTAL width, DIFFERENT
//       lane count. The store direction gated on total width; the load direction gated on lane
//       count as well.
//
// `in[i]` is 0x3F800000 + 4i, so every word this kernel bitcasts is a normal finite float and every
// half is a normal finite half -- no NaN a backend could quieten, so the bits round-trip exactly.
// Row 6 is never written and must still hold its fill. Rows 0 and 2 are LANE-WEIGHTED sums
// (`x0 + 2*x1 + 4*x2 + 8*x3`), so any permutation of the four lanes moves them; rows 1 and 3 are a
// single middle lane, which pins the actual bits.
//
//   out[0..7]   = 15w + 34   (mod 2^32)                  w = in[gid] = 0x3F800000 + 4*gid
//   out[8..15]  = w + 2
//   out[16..23] = 15w + 274  (mod 2^32)
//   out[24..31] = w + 18
//   out[32..39] = ((0x3D00 + gid) << 16) | (0x3C00 + gid)
//   out[40..47] = ((0x3F00 + gid) << 16) | (0x3E00 + gid)
//   out[48..55] = 0xCDCDCDCD                             untouched fill
#include <metal_stdlib>
using namespace metal;

struct HalfLead {
    half4 h;
    uint tail;
};

kernel void reinterpret_local_views(
    device const uint* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint w = in[gid];

    // (1) vector object stored through a scalar-array view
    alignas(16) uint a[4];
    *(thread float4*)a = float4(as_type<float>(w),
                                as_type<float>(w + 1u),
                                as_type<float>(w + 2u),
                                as_type<float>(w + 3u));
    out[gid] = a[0] + 2u * a[1] + 4u * a[2] + 8u * a[3];
    out[gid + 8] = a[2];

    // (2) scalar-slot view read as a vector of a different element type
    alignas(16) float b[4];
    b[0] = as_type<float>(w + 16u);
    b[1] = as_type<float>(w + 17u);
    b[2] = as_type<float>(w + 18u);
    b[3] = as_type<float>(w + 19u);
    uint4 r = *(thread uint4*)b;
    out[gid + 16] = r.x + 2u * r.y + 4u * r.z + 8u * r.w;
    out[gid + 24] = r.z;

    // (3) leading half4 member read as float2 -- same total width, half the lanes
    HalfLead s;
    s.h = half4(as_type<half>(ushort(0x3C00u + gid)),
                as_type<half>(ushort(0x3D00u + gid)),
                as_type<half>(ushort(0x3E00u + gid)),
                as_type<half>(ushort(0x3F00u + gid)));
    s.tail = 0u;
    float2 f = *(thread float2*)&s;
    out[gid + 32] = as_type<uint>(f.x);
    out[gid + 40] = as_type<uint>(f.y);
}
