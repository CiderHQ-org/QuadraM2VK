// Owned fixture: a kernel's explicit imageblock written to a texture through a slice with
// fewer lanes than the texture's format has channels. Metal writes the slice's lanes and then
// zeros, whatever the format: a 1-lane half slice lands in an RGBA16Float texel as (v, 0, 0, 0).
//
// Each threadgroup is 4x4 threads over a 4x4 imageblock; every thread fills its own cell from its
// grid position, then the threadgroup's first thread writes the whole slice at the threadgroup's
// first grid position. Every value is exact in its type.
#include <metal_stdlib>
using namespace metal;

struct TibHalfCell { half v; };
struct TibShort2Cell { short2 v; };
struct TibHalfTarget { texture2d<half, access::write> t; };

// A 1-lane half slice to the texture at slot 0.
kernel void tib_slot_half_slice(texture2d<half, access::write> out [[texture(0)]],
    imageblock<TibHalfCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock TibHalfCell *c = img.data(lid);
    c->v = half(1 + gid.x + 8 * gid.y) / 64.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0))) {
        out.write(img.slice(img.data(ushort2(0))->v), gid);
    }
}

// The same slice to the texture an argument buffer at slot 0 names: the texture's format is not
// known when the function compiles.
kernel void tib_argument_half_slice(device TibHalfTarget &target [[buffer(0)]],
    imageblock<TibHalfCell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock TibHalfCell *c = img.data(lid);
    c->v = half(1 + gid.x + 8 * gid.y) / 64.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0))) {
        target.t.write(img.slice(img.data(ushort2(0))->v), gid);
    }
}

// A 2-lane short slice, signed values, to the integer texture at slot 0.
kernel void tib_slot_short2_slice(texture2d<short, access::write> out [[texture(0)]],
    imageblock<TibShort2Cell, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 gid [[thread_position_in_grid]])
{
    threadgroup_imageblock TibShort2Cell *c = img.data(lid);
    c->v = short2(short(gid.x) - 3, -100 * short(gid.y) - 7);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0))) {
        out.write(img.slice(img.data(ushort2(0))->v), gid);
    }
}
