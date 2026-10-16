// Owned fixture: a kernel's explicit imageblock as threadgroup staging.
//
// Metal gives each threadgroup its own `width x height` imageblock, the compute encoder's
// `setImageblockWidth:height:`, of `sizeof(cell)` bytes per cell; the kernel reaches a cell only
// through `data(coord)`. Each kernel below stages a value per cell that names that cell and no
// other, so a cell stride, a row stride, or an extent taken from the wrong place lands a value
// that exists nowhere else in the block.
#include <metal_stdlib>
using namespace metal;

struct Cell {
    float value;
    ushort2 at;
};

// Every thread stages its own cell, then, after the imageblock barrier, reads its right-hand
// neighbour's (wrapping at the imageblock's width) and reports the extent the queries answer.
// Dispatched with an imageblock the size of the threadgroup (W x H):
//     out[y * W + x] = (bits(float((x+1)%W) + 100 * y), ((x+1)%W) | (y << 16), W, H)
kernel void imageblock_neighbours(
    imageblock<Cell, imageblock_layout_explicit> img,
    device uint4 *out [[buffer(0)]],
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 tg [[threads_per_threadgroup]])
{
    threadgroup_imageblock Cell *mine = img.data(lid);
    mine->value = float(lid.x) + 100.0f * float(lid.y);
    mine->at = lid;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    ushort w = img.get_width();
    ushort h = img.get_height();
    threadgroup_imageblock Cell *other = img.data(ushort2((lid.x + 1) % w, lid.y));
    uint i = uint(lid.y) * uint(tg.x) + uint(lid.x);
    out[i] = uint4(as_type<uint>(other->value), uint(other->at.x) | (uint(other->at.y) << 16),
                   uint(w), uint(h));
}

// Threadgroup memory and the imageblock share the threadgroup's memory: neither may overlap the
// other. Thread i writes scratch[i] and its cell, then reads both of thread (i + 1) mod N:
//     out[i] = (3 j + 1) + 1000 j, j = (i + 1) mod N, N = W * H
kernel void imageblock_after_threadgroup(
    imageblock<Cell, imageblock_layout_explicit> img,
    threadgroup uint *scratch [[threadgroup(0)]],
    device uint *out [[buffer(0)]],
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 tg [[threads_per_threadgroup]])
{
    uint n = uint(tg.x) * uint(tg.y);
    uint i = uint(lid.y) * uint(tg.x) + uint(lid.x);
    scratch[i] = i * 3u + 1u;
    img.data(lid)->value = float(i);
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_threadgroup_imageblock);
    uint j = (i + 1u) % n;
    threadgroup_imageblock Cell *other = img.data(ushort2(j % uint(tg.x), j / uint(tg.x)));
    out[i] = scratch[j] + uint(other->value) * 1000u;
}

// A two-lane plane copied out with an explicit size: a 3 x 2 block of `half2` cells from the
// imageblock's origin, written at (1, 1) of a two-component texture. The plane sits after a float
// in the cell, so the slice walks the cell stride from a nonzero field offset.
struct Plane {
    float pad;
    half2 uv;
};

kernel void imageblock_sized_slice(
    texture2d<half, access::write> out [[texture(0)]],
    imageblock<Plane, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock Plane *cell = img.data(lid);
    cell->pad = -1.0f;
    cell->uv = half2(half(lid.x), half(lid.y) * 0.5h);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        out.write(img.slice(cell->uv, ushort2(3, 2)), ushort2(1, 1));
    }
}

// Runtime block sizes and a slice pointer into another cell: the block starts at the imageblock's
// origin whatever cell the pointer names; the pointer selects the field. `short4` spells no
// signedness, so the texture's numeric type comes from its format. With a W x H threadgroup:
//     codes[x, y]   = code of cell (x, y) for x < W/2, y < H/2
//     weights[1 + x, y] = weight of cell (x, y) for x < W - 1, y < H - 1
struct Coded {
    short4 code;
    float weight;
};

kernel void imageblock_runtime_region(
    texture2d<short, access::write> codes [[texture(0)]],
    texture2d<float, access::write> weights [[texture(1)]],
    imageblock<Coded, imageblock_layout_explicit> img,
    ushort2 lid [[thread_position_in_threadgroup]],
    ushort2 tg [[threads_per_threadgroup]])
{
    threadgroup_imageblock Coded *mine = img.data(lid);
    mine->code = short4(short(lid.x) * 3 - 7, -short(lid.y), short(lid.x) + 10 * short(lid.y),
                        -300);
    mine->weight = float(lid.x) + 0.25f * float(lid.y);
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(lid == ushort2(0, 0))) {
        threadgroup_imageblock Coded *far = img.data(ushort2(2, 1));
        codes.write(img.slice(far->code, tg >> 1), ushort2(0, 0));
        weights.write(img.slice(far->weight, tg - ushort2(1, 1)), uint2(1, 0));
    }
}
