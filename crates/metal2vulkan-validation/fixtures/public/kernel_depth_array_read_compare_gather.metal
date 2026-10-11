// Six uncovered `depth2d_array` families in one kernel: `air.read_depth_2d_array.i16.f32`
// (2 corpus sources), `air.sample_compare_depth_2d_array.f32` (2),
// `air.gather_depth_2d_array.v4f32` (2), `air.get_width_depth_2d_array` (2) and
// `air.get_height_depth_2d_array` (2) -- 10 sources.
//
// Every one of them carries an array slice operand the non-array form does not, and in all three
// texel-fetching calls that operand sits directly beside another scalar: the read puts it between
// the coordinate and the offset, `sample_compare` puts it between the coordinate and the compare
// reference, and `gather` puts it between the coordinate and the has-offset flag. A lowering that
// dropped the slice, or read the neighbouring operand instead, still fetches a real texel; the
// point of this case is that it cannot fetch the RIGHT one.
//
// The two slices hold the same eight values in two different orders -- slice 0 is (1 + 2k)/32 and
// slice 1 is (1 + 2*(5k mod 8))/32 + 1/2 -- so no texel value occurs twice anywhere in the
// texture. Every value is a 32nd, exact in binary32, and the case compares exactly.
//
//   * `read` is asked at both slices of the same ushort2 coordinate, so the pair pins the slice
//     against the coordinate rather than either alone.
//   * `sample_compare` is asked at both slices of the same texel centre with the same reference
//     (shifted by the 1/2 that separates the slices), and the two references are permuted
//     differently from the two slices, so the pass/fail patterns differ: 0,0,0,1,0,0,1,1 at slice 0
//     and 0,1,0,1,0,0,1,0 at slice 1. A lowering that always samples slice 0 answers the first
//     pattern twice.
//   * `gather` is asked exactly on an interior texel corner (uv * size - 1/2 lands on .5 in both
//     axes), so the four texels are unambiguous and no addressing mode is involved. Its four
//     components are four distinct values, which pins the component ORDER as well as the slice.
//
// The queries answer 4 and 2, so a lowering that returned the other dimension, the slice count, or
// zero cannot match.
#include <metal_stdlib>
using namespace metal;

constexpr sampler csamp(coord::normalized, address::clamp_to_edge,
                        filter::nearest, mip_filter::none, compare_func::less);
constexpr sampler gsamp(coord::normalized, address::clamp_to_edge,
                        filter::nearest, mip_filter::none);

kernel void kernel_depth_array_read_compare_gather(
    depth2d_array<float> darr [[texture(0)]],
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    ushort x = ushort(tid % 4);
    ushort y = ushort(tid / 4);

    // Both slices of one coordinate.
    float r0 = darr.read(ushort2(x, y), ushort(0));
    float r1 = darr.read(ushort2(x, y), ushort(1));

    // Texel centres: 1/8, 3/8, 5/8, 7/8 across and 1/4, 3/4 down.
    float2 centre = float2((float(x) + 0.5f) / 4.0f, (float(y) + 0.5f) / 2.0f);
    float ref = (1.0f + 2.0f * float((3u * tid) % 8u)) / 32.0f;
    float c0 = darr.sample_compare(csamp, centre, 0u, ref);
    float c1 = darr.sample_compare(csamp, centre, 1u, ref + 0.5f);

    // Three interior corners across, one down; the slice alternates with the thread.
    float2 corner = float2(0.25f + 0.25f * float(tid % 3u), 0.5f);
    float4 g = darr.gather(gsamp, corner, tid % 2u);

    uint b = tid * 10;
    out[b + 0] = as_type<uint>(r0);
    out[b + 1] = as_type<uint>(r1);
    out[b + 2] = as_type<uint>(c0);
    out[b + 3] = as_type<uint>(c1);
    out[b + 4] = as_type<uint>(g.x);
    out[b + 5] = as_type<uint>(g.y);
    out[b + 6] = as_type<uint>(g.z);
    out[b + 7] = as_type<uint>(g.w);
    out[b + 8] = darr.get_width();
    out[b + 9] = darr.get_height();
}
