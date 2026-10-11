// Seven uncovered texture families in one kernel: three writes and four reads that need a sampler
// or a mip query. `air.sample_texture_1d_array.v4f32` (6 corpus sources),
// `air.write_texture_3d.i16.u.v4i16` (4), `air.write_texture_1d.i16.v4f32` (4),
// `air.sample_depth_cube.f32` (4), `air.get_num_mip_levels_depth_cube` (4),
// `air.get_num_mip_levels_depth_2d` (4) and `air.write_texture_3d.i16.v4f16` (3).
//
// The four threads each write a different voxel of the two 3D textures and a different texel of the
// 1D one, with values derived from the thread index, so the three writes pin their coordinate
// arithmetic against each other rather than all landing on texel zero. The `ushort3`/`ushort`
// coordinates are the `.i16` in the write symbols.
//
// Both samples are taken with a NEAREST sampler. The 1D array is sampled exactly on a texel centre,
// so its result is a stored value rather than an interpolation and it pins BOTH the array slice and
// the texel index. Every texel of a given cube FACE holds the same value, so the depth cube pins
// which face a direction selects without also depending on the intra-face texel convention, which
// is not what these four symbols are here to check. The mip queries are over single-level textures, which is
// what the case manifest can express, so they must both answer 1 -- they sit in adjacent words, so
// a lowering that returned zero or the texture's size instead cannot match.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_texture_writes_and_depth_samples(
    texture3d<ushort, access::write> vol [[texture(0)]],
    texture1d<float, access::write> line [[texture(1)]],
    texture3d<half, access::write> hvol [[texture(2)]],
    texture1d_array<float> larr [[texture(3)]],
    depthcube<float> dcube [[texture(4)]],
    depth2d<float> d2 [[texture(5)]],
    sampler samp [[sampler(0)]],
    device uint *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    ushort x = ushort(tid % 2);
    ushort y = ushort(tid / 2);

    vol.write(ushort4(ushort(100 + tid), ushort(200 + tid), ushort(300 + tid), ushort(400 + tid)),
              ushort3(x, y, x));
    line.write(float4(float(tid) + 0.5f, float(tid) + 1.5f, float(tid) + 2.5f, float(tid) + 3.5f),
               ushort(tid));
    hvol.write(half4(half(tid) + 0.5h, half(tid) + 1.5h, half(tid) + 2.5h, half(tid) + 3.5h),
               ushort3(y, x, y));

    // Four texels across, so texel centres are at 0.125, 0.375, 0.625 and 0.875.
    float u = 0.125f + 0.25f * float(tid);
    float4 s1 = larr.sample(samp, u, tid % 2);
    // One direction per cube face; +X, -X, +Y and -Y for the four threads. The minor components are
    // non-zero so the direction lands well inside a texel rather than on a face boundary.
    float3 dir = (tid == 0) ? float3( 1.0f,  0.5f,  0.5f)
               : (tid == 1) ? float3(-1.0f,  0.5f,  0.5f)
               : (tid == 2) ? float3( 0.5f,  1.0f,  0.5f)
                            : float3( 0.5f, -1.0f,  0.5f);
    float sd = dcube.sample(samp, dir);

    uint b = tid * 4;
    out[b + 0] = as_type<uint>(s1.x);
    out[b + 1] = as_type<uint>(sd);
    out[b + 2] = dcube.get_num_mip_levels();
    out[b + 3] = d2.get_num_mip_levels();
}
