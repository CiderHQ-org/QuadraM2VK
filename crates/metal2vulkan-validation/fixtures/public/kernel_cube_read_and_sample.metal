// A cube texel READ, on a cube the same kernel also direction-samples.
//
// SPIR-V has no cube texel fetch -- `OpImageFetch` forbids `Dim Cube` -- so `read()` is lowered as
// a SAMPLE along a direction aimed at the exact centre of the named texel, through a sampler the
// translator invents because no Metal argument supplies one. That descriptor is reported as
// `SynthesizedReadSampler`, and until recently the executor had no arm for the kind at all: no
// consumer could build a pipeline for a module containing one, so this family had no case and
// could not have had one. The state the sampler must be bound with -- nearest everywhere -- was
// recorded only in a comment.
//
// The cube stays `Dim Cube` because it is sampled as well as read; a cube only read is declared as
// something SPIR-V can fetch and no sampler is invented.
//
// Faces 0-3 carry a distinct value per texel, so the four reads on them measure the face index,
// the in-face coordinate, and the direction the translator builds to reach that texel. Faces 4 and
// 5 are uniform and are what the two samples aim at, along the exact +Z and -Z axes: a sample at a
// face centre of a uniform face is the same value under any filter or address mode, so the sample
// lanes measure the sampling path and nothing about the cube corner conventions.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_cube_read_and_sample(
    texturecube<float> tex [[texture(0)]],
    device float* out [[buffer(0)]])
{
    constexpr sampler s(filter::nearest, mip_filter::none, address::clamp_to_edge, coord::normalized);
    out[0] = tex.read(uint2(0, 0), 0).x;
    out[1] = tex.read(uint2(1, 0), 1).x;
    out[2] = tex.read(uint2(0, 1), 2).x;
    out[3] = tex.read(uint2(1, 1), 3).x;
    out[4] = tex.read(uint2(0, 0), 4).x;
    out[5] = tex.read(uint2(1, 1), 5).x;
    out[6] = tex.sample(s, float3(0.0, 0.0, 1.0)).x;
    out[7] = tex.sample(s, float3(0.0, 0.0, -1.0)).x;
}
