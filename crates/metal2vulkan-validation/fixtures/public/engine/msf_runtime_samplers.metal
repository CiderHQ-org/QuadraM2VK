// Owned MSL for the runtime-sampler probes: whether Metal
// filters a sample through a runtime sampler, bound to a slot or held in an argument buffer, as it
// filters one through a constexpr sampler of the same state. Each `msf_rs*` kernel samples the
// slot texture through its constexpr sampler and through the runtime sampler at slot 0, and the
// argument buffer's texture (the same texels) through the argument buffer's sampler, each at
// coordinate coord[i] at the base level, and writes the three results to out[3i], out[3i + 1] and
// out[3i + 2]. Each `msf_slot*` kernel samples only the slot texture, through its constexpr sampler
// and the runtime sampler at slot 0, and writes out[2i] and out[2i + 1]. A case binds the runtime
// samplers with the constexpr sampler's state. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Material1 {
    texture1d<float> t;
    sampler s;
};

struct Material2 {
    texture2d<float> t;
    sampler s;
};

struct Material3 {
    texture3d<float> t;
    sampler s;
};

struct MaterialCube {
    texturecube<float> t;
    sampler s;
};

// A 2D kernel `name` whose constexpr sampler has the state `...`.
#define PROBE2D(name, ...)                                                                    \
    kernel void name(texture2d<float> t [[texture(0)]], sampler r [[sampler(0)]],             \
                     constant Material2 &m [[buffer(0)]],                                     \
                     const device float2 *coord [[buffer(1)]],                                \
                     device float4 *out [[buffer(2)]], uint i [[thread_position_in_grid]])    \
    {                                                                                         \
        constexpr sampler c(__VA_ARGS__);                                                     \
        float2 x = coord[i];                                                                  \
        out[3 * i] = t.sample(c, x, level(0.0));                                              \
        out[3 * i + 1] = t.sample(r, x, level(0.0));                                          \
        out[3 * i + 2] = m.t.sample(m.s, x, level(0.0));                                      \
    }

// A 3D kernel `name` whose constexpr sampler has the state `...`.
#define PROBE3D(name, ...)                                                                    \
    kernel void name(texture3d<float> t [[texture(0)]], sampler r [[sampler(0)]],             \
                     constant Material3 &m [[buffer(0)]],                                     \
                     const device float4 *coord [[buffer(1)]],                                \
                     device float4 *out [[buffer(2)]], uint i [[thread_position_in_grid]])    \
    {                                                                                         \
        constexpr sampler c(__VA_ARGS__);                                                     \
        float3 x = coord[i].xyz;                                                              \
        out[3 * i] = t.sample(c, x, level(0.0));                                              \
        out[3 * i + 1] = t.sample(r, x, level(0.0));                                          \
        out[3 * i + 2] = m.t.sample(m.s, x, level(0.0));                                      \
    }

PROBE2D(msf_rs2d_linear_edge, filter::linear, address::clamp_to_edge)
PROBE2D(msf_rs2d_linear_repeat, filter::linear, address::repeat)
PROBE2D(msf_rs2d_linear_mirror, filter::linear, address::mirrored_repeat)
PROBE2D(msf_rs2d_linear_zero, filter::linear, address::clamp_to_zero)
PROBE2D(msf_rs2d_linear_edge_pixel, filter::linear, address::clamp_to_edge, coord::pixel)
PROBE2D(msf_rs2d_linear_zero_pixel, filter::linear, address::clamp_to_zero, coord::pixel)
PROBE2D(msf_rs2d_nearest_repeat, filter::nearest, address::repeat)
PROBE2D(msf_rs2d_nearest_edge_pixel, filter::nearest, address::clamp_to_edge, coord::pixel)

PROBE3D(msf_rs3d_linear_repeat, filter::linear, address::repeat)
PROBE3D(msf_rs3d_linear_zero, filter::linear, address::clamp_to_zero)

// The 1D form: a 1D texture has one level, so its sample takes no level.
kernel void msf_rs1d_linear_mirror(texture1d<float> t [[texture(0)]], sampler r [[sampler(0)]],
                                   constant Material1 &m [[buffer(0)]],
                                   const device float *coord [[buffer(1)]],
                                   device float4 *out [[buffer(2)]],
                                   uint i [[thread_position_in_grid]])
{
    constexpr sampler c(filter::linear, address::mirrored_repeat);
    float x = coord[i];
    out[3 * i] = t.sample(c, x);
    out[3 * i + 1] = t.sample(r, x);
    out[3 * i + 2] = m.t.sample(m.s, x);
}

// The cube form: a direction; Metal's cube sample ignores the address modes.
kernel void msf_rscube_linear(texturecube<float> t [[texture(0)]], sampler r [[sampler(0)]],
                              constant MaterialCube &m [[buffer(0)]],
                              const device float4 *coord [[buffer(1)]],
                              device float4 *out [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    constexpr sampler c(filter::linear, address::clamp_to_edge);
    float3 x = coord[i].xyz;
    out[3 * i] = t.sample(c, x, level(0.0));
    out[3 * i + 1] = t.sample(r, x, level(0.0));
    out[3 * i + 2] = m.t.sample(m.s, x, level(0.0));
}

// A slot-only 2D kernel `name` whose constexpr sampler has the state `...`.
#define SLOT2D(name, ...)                                                                     \
    kernel void name(texture2d<float> t [[texture(0)]], sampler r [[sampler(0)]],             \
                     const device float2 *coord [[buffer(0)]],                                \
                     device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])    \
    {                                                                                         \
        constexpr sampler c(__VA_ARGS__);                                                     \
        float2 x = coord[i];                                                                  \
        out[2 * i] = t.sample(c, x, level(0.0));                                              \
        out[2 * i + 1] = t.sample(r, x, level(0.0));                                          \
    }

// A slot-only 3D kernel `name` whose constexpr sampler has the state `...`.
#define SLOT3D(name, ...)                                                                     \
    kernel void name(texture3d<float> t [[texture(0)]], sampler r [[sampler(0)]],             \
                     const device float4 *coord [[buffer(0)]],                                \
                     device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])    \
    {                                                                                         \
        constexpr sampler c(__VA_ARGS__);                                                     \
        float3 x = coord[i].xyz;                                                              \
        out[2 * i] = t.sample(c, x, level(0.0));                                              \
        out[2 * i + 1] = t.sample(r, x, level(0.0));                                          \
    }

SLOT2D(msf_slot2d_linear_edge, filter::linear, address::clamp_to_edge)
SLOT2D(msf_slot2d_linear_repeat, filter::linear, address::repeat)
SLOT2D(msf_slot2d_linear_mirror, filter::linear, address::mirrored_repeat)
SLOT2D(msf_slot2d_linear_zero, filter::linear, address::clamp_to_zero)
SLOT2D(msf_slot2d_linear_edge_pixel, filter::linear, address::clamp_to_edge, coord::pixel)
SLOT2D(msf_slot2d_linear_zero_pixel, filter::linear, address::clamp_to_zero, coord::pixel)
SLOT2D(msf_slot2d_nearest_repeat, filter::nearest, address::repeat)
SLOT2D(msf_slot2d_nearest_edge_pixel, filter::nearest, address::clamp_to_edge, coord::pixel)

SLOT3D(msf_slot3d_linear_repeat, filter::linear, address::repeat)
SLOT3D(msf_slot3d_linear_mirror, filter::linear, address::mirrored_repeat)
SLOT3D(msf_slot3d_linear_zero, filter::linear, address::clamp_to_zero)

// The slot-only 1D form.
kernel void msf_slot1d_linear_mirror(texture1d<float> t [[texture(0)]], sampler r [[sampler(0)]],
                                     const device float *coord [[buffer(0)]],
                                     device float4 *out [[buffer(1)]],
                                     uint i [[thread_position_in_grid]])
{
    constexpr sampler c(filter::linear, address::mirrored_repeat);
    float x = coord[i];
    out[2 * i] = t.sample(c, x);
    out[2 * i + 1] = t.sample(r, x);
}

// The slot-only cube form.
kernel void msf_slotcube_linear(texturecube<float> t [[texture(0)]], sampler r [[sampler(0)]],
                                const device float4 *coord [[buffer(0)]],
                                device float4 *out [[buffer(1)]],
                                uint i [[thread_position_in_grid]])
{
    constexpr sampler c(filter::linear, address::clamp_to_edge);
    float3 x = coord[i].xyz;
    out[2 * i] = t.sample(c, x, level(0.0));
    out[2 * i + 1] = t.sample(r, x, level(0.0));
}

// The slot-only 2D form with a `half` result.
kernel void msf_slot2d_linear_repeat_half(texture2d<half> t [[texture(0)]],
                                          sampler r [[sampler(0)]],
                                          const device float2 *coord [[buffer(0)]],
                                          device half4 *out [[buffer(1)]],
                                          uint i [[thread_position_in_grid]])
{
    constexpr sampler c(filter::linear, address::repeat);
    float2 x = coord[i];
    out[2 * i] = t.sample(c, x, level(0.0));
    out[2 * i + 1] = t.sample(r, x, level(0.0));
}
