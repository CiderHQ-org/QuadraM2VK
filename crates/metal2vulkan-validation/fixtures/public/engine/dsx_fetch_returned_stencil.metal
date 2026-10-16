// Owned fixture: fragment functions without early fragment tests
// that read their color attachment (`[[color(0)]]`), write back half of it plus a quarter, and
// return the stencil reference (`[[stencil]]`), the sample mask (`[[sample_mask]]`), or both,
// that their buffer holds (`x` the stencil, `y` the mask). Beside a depth and stencil attachment
// their color store in the interlocked storage form precedes the tests, which then test with what
// they return. Every color value they take from a clear of dyadic fractions is a dyadic fraction a
// `float` holds exactly. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct FetchedStencil {
    float4 color [[color(0)]];
    uint stencil [[stencil]];
};

struct FetchedMask {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

struct FetchedStencilMask {
    float4 color [[color(0)]];
    uint stencil [[stencil]];
    uint mask [[sample_mask]];
};

fragment FetchedStencil fetch_returned_stencil(float4 color [[color(0)]],
                                               constant uint2 &returned [[buffer(0)]])
{
    FetchedStencil out;
    out.color = color * 0.5f + 0.25f;
    out.stencil = returned.x;
    return out;
}

fragment FetchedMask fetch_returned_mask(float4 color [[color(0)]],
                                         constant uint2 &returned [[buffer(0)]])
{
    FetchedMask out;
    out.color = color * 0.5f + 0.25f;
    out.mask = returned.y;
    return out;
}

fragment FetchedStencilMask fetch_returned_stencil_mask(float4 color [[color(0)]],
                                                        constant uint2 &returned [[buffer(0)]])
{
    FetchedStencilMask out;
    out.color = color * 0.5f + 0.25f;
    out.stencil = returned.x;
    out.mask = returned.y;
    return out;
}
