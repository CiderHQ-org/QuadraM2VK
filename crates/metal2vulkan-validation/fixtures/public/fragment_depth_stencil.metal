#include <metal_stdlib>
using namespace metal;

struct DepthStencilOutput {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

fragment DepthStencilOutput fragment_depth_stencil()
{
    return {0.5f, 7u};
}
