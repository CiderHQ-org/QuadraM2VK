// Owned fixture: `[[front_facing]]` under the executors' fullscreen draw
//. Both executors draw one triangle, (-1, -1), (3, -1), (-1, 3) in clip space,
// with Metal's default clockwise front-face winding and no culling; each fragment writes the
// colour its face selects from the constant buffer (element 0 for a front face, 1 for a back
// face). Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 a2_front_facing(bool front [[front_facing]],
                                constant float4 *colors [[buffer(0)]])
{
    return colors[front ? 0 : 1];
}
