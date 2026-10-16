// Owned fixture: a blend into a memoryless color 2 observed through
// color 0. `xrb_color2_fill` writes color 2 alone, the `half4` of its pixel in a buffer of rows
// `width` long (a draw unblended sets the attachment's texels, a draw blended blends over them);
// `xrb_color2_copy` reads color 2 and writes it to color 0, the stored attachment a test reads.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct XrbColor2Vertex {
    float4 position [[position]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3).
vertex XrbColor2Vertex xrb_color2_vertex(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return XrbColor2Vertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

struct XrbColor2Out {
    half4 color [[color(2)]];
};

fragment XrbColor2Out xrb_color2_fill(XrbColor2Vertex input [[stage_in]],
        const device half4 *values [[buffer(0)]], constant uint &width [[buffer(1)]]) {
    uint x = uint(input.position.x);
    uint y = uint(input.position.y);
    return XrbColor2Out { values[y * width + x] };
}

struct XrbColor0Out {
    half4 color [[color(0)]];
};

fragment XrbColor0Out xrb_color2_copy(XrbColor2Vertex input [[stage_in]],
        half4 color2 [[color(2)]]) {
    return XrbColor0Out { color2 };
}
