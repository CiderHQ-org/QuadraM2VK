// Owned fixture: reads of `read_write` textures per bound pixel format
//. A storage read of a format the shader does not declare needs
// `shaderStorageImageReadWithoutFormat` and the format's `STORAGE_READ_WITHOUT_FORMAT` feature in
// Vulkan; what Metal returns for each format (unorm conversion, half denormals, signed zero,
// infinities, NaN, 8- and 16-bit integer sign extension) is the reading. The cases bind different
// formats to the same float, uint, and int slots; each lane stores the raw bits of its texel.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void a2_storage_read_formats(texture2d<float, access::read_write> f [[texture(0)]],
                                    texture2d<uint, access::read_write> u [[texture(1)]],
                                    texture2d<int, access::read_write> i [[texture(2)]],
                                    device uint4 *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]])
{
    uint2 texel = uint2(tid, 0);
    out[3 * tid + 0] = as_type<uint4>(f.read(texel));
    out[3 * tid + 1] = u.read(texel);
    out[3 * tid + 2] = as_type<uint4>(i.read(texel));
}
