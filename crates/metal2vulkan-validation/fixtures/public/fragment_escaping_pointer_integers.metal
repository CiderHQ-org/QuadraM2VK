#include <metal_stdlib>
using namespace metal;

fragment void escaping_pointer_integers(device uchar* data [[buffer(0)]],
                                      constant uchar* constants [[buffer(1)]],
                                      device ulong* output [[buffer(2)]]) {
    output[1] = ulong(reinterpret_cast<uintptr_t>(data + 12)) - ulong(reinterpret_cast<uintptr_t>(data));
    output[2] = ulong(reinterpret_cast<uintptr_t>(constants + 28)) - ulong(reinterpret_cast<uintptr_t>(constants));
    output[3] = uint(reinterpret_cast<uintptr_t>(data + 12)) - uint(reinterpret_cast<uintptr_t>(data));
    output[4] = uint(reinterpret_cast<uintptr_t>(constants + 28)) - uint(reinterpret_cast<uintptr_t>(constants));
}
