#include <metal_stdlib>
using namespace metal;

kernel void store_const(device int *out [[buffer(0)]]) {
    out[0] = 42;
}
