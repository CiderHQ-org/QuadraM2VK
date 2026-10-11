#include <metal_stdlib>
using namespace metal;

// Wrapping the array in its own struct is what lets the twin spell the copy without a loop: C++
// struct assignment copies the array member, which is the same twelve bytes the AIR beside this file
// moves with two `llvm.memcpy` calls through element-ZERO `float` pointers.
struct Payload {
    float b[3];
};

struct Blob {
    uint tag;
    Payload payload;
};

kernel void copy_a_float_run(device Blob* blobs [[buffer(0)]])
{
    Payload tmp = blobs[0].payload;
    blobs[1].payload = tmp;
}
