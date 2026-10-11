#include <metal_stdlib>
using namespace metal;

// `Corner` is twelve bytes of members in sixteen bytes of object: `float2` forces eight-byte
// alignment, so the trailing `float` leaves four bytes of padding the struct assignment below still
// copies. Metal's own front end lowers all three assignments to `llvm.memcpy ... i64 16`, which is
// exactly the AIR spelled by the hand-written twin beside this file.
struct Corner {
    float2 v;
    float f;
};

kernel void swap_corners_through_a_local(device Corner* items [[buffer(0)]])
{
    Corner tmp = items[0];
    items[0] = items[1];
    items[1] = tmp;
}
