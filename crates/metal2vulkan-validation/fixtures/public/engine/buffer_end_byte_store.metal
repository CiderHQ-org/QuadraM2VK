// Owned fixture: store one byte at the end of a 6-byte buffer.
// Bind `buf` to a 6-byte allocation with known initial bytes. Only byte 5 changes to 0xA5; a
// word-composed store must preserve bytes 0..4 (and the padding). Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void buffer_end_byte_store(device uchar *buf [[buffer(0)]]) {
    buf[5] = uchar(0xA5);
}
