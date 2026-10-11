// Metal's float-to-integer conversion SATURATES at every destination width and sends NaN to zero.
// It is not the C contract, and it is not SPIR-V's: `OpConvertFToU`/`OpConvertFToS` leave an
// out-of-range or NaN input undefined, so nothing but the device can say what a shader gets.
//
// Device-measured on Apple M3 Max / macOS 26.5.2, over the edge set below and exhaustively over
// all 65536 halves for the narrow destinations:
//
//   uint(NaN)=0   uint(+inf)=4294967295   uint(-inf)=0    uint(4.3e9)=4294967295
//   int(NaN)=0    int(+inf)=2147483647    int(-inf)=-2147483648
//   ushort(70000)=65535   short(-32769)=-32768   uchar(300)=255   char(255)=127
//
// The two rows this case exists to pin are the ones a wrapping clamp gets wrong:
//   short(NaN) is 0, not -32768 -- `clamp(NaN, lo, hi)` on Metal returns `lo`.
//   short(half(40000)) is 32767, not 32752 -- the largest half below 32767 is 32752, so a clamp
//   whose edges are spelled in the SOURCE float type cannot reach the destination's own maximum.
//
// Both source widths are here because the second row only exists at half: an f32 holds 65535 and
// 32767 exactly, an f16 holds neither.
//
// Every argument is read from a buffer so the frontend cannot fold the conversion away, and every
// result is widened to `int` so one output buffer can carry all six destination types. An unsigned
// 32-bit result is bit-cast rather than converted, so 4294967295 reads back as -1.
#include <metal_stdlib>
using namespace metal;

kernel void kernel_float_to_integer_saturates(
    device const float* fin [[buffer(0)]],
    device const half* hin [[buffer(1)]],
    device int* out [[buffer(2)]])
{
    float f_not_a_number = fin[0];
    float f_positive_infinity = fin[1];
    float f_negative_infinity = fin[2];
    float f_far_above = fin[3];
    float f_far_below = fin[4];
    float f_in_range = fin[5];
    half h_not_a_number = hin[0];
    half h_positive_infinity = hin[1];
    half h_negative_infinity = hin[2];
    half h_far_above = hin[3];
    half h_far_below = hin[4];
    half h_in_range = hin[5];

    out[0] = int(f_not_a_number);
    out[1] = as_type<int>(uint(f_not_a_number));
    out[2] = int(short(f_not_a_number));
    out[3] = int(ushort(f_not_a_number));
    out[4] = int(char(f_not_a_number));
    out[5] = int(uchar(f_not_a_number));
    out[6] = int(f_positive_infinity);
    out[7] = as_type<int>(uint(f_positive_infinity));
    out[8] = int(short(f_positive_infinity));
    out[9] = int(ushort(f_positive_infinity));
    out[10] = int(char(f_positive_infinity));
    out[11] = int(uchar(f_positive_infinity));
    out[12] = int(f_negative_infinity);
    out[13] = as_type<int>(uint(f_negative_infinity));
    out[14] = int(short(f_negative_infinity));
    out[15] = int(ushort(f_negative_infinity));
    out[16] = int(char(f_negative_infinity));
    out[17] = int(uchar(f_negative_infinity));
    out[18] = int(f_far_above);
    out[19] = as_type<int>(uint(f_far_above));
    out[20] = int(short(f_far_above));
    out[21] = int(ushort(f_far_above));
    out[22] = int(char(f_far_above));
    out[23] = int(uchar(f_far_above));
    out[24] = int(f_far_below);
    out[25] = as_type<int>(uint(f_far_below));
    out[26] = int(short(f_far_below));
    out[27] = int(ushort(f_far_below));
    out[28] = int(char(f_far_below));
    out[29] = int(uchar(f_far_below));
    out[30] = int(f_in_range);
    out[31] = as_type<int>(uint(f_in_range));
    out[32] = int(short(f_in_range));
    out[33] = int(ushort(f_in_range));
    out[34] = int(char(f_in_range));
    out[35] = int(uchar(f_in_range));

    out[36] = int(h_not_a_number);
    out[37] = as_type<int>(uint(h_not_a_number));
    out[38] = int(short(h_not_a_number));
    out[39] = int(ushort(h_not_a_number));
    out[40] = int(char(h_not_a_number));
    out[41] = int(uchar(h_not_a_number));
    out[42] = int(h_positive_infinity);
    out[43] = as_type<int>(uint(h_positive_infinity));
    out[44] = int(short(h_positive_infinity));
    out[45] = int(ushort(h_positive_infinity));
    out[46] = int(char(h_positive_infinity));
    out[47] = int(uchar(h_positive_infinity));
    out[48] = int(h_negative_infinity);
    out[49] = as_type<int>(uint(h_negative_infinity));
    out[50] = int(short(h_negative_infinity));
    out[51] = int(ushort(h_negative_infinity));
    out[52] = int(char(h_negative_infinity));
    out[53] = int(uchar(h_negative_infinity));
    out[54] = int(h_far_above);
    out[55] = as_type<int>(uint(h_far_above));
    out[56] = int(short(h_far_above));
    out[57] = int(ushort(h_far_above));
    out[58] = int(char(h_far_above));
    out[59] = int(uchar(h_far_above));
    out[60] = int(h_far_below);
    out[61] = as_type<int>(uint(h_far_below));
    out[62] = int(short(h_far_below));
    out[63] = int(ushort(h_far_below));
    out[64] = int(char(h_far_below));
    out[65] = int(uchar(h_far_below));
    out[66] = int(h_in_range);
    out[67] = as_type<int>(uint(h_in_range));
    out[68] = int(short(h_in_range));
    out[69] = int(ushort(h_in_range));
    out[70] = int(char(h_in_range));
    out[71] = int(uchar(h_in_range));
}
