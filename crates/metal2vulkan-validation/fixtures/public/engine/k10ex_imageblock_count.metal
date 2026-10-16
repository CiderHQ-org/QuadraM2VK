// Owned fixture: a read-modify-write of a whole explicit imageblock layout
// (no master): `tally` (a uchar4 at byte 0) becomes `tally * 3 + (1, 2, 3, 4)` and `n` (a ushort
// at byte 4) `n * 5 + 7`, both wrapping, and `h` (a half at byte 6, the high half of the word `n`
// shares) `(h + h) - 1`, exact for the authored values; the color is the `h` the fragment read.
// None of the steps is idempotent, so `k` overlapping fragments of a pixel leave the step applied
// `k` times exactly when each reads the cell after the previous one wrote it. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Count {
    uchar4 tally [[raster_order_group(0)]];
    ushort n [[raster_order_group(0)]];
    half h [[raster_order_group(0)]];
};

struct CountOut {
    half4 color [[color(0)]];
    Count cell [[imageblock_data]];
};

fragment CountOut k10ex_imageblock_count(Count in [[imageblock_data]])
{
    CountOut out;
    out.color = half4(in.h, 0.0h, 0.0h, 1.0h);
    out.cell.tally = in.tally * uchar4(3) + uchar4(1, 2, 3, 4);
    out.cell.n = ushort(in.n * ushort(5) + ushort(7));
    out.cell.h = (in.h + in.h) - 1.0h;
    return out;
}
