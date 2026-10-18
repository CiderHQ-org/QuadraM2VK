// Owned fixture: a read-modify-write of color attachment 1 alone. `mli_tag_step` reads the
// attachment as an unsigned `uint4` and writes back what it read plus (1, 2, 3, 4) (wrapping),
// writing no other attachment, so each lane counts the steps a pixel took from what the attachment
// held before them. Beside `exf2_fetch_unwritten`, which folds attachment 1 into attachment 0, a
// memoryless attachment 1 that starts at zero leaves `n · (k + 1)` in lane `k` after `n` steps,
// and one that started from what an earlier pass left there leaves more. `mli_count_seed` writes
// (1, 2, 3, 4) into attachment 0 alone, reading none: a draw that fetches nothing before them.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct MliTag {
    uint4 tag [[color(1)]];
};

fragment MliTag mli_tag_step(uint4 tag [[color(1)]])
{
    MliTag out;
    out.tag = tag + uint4(1u, 2u, 3u, 4u);
    return out;
}

struct MliCount {
    uint4 count [[color(0)]];
};

fragment MliCount mli_count_seed()
{
    MliCount out;
    out.count = uint4(1u, 2u, 3u, 4u);
    return out;
}
