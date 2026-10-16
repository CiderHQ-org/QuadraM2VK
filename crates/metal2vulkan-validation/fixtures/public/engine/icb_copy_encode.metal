// Owned fixture: a kernel that copies compute commands between
// three indirect command buffers: `source`, whose commands the host encodes; `commands`, of the
// same descriptor; and `inheriting`, which inherits its pipeline and buffers. Thread `tid` reads
// its operation `ops[tid]` and, in order: sets kernel buffer 0 of command `within` of `commands`
// to the `params` window `window` (the copy after it writes over it); copies command
// `from_source` of `source` onto command `to` of `commands`; sets kernel buffer 1 of that copied
// command to the `out` window `window` (over the copied one); copies command `from_within` of
// `commands` onto its command `within`; and copies command `into` of `source` onto command
// `inherited` of `inheriting`. `params` and `out` are the buffers the commands bind. Only
// executing the commands observes them. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
    command_buffer source;
    command_buffer inheriting;
};

struct Copy {
    uint to;
    uint from_source;
    uint window;
    uint within;
    uint from_within;
    uint inherited;
    uint into;
};

kernel void icb_copy_encode(
    device const Encoder &encoder [[buffer(0)]],
    constant Copy *ops [[buffer(1)]],
    constant uint *params [[buffer(2)]],
    device uint *out [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    Copy op = ops[tid];
    compute_command within(encoder.commands, op.within);
    within.set_kernel_buffer(params + 64 * op.window, 0);
    compute_command to(encoder.commands, op.to);
    to.copy_command(compute_command(encoder.source, op.from_source));
    to.set_kernel_buffer(out + 64 * op.window, 1);
    within.copy_command(compute_command(encoder.commands, op.from_within));
    compute_command(encoder.inheriting, op.inherited)
        .copy_command(compute_command(encoder.source, op.into));
}
