//!   ([`RecorderShared::settle_all`]), and a commit whose pages' buffer let go of them (released
//!   or detached while an encoder took them, on any thread) replays its command buffer itself,
//!   so the pages' deallocator runs in the call it runs in without a recorder. Each marks its
//!   side, then reads the other's, so a let-go and a commit that race agree on which replays
//!   (`HostPages::let_go_by_buffer`). Its release (completion, failure, drop) lets go of the
//!   rest, after any replay in progress, replaying nothing more.
