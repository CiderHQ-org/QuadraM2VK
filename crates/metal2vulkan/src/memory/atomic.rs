//! Atomics.
//!
//! Every memory an atomic reaches is a `u32` word array (descriptor views, device addresses, and
//! the threadgroup arena), so atomics need no memory-side representation of their own: the
//! lowering is `lower::access::atomic`, on the word pointers of `memory::views`,
//! `lower::access`'s PSB leaf, and `memory::arena`.
