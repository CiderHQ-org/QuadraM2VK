//! Memcpy, memset, and memmove.
//!
//! A memory intrinsic is a sequence of the ordinary word-composed leaves (`memory::leaf`) on its
//! two endpoints, so it needs no memory-side representation of its own: the lowering is
//! `lower::access::copy`. Async copies (`air.simdgroup_async_copy_2d`) are not lowered yet.
