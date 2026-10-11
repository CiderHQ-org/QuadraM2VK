//! Cutover criterion C1's reviewed list, which C3 reads for its device gaps ([`reviewed`]).
//!
//! C1 compared the engine with the old engine passing L3 under each profile: the old engine's
//! per-profile L3 lane and C1's report over the two engines' censuses. Both are deleted with the
//! old engine; the C3F snapshot holds their final measurement. The reviewed list stays: its
//! signed-off device-capability entries are C3's device gaps.

pub mod reviewed;
