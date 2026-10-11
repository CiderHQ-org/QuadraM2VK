//! The testkit's decision scheme (see the module docs), shared by every interpreter that must
//! decide like [`run_test_cfg`](super::interp::run_test_cfg): the CFG and tree interpreters here,
//! the 32-lane SIMT models, and the SPIR-V emitters that compute the same hashes in the output.
//!
//! A run's state is one `u64`, [`mix`] of the seed. Executing original block `b` replaces it with
//! `mix(state ^ b·EXEC_MUL)`. The condition, switch selector, or return value named by value `v`
//! is `mix(state ^ key(v))`: its low bit, its residue modulo [`SELECTOR_RANGE`], or all of it.

use super::view::SELECTOR_RANGE;
use crate::air::ir::{BlockId, ValueId};

/// SplitMix64's increment and finalizer multipliers: [`mix`] adds `MIX[0]`, then multiplies by
/// `MIX[1]` and `MIX[2]`.
pub const MIX: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
];

/// The multiplier that spreads a block id before it is mixed into the state.
pub const EXEC_MUL: u64 = 0x2545_f491_4f6c_dd1d;

/// SplitMix64 of `z`: the increment, then the finalizer.
pub fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(MIX[0]);
    z = (z ^ (z >> 30)).wrapping_mul(MIX[1]);
    z = (z ^ (z >> 27)).wrapping_mul(MIX[2]);
    z ^ (z >> 31)
}

/// The key of value `v`: `state ^ key(v)` is hashed for its condition, selector, or return value.
pub fn key(v: ValueId) -> u64 {
    u64::from(v.raw()).wrapping_add(1).rotate_left(32)
}

/// One run's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hash(u64);

impl Hash {
    /// The state of a run seeded with `seed`.
    pub fn seeded(seed: u64) -> Hash {
        Hash(mix(seed))
    }

    /// The state's bits.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Execute original block `b`: mix it into the state.
    pub fn exec(&mut self, b: BlockId) {
        self.0 = mix(self.0 ^ u64::from(b.raw()).wrapping_mul(EXEC_MUL));
    }

    /// The value named by `v` (a return value, and the source of conditions and selectors).
    pub fn value(self, v: ValueId) -> u64 {
        mix(self.0 ^ key(v))
    }

    /// The condition named by `v`: the value's low bit.
    pub fn cond(self, v: ValueId) -> bool {
        self.value(v) & 1 == 1
    }

    /// The switch selector named by `v`, in `0..SELECTOR_RANGE`.
    pub fn selector(self, v: ValueId) -> u64 {
        self.value(v) % SELECTOR_RANGE
    }
}
