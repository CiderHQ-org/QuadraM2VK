//! The oracle's status buffer, written before the dispatch and read after it, before any output
//! ("exhaustion is never evidence").
//!
//! The layout is the oracle's own, not the product's `ExecutionStatus`: eight little-endian `u32`
//! words, of which the fuel transform reads and writes the four of the fuel model below. Every
//! other word stays 0.

use std::fmt;

/// Per-invocation fuel when a case declares none (2^20 loop checks).
pub const DEFAULT_INVOCATION_FUEL: u32 = 1 << 20;

/// The status buffer's size in bytes.
pub const STATUS_BYTES: usize = 32;

/// Word 0: status bits ([`EXHAUSTED`]). Initial 0.
pub const WORD_STATUS: u64 = 0;
/// Word 1: pool chunks consumed so far. Initial 0.
pub const WORD_CONSUMED_CHUNKS: u64 = 1;
/// Word 2: the shared pool's size in chunks. Initial [`POOL_CHUNKS`].
pub const WORD_POOL_CHUNKS: u64 = 2;
/// Word 3: per-invocation fuel. Initial the case's fuel.
pub const WORD_INVOCATION_FUEL: u64 = 3;

/// The bit of [`WORD_STATUS`] a forced exit sets.
pub const EXHAUSTED: u32 = 1;

/// Local loop iterations per pool chunk: an invocation draws from the shared pool every `2^10`
/// local iterations.
pub const CHUNK_SHIFT: u32 = 10;

/// The shared pool's size in chunks: about `2^30` loop iterations per dispatch.
pub const POOL_CHUNKS: u32 = 1 << 20;

/// The `index`th word of `bytes`.
fn word(bytes: &[u8; STATUS_BYTES], index: u64) -> u32 {
    let at = 4 * index as usize;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// The status buffer's initial bytes: the pool, and `invocation_fuel` as the per-invocation fuel
/// word.
pub fn initial_status(invocation_fuel: u32) -> [u8; STATUS_BYTES] {
    let mut bytes = [0u8; STATUS_BYTES];
    for (index, value) in [
        (WORD_POOL_CHUNKS, POOL_CHUNKS),
        (WORD_INVOCATION_FUEL, invocation_fuel),
    ] {
        let at = 4 * index as usize;
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// A run that exhausted its fuel or the pool. Its outputs are not Metal's semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFuelExhausted {
    /// Pool chunks the run consumed.
    pub consumed_chunks: u32,
    /// Every status word as read back. The transform writes only the first four, so a word past
    /// them that is not 0, or a pool or fuel word that is not its initial value, is a write the
    /// transform did not make.
    pub words: [u32; STATUS_BYTES / 4],
}

impl fmt::Display for LoopFuelExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LoopFuelExhausted: a loop ran out of fuel ({} pool chunks consumed; status words \
             {:#010x?}); the output is not a semantic observation",
            self.consumed_chunks, self.words
        )
    }
}

impl std::error::Error for LoopFuelExhausted {}

/// Check a status buffer read back after the run. A buffer of the wrong size is an error string
/// (an executor bug), exhaustion is [`LoopFuelExhausted`].
pub fn check_status(bytes: &[u8]) -> Result<Result<(), LoopFuelExhausted>, String> {
    let bytes: &[u8; STATUS_BYTES] = bytes.try_into().map_err(|_| {
        format!(
            "status buffer holds {} bytes, expected {STATUS_BYTES}",
            bytes.len()
        )
    })?;
    Ok(if word(bytes, WORD_STATUS) & EXHAUSTED != 0 {
        Err(LoopFuelExhausted {
            consumed_chunks: word(bytes, WORD_CONSUMED_CHUNKS),
            words: std::array::from_fn(|index| word(bytes, index as u64)),
        })
    } else {
        Ok(())
    })
}
