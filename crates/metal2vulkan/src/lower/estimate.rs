//! `SizeEstimate` and the `SpirvWords` checks.
//!
//! `select` sums a word cost per strategy (each family states its variants' costs,
//! `FamilyStrategy::words`) plus the fixed costs of the module frame, the declarations, and the
//! structured control flow, and refuses the function with `LimitExceeded(SpirvWords)` before
//! anything is emitted when the sum exceeds the cap (`budget::SPIRV_WORDS`, provisional until
//! phase 6 re-sets it from the old translator's output).
//!
//! The sum is a forecast, not a bound: each family states its costs from its own lowering, and
//! a function can emit more than their sum (one corpus row emits 744 words on `moltenvk-m3`
//! against an estimate of 581, 512 of them in its body, where its texture and arithmetic
//! strategies estimate 361). The early check saves emitting a function whose forecast is
//! already too large; the cap itself holds on the serialized words ([`check_emitted`]), so no
//! artifact above it is kept whatever the estimate said.

use crate::error::{Error, Limit};

/// Header, capabilities, memory model, entry point, the function frame, and the base types.
pub const MODULE_WORDS: u64 = 128;
/// One descriptor (a buffer view, an image, a sampler, the status binding): its types, variable,
/// and decorations.
pub const SLOT_WORDS: u64 = 40;
/// One push member: its type share, offset decoration, and access.
pub const PUSH_FIELD_WORDS: u64 = 12;
/// One AIR block after structurization: labels, merges, branches, and its share of the
/// structurizer's selector traffic (`Limit::OutputBlocks` bounds the block count).
pub const BLOCK_WORDS: u64 = 24;
/// One demoted value or phi variable: its declaration, stores, and loads.
pub const VARIABLE_WORDS: u64 = 16;

/// The planned output size of one function, in SPIR-V words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SizeEstimate {
    pub words: u64,
}

impl SizeEstimate {
    /// Add `words`, saturating.
    pub fn add(&mut self, words: u64) {
        self.words = self.words.saturating_add(words);
    }

    /// Refuse an estimate above `cap` (`LimitExceeded(SpirvWords)`).
    pub fn check(&self, cap: u64) -> Result<(), Error> {
        if self.words > cap {
            Err(Error::LimitExceeded(Limit::SpirvWords))
        } else {
            Ok(())
        }
    }
}

/// Refuse a serialized module of `words` words above `cap` (`LimitExceeded(SpirvWords)`): the exact
/// form of [`SizeEstimate::check`], which the estimate cannot guarantee.
pub fn check_emitted(words: usize, cap: u64) -> Result<(), Error> {
    if u64::try_from(words).map_or(true, |w| w > cap) {
        Err(Error::LimitExceeded(Limit::SpirvWords))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_estimate_above_the_cap_is_refused() {
        let mut e = SizeEstimate::default();
        e.add(MODULE_WORDS);
        assert_eq!(e.check(MODULE_WORDS), Ok(()));
        e.add(1);
        assert_eq!(
            e.check(MODULE_WORDS),
            Err(Error::LimitExceeded(Limit::SpirvWords))
        );
        e.add(u64::MAX);
        assert_eq!(e.words, u64::MAX, "saturates");
    }

    #[test]
    fn emitted_words_above_the_cap_are_refused() {
        assert_eq!(check_emitted(128, 128), Ok(()));
        assert_eq!(
            check_emitted(129, 128),
            Err(Error::LimitExceeded(Limit::SpirvWords))
        );
    }
}
