//! The linear sanity scan a decoded artifact passes before it is trusted.
//!
//! A cache load is trusted by identity (magic, versions, key, profile fingerprint, checksum), not
//! re-verified: the owned verifier needs loop metadata and dominator trees that do not survive
//! serialization. The scan is the cheap structural floor under that trust, one
//! pass over the words:
//!
//! - the header: SPIR-V magic, the version every profile emits (1.5), a nonzero id bound no
//!   larger than the instruction count allows, and a zero schema word;
//! - every instruction's word count is nonzero and stays inside the module;
//! - (relocatable modules) every relocation, in strictly increasing word order, is the literal
//!   operand of a four-word `OpDecorate` whose decoration is the one its symbol names, and still
//!   holds 0 as the serializer wrote it.

use crate::spv::{Decoration, Op};

use crate::pipeline::sym::Symbol;
use crate::spirv::serialize::{RelocatableModule, SPIRV_VERSION};

use super::codec::Malformed;

/// Words before the first instruction.
const HEADER_WORDS: usize = 5;

/// Walk `words`' header and instruction lengths, calling `each(start, word_count)` per
/// instruction in order.
fn walk(
    words: &[u32],
    mut each: impl FnMut(usize, usize) -> Result<(), Malformed>,
) -> Result<(), Malformed> {
    let header = words.get(..HEADER_WORDS).ok_or(Malformed)?;
    if header[0] != crate::spv::MAGIC_NUMBER || header[1] != SPIRV_VERSION || header[4] != 0 {
        return Err(Malformed);
    }
    let bound = header[3] as usize;
    let mut at = HEADER_WORDS;
    let mut instructions = 0usize;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || count > words.len() - at {
            return Err(Malformed);
        }
        each(at, count)?;
        instructions += 1;
        at += count;
    }
    // Every result id below the bound is defined by its own instruction.
    if bound == 0 || bound - 1 > instructions {
        return Err(Malformed);
    }
    Ok(())
}

/// The scan of final (relocated) words: header and instruction lengths.
pub(crate) fn scan_words(words: &[u32]) -> Result<(), Malformed> {
    walk(words, |_, _| Ok(()))
}

/// The decoration a relocation of `sym` writes the literal of.
fn decoration(sym: &Symbol) -> Decoration {
    match sym {
        Symbol::Binding(_) => Decoration::Binding,
        Symbol::DescriptorSet(_) => Decoration::DescriptorSet,
        Symbol::Location(_) => Decoration::Location,
        Symbol::Component(_) => Decoration::Component,
        Symbol::SpecId(_) => Decoration::SpecId,
    }
}

/// The scan of a relocatable module: [`scan_words`] plus every relocation landing on the zeroed
/// literal of the decoration its symbol names.
pub(crate) fn scan_relocatable(m: &RelocatableModule) -> Result<(), Malformed> {
    let words = &m.words;
    let mut relocs = m.relocs.iter().peekable();
    walk(words, |start, count| {
        while let Some((sym, at)) = relocs.next_if(|(_, at)| (*at as usize) < start + count) {
            let at = *at as usize;
            let decorate = words[start] == (4 << 16) | Op::Decorate as u32;
            if !decorate
                || count != 4
                || at != start + 3
                || words[start + 2] != decoration(sym) as u32
                || words[at] != 0
            {
                return Err(Malformed);
            }
        }
        Ok(())
    })?;
    // Every relocation was consumed by some instruction, and no two share a literal (a reloc
    // before the first instruction or out of order already failed the check above).
    let strictly_increasing = m.relocs.windows(2).all(|p| p[0].1 < p[1].1);
    if relocs.next().is_some() || !strictly_increasing {
        return Err(Malformed);
    }
    Ok(())
}
