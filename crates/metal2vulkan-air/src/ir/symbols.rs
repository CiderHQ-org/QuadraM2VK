//! The module's interned byte strings: symbol names and metadata strings.

use std::collections::HashMap;

use super::id::{Arena, Sym};
use crate::budget::Meter;
use crate::error::Error;

/// Interned byte strings. Equal bytes intern to the same [`Sym`]; ids follow first-intern order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Symbols {
    strings: Arena<Sym, Box<[u8]>>,
    index: HashMap<Box<[u8]>, Sym>,
}

impl Symbols {
    /// An empty table.
    pub fn new() -> Symbols {
        Symbols::default()
    }

    /// Intern `bytes`, charging the meter for a new entry (stored once in the arena and once as
    /// the lookup key).
    pub fn intern(&mut self, bytes: &[u8], meter: &mut Meter) -> Result<Sym, Error> {
        if let Some(&sym) = self.index.get(bytes) {
            return Ok(sym);
        }
        meter.charge(bytes.len().saturating_mul(2))?;
        let sym = self.strings.push(bytes.into(), meter)?;
        self.index.insert(bytes.into(), sym);
        Ok(sym)
    }

    /// The bytes of `sym`.
    pub fn get(&self, sym: Sym) -> Option<&[u8]> {
        self.strings.get(sym).map(|b| &**b)
    }

    /// The bytes of `sym` as UTF-8, if they are.
    pub fn as_str(&self, sym: Sym) -> Option<&str> {
        self.get(sym).and_then(|b| std::str::from_utf8(b).ok())
    }

    /// The symbol already interned for `bytes`, if any.
    pub fn lookup(&self, bytes: &[u8]) -> Option<Sym> {
        self.index.get(bytes).copied()
    }

    /// Whether `sym` is in range.
    pub fn contains(&self, sym: Sym) -> bool {
        self.strings.contains(sym)
    }

    /// The number of interned strings.
    pub(crate) fn len(&self) -> usize {
        self.strings.len()
    }
}
