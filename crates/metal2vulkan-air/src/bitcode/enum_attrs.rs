//! The enum attributes decode does not interpret, kept apart from the IR.
//!
//! The kinds the IR interprets are decoded into the named fields of `FnAttrs` and `ParamAttrs`.
//! Every other enum-form kind is part of the module as written but means nothing to the
//! translation, so the IR does not carry it. A decode asked for them
//! ([`DecodeOptions::enum_attrs`](super::DecodeOptions::enum_attrs)) records them here instead,
//! by position, for a writer that must hand the module back as written: Metal's compiler relies on
//! some of them, e.g. it deletes an unused call to a `readnone` function only when the call is
//! `nounwind` and `willreturn`, and a dead call to a `"deferred-static-alloca-size"` function that
//! survives is a pipeline error. A translation's decode records nothing.

use std::collections::BTreeMap;

use crate::ir::{FuncId, InstId};

/// The enum attributes of one position that the IR does not interpret, as a set of LLVM attribute
/// kind codes (`AttributeKindCodes`): every enum-form kind except the ones decoded into named
/// fields (`convergent`, the memory spellings, `noalias`, `nocapture`, `byval`, `sret`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnumAttrs([u64; 2]);

impl EnumAttrs {
    /// The largest kind code a set holds (LLVM's table ends at 109).
    pub const MAX_KIND: u32 = 127;
    /// `nounwind`.
    pub const NO_UNWIND: u32 = 18;
    /// `speculatable`.
    pub const SPECULATABLE: u32 = 53;
    /// `willreturn`.
    pub const WILL_RETURN: u32 = 61;
    /// `nofree`.
    pub const NO_FREE: u32 = 62;
    /// `nosync`.
    pub const NO_SYNC: u32 = 63;
    /// `mustprogress`.
    pub const MUST_PROGRESS: u32 = 70;

    /// Add `kind`; `false` (and no change) when it is above [`EnumAttrs::MAX_KIND`].
    pub fn insert(&mut self, kind: u32) -> bool {
        match Self::slot(kind).and_then(|(w, bit)| Some((self.0.get_mut(w)?, bit))) {
            Some((word, bit)) => {
                *word |= bit;
                true
            }
            None => false,
        }
    }

    /// Remove `kind`.
    pub fn remove(&mut self, kind: u32) {
        if let Some((word, bit)) =
            Self::slot(kind).and_then(|(w, bit)| Some((self.0.get_mut(w)?, bit)))
        {
            *word &= !bit;
        }
    }

    /// Whether the set holds `kind` (`false` for a kind above [`EnumAttrs::MAX_KIND`]).
    pub fn contains(&self, kind: u32) -> bool {
        Self::slot(kind)
            .and_then(|(w, bit)| self.0.get(w).map(|word| word & bit != 0))
            .unwrap_or(false)
    }

    /// Whether the set holds no kind.
    pub fn is_empty(&self) -> bool {
        self.0 == [0, 0]
    }

    /// Every kind of `self` or `other`.
    pub fn union(self, other: EnumAttrs) -> EnumAttrs {
        let [a, b] = self.0;
        let [c, d] = other.0;
        EnumAttrs([a | c, b | d])
    }

    /// The kinds, in increasing order.
    pub fn kinds(self) -> impl Iterator<Item = u32> {
        (0..=Self::MAX_KIND).filter(move |&k| self.contains(k))
    }

    /// The word and bit of `kind`.
    fn slot(kind: u32) -> Option<(usize, u64)> {
        (kind <= Self::MAX_KIND).then(|| ((kind / 64) as usize, 1u64 << (kind % 64)))
    }
}

/// The uninterpreted enum attributes of one function or call site: the function's (or the
/// call's function attributes), the return value's, and each parameter's.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SiteEnums {
    /// The function's own attributes (a call's function attributes).
    pub func: EnumAttrs,
    /// The return value's.
    pub ret: EnumAttrs,
    /// By parameter position, up to the last that has any; a missing position has none.
    pub params: Vec<EnumAttrs>,
}

impl SiteEnums {
    /// The kinds of parameter `index`.
    pub fn param(&self, index: usize) -> EnumAttrs {
        self.params.get(index).copied().unwrap_or_default()
    }

    /// Whether no position has a kind.
    pub fn is_empty(&self) -> bool {
        self.func.is_empty() && self.ret.is_empty() && self.params.iter().all(EnumAttrs::is_empty)
    }
}

/// Every uninterpreted enum attribute of a module, by function and by call site (the calling
/// function and the call's instruction). Only positions that have one are recorded.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnumAttrTable {
    funcs: BTreeMap<FuncId, SiteEnums>,
    calls: BTreeMap<(FuncId, InstId), SiteEnums>,
}

impl EnumAttrTable {
    /// Whether no position has a kind.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.funcs.is_empty() && self.calls.is_empty()
    }

    /// The kinds of function `func`'s positions.
    pub fn func(&self, func: FuncId) -> Option<&SiteEnums> {
        self.funcs.get(&func)
    }

    /// The kinds of function `func`'s positions, to edit.
    pub fn func_mut(&mut self, func: FuncId) -> Option<&mut SiteEnums> {
        self.funcs.get_mut(&func)
    }

    /// The kinds of call `call`'s positions, in function `func`.
    pub fn call(&self, func: FuncId, call: InstId) -> Option<&SiteEnums> {
        self.calls.get(&(func, call))
    }

    /// The kinds of call `call`'s positions, in function `func`, to edit.
    pub fn call_mut(&mut self, func: FuncId, call: InstId) -> Option<&mut SiteEnums> {
        self.calls.get_mut(&(func, call))
    }

    /// Record function `func`'s kinds (nothing when `site` has none).
    pub fn set_func(&mut self, func: FuncId, site: SiteEnums) {
        if site.is_empty() {
            self.funcs.remove(&func);
        } else {
            self.funcs.insert(func, site);
        }
    }

    /// Record call `call`'s kinds, in function `func` (nothing when `site` has none).
    pub fn set_call(&mut self, func: FuncId, call: InstId, site: SiteEnums) {
        if site.is_empty() {
            self.calls.remove(&(func, call));
        } else {
            self.calls.insert((func, call), site);
        }
    }

    /// Renumber the calls of function `func` after its body was rebuilt: `to` maps each old
    /// instruction id to its new one, `None` for an instruction that is gone.
    pub fn renumber_calls(&mut self, func: FuncId, to: impl Fn(InstId) -> Option<InstId>) {
        let old: Vec<InstId> = self
            .calls
            .range((func, InstId::new(0))..=(func, InstId::new(u32::MAX)))
            .map(|(&(_, call), _)| call)
            .collect();
        let mut moved = Vec::with_capacity(old.len());
        for call in old {
            if let Some(site) = self.calls.remove(&(func, call)) {
                moved.push((to(call), site));
            }
        }
        for (call, site) in moved {
            if let Some(call) = call {
                self.calls.insert((func, call), site);
            }
        }
    }

    /// The table with its functions renumbered: `to` maps each old function id to its new one,
    /// `None` for a function that is gone (with its calls).
    pub(crate) fn renumbered_funcs(self, to: impl Fn(FuncId) -> Option<FuncId>) -> EnumAttrTable {
        let EnumAttrTable { funcs, calls } = self;
        EnumAttrTable {
            funcs: funcs
                .into_iter()
                .filter_map(|(f, site)| Some((to(f)?, site)))
                .collect(),
            calls: calls
                .into_iter()
                .filter_map(|((f, call), site)| Some(((to(f)?, call), site)))
                .collect(),
        }
    }
}
