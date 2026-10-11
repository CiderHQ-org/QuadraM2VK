//! Typed reads over the metadata graph, and the ABI error constructors.

use super::AbiInt;
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{AbiError, DiagName, InputError, InputErrorKind, Location};
use crate::ir::{CastOp, Const, ConstExpr, ConstId, FuncId, GlobalId, Md, MdId, Module, Type};

/// An ABI error of `kind` at `at`.
pub(super) fn abi_error(kind: AbiError, at: Location) -> Error {
    Error::InvalidInput(InputError::new(InputErrorKind::Abi(kind), at))
}

/// Where an error is reported: the entry (by name), the argument, and the key being read.
#[derive(Clone, Debug, Default)]
pub(super) struct Site {
    pub(super) entry: Option<DiagName>,
    pub(super) arg: Option<u32>,
}

impl Site {
    pub(super) fn module() -> Site {
        Site::default()
    }

    pub(super) fn at(&self, key: Option<&'static str>) -> Location {
        Location::metadata(self.entry.clone(), self.arg, key)
    }

    pub(super) fn err(&self, kind: AbiError, key: Option<&'static str>) -> Error {
        abi_error(kind, self.at(key))
    }
}

/// One metadata operand, resolved.
#[derive(Clone, Copy, Debug)]
pub(super) enum Tok<'m> {
    Null,
    Str(&'m [u8]),
    /// An integer constant: its bit width and zero-extended value.
    Int {
        bits: u8,
        value: u64,
    },
    Global(GlobalId),
    Func(FuncId),
    /// Any other constant (aggregates, floats, other expressions).
    Other,
    Node(MdId),
}

/// Resolve metadata operand `op` of `m`.
pub(super) fn tok(m: &Module, op: Option<MdId>) -> Option<Tok<'_>> {
    let Some(id) = op else {
        return Some(Tok::Null);
    };
    Some(match m.md(id)? {
        Md::String(s) => Tok::Str(m.symbols().get(*s)?),
        Md::Value(c) => value_tok(m, *c)?,
        Md::Node { .. } => Tok::Node(id),
        Md::Debug => return None,
    })
}

/// A constant as a token; a pointer bitcast (typed-pointer spelling) resolves to its operand.
fn value_tok(m: &Module, mut c: ConstId) -> Option<Tok<'_>> {
    // Casts chain at most as deep as the constant table; each step moves to an older constant.
    loop {
        let constant = m.consts().get(c)?;
        match &constant.value {
            Const::Int(v) => {
                let bits = match m.types().get(constant.ty)? {
                    Type::Int(bits) => *bits,
                    _ => return Some(Tok::Other),
                };
                return Some(Tok::Int { bits, value: *v });
            }
            Const::GlobalAddr(g) => return Some(Tok::Global(*g)),
            Const::FuncAddr(f) => return Some(Tok::Func(*f)),
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast,
                ops: [inner],
            }) if *inner < c => c = *inner,
            _ => return Some(Tok::Other),
        }
    }
}

/// The operands of node `id`.
pub(super) fn node_ops(m: &Module, id: MdId) -> Option<&[Option<MdId>]> {
    match m.md(id)? {
        Md::Node { ops, .. } => Some(ops),
        _ => None,
    }
}

/// A cursor over one node's operands.
pub(super) struct Ops<'m> {
    m: &'m Module,
    ops: &'m [Option<MdId>],
    at: usize,
    pub(super) site: Site,
}

impl<'m> Ops<'m> {
    /// The operands of node `id`; not a node is `BadValue`.
    pub(super) fn node(m: &'m Module, id: MdId, site: Site) -> Result<Ops<'m>, Error> {
        let ops = node_ops(m, id).ok_or_else(|| site.err(AbiError::BadValue, None))?;
        Ok(Ops {
            m,
            ops,
            at: 0,
            site,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.ops.len()
    }

    pub(super) fn is_done(&self) -> bool {
        self.at >= self.ops.len()
    }

    fn bad(&self, key: Option<&'static str>) -> Error {
        self.site.err(AbiError::BadValue, key)
    }

    /// The next operand without consuming it.
    pub(super) fn peek(&self) -> Option<Tok<'m>> {
        tok(self.m, *self.ops.get(self.at)?)
    }

    /// The next operand; running out is `MissingKey`, an unresolvable operand is `BadValue`.
    pub(super) fn next(&mut self, key: Option<&'static str>) -> Result<Tok<'m>, Error> {
        let op = *self
            .ops
            .get(self.at)
            .ok_or_else(|| self.site.err(AbiError::MissingKey, key))?;
        self.at += 1;
        tok(self.m, op).ok_or_else(|| self.bad(key))
    }

    /// The next operand as a string.
    pub(super) fn string(&mut self, key: Option<&'static str>) -> Result<&'m [u8], Error> {
        match self.next(key)? {
            Tok::Str(s) => Ok(s),
            _ => Err(self.bad(key)),
        }
    }

    /// The next operand as a string, kept for diagnostics. The name is retained in the ABI, so
    /// its bytes are charged to `meter` before it is allocated.
    pub(super) fn diag(
        &mut self,
        key: Option<&'static str>,
        meter: &mut Meter,
    ) -> Result<DiagName, Error> {
        let s = self.string(key)?;
        kept_diag(s, meter)
    }

    /// The next operand as an unsigned integer of at most 32 bits.
    pub(super) fn u32(&mut self, key: Option<&'static str>) -> Result<u32, Error> {
        match self.next(key)? {
            Tok::Int { bits, value } if bits <= 32 => {
                u32::try_from(value).map_err(|_| self.bad(key))
            }
            _ => Err(self.bad(key)),
        }
    }

    /// The next operand as an `i1`.
    pub(super) fn bool(&mut self, key: Option<&'static str>) -> Result<bool, Error> {
        match self.next(key)? {
            Tok::Int { bits: 1, value } => Ok(value != 0),
            _ => Err(self.bad(key)),
        }
    }

    /// The next operand as an integer or a global computed by the static initializer.
    pub(super) fn abi_int(&mut self, key: Option<&'static str>) -> Result<AbiInt, Error> {
        match self.next(key)? {
            Tok::Int { bits, value } if bits <= 32 => u32::try_from(value)
                .map(AbiInt::Const)
                .map_err(|_| self.bad(key)),
            Tok::Global(g) => Ok(AbiInt::Global(g)),
            _ => Err(self.bad(key)),
        }
    }

    /// The next operand as a global's address.
    pub(super) fn global(&mut self, key: Option<&'static str>) -> Result<GlobalId, Error> {
        match self.next(key)? {
            Tok::Global(g) => Ok(g),
            _ => Err(self.bad(key)),
        }
    }

    /// The next operand as a node.
    pub(super) fn child(&mut self, key: Option<&'static str>) -> Result<MdId, Error> {
        match self.next(key)? {
            Tok::Node(id) => Ok(id),
            _ => Err(self.bad(key)),
        }
    }

    /// Require that every operand was read.
    pub(super) fn finish(&self) -> Result<(), Error> {
        if self.is_done() {
            Ok(())
        } else {
            Err(self.site.err(AbiError::UnknownKey, None))
        }
    }
}

/// Bytes as a diagnostic name (for an error report; a name the ABI retains goes through
/// [`kept_diag`]).
pub(super) fn diag(bytes: &[u8]) -> DiagName {
    DiagName::new(String::from_utf8_lossy(bytes).into_owned())
}

/// Bytes kept as a diagnostic name in the decoded ABI: the length of the name is charged to
/// `meter` before the name is allocated.
pub(super) fn kept_diag(bytes: &[u8], meter: &mut Meter) -> Result<DiagName, Error> {
    meter.charge(lossy_len(bytes))?;
    Ok(diag(bytes))
}

/// The byte length of `bytes` after lossy UTF-8 conversion: each invalid sequence becomes one
/// U+FFFD (3 bytes).
fn lossy_len(bytes: &[u8]) -> usize {
    bytes
        .utf8_chunks()
        .map(|c| {
            let replaced = if c.invalid().is_empty() { 0 } else { 3 };
            c.valid().len().saturating_add(replaced)
        })
        .fold(0, usize::saturating_add)
}

/// Bytes kept as a user key in the decoded ABI, charged to `meter` before they are copied.
pub(super) fn kept_bytes(bytes: &[u8], meter: &mut Meter) -> Result<Box<[u8]>, Error> {
    meter.charge(bytes.len())?;
    Ok(bytes.into())
}
