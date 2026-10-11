//! The byte layout of LLVM types in AIR module text: what a constant `getelementptr` adds to its
//! base pointer.
//!
//! The rules are LLVM's for AIR's one data layout ([`AIR_DATA_LAYOUT`]): a pointer of any address
//! space is 8 bytes, an integer or float its natural width, a vector the next power of two of its
//! bytes, an array its elements' allocation sizes, and a struct its fields at their alignments (a
//! packed struct without padding). A harvested row's text no longer states the layout (the
//! sanitizer removes it) and is AIR by construction; a module that states another layout is laid
//! out only where every layout agrees (bytes and zero offsets). A type outside these rules (an
//! opaque struct, `i1` vectors, a target type) has no layout, so a step over it has no offset.

#[cfg(test)]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

/// The data layout of every AIR module (`air64`).
const AIR_DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-\
                               f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-\
                               v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-\
                               v1024:1024:1024-n8:16:32";

/// How deep a type's named structs may nest before it has no layout.
const MAX_NESTING: usize = 32;

/// A parsed type of the module text.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ty<'a> {
    Int(u32),
    Float(u64),
    /// A pointer, in its address space.
    Ptr(u32),
    Vector(u64, Box<Ty<'a>>),
    Array(u64, Box<Ty<'a>>),
    Struct {
        packed: bool,
        fields: Vec<Ty<'a>>,
    },
    Named(&'a str),
}

/// An address space that stands for every one: what a type nested past [`MAX_NESTING`] may hold.
pub(super) const ANY_SPACE: u32 = u32::MAX;

/// The named struct types of one module and whether its layout is AIR's, with what each named
/// type's size and pointers were found to be, so each is worked out once however often a type
/// tree names it.
pub(super) struct Layouts<'a> {
    named: HashMap<&'a str, &'a str>,
    air: bool,
    sizes: RefCell<HashMap<&'a str, Option<(u64, u64)>>>,
    spaces: RefCell<HashMap<&'a str, BTreeSet<u32>>>,
    aligned: RefCell<HashMap<&'a str, bool>>,
    /// Named types worked out (not found in the memos), for the boundedness tests.
    #[cfg(test)]
    expansions: Cell<usize>,
}

impl<'a> Layouts<'a> {
    pub(super) fn new(source_ll: &'a str) -> Layouts<'a> {
        let mut named = HashMap::new();
        let mut air = true;
        for line in source_ll.lines() {
            if let Some(layout) = line.strip_prefix("target datalayout = ") {
                air = layout.trim().trim_matches('"') == AIR_DATA_LAYOUT;
            } else if line.starts_with('%') {
                if let Some((name, body)) = line.split_once(" = type ") {
                    if let Some((name, "")) = type_name(name) {
                        named.insert(name, body.trim());
                    }
                }
            }
        }
        Layouts {
            named,
            air,
            sizes: RefCell::new(HashMap::new()),
            spaces: RefCell::new(HashMap::new()),
            aligned: RefCell::new(HashMap::new()),
            #[cfg(test)]
            expansions: Cell::new(0),
        }
    }

    /// How many named types have been worked out.
    #[cfg(test)]
    pub(super) fn expansions(&self) -> usize {
        self.expansions.get()
    }

    #[cfg(test)]
    fn expanded(&self) {
        self.expansions.set(self.expansions.get() + 1);
    }

    #[cfg(not(test))]
    fn expanded(&self) {}

    /// The constant byte offset a `getelementptr` over `source` (its source element type's text)
    /// adds with `indices` (each a `<type> <value>` operand), or `None` when an index is not an
    /// integer constant or a step has no layout.
    #[cfg(test)]
    pub(super) fn gep_offset(&self, source: &str, indices: &[&str]) -> Option<i64> {
        self.gep_range(source, indices)
            .and_then(|(low, high)| (low == high).then_some(low))
    }

    /// The closed range of byte offsets a `getelementptr` over `source` may add with `indices`:
    /// a dynamic index into an array or a vector ranges over its extent (an index past it is
    /// undefined in the source language), so the range is one offset when every index is a
    /// constant. `None` when a step has no layout or a dynamic index steps over the source type
    /// itself (unbounded).
    pub(super) fn gep_range(&self, source: &str, indices: &[&str]) -> Option<(i64, i64)> {
        self.steps(source, indices)?
            .into_iter()
            .try_fold((0i64, 0i64), |(low, high), step| match step {
                Step::Bytes(bytes) => Some((low.checked_add(bytes)?, high.checked_add(bytes)?)),
                Step::Scaled {
                    size,
                    count: Some(count @ 1..),
                } => {
                    let span = i64::try_from((count - 1).checked_mul(size)?).ok()?;
                    Some((low, high.checked_add(span)?))
                }
                Step::Scaled { .. } => None,
            })
    }

    /// The address spaces of every pointer the type at the start of `text` holds (itself, a
    /// vector's elements, or an aggregate's fields at any depth; a type, or an operand's
    /// `<type> [attributes] <value>`). Empty when it holds none, or names a type the module does
    /// not define; [`ANY_SPACE`] among them when its types nest past [`MAX_NESTING`].
    pub(super) fn pointer_spaces(&self, text: &str) -> BTreeSet<u32> {
        let mut spaces = BTreeSet::new();
        if let Some((ty, _)) = parse(text) {
            self.collect_spaces(&ty, 0, &mut spaces);
        }
        spaces
    }

    fn collect_spaces(&self, ty: &Ty<'_>, depth: usize, spaces: &mut BTreeSet<u32>) {
        if depth >= MAX_NESTING {
            spaces.insert(ANY_SPACE);
            return;
        }
        match ty {
            Ty::Ptr(space) => {
                spaces.insert(*space);
            }
            Ty::Vector(_, element) | Ty::Array(_, element) => {
                self.collect_spaces(element, depth + 1, spaces);
            }
            Ty::Struct { fields, .. } => {
                for field in fields {
                    self.collect_spaces(field, depth + 1, spaces);
                }
            }
            Ty::Named(name) => {
                let Some((&name, &body)) = self.named.get_key_value(*name) else {
                    return;
                };
                if let Some(known) = self.spaces.borrow().get(name) {
                    spaces.extend(known);
                    return;
                }
                self.expanded();
                let mut own = BTreeSet::new();
                if let Some((definition, _)) = parse(body) {
                    self.collect_spaces(&definition, depth + 1, &mut own);
                }
                spaces.extend(&own);
                self.spaces.borrow_mut().insert(name, own);
            }
            Ty::Int(_) | Ty::Float(_) => {}
        }
    }

    /// Whether every pointer the type at the start of `text` holds lies a whole number of
    /// pointers (8 bytes) from its start, as a type holding none does: `false` when a pointer's
    /// place has no layout, or a packed struct holds one between two 8-byte slots.
    pub(super) fn pointers_aligned(&self, text: &str) -> bool {
        parse(text).is_some_and(|(ty, _)| self.aligned(&ty, 0))
    }

    fn aligned(&self, ty: &Ty<'_>, depth: usize) -> bool {
        if depth >= MAX_NESTING {
            return false;
        }
        let holds = |ty: &Ty<'_>| {
            let mut spaces = BTreeSet::new();
            self.collect_spaces(ty, depth + 1, &mut spaces);
            !spaces.is_empty()
        };
        let slot = |bytes: u64| bytes.is_multiple_of(8);
        match ty {
            Ty::Int(_) | Ty::Float(_) | Ty::Ptr(_) => true,
            // A vector's pointers are 8 bytes apart.
            Ty::Vector(_, element) => self.aligned(element, depth + 1),
            Ty::Array(_, element) => {
                !holds(element)
                    || self
                        .size_align(element, depth + 1)
                        .is_some_and(|(size, _)| slot(size))
                        && self.aligned(element, depth + 1)
            }
            Ty::Struct { packed, fields } => match self.field_offsets(*packed, fields) {
                Some(offsets) => fields
                    .iter()
                    .zip(offsets)
                    .all(|(field, at)| !holds(field) || slot(at) && self.aligned(field, depth + 1)),
                None => !fields.iter().any(holds),
            },
            Ty::Named(name) => {
                let Some((&name, &body)) = self.named.get_key_value(*name) else {
                    return true;
                };
                if let Some(&known) = self.aligned.borrow().get(name) {
                    return known;
                }
                self.expanded();
                let own =
                    parse(body).is_some_and(|(definition, _)| self.aligned(&definition, depth + 1));
                self.aligned.borrow_mut().insert(name, own);
                own
            }
        }
    }

    /// The allocation size of the type `ty` (its text), or `None` when it has no layout.
    pub(super) fn size(&self, ty: &str) -> Option<u64> {
        let (ty, rest) = parse(ty.trim())?;
        if !rest.trim().is_empty() {
            return None;
        }
        self.size_align(&ty, 0).map(|(size, _)| size)
    }

    /// What a `getelementptr` over `source` adds with `indices`, modulo `modulus`: `None` when a
    /// step has no layout or a dynamic index steps over elements whose size is no multiple of
    /// `modulus`.
    pub(super) fn gep_residue(&self, source: &str, indices: &[&str], modulus: u64) -> Option<u64> {
        let modulus_i64 = i64::try_from(modulus).ok()?;
        self.steps(source, indices)?
            .into_iter()
            .try_fold(0u64, |residue, step| match step {
                Step::Bytes(bytes) => {
                    let bytes = u64::try_from(bytes.rem_euclid(modulus_i64)).ok()?;
                    Some((residue + bytes) % modulus)
                }
                Step::Scaled { size, .. } => (size % modulus == 0).then_some(residue),
            })
    }

    /// Each index's step of a `getelementptr` over `source`.
    fn steps(&self, source: &str, indices: &[&str]) -> Option<Vec<Step>> {
        let (mut ty, rest) = parse(source)?;
        if !rest.trim().is_empty() {
            return None;
        }
        let mut steps = Vec::with_capacity(indices.len());
        for (position, operand) in indices.iter().enumerate() {
            let index = index_operand(operand)?;
            let (element, count) = if position == 0 {
                (ty.clone(), None)
            } else {
                match self.resolve(ty, 0)? {
                    Ty::Struct { packed, fields } => {
                        let Index::Constant(field) = index else {
                            return None;
                        };
                        let field = usize::try_from(field).ok()?;
                        let at = self.field_offsets(packed, &fields)?.get(field).copied()?;
                        steps.push(Step::Bytes(i64::try_from(at).ok()?));
                        ty = fields.into_iter().nth(field)?;
                        continue;
                    }
                    Ty::Array(count, element) | Ty::Vector(count, element) => {
                        (*element, Some(count))
                    }
                    _ => return None,
                }
            };
            steps.push(match index {
                Index::Constant(0) => Step::Bytes(0),
                Index::Constant(index) => {
                    let (size, _) = self.size_align(&element, 0)?;
                    Step::Bytes(index.checked_mul(i64::try_from(size).ok()?)?)
                }
                Index::Dynamic => Step::Scaled {
                    size: self.size_align(&element, 0)?.0,
                    count,
                },
            });
            ty = element;
        }
        Some(steps)
    }

    /// A named type's definition, or the type itself.
    fn resolve<'t>(&self, ty: Ty<'t>, depth: usize) -> Option<Ty<'t>>
    where
        'a: 't,
    {
        match ty {
            Ty::Named(name) => {
                if depth >= MAX_NESTING {
                    return None;
                }
                let (definition, rest) = parse(self.named.get(name)?)?;
                if !rest.trim().is_empty() {
                    return None;
                }
                self.resolve(definition, depth + 1)
            }
            ty => Some(ty),
        }
    }

    /// Each field's byte offset in a struct of `fields`.
    fn field_offsets(&self, packed: bool, fields: &[Ty<'_>]) -> Option<Vec<u64>> {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut end = 0u64;
        for field in fields {
            let (size, align) = self.size_align(field, 0)?;
            let at = if packed {
                end
            } else {
                end.next_multiple_of(align)
            };
            offsets.push(at);
            end = at.checked_add(size)?;
        }
        Some(offsets)
    }

    /// A type's allocation size and alignment, in bytes.
    fn size_align(&self, ty: &Ty<'_>, depth: usize) -> Option<(u64, u64)> {
        if depth >= MAX_NESTING {
            return None;
        }
        let natural = |bytes: u64| Some((bytes, bytes));
        match ty {
            Ty::Int(8) => natural(1),
            _ if !self.air => None,
            Ty::Int(1) => natural(1),
            Ty::Int(16) => natural(2),
            Ty::Int(32) => natural(4),
            Ty::Int(64) => natural(8),
            Ty::Int(_) => None,
            Ty::Float(bits) => natural(bits / 8),
            Ty::Ptr(_) => natural(8),
            Ty::Vector(lanes, element) => {
                let bits = match **element {
                    Ty::Int(bits @ (8 | 16 | 32 | 64)) => u64::from(bits),
                    Ty::Float(bits) => bits,
                    Ty::Ptr(_) => 64,
                    _ => return None,
                };
                let bytes = lanes.checked_mul(bits)? / 8;
                let align = bytes.checked_next_power_of_two()?;
                (1..=128)
                    .contains(&align)
                    .then(|| (bytes.next_multiple_of(align), align))
            }
            Ty::Array(length, element) => {
                let (size, align) = self.size_align(element, depth + 1)?;
                Some((length.checked_mul(size)?, align))
            }
            Ty::Struct { packed, fields } => {
                let mut end = 0u64;
                let mut align = 1u64;
                for field in fields {
                    let (size, field_align) = self.size_align(field, depth + 1)?;
                    let field_align = if *packed { 1 } else { field_align };
                    end = end.next_multiple_of(field_align).checked_add(size)?;
                    align = align.max(field_align);
                }
                Some((end.next_multiple_of(align), align))
            }
            Ty::Named(name) => {
                let (&name, &body) = self.named.get_key_value(*name)?;
                if let Some(&known) = self.sizes.borrow().get(name) {
                    return known;
                }
                self.expanded();
                let layout = parse(body).and_then(|(definition, rest)| {
                    rest.trim()
                        .is_empty()
                        .then(|| self.size_align(&definition, depth + 1))
                        .flatten()
                });
                self.sizes.borrow_mut().insert(name, layout);
                layout
            }
        }
    }
}

/// One index of a `getelementptr`.
enum Index {
    Constant(i64),
    /// A value only known at run time.
    Dynamic,
}

/// One index's step of a `getelementptr`: a constant byte count, or a dynamic index over elements
/// of `size` bytes, `count` of them in its array or vector (`None`: the source type itself).
enum Step {
    Bytes(i64),
    Scaled { size: u64, count: Option<u64> },
}

/// An integer index operand (`<type> <value>`).
fn index_operand(operand: &str) -> Option<Index> {
    let (ty, value) = operand.trim().rsplit_once(' ')?;
    if !ty.trim().starts_with('i') {
        return None;
    }
    match value {
        "zeroinitializer" | "null" => Some(Index::Constant(0)),
        value if value.starts_with('%') => Some(Index::Dynamic),
        value => value.parse().ok().map(Index::Constant),
    }
}

/// A type's name at the start of `text` (after its `%`), quoted or not, and the text after it.
fn type_name(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('%')?;
    if let Some(quoted) = text.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some((&text[..end + 2], &quoted[end + 1..]));
    }
    let end = text
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-')))
        .unwrap_or(text.len());
    (end != 0).then(|| (&text[..end], &text[end..]))
}

/// The type at the start of `text` and the text after it.
fn parse(text: &str) -> Option<(Ty<'_>, &str)> {
    let text = text.trim_start();
    let (mut ty, mut rest) = if let Some(rest) = text.strip_prefix("<{") {
        let (fields, rest) = fields(rest, "}>")?;
        (
            Ty::Struct {
                packed: true,
                fields,
            },
            rest,
        )
    } else if let Some(rest) = text.strip_prefix('{') {
        let (fields, rest) = fields(rest, "}")?;
        (
            Ty::Struct {
                packed: false,
                fields,
            },
            rest,
        )
    } else if let Some(rest) = text.strip_prefix('<') {
        let (length, element, rest) = sequence(rest, '>')?;
        (Ty::Vector(length, Box::new(element)), rest)
    } else if let Some(rest) = text.strip_prefix('[') {
        let (length, element, rest) = sequence(rest, ']')?;
        (Ty::Array(length, Box::new(element)), rest)
    } else if text.starts_with('%') {
        let (name, rest) = type_name(text)?;
        (Ty::Named(name), rest)
    } else {
        let end = text
            .find(|ch: char| !ch.is_ascii_alphanumeric())
            .unwrap_or(text.len());
        let (word, rest) = text.split_at(end);
        let ty = match word {
            "half" | "bfloat" => Ty::Float(16),
            "float" => Ty::Float(32),
            "double" => Ty::Float(64),
            "ptr" => Ty::Ptr(0),
            _ => Ty::Int(word.strip_prefix('i')?.parse().ok()?),
        };
        (ty, rest)
    };
    // `ptr addrspace(N)`, and a typed pointer's `addrspace(N)*` or `*` after its pointee.
    let mut opaque = ty == Ty::Ptr(0);
    loop {
        let trimmed = rest.trim_start();
        if let Some(after) = trimmed.strip_prefix("addrspace(") {
            let (space, after) = after.split_once(')')?;
            let space = space.trim().parse().ok()?;
            if opaque {
                (ty, rest, opaque) = (Ty::Ptr(space), after, false);
                continue;
            }
            let after = after.trim_start().strip_prefix('*')?;
            (ty, rest) = (Ty::Ptr(space), after);
        } else if let Some(after) = trimmed.strip_prefix('*') {
            (ty, rest, opaque) = (Ty::Ptr(0), after, false);
        } else {
            return Some((ty, rest));
        }
    }
}

/// A pointer type, or a vector of pointers, and the address space it points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PointerType {
    pub(super) space: u32,
    pub(super) vector: bool,
}

/// The text after the type at the start of `text`, when it starts with one.
pub(super) fn after_type(text: &str) -> Option<&str> {
    parse(text).map(|(_, rest)| rest)
}

/// The pointer type at the start of `text` (a type's text, perhaps followed by attributes or an
/// operand's value): an opaque `ptr addrspace(N)` or a typed pointer, alone or the element of a
/// vector. `None` when `text` starts with any other type.
pub(super) fn pointer_type(text: &str) -> Option<PointerType> {
    match parse(text)?.0 {
        Ty::Ptr(space) => Some(PointerType {
            space,
            vector: false,
        }),
        Ty::Vector(_, element) => match *element {
            Ty::Ptr(space) => Some(PointerType {
                space,
                vector: true,
            }),
            _ => None,
        },
        _ => None,
    }
}

/// The comma-separated field types of a struct up to its `close`, and the text after it.
fn fields<'a>(mut text: &'a str, close: &str) -> Option<(Vec<Ty<'a>>, &'a str)> {
    let mut fields = Vec::new();
    loop {
        text = text.trim_start();
        if let Some(rest) = text.strip_prefix(close) {
            return Some((fields, rest));
        }
        if !fields.is_empty() {
            text = text.strip_prefix(',')?;
        }
        let (field, rest) = parse(text)?;
        fields.push(field);
        text = rest;
    }
}

/// `<length> x <element>` up to its `close`, and the text after it.
fn sequence(text: &str, close: char) -> Option<(u64, Ty<'_>, &str)> {
    let (length, rest) = text.trim_start().split_once(" x ")?;
    let (element, rest) = parse(rest)?;
    let rest = rest.trim_start().strip_prefix(close)?;
    Some((length.trim().parse().ok()?, element, rest))
}
