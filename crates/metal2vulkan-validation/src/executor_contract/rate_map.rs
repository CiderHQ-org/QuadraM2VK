//! Where an entry's coordinate-mapping calls read their rasterization-rate map.
//!
//! `air.map_*_coordinates(coordinate, map, layer)` reads the map's parameter data through its
//! pointer operand, and that data's layout is each implementation's own: the Metal oracle binds
//! what Apple's `copyParameterDataToBuffer:offset:` writes, and the engine's module decodes
//! this crate's (`bind::RasterizationRateMap`). A case states each map once, as the
//! `rasterization_rate_map` literal of the resource the shader reads it from, so that each
//! executor writes its own implementation's data there. The literal is a whole buffer, read at
//! offset 0: an input buffer ([`MapPlace::Buffer`]) or the buffer an argument-buffer member holds
//! ([`MapPlace::ArgumentBufferMember`]). Map data anywhere else (another offset, a buffer reached
//! through two pointers, memory no entry parameter names) stays authored bytes, which hold one
//! implementation's layout and are decoded as the other's by the engine's candidate:
//! [`ToolingRequirement::RasterizationRateMapPlacement`].
//!
//! [`rate_map_placement`] is the one decision of where maps are read, shared by classification
//! (through `unsupported_source_requirements`) and the checker (which requires the literal on
//! exactly its [`RateMapPlacement::places`]). It reads the module's text, as classification does,
//! with the declared reflection of the entry, which names the argument buffers' buffer members.
//!
//! What each defined function's values can be is solved by a worklist over its definitions
//! (`solve`), which evaluates a value again only when one it names, or its array's summary,
//! changed. A pointer is a set of roots, each a parameter, a pointer loaded from a parameter's
//! memory, or memory the module holds (an `alloca`, a global it defines), at a span of byte
//! offsets (the `layout` module computes what a `getelementptr` adds; a
//! dynamic index ranges over its array's or vector's extent, whole pointer slots apart when its
//! elements' sizes are multiples of 8), plus whether it may also be
//! something unknown, and whether it may be forged: made from an integer or a constant, loaded from
//! memory not known to hold authored pointers, or a call's result. An unknown part never drops the
//! known roots beside it. `bitcast`, `addrspacecast`, `freeze`, `getelementptr`, `select`, `phi`,
//! and the vector operations keep their operands' roots (a vector of pointers is the union of its
//! elements, and one parser, `layout::pointer_type`, names every pointer type). A load is followed
//! through a parameter's memory or through a thread-memory array (an `alloca`) that only holds
//! pointers, read and written by whole pointer slots: a load from one reads its array's summary of
//! every pointer stored into it, provided the array's address goes nowhere but its own loads,
//! stores, and lifetime markers. A value shared by two paths holds what both make of it; past
//! `MAX_POINTERS` roots, or after `MAX_POINTERS` changes around a loop, its offsets are forgotten.
//! Parameters are then resolved across direct calls to a fixed point: the entry's are its own
//! arguments, and another function's are every value its call sites, in any defined function, pass.
//! A function some call may reach indirectly (its address is taken) or that no direct call reaches
//! may be passed anything.
//!
//! A map operand is placed when every value it can be is known and is offset 0 of a constant entry
//! buffer that is not an argument buffer, or offset 0 of the buffer a buffer member of a constant
//! argument buffer holds, at the member's byte offset, and no writable view reaches that memory:
//! the shader cannot write constant memory through a constant pointer, so the map and the member's
//! pointer are what the case authored. The writable views are every root, at any offset, of a
//! pointer value that is not constant memory, and of a pointer that escapes the analysis (stored
//! anywhere but a followed thread-memory array, returned, turned into an integer or an aggregate,
//! passed to an intrinsic in a vector, or passed to any other function the module does not define,
//! or to an intrinsic that returns a pointer); a loaded root views the slots it was loaded from,
//! and a parameter's memory read as bytes (a load of anything but one pointer, or any pointer
//! operand of a call the module does not define) views the slots it touches.
//!
//! The lattice is closed: a value is nothing only until the trace reaches it, and whatever the
//! trace does not follow (a global the module only declares, `null`, `poison`, a parameter no call
//! fills, a constant expression but an offset or a cast of a global the module defines) is
//! unknown or forged, never nothing. A value or parameter still nothing when the trace settles,
//! which no path gives a value (a pointer only ever stored back into the array it was loaded
//! from, a cycle of `phi`s or of calls no value enters), is unknown, and what names it, loads it,
//! or is passed it takes that in, so nothing that may be a pointer vanishes in a join. An
//! operand's value is read by the text's grammar (the `value` module).
//!
//! No map is placed when a function the entry may reach calls code the module does not hold (a
//! function it only declares, other than an `air.*` or `llvm.*` intrinsic; an indirect call; a
//! constant-expression callee), or writes through a pointer of any space but constant that may be
//! something unknown, forged, or another root than an entry parameter or a pointer loaded from a
//! buffer member's slot of a constant argument buffer no writable view reaches: it may be any
//! address, a map's included. A store's, an atomic's, and an intrinsic's operands are written
//! through, but one whose attributes promise the intrinsic only reads through it (`readonly` with
//! `captures(none)`). A device pointer cast to thread or threadgroup space keeps its roots, so a
//! write through the cast is a write through it. Anything else (device memory, an unknown offset,
//! memory the module holds, a constant expression, a call result, an integer cast to a pointer, a
//! pointer loaded through a loaded pointer) is not placed, so the scan errs towards the
//! requirement, short of the stated limit.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use crate::air::reflect::{ArgumentBufferResourceKind, ShaderInterface};

use self::layout::pointer_type;
use self::value::{value, value_text, Value};
use super::{instruction, split_top_level, ssa_names};
use crate::requirement::ToolingRequirement;

mod layout;
mod value;

/// The callee prefixes of the coordinate-mapping calls, every overload of which takes the map
/// operand second.
const MAP_CALLS: [&str; 2] = [
    "air.map_screen_to_physical_coordinates.",
    "air.map_physical_to_screen_coordinates.",
];

/// The position of the map operand in a coordinate-mapping call.
const MAP_OPERAND: usize = 1;

/// How many pointers one value or parameter may be, and how many times it may change, before its
/// offsets and slots are forgotten.
const MAX_POINTERS: usize = 64;

/// Where a coordinate-mapping call reads its map: a resource a case authors as
/// `rasterization_rate_map`, read at its offset 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapPlace {
    /// The buffer the entry's parameter `param` binds (a `buffers` literal).
    Buffer { param: u32 },
    /// The buffer an argument-buffer member holds: the member at byte `offset` of the argument
    /// buffer the entry's parameter `param` binds (an `argument_buffer_buffers` literal).
    ArgumentBufferMember { param: u32, offset: u32 },
}

/// Where the coordinate-mapping calls of a module's entry read their maps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RateMapPlacement {
    /// The places the placed calls read a map from, each of which a case authors as the map.
    pub places: BTreeSet<MapPlace>,
    /// Some call may read its map from somewhere else: the requirement.
    pub unplaced: bool,
}

impl RateMapPlacement {
    /// [`ToolingRequirement::RasterizationRateMapPlacement`] when some call is not placed.
    pub fn requirement(&self) -> Option<ToolingRequirement> {
        self.unplaced
            .then_some(ToolingRequirement::RasterizationRateMapPlacement)
    }
}

/// Whether a module whose `air.*` calls are `calls` maps coordinates through a
/// rasterization-rate map (any overload): the descriptive fact of the family.
pub fn maps_coordinates(calls: &BTreeMap<String, usize>) -> bool {
    calls
        .keys()
        .any(|call| MAP_CALLS.iter().any(|prefix| call.starts_with(prefix)))
}

/// [`rate_map_placement`]'s requirement.
pub fn rate_map_placement_requirement(
    source_ll: &str,
    reflection: &ShaderInterface,
) -> Option<ToolingRequirement> {
    rate_map_placement(source_ll, reflection).requirement()
}

/// Where the coordinate-mapping calls of `source_ll` read their maps, for the entry `reflection`
/// describes; see the module documentation.
pub fn rate_map_placement(source_ll: &str, reflection: &ShaderInterface) -> RateMapPlacement {
    analyze(source_ll, reflection, &mut Work::default())
}

/// The analysis's work: what its boundedness tests count.
#[derive(Debug, Default)]
struct Work {
    /// Definitions whose values were computed, each time one was.
    traced: usize,
    /// Users visited after a value changed, and users visited deriving thread-memory arrays'
    /// addresses.
    visited: usize,
    /// Pointer sets resolved against their function's parameters.
    resolved: usize,
    /// Operand values joined into a value: each operand a definition's evaluation reads, each
    /// changed operand a `select` or `phi` takes in, and each changed value its arrays' summaries
    /// take in.
    joined: usize,
}

/// A byte offset from a root: the offsets it may be ([`Span`]; one offset when it is a constant),
/// or `None` when it is unbounded (an index over the source type itself, or forgotten).
type Offset = Option<Span>;

/// The byte offsets from `low` to `high`; when `stepped`, only those a whole number of pointer
/// slots ([`SLOT_BYTES`]) past `low` (one offset, or a dynamic index over elements whose sizes are
/// multiples of a slot), so a pointer loaded at any of them is one slot's, not bytes of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Span {
    low: i64,
    high: i64,
    stepped: bool,
}

impl Span {
    /// The offsets from `low` to `high`, slots apart when `stepped` (one offset always is).
    fn new(low: i64, high: i64, stepped: bool) -> Span {
        Span {
            low,
            high,
            stepped: stepped || low == high,
        }
    }

    /// The one offset, if the span is one.
    fn single(self) -> Option<i64> {
        (self.low == self.high).then_some(self.low)
    }

    /// Every offset a pointer loaded at the span may be read from, when they are slots apart.
    fn slots(self) -> Option<impl Iterator<Item = i64>> {
        let step = usize::try_from(SLOT_BYTES).ok()?;
        self.stepped.then(|| (self.low..=self.high).step_by(step))
    }
}

/// The constant offset `offset`.
fn exact(offset: i64) -> Offset {
    Some(Span::new(offset, offset, true))
}

/// Whether every offset `inner` may be is one `outer` may be.
fn within(outer: Offset, inner: Offset) -> bool {
    match (outer, inner) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(outer), Some(inner)) => {
            outer.low <= inner.low
                && inner.high <= outer.high
                && (!outer.stepped || (inner.stepped && (inner.low - outer.low) % SLOT_BYTES == 0))
        }
    }
}

/// A pointer's root in one function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Local {
    /// The function's parameter.
    Param(usize),
    /// The pointer loaded from the parameter's memory at these byte offsets (its slots).
    Loaded { param: usize, at: Offset },
    /// An object the module holds: an `alloca`, or a global it defines ([`held_globals`]).
    Held,
}

/// A pointer's root in the entry's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Root {
    /// The entry's parameter.
    Param(usize),
    /// The pointer loaded from the entry parameter's memory at these byte offsets (its slots).
    Loaded { param: usize, at: Offset },
    /// An object the module holds.
    Held,
}

/// The two kinds of root: what [`Pointers`] needs of them.
trait Base: Copy + Ord {
    /// The same root with a loaded pointer's slot forgotten.
    fn coarse(self) -> Self;
    /// Whether every pointer `other` roots is one this roots too.
    fn covers(self, other: Self) -> bool;
    /// Whether the root covers no other root: it is not a loaded pointer at more than one slot.
    fn single(self) -> bool;
}

macro_rules! base {
    ($root:ident) => {
        impl Base for $root {
            fn coarse(self) -> Self {
                match self {
                    $root::Loaded { param, .. } => $root::Loaded { param, at: None },
                    root => root,
                }
            }

            fn single(self) -> bool {
                match self {
                    $root::Loaded { at, .. } => at.and_then(Span::single).is_some(),
                    _ => true,
                }
            }

            fn covers(self, other: Self) -> bool {
                match (self, other) {
                    ($root::Param(left), $root::Param(right)) => left == right,
                    ($root::Held, $root::Held) => true,
                    (
                        $root::Loaded { param, at },
                        $root::Loaded {
                            param: other,
                            at: slot,
                        },
                    ) => param == other && within(at, slot),
                    _ => false,
                }
            }
        }
    };
}
base!(Local);
base!(Root);

/// What a value can be: pointers, each a root at an offset, and whether it may also be anything
/// else. A root is never dropped for an unknown one beside it: the writable relation needs
/// every known root, and a map is placed only when nothing else is possible.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Pointers<R> {
    roots: BTreeSet<(R, Offset)>,
    /// The value may be a pointer the analysis does not follow.
    other: bool,
    /// The value may be a pointer the case did not author: made from an integer or a constant,
    /// loaded from memory not known to hold authored pointers (a thread-memory array the trace
    /// does not follow, a global, memory reached through a loaded or a forged pointer), or a
    /// call's result. It may be any address of its space, a map's included.
    forged: bool,
    /// The roots' offsets and slots were forgotten (past [`MAX_POINTERS`] pointers), and any
    /// added later are forgotten too.
    coarse: bool,
}

impl<R> Default for Pointers<R> {
    fn default() -> Self {
        Pointers {
            roots: BTreeSet::new(),
            other: false,
            forged: false,
            coarse: false,
        }
    }
}

impl<R: Base> Pointers<R> {
    /// A value the analysis does not follow.
    fn anything() -> Self {
        Pointers {
            other: true,
            ..Pointers::default()
        }
    }

    /// A pointer that may be any address ([`Pointers::forged`]).
    fn forged() -> Self {
        Pointers {
            other: true,
            forged: true,
            ..Pointers::default()
        }
    }

    /// Whether the value is nothing yet: no root, and neither unknown nor forged. Only a value
    /// the trace has not reached is: every constant is something, and a definition or parameter
    /// still nothing when the trace settles becomes unknown ([`solve`], [`analyze`]).
    fn is_nothing(&self) -> bool {
        self.roots.is_empty() && !self.other && !self.forged
    }

    /// The one pointer `root` at `offset`.
    fn root(root: R, offset: Offset) -> Self {
        Pointers {
            roots: BTreeSet::from([(root, offset)]),
            ..Pointers::default()
        }
    }

    /// Add `add`; whether this grew. Past [`MAX_POINTERS`] pointers, offsets and slots are
    /// forgotten for good ([`widen`] bounds the changes of a value whose ranges keep growing).
    fn join(&mut self, add: &Pointers<R>) -> bool {
        // Most joins add nothing new: no root the value lacks, and no flag.
        let flags = (add.other && !self.other)
            || (add.forged && !self.forged)
            || (add.coarse && !self.coarse);
        let fresh = add
            .roots
            .iter()
            .map(|&(root, offset)| match self.coarse {
                true => (root.coarse(), None),
                false => (root, offset),
            })
            .filter(|entry| !self.roots.contains(entry))
            .collect::<Vec<_>>();
        if !flags && fresh.is_empty() {
            return false;
        }
        let mut roots = self.roots.clone();
        roots.extend(fresh);
        let coarse = self.coarse || add.coarse || roots.len() > MAX_POINTERS;
        let next = Pointers {
            roots: normalized(roots, coarse),
            other: self.other || add.other,
            forged: self.forged || add.forged,
            coarse,
        };
        let grew = next != *self;
        *self = next;
        grew
    }
}

/// Count one more change of `value`; past [`MAX_POINTERS`] changes its offsets and slots are
/// forgotten for good (a range that grows each time round a loop). A value so changes at most
/// `MAX_POINTERS + 1` times until they are forgotten (the last change forgets them), then at most
/// twice for each parameter its roots may name (the parameter, and a pointer loaded from its
/// memory), once for memory the module holds, and once each for its `other` and `forged` flags:
/// `MAX_POINTERS + 4 + 2P` in all.
fn widen<R: Base>(value: &mut Pointers<R>, changes: &mut usize) {
    *changes += 1;
    if *changes > MAX_POINTERS && !value.coarse {
        value.join(&Pointers {
            coarse: true,
            ..Pointers::default()
        });
    }
}

/// `roots` without those another covers (at its offset, or at every offset), each coarsened when
/// `coarse`. Coarsened roots cover only equal ones, which the set already merges, so only a set
/// of at most `MAX_POINTERS` roots is compared, and each root only with the wide ones (a range of
/// offsets or slots): a root at one offset and one slot covers only itself.
fn normalized<R: Base>(roots: BTreeSet<(R, Offset)>, coarse: bool) -> BTreeSet<(R, Offset)> {
    if coarse {
        return roots
            .into_iter()
            .map(|(root, _)| (root.coarse(), None))
            .collect();
    }
    let wide = roots
        .iter()
        .copied()
        .filter(|&(root, offset)| !root.single() || offset.and_then(Span::single).is_none())
        .collect::<Vec<_>>();
    if wide.is_empty() {
        return roots;
    }
    #[cfg(test)]
    tests::compared(roots.len() * wide.len());
    roots
        .iter()
        .copied()
        .filter(|&(root, offset)| {
            !wide.iter().any(|&(by, at)| {
                (by, at) != (root, offset) && by.covers(root) && within(at, offset)
            })
        })
        .collect()
}

/// The sum of two offsets, when both are bounded and it fits: slots apart when both are (or are
/// one offset).
fn add(left: Offset, right: Offset) -> Offset {
    let (left, right) = (left?, right?);
    Some(Span::new(
        left.low.checked_add(right.low)?,
        left.high.checked_add(right.high)?,
        left.stepped && right.stepped,
    ))
}

fn analyze(source_ll: &str, reflection: &ShaderInterface, work: &mut Work) -> RateMapPlacement {
    if !MAP_CALLS
        .iter()
        .any(|prefix| source_ll.contains(&format!("@{prefix}")))
    {
        return RateMapPlacement::default();
    }
    let module = Module::new(source_ll);
    let functions = functions(source_ll, &module, work);
    let entry = reflection
        .entry_point
        .as_deref()
        .filter(|entry| functions.contains_key(entry));

    // Each function's direct calls of defined functions, with what every pointer operand can be,
    // and its map operands (each function's values are solved once, by `Function::new`).
    let mut sites = BTreeMap::<&str, Vec<(&str, Vec<(usize, Pointers<Local>)>)>>::new();
    let mut maps = Vec::<(&str, Pointers<Local>)>::new();
    let mut direct_calls = HashMap::<&str, usize>::new();
    for (&name, function) in &functions {
        for call in function.calls() {
            if MAP_CALLS
                .iter()
                .any(|prefix| call.callee.starts_with(prefix))
            {
                let map = call
                    .args
                    .get(MAP_OPERAND)
                    .map_or_else(Pointers::anything, |map| function.pointers(map, &module));
                maps.push((name, map));
            } else if let Some(callee) = functions.get(call.callee) {
                *direct_calls.entry(call.callee).or_default() += 1;
                // Each pointer parameter of the callee is what the call passes there; one the
                // call leaves out, or fills with no pointer, may be anything.
                let args = callee
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(_, param)| param.is_some_and(|(ty, _)| points(ty)))
                    .map(|(index, _)| {
                        let arg = call.args.get(index).filter(|arg| points(arg));
                        (
                            index,
                            arg.map_or_else(Pointers::anything, |arg| {
                                function.pointers(arg, &module)
                            }),
                        )
                    })
                    .collect();
                sites.entry(name).or_default().push((call.callee, args));
            }
        }
    }

    // The entry's parameters are its own arguments. A function no direct call reaches, or one
    // whose address is taken (an indirect call or a table may reach it), may be passed anything
    // as well as what its direct calls pass.
    let mut params = HashMap::<(&str, usize), Pointers<Root>>::new();
    let references = global_references(source_ll);
    let mut address_taken = Vec::new();
    for (&name, function) in &functions {
        let calls = direct_calls.get(name).copied().unwrap_or_default();
        // The `define` line names the function once.
        let taken = references.get(name).copied().unwrap_or_default() > calls + 1;
        if taken {
            address_taken.push(name);
        }
        if Some(name) == entry {
            for index in 0..function.params.len() {
                params.insert((name, index), Pointers::root(Root::Param(index), exact(0)));
            }
        } else if calls == 0 || taken {
            for index in 0..function.params.len() {
                params.insert((name, index), Pointers::anything());
            }
        }
    }

    // Resolve the parameters to a fixed point: a function's call sites are re-resolved only when
    // one of its own parameters grew, and a parameter grows a bounded number of times. A
    // parameter still nothing when it settles is one no call fills (one that is no pointer, or
    // one only calls in a cycle no value enters reach): it may be anything, which the calls that
    // pass it on then hand to their callees in turn.
    let mut pending = functions.keys().copied().collect::<Vec<_>>();
    let mut queued = pending.iter().copied().collect::<HashSet<_>>();
    let mut changes = HashMap::<(&str, usize), usize>::new();
    loop {
        while let Some(caller) = pending.pop() {
            queued.remove(caller);
            for &(callee, ref args) in sites.get(caller).into_iter().flatten() {
                for (index, pointers) in args {
                    work.resolved += 1;
                    let resolved = resolve(caller, pointers, &params);
                    let param = params.entry((callee, *index)).or_default();
                    if param.join(&resolved) {
                        widen(param, changes.entry((callee, *index)).or_default());
                        if queued.insert(callee) {
                            pending.push(callee);
                        }
                    }
                }
            }
        }
        for (&name, function) in &functions {
            for index in 0..function.params.len() {
                let param = params.entry((name, index)).or_default();
                if param.is_nothing() {
                    *param = Pointers::anything();
                    if queued.insert(name) {
                        pending.push(name);
                    }
                }
            }
        }
        if pending.is_empty() {
            break;
        }
    }

    // The memory a pointer that is not constant memory can view, or that a pointer the analysis
    // loses can reach, in the entry's terms: a map there may not hold what the case authored. Each
    // known root counts, whatever else the same value may be.
    let mut writable = BTreeSet::new();
    for (&name, function) in &functions {
        if function.writable.is_empty() {
            continue;
        }
        work.resolved += 1;
        let views = Pointers {
            roots: function.writable.clone(),
            ..Pointers::default()
        };
        writable.extend(
            resolve(name, &views, &params)
                .roots
                .into_iter()
                .map(|(root, _)| root),
        );
    }

    // A write in a function the entry may reach, through a pointer of any space but constant or
    // by an intrinsic it is handed to, must land where a known root says: an entry parameter,
    // memory the module holds (an `alloca`, a global), or the buffer a buffer member the product
    // reflects holds, at every slot the pointer may be loaded from, of a constant argument buffer
    // no writable view reaches, which each executor encodes from what the case declared. A write
    // through any other pointer (forged, or unknown) may land anywhere, a map included, and so may
    // a call to code the module does not hold: no map is then placed.
    let entry_params = entry.map(|entry| &functions[entry].params);
    // Each argument buffer's trusted slots, worked out once: its buffer members' byte offsets,
    // when it is constant memory no writable view reaches.
    let mut trusted_slots = HashMap::<usize, BTreeSet<i64>>::new();
    for field in &reflection.argument_buffer_fields {
        let Ok(param) = usize::try_from(field.buffer_param_index) else {
            continue;
        };
        let constant = entry_params
            .and_then(|params| params.get(param))
            .is_some_and(|param| param.is_some_and(|(ty, _)| constant_buffer_type(ty)));
        if field.resource_kind == Some(ArgumentBufferResourceKind::Buffer)
            && constant
            && !writable.contains(&Root::Param(param))
        {
            trusted_slots
                .entry(param)
                .or_default()
                .insert(i64::from(field.field_offset));
        }
    }
    let trusted = |root: Root| match root {
        Root::Param(_) | Root::Held => true,
        // Every slot the pointer may be loaded from is a buffer member's, of an argument buffer
        // the shader cannot write. Offsets not slots apart (a dynamic index over 4-byte elements)
        // may read a pointer's bytes from two slots, which is no member's pointer.
        Root::Loaded { param, at } => trusted_slots.get(&param).is_some_and(|members| {
            at.and_then(Span::slots)
                .is_some_and(|mut slots| slots.all(|slot| members.contains(&slot)))
        }),
    };
    let mut reached = entry.into_iter().chain(address_taken).collect::<Vec<_>>();
    let mut seen = reached.iter().copied().collect::<HashSet<_>>();
    let mut untrusted = false;
    while let Some(name) = reached.pop() {
        work.resolved += 1;
        let function = &functions[name];
        let writes = resolve(name, &function.writes, &params);
        untrusted |= function.calls_unknown
            || writes.forged
            || writes.other
            || writes.roots.iter().any(|&(root, _)| !trusted(root));
        for &(callee, _) in sites.get(name).into_iter().flatten() {
            if seen.insert(callee) {
                reached.push(callee);
            }
        }
    }

    let mut placement = RateMapPlacement::default();
    for &(function, ref map) in &maps {
        work.resolved += 1;
        let resolved = resolve(function, map, &params);
        let places = (!untrusted && !resolved.other && !resolved.roots.is_empty())
            .then(|| {
                resolved
                    .roots
                    .iter()
                    .map(|&(root, offset)| {
                        (offset == exact(0))
                            .then(|| place(root, entry_params?, reflection, &writable))
                            .flatten()
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .flatten();
        match places {
            Some(places) => placement.places.extend(places),
            None => placement.unplaced = true,
        }
    }
    placement
}

/// What a function's local pointers are in the entry's terms, given its parameters'. A parameter
/// no call has reached yet adds nothing; one that may be anything, or a pointer loaded through a
/// loaded pointer, makes the result possibly anything, beside every root it does know.
fn resolve(
    function: &str,
    pointers: &Pointers<Local>,
    params: &HashMap<(&str, usize), Pointers<Root>>,
) -> Pointers<Root> {
    let mut resolved = Pointers {
        other: pointers.other,
        forged: pointers.forged,
        coarse: pointers.coarse,
        ..Pointers::default()
    };
    for &(local, offset) in &pointers.roots {
        let index = match local {
            Local::Param(index) | Local::Loaded { param: index, .. } => index,
            Local::Held => {
                resolved.roots.insert((Root::Held, offset));
                continue;
            }
        };
        let Some(param) = params.get(&(function, index)) else {
            continue;
        };
        resolved.other |= param.other;
        resolved.forged |= param.forged;
        resolved.coarse |= param.coarse;
        for &(root, base) in &param.roots {
            match (local, root) {
                (Local::Param(_), root) => {
                    resolved.roots.insert((root, add(base, offset)));
                }
                (Local::Loaded { at, .. }, Root::Param(param)) => {
                    let at = add(base, at);
                    resolved.roots.insert((Root::Loaded { param, at }, offset));
                }
                // A pointer loaded through a loaded pointer, or from memory the module holds.
                (Local::Loaded { .. }, Root::Loaded { .. } | Root::Held) => {
                    resolved.other = true;
                    resolved.forged = true;
                }
                (Local::Held, _) => {}
            }
        }
    }
    resolved
}

/// Whether a writable view reaches `root`'s memory: a view of the parameter's own memory, or of
/// an argument buffer's pointer at a slot whose 8 bytes overlap the member's.
fn viewed(writable: &BTreeSet<Root>, root: Root) -> bool {
    writable.iter().any(|&view| match (view, root) {
        (Root::Param(view), Root::Param(param)) => view == param,
        (
            Root::Loaded { param: view, at },
            Root::Loaded {
                param,
                at: Some(Span { low: slot, .. }),
            },
        ) => {
            view == param
                && at.is_none_or(|Span { low, high, .. }| {
                    low < slot.saturating_add(SLOT_BYTES) && high > slot.saturating_sub(SLOT_BYTES)
                })
        }
        _ => false,
    })
}

/// The place an entry pointer at offset 0 of `root` is, when a case can author a map there and
/// no writable view reaches it.
fn place(
    root: Root,
    params: &[Option<(&str, &str)>],
    reflection: &ShaderInterface,
    writable: &BTreeSet<Root>,
) -> Option<MapPlace> {
    let members = || {
        reflection
            .argument_buffer_fields
            .iter()
            .filter(|field| field.resource_kind == Some(ArgumentBufferResourceKind::Buffer))
    };
    let viewed = |root: Root| viewed(writable, root);
    match root {
        Root::Param(index) => {
            let param = u32::try_from(index).ok()?;
            let constant = params
                .get(index)?
                .is_some_and(|(ty, _)| constant_buffer_type(ty));
            // An argument buffer's own memory holds handles, which each executor encodes.
            let argument_buffer = reflection
                .argument_buffer_fields
                .iter()
                .any(|field| field.buffer_param_index == param);
            (constant && !argument_buffer && !viewed(root)).then_some(MapPlace::Buffer { param })
        }
        Root::Loaded { param: index, at } => {
            let param = u32::try_from(index).ok()?;
            let offset = u32::try_from(at?.single()?).ok()?;
            // A shader may store another pointer into a device argument buffer's member, or
            // through a writable view of the argument buffer or of the member's buffer.
            let constant = params
                .get(index)?
                .is_some_and(|(ty, _)| constant_buffer_type(ty));
            (constant
                && !viewed(Root::Param(index))
                && !viewed(root)
                && members()
                    .any(|field| field.buffer_param_index == param && field.field_offset == offset))
            .then_some(MapPlace::ArgumentBufferMember { param, offset })
        }
        // A map the module holds is its own bytes, not what the case authored.
        Root::Held => None,
    }
}

/// Constant memory, which a shader cannot write.
const CONSTANT_SPACE: u32 = 2;

/// Whether a parameter of type text `ty` (its attributes after it) points to constant memory,
/// which the shader cannot write: not a device buffer, a table (`array_ref`, a thread-space
/// pointer), a vector, or a value.
fn constant_buffer_type(ty: &str) -> bool {
    pointer_type(ty).is_some_and(|pointer| !pointer.vector && pointer.space == CONSTANT_SPACE)
}

/// Whether the type at the start of `text` (a type, or an operand's `<type> [attributes]
/// <value>`) is a pointer or a vector of pointers, which the trace follows as the union of its
/// elements.
fn points(text: &str) -> bool {
    pointer_type(text).is_some()
}

/// Whether the type at the start of `text` is one pointer: what a thread-memory array's slot and
/// a followed load hold.
fn one_pointer(text: &str) -> bool {
    pointer_type(text).is_some_and(|pointer| !pointer.vector)
}

/// One defined function of the module text.
struct Function<'a> {
    /// Each parameter's type text (with its attributes) and `%name`; `None` when it has no name.
    params: Vec<Option<(&'a str, &'a str)>>,
    /// Every instruction's text.
    body: Vec<&'a str>,
    /// What each pointer parameter is: itself at offset 0.
    param_pointers: HashMap<&'a str, Pointers<Local>>,
    /// What each named value can be ([`solve`]).
    values: HashMap<&'a str, Pointers<Local>>,
    /// The roots of every pointer of the function that is not constant memory, and of every
    /// pointer the function lets escape ([`escaping`]), each at any offset: memory it may write.
    writable: BTreeSet<(Local, Offset)>,
    /// What the pointers the function writes through, or hands to an intrinsic that may write
    /// through them, can be, with their roots' offsets forgotten and their slots kept
    /// ([`written_through`]).
    writes: Pointers<Local>,
    /// The function calls code the module does not hold: an external function, an indirect
    /// call, or a constant-expression callee ([`unknown_call`]).
    calls_unknown: bool,
}

/// One function's thread-memory arrays (its `alloca`s): see [`ThreadArrays::new`].
struct ThreadArrays<'a> {
    /// Each value derived from an array's address, by the array, with its byte offset from the
    /// array's start modulo the pointer size (`None` when not constant modulo it).
    array_of: HashMap<&'a str, (&'a str, Option<u64>)>,
    /// Each array's stored pointers, or `None` when it may hold anything else.
    held: HashMap<&'a str, Option<Vec<&'a str>>>,
}

/// A direct call: its callee's name and its operands' texts.
struct Call<'a> {
    callee: &'a str,
    args: Vec<&'a str>,
}

/// The size of a pointer in AIR, and so of a thread-memory array's slot.
const POINTER_BYTES: u64 = 8;

/// [`POINTER_BYTES`] as a byte offset: the size of an argument buffer's pointer slot.
const SLOT_BYTES: i64 = 8;

impl<'a> Function<'a> {
    fn new(
        params: Vec<Option<(&'a str, &'a str)>>,
        lines: Vec<&'a str>,
        module: &Module<'_>,
        work: &mut Work,
    ) -> Function<'a> {
        let (defined, layouts) = (&module.defined, &module.layouts);
        let mut definitions = HashMap::new();
        let mut order = Vec::new();
        let mut body = Vec::new();
        for line in lines {
            if let Some((result, text)) = instruction(line) {
                if let Some(result) = result {
                    definitions.insert(result, text);
                    order.push(result);
                }
                body.push(text);
            }
        }
        // Each definition's users: the definitions that name it.
        let mut users = HashMap::<&str, Vec<&str>>::new();
        for &result in &order {
            for operand in ssa_names(definitions[result]) {
                if definitions.contains_key(operand) {
                    users.entry(operand).or_default().push(result);
                }
            }
        }
        let param_pointers = params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| {
                let (ty, name) = (*param)?;
                points(ty).then(|| (name, Pointers::root(Local::Param(index), exact(0))))
            })
            .collect::<HashMap<_, _>>();
        let arrays = ThreadArrays::new(&definitions, &users, &body, layouts, work);
        let values = solve(
            &param_pointers,
            &definitions,
            &order,
            users,
            &arrays,
            module,
            work,
        );
        let roots = |value: &str| {
            param_pointers
                .get(value)
                .or_else(|| values.get(value))
                .map(|pointers| &pointers.roots)
        };
        let mut writable = BTreeSet::new();
        for &result in &order {
            let view = result_type(definitions[result]).is_some_and(|ty| {
                layouts
                    .pointer_spaces(ty)
                    .iter()
                    .any(|&space| space != CONSTANT_SPACE)
            });
            if view {
                writable.extend(
                    roots(result)
                        .into_iter()
                        .flatten()
                        .map(|&(local, _)| (local, None)),
                );
            }
        }
        // Every write through a pointer of a space the shader can write (a device pointer cast to
        // thread space still reaches device memory; constant memory cannot be written through a
        // constant pointer): what it may land on is checked, and its memory is written.
        let mut writes = Pointers::default();
        for &text in &body {
            for operand in written_through(text, defined) {
                if layouts
                    .pointer_spaces(operand)
                    .iter()
                    .all(|&space| space == CONSTANT_SPACE)
                {
                    continue;
                }
                let value = known(&param_pointers, &values, operand, module);
                writes.other |= value.other;
                writes.forged |= value.forged;
                writes
                    .roots
                    .extend(value.roots.iter().map(|&(local, _)| (local, None)));
                writable.extend(value.roots.iter().map(|&(local, _)| (local, None)));
            }
        }
        let calls_unknown = body.iter().any(|text| unknown_call(text, defined));
        for &text in &body {
            for value in escaping(text, &arrays, defined) {
                writable.extend(
                    roots(value)
                        .into_iter()
                        .flatten()
                        .map(|&(local, _)| (local, None)),
                );
            }
            // A parameter's memory read as bytes may hand on a member's pointer, which escapes:
            // its pointee is viewed over every slot the read touches.
            for (address, size) in byte_reads(text, layouts, defined) {
                writable.extend(roots(address).into_iter().flatten().filter_map(
                    |&(local, offset)| match local {
                        Local::Param(param) => Some((
                            Local::Loaded {
                                param,
                                at: touched_slots(offset, size),
                            },
                            None,
                        )),
                        // A member's pointee, and memory the module holds, hold no placed map's
                        // pointer.
                        Local::Loaded { .. } | Local::Held => None,
                    },
                ));
            }
        }
        Function {
            params,
            body,
            param_pointers,
            values,
            writable,
            writes,
            calls_unknown,
        }
    }

    fn calls(&self) -> Vec<Call<'a>> {
        self.body.iter().filter_map(|text| call(text)).collect()
    }

    /// The pointers the operand `operand` (its text, perhaps with its type) can be: a pointer
    /// parameter, a value [`solve`] followed, or a constant ([`constant`]).
    fn pointers(&self, operand: &str, module: &Module<'_>) -> Pointers<Local> {
        known(&self.param_pointers, &self.values, operand, module)
    }
}

/// What each named value of one function can be, to a fixed point over its definitions: `bitcast`,
/// `addrspacecast`, `freeze`, `getelementptr` (over its offsets' range), `select`, `phi`, and the
/// vector operations keep their operands' roots; a pointer loaded through a parameter's memory is
/// the pointer that memory holds at those slots; one loaded from a thread-memory array is its
/// array's summary, every pointer stored into it; `inttoptr` may be any address. Anything else may
/// be anything.
///
/// Every value and summary only grows ([`Pointers::join`]) and forgets its offsets after
/// [`MAX_POINTERS`] changes ([`widen`]), so a cycle that moves a pointer (a pointer stepped
/// around a loop) ends, one that keeps its offsets settles, and a value shared by two paths holds
/// both. A value is evaluated again only when one it names, or its array's summary, changed; a
/// `select` or `phi`, the union of its operands, is evaluated once and then takes in each changed
/// operand alone; and a stored value grows its array's summary rather than each load. Each value
/// and summary changes a bounded number of times, and each change joins one value into each of
/// its users' or summaries', so the operand joins are linear in the function's uses. A value
/// still nothing when the rest settles is one no path gives a value: it may be anything, which
/// is handed on as any change is.
fn solve<'a>(
    param_pointers: &HashMap<&'a str, Pointers<Local>>,
    definitions: &HashMap<&'a str, &'a str>,
    order: &[&'a str],
    users: HashMap<&'a str, Vec<&'a str>>,
    arrays: &ThreadArrays<'a>,
    module: &Module<'_>,
    work: &mut Work,
) -> HashMap<&'a str, Pointers<Local>> {
    let layouts = &module.layouts;
    // What each followed array holds: one summary of every pointer stored into it, which a value
    // stored into it grows when it changes and which its loads read.
    let mut held = HashMap::<&str, Pointers<Local>>::new();
    let mut stored_into = HashMap::<&str, Vec<&str>>::new();
    for (array, stored) in arrays.held() {
        let summary = held.entry(array).or_default();
        for &value in stored {
            if definitions.contains_key(value) {
                stored_into.entry(value).or_default().push(array);
            } else {
                summary.join(
                    &param_pointers
                        .get(value)
                        .cloned()
                        .unwrap_or_else(|| constant(value, module)),
                );
            }
        }
    }
    let mut loads = HashMap::<&str, Vec<&str>>::new();
    for &result in order {
        if let Some(Some(array)) = arrays.load(definitions[result]) {
            loads.entry(array).or_default().push(result);
        }
    }
    // A `select` or a `phi` is the union of its operands, so once evaluated it takes in a changed
    // operand alone rather than joining every operand again: each definition's unions, the
    // `select`s and `phi`s it is an operand of.
    let mut unions = HashSet::<&str>::new();
    let mut joined_into = HashMap::<&str, Vec<&str>>::new();
    for &result in order {
        let Some(operands) = union_operands(definitions[result]) else {
            continue;
        };
        unions.insert(result);
        for operand in operands.into_iter().flatten() {
            let name = value_text(operand);
            if definitions.contains_key(name) {
                let into = joined_into.entry(name).or_default();
                if into.last() != Some(&result) {
                    into.push(result);
                }
            }
        }
    }
    let mut values = HashMap::<&str, Pointers<Local>>::with_capacity(order.len());
    let mut changes = HashMap::<&str, usize>::new();
    let mut summary_changes = HashMap::<&str, usize>::new();
    let mut pending = Pending {
        queue: order.iter().copied().collect(),
        queued: order.iter().copied().collect(),
    };
    // The first definition, in order, that may still be nothing: a value only grows.
    let mut unset = 0;
    loop {
        let (result, next) = if let Some(result) = pending.queue.pop_front() {
            pending.queued.remove(result);
            work.traced += 1;
            let read = Cell::new(0);
            let operand = |operand: &str| -> Cow<'_, Pointers<Local>> {
                read.set(read.get() + 1);
                let value = value_text(operand);
                if let Some(pointers) = param_pointers.get(value) {
                    return Cow::Borrowed(pointers);
                }
                if !definitions.contains_key(value) {
                    return Cow::Owned(constant(operand, module));
                }
                // A definition not yet reached adds nothing yet.
                values
                    .get(value)
                    .map_or_else(|| Cow::Owned(Pointers::default()), Cow::Borrowed)
            };
            let next = transfer(definitions[result], operand, arrays, &held, layouts);
            work.joined += read.get();
            (result, next)
        } else {
            // Settled. A definition still nothing is one no path gives a value (a pointer only
            // ever stored back into the array it was loaded from, a cycle of `phi`s no value
            // enters): it may be anything, and what names it or loads it takes that in.
            while order
                .get(unset)
                .is_some_and(|result| values.get(result).is_some_and(|value| !value.is_nothing()))
            {
                unset += 1;
            }
            let Some(&result) = order.get(unset) else {
                break;
            };
            work.traced += 1;
            (result, Pointers::anything())
        };
        let value = values.entry(result).or_default();
        if !value.join(&next) {
            continue;
        }
        widen(value, changes.entry(result).or_default());
        // Hand each change on: to the arrays the value is stored into (whose loads read it), to
        // the users that evaluate again, and to the unions that take it in.
        let mut changed = vec![result];
        while let Some(result) = changed.pop() {
            for &user in users.get(result).into_iter().flatten() {
                if !unions.contains(user) {
                    pending.visit(user, work);
                }
            }
            if !stored_into.contains_key(result) && !joined_into.contains_key(result) {
                continue;
            }
            let value = values[result].clone();
            for &array in stored_into.get(result).into_iter().flatten() {
                work.joined += 1;
                let summary = held.entry(array).or_default();
                if summary.join(&value) {
                    widen(summary, summary_changes.entry(array).or_default());
                    for &load in loads.get(array).into_iter().flatten() {
                        pending.visit(load, work);
                    }
                }
            }
            for &union in joined_into.get(result).into_iter().flatten() {
                // A union not evaluated yet reads every operand when it is.
                if pending.queued.contains(union) {
                    continue;
                }
                work.visited += 1;
                work.joined += 1;
                let into = values.entry(union).or_default();
                if into.join(&value) {
                    widen(into, changes.entry(union).or_default());
                    changed.push(union);
                }
            }
        }
    }
    values
}

/// The definitions [`solve`] will evaluate again, in order, each at most once at a time.
struct Pending<'a> {
    queue: VecDeque<&'a str>,
    queued: HashSet<&'a str>,
}

impl<'a> Pending<'a> {
    /// Visit `user`, which names a value that changed: evaluate it again unless it is queued.
    fn visit(&mut self, user: &'a str, work: &mut Work) {
        work.visited += 1;
        if self.queued.insert(user) {
            self.queue.push_back(user);
        }
    }
}

/// What the definition `text` can be, given what its operands can be and what each followed
/// thread-memory array `held` ([`solve`]).
fn transfer<'a, 'v>(
    text: &'a str,
    operand: impl Fn(&str) -> Cow<'v, Pointers<Local>>,
    arrays: &ThreadArrays<'a>,
    held: &HashMap<&'a str, Pointers<Local>>,
    layouts: &layout::Layouts<'_>,
) -> Pointers<Local> {
    let Some((opcode, rest)) = text.split_once(' ') else {
        return Pointers::forged();
    };
    let mut pointers = Pointers::default();
    match opcode {
        "bitcast" | "addrspacecast" => {
            let source = rest.split(" to ").next().unwrap_or_default();
            pointers = operand(source).into_owned();
        }
        "freeze" => pointers = operand(rest).into_owned(),
        "getelementptr" => {
            let operands = split_top_level(rest);
            let Some(base) = operands.get(1) else {
                return Pointers::forged();
            };
            let delta = operands.get(2..).and_then(|indices| {
                let source = gep_source_type(operands[0]);
                let (low, high) = layouts.gep_range(source, indices)?;
                let stepped = layouts
                    .gep_residue(source, indices, POINTER_BYTES)
                    .is_some();
                Some(Span::new(low, high, stepped))
            });
            let base = operand(base).into_owned();
            pointers = Pointers {
                roots: base
                    .roots
                    .into_iter()
                    .map(|(root, offset)| (root, add(offset, delta)))
                    .collect(),
                // An offset from a pointer that may be null is an address made from an integer.
                forged: base.forged || (base.other && delta != exact(0)),
                ..base
            };
        }
        "select" | "phi" => {
            for value in union_operands(text).into_iter().flatten() {
                match value {
                    Some(value) => pointers.join(&operand(value)),
                    None => pointers.join(&Pointers::forged()),
                };
            }
        }
        "load" => {
            let operands = memory_operands(rest);
            let (Some(ty), Some(address)) = (operands.first(), operands.get(1)) else {
                return Pointers::forged();
            };
            match arrays.load(text) {
                // A pointer held in a thread-memory array: any pointer stored into it.
                Some(Some(array)) if one_pointer(ty) => {
                    pointers = held.get(array).cloned().unwrap_or_default();
                }
                // An array that may hold anything else holds no pointer the case authored.
                Some(_) => return Pointers::forged(),
                None => {
                    let address = operand(address).into_owned();
                    // An address not reached yet loads nothing yet.
                    if address.is_nothing() {
                        return Pointers::default();
                    }
                    // Only a parameter's memory holds pointers the case authored: memory reached
                    // through a loaded, a forged, or an unknown pointer, or memory the module
                    // holds (an `alloca` the trace does not follow, a global), does not.
                    let elsewhere = address
                        .roots
                        .iter()
                        .any(|(root, _)| !matches!(root, Local::Param(_)));
                    let forged = address.forged || address.other || elsewhere;
                    pointers.other = forged;
                    pointers.forged = forged;
                    // One pointer is the one at its slot; an aggregate or a vector (of pointers,
                    // or of bytes) is any of the pointers at the slots it covers, when it is
                    // loaded at slots and holds its pointers at slots (else one may be bytes of
                    // two, at no known slot).
                    let one = one_pointer(ty);
                    let size = layouts.size(ty);
                    let aligned = layouts.pointers_aligned(ty);
                    for (root, at) in address.roots {
                        if let Local::Param(param) = root {
                            let at = if one {
                                at
                            } else {
                                at.filter(|at| aligned && at.stepped && at.low % SLOT_BYTES == 0)
                                    .and_then(|at| touched_slots(Some(at), size))
                            };
                            pointers
                                .roots
                                .insert((Local::Loaded { param, at }, exact(0)));
                        }
                    }
                }
            }
        }
        // A pointer made from an integer may be any address.
        "inttoptr" => return Pointers::forged(),
        // A vector of pointers, or an aggregate, is followed as the union of its elements.
        "extractelement" | "extractvalue" => {
            let operands = split_top_level(rest);
            let Some(whole) = operands.first() else {
                return Pointers::forged();
            };
            pointers = operand(whole).into_owned();
        }
        "insertelement" | "shufflevector" | "insertvalue" => {
            let operands = split_top_level(rest);
            let (Some(left), Some(right)) = (operands.first(), operands.get(1)) else {
                return Pointers::forged();
            };
            pointers.join(&operand(left));
            pointers.join(&operand(right));
        }
        // Thread memory the function holds: an object of its own, no map's.
        "alloca" => return Pointers::root(Local::Held, None),
        // A call's result, and whatever else the trace does not follow, may be any address.
        _ => return Pointers::forged(),
    }
    pointers
}

/// What the operand `operand` (its text) can be, given the function's parameters and solved
/// values.
fn known(
    params: &HashMap<&str, Pointers<Local>>,
    values: &HashMap<&str, Pointers<Local>>,
    operand: &str,
    module: &Module<'_>,
) -> Pointers<Local> {
    let value = value_text(operand);
    params
        .get(value)
        .or_else(|| values.get(value))
        .cloned()
        .unwrap_or_else(|| constant(operand, module))
}

/// The operands (their texts) a `select` or a `phi`, the instruction `text`, is the union of: a
/// `select`'s two values, not its condition, and each of a `phi`'s incoming values, not their
/// blocks; `None` for one the text lacks, which may be any address. `None` for any other
/// instruction.
fn union_operands(text: &str) -> Option<Vec<Option<&str>>> {
    let (opcode, rest) = text.split_once(' ')?;
    match opcode {
        "select" => {
            let operands = split_top_level(rest);
            Some(vec![operands.get(1).copied(), operands.get(2).copied()])
        }
        "phi" => Some(match rest.find('[') {
            Some(start) => split_top_level(&rest[start..])
                .into_iter()
                .map(|pair| {
                    pair.trim()
                        .strip_prefix('[')
                        .and_then(|pair| split_top_level(pair).first().copied())
                        .map(str::trim)
                })
                .collect(),
            None => vec![None],
        }),
        _ => None,
    }
}

/// What an operand that names no parameter or definition can be (its text). A global the module
/// defines is an object it holds ([`Local::Held`]), and so is a `getelementptr`, `bitcast`, or
/// `addrspacecast` of one: an offset keeps its base's object. `null`, `undef`, `poison`, a
/// global the module only declares, and a function may be anything (no write through one is
/// trusted, and no map is placed there). An offset from one of those, an `inttoptr`, and every
/// other constant may be any address ([`Pointers::forged`]).
fn constant(operand: &str, module: &Module<'_>) -> Pointers<Local> {
    match value(operand) {
        Value::Global(name) if module.globals.contains(name) => Pointers::root(Local::Held, None),
        Value::Global(_) | Value::Empty => Pointers::anything(),
        Value::Expression {
            opcode: "bitcast" | "addrspacecast",
            operands,
        } => operands
            .rsplit_once(" to ")
            .map_or_else(Pointers::forged, |(cast, _)| constant(cast, module)),
        Value::Expression {
            opcode: "getelementptr",
            operands,
        } => {
            let base = split_top_level(operands)
                .get(1)
                .map(|base| constant(base, module));
            match base {
                Some(base) if base == Pointers::root(Local::Held, None) => base,
                _ => Pointers::forged(),
            }
        }
        Value::Expression { .. } | Value::Local(_) | Value::Literal => Pointers::forged(),
    }
}

/// The operands (`<type> [attributes] <value>`) through which the instruction `text` may write: a
/// `store`'s, `atomicrmw`'s, or `cmpxchg`'s address, and every operand of an intrinsic but one
/// the call promises only to read ([`only_read`]). A defined function's own instructions are
/// its, and a call to code the module does not hold is refused whole ([`unknown_call`]).
fn written_through<'a>(text: &'a str, defined: &HashSet<&str>) -> Vec<&'a str> {
    let (opcode, rest) = text.split_once(' ').unwrap_or((text, ""));
    let mut rest = rest.trim_start();
    match opcode {
        "store" => memory_operands(rest).get(1).copied().into_iter().collect(),
        "atomicrmw" | "cmpxchg" => {
            while let Some(after) = ["weak ", "volatile "]
                .iter()
                .find_map(|flag| rest.strip_prefix(flag))
            {
                rest = after.trim_start();
            }
            if opcode == "atomicrmw" {
                rest = rest.split_once(' ').map_or("", |(_, after)| after);
            }
            split_top_level(rest).first().copied().into_iter().collect()
        }
        _ => match call(text) {
            Some(call) if !defined.contains(call.callee) && intrinsic(call.callee) => call
                .args
                .into_iter()
                .filter(|arg| !only_read(arg))
                .collect(),
            _ => Vec::new(),
        },
    }
}

/// Whether `callee` is an AIR or LLVM intrinsic: a stable ABI symbol whose declaration states
/// what it does.
fn intrinsic(callee: &str) -> bool {
    callee.starts_with("air.") || callee.starts_with("llvm.")
}

/// Whether the instruction `text` calls code the module does not hold: a function it declares
/// but does not define (an intrinsic aside), an indirect call, or a callee it cannot name (a
/// constant expression, inline assembly).
fn unknown_call(text: &str, defined: &HashSet<&str>) -> bool {
    let opcode = text.split_whitespace().next().unwrap_or_default();
    if !matches!(
        opcode,
        "call" | "tail" | "musttail" | "notail" | "invoke" | "callbr"
    ) {
        return false;
    }
    call(text).is_none_or(|call| !defined.contains(call.callee) && !intrinsic(call.callee))
}

/// Whether a call's operand text carries LLVM's promise that the callee only reads through it
/// and keeps no copy of it: `readonly` or `readnone`, with `captures(none)` or `nocapture`.
fn only_read(operand: &str) -> bool {
    let attribute = |names: [&str; 2]| {
        operand
            .split_whitespace()
            .any(|token| names.contains(&token))
    };
    attribute(["readonly", "readnone"]) && attribute(["captures(none)", "nocapture"])
}

/// The values whose pointers the instruction `text` lets escape the analysis: where it no longer
/// follows them, so any pointer it does not follow may be one of them and write through it.
/// Followed instructions, reads and writes through an address, comparisons, a pointer stored into
/// a followed thread-memory array, a defined function's arguments, and an intrinsic's one pointers
/// when it returns no pointer let nothing escape. A pointer stored anywhere else, returned, turned
/// into an integer or an aggregate, passed to an intrinsic in a vector, or passed to any other
/// call escapes.
fn escaping<'a>(text: &'a str, arrays: &ThreadArrays<'a>, defined: &HashSet<&str>) -> Vec<&'a str> {
    let (opcode, rest) = text.split_once(' ').unwrap_or((text, ""));
    match opcode {
        "bitcast" | "addrspacecast" | "freeze" | "getelementptr" | "select" | "phi" | "load"
        | "extractelement" | "insertelement" | "shufflevector" | "extractvalue" | "icmp"
        | "alloca" | "br" | "switch" | "unreachable" => Vec::new(),
        "store" => {
            let operands = memory_operands(rest);
            let into_array = operands
                .get(1)
                .is_some_and(|address| arrays.followed(value_text(address)));
            match (into_array, operands.first()) {
                (false, Some(value)) => vec![value_text(value)],
                _ => Vec::new(),
            }
        }
        _ => match call(text) {
            Some(call) if defined.contains(call.callee) => Vec::new(),
            // An intrinsic reads and writes through its pointers, but may store a vector of
            // pointers' elements (`llvm.masked.store`, `llvm.masked.scatter`).
            Some(call) if intrinsic(call.callee) && !returns_pointer(text) => call
                .args
                .iter()
                .filter(|arg| pointer_type(arg).is_some_and(|pointer| pointer.vector))
                .map(|arg| value_text(arg))
                .collect(),
            _ => ssa_names(text).collect(),
        },
    }
}

/// The addresses an instruction `text` may read as bytes rather than as one pointer, each with
/// how many bytes (`None` when unknown): a `load` of anything but one pointer, and every pointer
/// operand of a call to a function the module does not define (an intrinsic may read through
/// any of them, whatever its name).
fn byte_reads<'a>(
    text: &'a str,
    layouts: &layout::Layouts<'_>,
    defined: &HashSet<&str>,
) -> Vec<(&'a str, Option<u64>)> {
    if let Some(rest) = text.strip_prefix("load ") {
        let operands = memory_operands(rest);
        return match (operands.first(), operands.get(1)) {
            (Some(ty), Some(address)) if !one_pointer(ty) => {
                vec![(value_text(address), layouts.size(ty))]
            }
            _ => Vec::new(),
        };
    }
    match call(text) {
        Some(call) if !defined.contains(call.callee) => call
            .args
            .iter()
            .filter(|arg| points(arg))
            .map(|arg| (value_text(arg), None))
            .collect(),
        _ => Vec::new(),
    }
}

/// The pointer slots a read of `size` bytes at `offset` touches, as the offsets an 8-byte pointer
/// read there could start at: the slots holding its first and last bytes, and every slot between
/// (exact for slots at multiples of 8, and never fewer for others). `None` when the offset or the
/// size is unknown.
fn touched_slots(offset: Offset, size: Option<u64>) -> Offset {
    let Span { low, high, .. } = offset?;
    let last = high.checked_add(i64::try_from(size?).ok()?.checked_sub(1)?)?;
    Some(Span::new(
        low.div_euclid(SLOT_BYTES) * SLOT_BYTES,
        last.max(low).div_euclid(SLOT_BYTES) * SLOT_BYTES,
        true,
    ))
}

/// Whether a call's text returns a pointer (its return type, before the callee, names one).
fn returns_pointer(text: &str) -> bool {
    text.find('@')
        .is_some_and(|at| text[..at].contains("ptr") || text[..at].contains('*'))
}

/// A `load`'s or `store`'s operands, after its `atomic` and `volatile` flags.
fn memory_operands(rest: &str) -> Vec<&str> {
    let mut rest = rest.trim_start();
    while let Some(after) = rest
        .strip_prefix("atomic ")
        .or_else(|| rest.strip_prefix("volatile "))
    {
        rest = after.trim_start();
    }
    split_top_level(rest)
}

/// The type of the value the definition `text` makes (its text, perhaps with a value after it),
/// for the definitions [`transfer`] follows, `inttoptr`, and an aggregate an `insertvalue` makes;
/// an `extractelement`'s and an `extractvalue`'s are their whole operand's.
fn result_type(text: &str) -> Option<&str> {
    let (opcode, rest) = text.split_once(' ')?;
    match opcode {
        "load" => memory_operands(rest).first().copied(),
        "bitcast" | "addrspacecast" | "inttoptr" => rest.split(" to ").nth(1),
        "freeze" | "phi" => Some(rest),
        "getelementptr" | "select" => split_top_level(rest).get(1).copied(),
        "extractelement" | "extractvalue" | "insertelement" | "shufflevector" | "insertvalue" => {
            split_top_level(rest).first().copied()
        }
        _ => None,
    }
}

impl<'a> ThreadArrays<'a> {
    /// The values derived from each array's address, and what each array holds: its stored
    /// pointers, or `None` when its address reaches anything but a `bitcast`, `addrspacecast`, or
    /// `getelementptr` of it, the address of a `load` or a `store` of a pointer, or a lifetime
    /// marker, when a load or store is not at a whole pointer slot (a byte offset from the
    /// array's start that is not a multiple of the pointer's size), or when nothing is stored
    /// into it. The derived addresses are found by a worklist over `users`, each user visited
    /// once per mention.
    fn new(
        definitions: &HashMap<&'a str, &'a str>,
        users: &HashMap<&'a str, Vec<&'a str>>,
        body: &[&'a str],
        layouts: &layout::Layouts<'_>,
        work: &mut Work,
    ) -> ThreadArrays<'a> {
        let mut array_of = definitions
            .iter()
            .filter(|(_, text)| text.starts_with("alloca "))
            .map(|(&name, _)| (name, (name, Some(0))))
            .collect::<HashMap<_, _>>();
        if array_of.is_empty() {
            return ThreadArrays {
                array_of,
                held: HashMap::new(),
            };
        }
        let mut pending = array_of.keys().copied().collect::<Vec<_>>();
        while let Some(value) = pending.pop() {
            let (array, residue) = array_of[value];
            for &user in users.get(value).into_iter().flatten() {
                work.visited += 1;
                if array_of.contains_key(user) {
                    continue;
                }
                let text = definitions[user];
                let (opcode, rest) = text.split_once(' ').unwrap_or((text, ""));
                let derived = match opcode {
                    "bitcast" | "addrspacecast" => (rest.split(" to ").next().map(value_text)
                        == Some(value))
                    .then_some(residue),
                    "getelementptr" => {
                        let operands = split_top_level(rest);
                        (operands.get(1).map(|base| value_text(base)) == Some(value)).then(|| {
                            let step = operands.get(2..).and_then(|indices| {
                                layouts.gep_residue(
                                    gep_source_type(operands[0]),
                                    indices,
                                    POINTER_BYTES,
                                )
                            });
                            residue
                                .zip(step)
                                .map(|(base, step)| (base + step) % POINTER_BYTES)
                        })
                    }
                    _ => None,
                };
                if let Some(residue) = derived {
                    array_of.insert(user, (array, residue));
                    pending.push(user);
                }
            }
        }
        let mut held = array_of
            .values()
            .map(|&(array, _)| (array, Some(Vec::new())))
            .collect::<HashMap<_, _>>();
        for &text in body {
            let mentioned = ssa_names(text)
                .filter_map(|name| array_of.get(name).map(|&(array, _)| array))
                .collect::<BTreeSet<_>>();
            if mentioned.is_empty() {
                continue;
            }
            let (opcode, rest) = text.split_once(' ').unwrap_or((text, ""));
            let operands = match opcode {
                "load" | "store" => memory_operands(rest),
                _ => split_top_level(rest),
            };
            // Only the operand at `position` names an array's address.
            let only = |position: usize| {
                operands.iter().enumerate().all(|(index, operand)| {
                    index == position || !ssa_names(operand).any(|name| array_of.contains_key(name))
                })
            };
            // The address operand is a whole slot of its array.
            let slot = || {
                operands
                    .get(1)
                    .and_then(|address| array_of.get(value_text(address)))
                    .is_some_and(|&(_, residue)| residue == Some(0))
            };
            let allowed = match opcode {
                "alloca" | "bitcast" | "addrspacecast" => true,
                "getelementptr" => only(1),
                "load" => {
                    only(1) && slot() && operands.first().is_some_and(|loaded| one_pointer(loaded))
                }
                "store" => {
                    let stored = operands.first().copied().unwrap_or_default();
                    let pointer = one_pointer(stored) && only(1) && slot();
                    let array = operands
                        .get(1)
                        .and_then(|address| array_of.get(value_text(address)));
                    if let (true, Some(Some(values))) =
                        (pointer, array.and_then(|&(array, _)| held.get_mut(array)))
                    {
                        values.push(value_text(stored));
                    }
                    pointer
                }
                _ => call(text).is_some_and(|call| call.callee.starts_with("llvm.lifetime.")),
            };
            if !allowed {
                for array in mentioned {
                    held.insert(array, None);
                }
            }
        }
        for stored in held.values_mut() {
            if stored.as_ref().is_some_and(Vec::is_empty) {
                *stored = None;
            }
        }
        ThreadArrays { array_of, held }
    }

    /// For a load `text` from an array: the array, or `Some(None)` when the array may hold
    /// anything but the pointers stored into it. `None` when `text` is no load from an array.
    fn load(&self, text: &str) -> Option<Option<&'a str>> {
        let rest = text.strip_prefix("load ")?;
        let address = value_text(memory_operands(rest).get(1)?);
        let &(array, _) = self.array_of.get(address)?;
        Some(matches!(self.held.get(array), Some(Some(_))).then_some(array))
    }

    /// Each followed array and the pointers stored into it.
    fn held(&self) -> impl Iterator<Item = (&'a str, &[&'a str])> {
        self.held
            .iter()
            .filter_map(|(&array, stored)| Some((array, stored.as_deref()?)))
    }

    /// Whether `address` is in an array whose stored pointers are followed.
    fn followed(&self, address: &str) -> bool {
        self.array_of
            .get(address)
            .is_some_and(|&(array, _)| matches!(self.held.get(array), Some(Some(_))))
    }
}

/// A `getelementptr`'s source element type, after its flags.
fn gep_source_type(operand: &str) -> &str {
    let mut operand = operand.trim_start();
    loop {
        if let Some(rest) = ["inbounds ", "nuw ", "nusw "]
            .iter()
            .find_map(|flag| operand.strip_prefix(flag))
        {
            operand = rest.trim_start();
        } else if let Some(rest) = operand.strip_prefix("inrange(") {
            operand = rest
                .split_once(')')
                .map_or("", |(_, rest)| rest.trim_start());
        } else {
            return operand;
        }
    }
}

/// The global name at the start of `text` (after its `@`), quoted or not, and the text after it.
fn global_name(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('@')?;
    if let Some(quoted) = text.strip_prefix('"') {
        let end = quoted.find('"')?;
        return Some((&quoted[..end], &quoted[end + 1..]));
    }
    let end = text
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-')))
        .unwrap_or(text.len());
    Some((&text[..end], &text[end..]))
}

/// How many times the module text names each global.
fn global_references(source_ll: &str) -> HashMap<&str, usize> {
    let mut references = HashMap::new();
    for (at, _) in source_ll.match_indices('@') {
        if let Some((name, _)) = global_name(&source_ll[at..]) {
            *references.entry(name).or_default() += 1;
        }
    }
    references
}

/// The text inside the parenthesized list that `text` starts with, and the text after it. A
/// parenthesis inside a quoted name or string is the name's (the text escapes no `"` inside one).
fn parenthesized(text: &str) -> Option<(&str, &str)> {
    let inner = text.strip_prefix('(')?;
    let mut depth = 1usize;
    let mut quoted = false;
    for (index, ch) in inner.char_indices() {
        match ch {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&inner[..index], &inner[index + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

/// The call an instruction's text makes: its callee (the first global named right before a
/// parenthesized list, or for an indirect call the first local value so named, `%` included) and
/// its operands' texts.
fn call(text: &str) -> Option<Call<'_>> {
    let opcode = text.split_whitespace().next()?;
    if !matches!(opcode, "call" | "tail" | "musttail" | "notail") {
        return None;
    }
    let mut from = 0;
    while let Some(at) = text[from..].find('@') {
        let at = from + at;
        let (callee, rest) = global_name(&text[at..])?;
        if let Some((args, _)) = parenthesized(rest) {
            let args = if args.trim().is_empty() {
                Vec::new()
            } else {
                split_top_level(args)
            };
            return Some(Call { callee, args });
        }
        from = at + 1;
    }
    let mut rest = text;
    while let Some(at) = rest.find('%') {
        let (callee, after) = local_name(&rest[at..]);
        if let Some((args, _)) = parenthesized(after) {
            let args = if args.trim().is_empty() {
                Vec::new()
            } else {
                split_top_level(args)
            };
            return Some(Call { callee, args });
        }
        rest = &rest[at + 1..];
    }
    None
}

/// The local name at the start of `text` (`%` included, and its quotes when quoted) and the text
/// after it.
fn local_name(text: &str) -> (&str, &str) {
    let after = &text[1..];
    let length = match after.strip_prefix('"') {
        Some(quoted) => quoted.find('"').map_or(after.len(), |end| end + 2),
        None => after
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '$' | '-')))
            .unwrap_or(after.len()),
    };
    text.split_at(1 + length)
}

/// What the trace knows of the module as a whole: the functions it defines (so a call's
/// arguments are known followed or not), the globals it defines ([`held_globals`]), and its
/// types' layouts.
struct Module<'a> {
    defined: HashSet<&'a str>,
    globals: HashSet<&'a str>,
    layouts: layout::Layouts<'a>,
}

impl<'a> Module<'a> {
    fn new(source_ll: &'a str) -> Module<'a> {
        let defined = source_ll
            .lines()
            .filter_map(|line| line.strip_prefix("define "))
            .filter_map(|define| global_name(&define[define.find('@')?..]))
            .map(|(name, _)| name)
            .collect();
        Module {
            defined,
            globals: held_globals(source_ll),
            layouts: layout::Layouts::new(source_ll),
        }
    }
}

/// The global variables the module defines: objects it holds, which no case authors. A
/// declaration (`external` or `extern_weak`, with no initializer), an alias, and an ifunc name
/// memory the module does not hold, and a function is code.
fn held_globals(source_ll: &str) -> HashSet<&str> {
    source_ll
        .lines()
        .filter_map(|line| {
            let (name, rest) = global_name(line)?;
            let rest = rest.trim_start().strip_prefix('=')?;
            let mut words = rest.split_whitespace();
            let before = words
                .by_ref()
                .take_while(|word| !matches!(*word, "global" | "constant" | "alias" | "ifunc"))
                .collect::<Vec<_>>();
            let kind = rest
                .split_whitespace()
                .find(|word| matches!(*word, "global" | "constant" | "alias" | "ifunc"))?;
            let declared = before
                .iter()
                .any(|word| matches!(*word, "external" | "extern_weak"));
            (matches!(kind, "global" | "constant") && !declared).then_some(name)
        })
        .collect()
}

/// Every defined function of the module text, by name.
fn functions<'a>(
    source_ll: &'a str,
    module: &Module<'_>,
    work: &mut Work,
) -> BTreeMap<&'a str, Function<'a>> {
    let mut functions = BTreeMap::new();
    let mut lines = source_ll.lines();
    while let Some(line) = lines.next() {
        let Some(define) = line.strip_prefix("define ") else {
            continue;
        };
        let header = define
            .find('@')
            .and_then(|at| global_name(&define[at..]))
            .and_then(|(name, rest)| Some((name, parenthesized(rest)?.0)));
        let body = lines
            .by_ref()
            .take_while(|line| !line.starts_with('}'))
            .collect::<Vec<_>>();
        let Some((name, params)) = header else {
            continue;
        };
        let params = if params.trim().is_empty() {
            Vec::new()
        } else {
            split_top_level(params)
                .into_iter()
                .map(|param| {
                    let name = value_text(param);
                    let ty = param.strip_suffix(name)?.trim();
                    (name.starts_with('%') && !ty.is_empty()).then_some((ty, name))
                })
                .collect()
        };
        functions.insert(name, Function::new(params, body, module, work));
    }
    functions
}

#[cfg(test)]
mod tests;
