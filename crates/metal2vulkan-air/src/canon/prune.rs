//! Entry pruning: the module keeps exactly what the entry reaches.
//!
//! After specialization the entry's body decides everything else: the declarations it calls, the
//! globals and constants its operands name (transitively through initializers), and the types all
//! of those use. Everything else (other entries, the static initializer, `llvm.global_ctors`,
//! unreferenced function-constant globals, the raw metadata the ABI was decoded from) is dropped,
//! and every id is renumbered densely. Functions and globals are renumbered in the order the mark
//! first reaches them, not the module's order, which carries no meaning: functions the entry, then
//! those its body calls in order, then those its constants name; globals as the walk of the
//! constants reaches them, depth first in the order the body names them and each constant names
//! its parts. LLVM's upgrade of a typed-pointer intrinsic's name declares it anew at the module's
//! end, so the two spellings of one program can list their declarations in other orders. Types and
//! constants are renumbered in first-use order of one fixed walk (kept functions in order, each
//! signature before its body, then kept globals), children first, so the pruned module depends only
//! on the entry's structure: typed- and opaque-pointer bitcode of one program prune to the same
//! module.

use crate::budget::Meter;
use crate::error::DiagName;
use crate::error::{Error, Feature, Unsupported};
use crate::ir::TypeTable;
use crate::ir::{Arena, Const, ConstId, ConstTable, FuncId, GlobalId, Idx, ModuleParts};

use super::renumber::{inst_refs, Renumber, Syms};
use super::rewrite::{filled, invariant, scoped};

/// The pruned module's parts and the renumbering of the kept globals and functions.
pub(super) struct Pruned {
    pub parts: ModuleParts,
    pub func: FuncId,
    pub gmap: Vec<Option<GlobalId>>,
    /// The renumbering of the kept functions (a post-tessellation entry's control-point
    /// function).
    pub fmap: Vec<Option<FuncId>>,
}

/// Keep what `entry` reaches in `parts`, renumbered.
pub(super) fn prune(parts: ModuleParts, entry: FuncId, m: &mut Meter) -> Result<Pruned, Error> {
    scoped(m, |m, held| prune_in(parts, entry, m, held))
}

fn prune_in(
    parts: ModuleParts,
    entry: FuncId,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Pruned, Error> {
    let ModuleParts {
        symbols,
        types,
        consts,
        globals,
        funcs,
        metadata: _,
        named_md: _,
        triple,
        source_filename,
        data_layout,
    } = parts;
    let entry_fn = funcs
        .get(entry)
        .ok_or_else(|| invariant("entry out of range"))?;
    let body = entry_fn
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry has no body"))?;

    // ---- Mark: what the entry reaches, in the order it first reaches it. ----
    let mut keep_c = filled(m, held, consts.len(), false)?;
    let mut keep_g = filled(m, held, globals.len(), false)?;
    let mut keep_f = filled(m, held, funcs.len(), false)?;
    let mut gorder = ordered(m, held, globals.len())?;
    let mut forder = ordered(m, held, funcs.len())?;
    let mut stack: Vec<ConstId> = Vec::new();
    reach(&mut keep_f, &mut forder, entry);
    for inst in body.insts.values() {
        m.tick(1)?;
        inst_refs(
            &inst.kind,
            inst.ty,
            |c| stack.push(c),
            |f| reach(&mut keep_f, &mut forder, f),
            |_| {},
        );
    }
    // The walk pops the last pushed first: the body's constants, and below each constant its
    // parts, go on the stack last first, so the walk reaches them in the order they are named.
    stack.reverse();
    while let Some(c) = stack.pop() {
        m.tick(1)?;
        let Some(slot) = keep_c.get_mut(c.index()) else {
            return Err(invariant("constant out of range"));
        };
        if *slot {
            continue;
        }
        *slot = true;
        let k = consts
            .get(c)
            .ok_or_else(|| invariant("constant out of range"))?;
        let bytes = std::mem::size_of_val(k.value.children());
        m.charge(bytes)?;
        *held += bytes;
        stack.extend(k.value.children().iter().rev());
        match k.value {
            Const::GlobalAddr(g) => {
                let Some(kg) = keep_g.get_mut(g.index()) else {
                    return Err(invariant("global out of range"));
                };
                if !*kg {
                    *kg = true;
                    gorder.push(g);
                    if let Some(init) = globals.get(g).and_then(|g| g.init) {
                        stack.push(init);
                    }
                }
            }
            Const::FuncAddr(f) => reach(&mut keep_f, &mut forder, f),
            _ => {}
        }
    }
    // A defined function other than the entry that is still reachable has its address taken
    // (direct calls were refused before pruning): a function pointer, named by its symbol.
    for (f, func) in funcs.iter() {
        if f != entry && keep_f.get(f.index()).copied().unwrap_or(false) && func.body.is_some() {
            let name = symbols.get(func.name).unwrap_or(b"?");
            let name = DiagName::new(String::from_utf8_lossy(name));
            return Err(Unsupported::not_implemented(Feature::DynamicCall)
                .naming(name)
                .into());
        }
    }

    // ---- Renumber globals and functions in the order the mark reached them. ----
    let mut gmap = filled(m, held, globals.len(), None)?;
    for (n, g) in (0u32..).zip(&gorder) {
        if let Some(slot) = gmap.get_mut(g.index()) {
            *slot = Some(GlobalId::new(n));
        }
    }
    let mut fmap = filled(m, held, funcs.len(), None)?;
    for (n, f) in (0u32..).zip(&forder) {
        if let Some(slot) = fmap.get_mut(f.index()) {
            *slot = Some(FuncId::new(n));
        }
    }

    let mut r = Renumber::new(
        (&types, &consts),
        (TypeTable::new(), ConstTable::new()),
        (&gmap, &fmap),
        Syms::Same,
        m,
        held,
    )?;

    let mut new_funcs = Arena::new();
    for &f in &forder {
        m.tick(1)?;
        let func = funcs
            .get(f)
            .ok_or_else(|| invariant("function out of range"))?;
        let out = r.function(func, m)?;
        new_funcs.push(out, m)?;
    }
    let mut new_globals = Arena::new();
    for &g in &gorder {
        m.tick(1)?;
        let global = globals
            .get(g)
            .ok_or_else(|| invariant("global out of range"))?;
        let out = r.global(global, m)?;
        new_globals.push(out, m)?;
    }
    let Renumber {
        types: new_types,
        consts: new_consts,
        ..
    } = r;
    // The old functions' bodies are freed here, the kept ones renumbered.
    for func in funcs.into_vec() {
        if let Some(body) = func.body {
            body.release(m);
        }
    }
    let func = fmap
        .get(entry.index())
        .copied()
        .flatten()
        .ok_or_else(|| invariant("the entry was pruned"))?;
    Ok(Pruned {
        parts: ModuleParts {
            symbols,
            types: new_types,
            consts: new_consts,
            globals: new_globals,
            funcs: new_funcs,
            metadata: Arena::new(),
            named_md: Vec::new(),
            triple,
            source_filename,
            data_layout,
        },
        func,
        gmap,
        fmap,
    })
}

/// An empty list of up to `n` ids, its room charged to the ledger: each id is pushed at most once
/// (the mark's flag guards it), so it never grows past `n`.
fn ordered<T>(m: &mut Meter, held: &mut usize, n: usize) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, n)?;
    *held = held.saturating_add(n.saturating_mul(std::mem::size_of::<T>()));
    Ok(out)
}

/// Mark `f` kept, recording it in `order` the first time.
fn reach(keep: &mut [bool], order: &mut Vec<FuncId>, f: FuncId) {
    if let Some(k) = keep.get_mut(f.index()) {
        if !*k {
            *k = true;
            order.push(f);
        }
    }
}
