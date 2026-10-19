//! The bound values a function reads as data (`SpecKey::read_as_data`).
//!
//! The format class of the texture bound at a slot (`SpecKey::TexelFormat`), and the render
//! pass's extent a tile function stages its tile within (`SpecKey::RenderAreaWidth`,
//! `SpecKey::RenderAreaHeight`) and a tile or fragment function places its tile's threadgroup
//! memory by (`SpecKey::TileGridWidth`), were specialization constants, so each new format a guest
//! bound, and each new pass size a tile function ran in, made a new `VkPipeline`, a whole driver
//! compile each (a guest's tile function over passes 64 to 1984 pixels wide made 32). A function
//! reads them from its own push fields, `PushField::Specialized(key)`, and declares no
//! specialization constant for them:
//!
//! - `select` asks a field for each such key the function's families declare ([`request`]);
//! - the families still declare each key's constant and build their operations over it as
//!   before, so every arm and every test reads the value it read;
//! - the entry hook ([`emit_entry`]) reads each field in the function's first block, right
//!   after the push fields load, and computes there each specialization-constant operation over
//!   them, as instructions; a tile function's entry wrapper, which stages its tile before and
//!   after the function (`access::tile::wrap_entry`), runs the same hook in its own first block;
//! - once the module is finalized, every read in each of those functions takes its local value
//!   (`TypedModule::localize_specs`), and the constants nothing else reads leave the module.
//!
//! The field holds the value the binder specialized the constant to before, so the module
//! computes what it did; only the driver no longer folds it. Every other bound value stays a
//! specialization: read as data, they cost NVIDIA's making far more.

use std::collections::{BTreeMap, BTreeSet};

use crate::lower::access::tile;
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, SelectCx};
use crate::pipeline::sym::{PushField, SpecKey};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstKind, SpecOp, Type};

/// The keys `families` declare constants of that a function reads as data
/// (`SpecKey::read_as_data`): the format class of every texture slot whose texture family
/// declares one; a tile function's render area where it stages its tile's pixels, and its tile
/// grid's width where it stages its tile's threadgroup memory (`access::tile::declare`); a
/// fragment function's tile grid's width where its arena is its tile's memory
/// (`memory::arena::declare_threadgroup`).
fn keys(families: &FamilyPlans) -> BTreeSet<SpecKey> {
    let access = &families.access;
    let texel = families
        .texture
        .slots
        .values()
        .filter(|s| s.declares_texel_format())
        .map(|s| SpecKey::TexelFormat(s.slot));
    let area = tile::stages(access.tile.as_ref())
        .then_some([SpecKey::RenderAreaWidth, SpecKey::RenderAreaHeight])
        .into_iter()
        .flatten();
    let fragment_tile = access
        .local
        .tg
        .as_ref()
        .is_some_and(|s| s.fragment_tile().is_some());
    let grid = (tile::has_memory(access) || fragment_tile).then_some(SpecKey::TileGridWidth);
    texel.chain(area).chain(grid).collect()
}

/// Ask a push field for each key the function reads as data, once the texture and sampler
/// families are complete and before `select` places the fields.
pub(crate) fn request(cx: &mut SelectCx<'_>) {
    let keys = keys(&cx.families);
    cx.push.extend(keys.into_iter().map(PushField::Specialized));
}

/// The entry hook: read each field the plan places for a key the function reads as data, and
/// compute from it, in the first block of `fb`'s function (the function's own, or its entry
/// wrapper's), the value of the key's constant and of every specialization-constant operation over
/// such constants, recording them in `cx` for `TypedModule::localize_specs`. A placed field whose
/// key the module does not declare, or a declared constant of such a key without its field, is
/// the plan and the declarations disagreeing: an invariant failure.
pub(crate) fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let placed: BTreeSet<SpecKey> = cx
        .plan
        .push
        .fields
        .iter()
        .filter_map(|f| match f.field {
            PushField::Specialized(key) => Some(key),
            _ => None,
        })
        .collect();
    if placed.is_empty() {
        return Ok(());
    }
    let mut declared: BTreeMap<SpecKey, ConstId> = BTreeMap::new();
    let mut ops: Vec<(ConstId, SpecOp, Vec<ConstId>)> = Vec::new();
    let mut stray = None;
    for (c, def) in fb.module().consts().iter() {
        match &def.kind {
            ConstKind::Spec { key, .. } if placed.contains(key) => {
                declared.insert(*key, c);
            }
            ConstKind::Spec { key, .. } if key.read_as_data() => stray = Some(*key),
            ConstKind::SpecOp { op, operands } => ops.push((c, *op, operands.to_vec())),
            _ => {}
        }
    }
    if let Some(key) = stray {
        return Err(cx.fail(format!("{key:?} is declared without its push field")));
    }
    if let Some(key) = placed.iter().find(|k| !declared.contains_key(k)) {
        return Err(cx.fail(format!(
            "a push field for {key:?}, which no constant declares"
        )));
    }
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let mut local: BTreeMap<ConstId, SValue> = BTreeMap::new();
    for (key, c) in &declared {
        if fb.module().consts().get(*c)?.ty != u32_ {
            return Err(cx.fail(format!("{key:?} is not a `u32`")));
        }
        let word = cx.push_field(fb, PushField::Specialized(*key))?;
        local.insert(*c, word);
    }
    // In declaration order, so an operation's operands come before it.
    for (c, op, operands) in ops {
        if !operands.iter().any(|o| local.contains_key(o)) {
            continue;
        }
        let mut vals = Vec::with_capacity(operands.len());
        for o in &operands {
            vals.push(match local.get(o) {
                Some(v) => *v,
                None => fb.module().const_value(*o)?,
            });
        }
        let v = match (op, vals.as_slice()) {
            (SpecOp::IAdd, [a, b]) => fb.binary(BinOp::IAdd, *a, *b)?,
            (SpecOp::IMul, [a, b]) => fb.binary(BinOp::IMul, *a, *b)?,
            (SpecOp::BitwiseAnd, [a, b]) => fb.binary(BinOp::BitwiseAnd, *a, *b)?,
            (SpecOp::ShiftRightLogical, [a, b]) => fb.binary(BinOp::ShiftRightLogical, *a, *b)?,
            (SpecOp::ULessThan, [a, b]) => fb.binary(BinOp::ULessThan, *a, *b)?,
            (SpecOp::Select, [cond, a, b]) => fb.select(*cond, *a, *b)?,
            _ => {
                return Err(cx.fail(format!(
                    "a specialization operation {op:?} of {} operands",
                    vals.len()
                )))
            }
        };
        local.insert(c, v);
    }
    cx.localized = Some((fb.id(), local.into_iter().collect()));
    Ok(())
}
