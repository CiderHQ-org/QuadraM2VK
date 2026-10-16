//! Values a loop reads that it does not define. Where the profile takes
//! `DriverWorkaround::CompositeInsertsWrittenInPlace` (MoltenVK 1.4.1, whose SPIRV-Cross writes an
//! insert into the composite it reads), `serialize_for` makes an `OpCompositeInsert` in a loop that
//! does not define its composite insert into an `OpCopyObject` of the composite written just
//! before the insert, and materializes the composite where it is defined. Where it takes
//! `DriverWorkaround::LoadsForwardedIntoLoops` (the same SPIRV-Cross, which writes a loaded value
//! into each reader's text), a value read from memory before a loop that the loop reads is
//! materialized where it is defined. An insert or a load in the loop that defines its value, and
//! one outside every loop, are written as before, and every other profile writes the module as
//! `serialize` does. Through the MSL `spirv-cross` writes with MoltenVK's options, the loop then
//! writes no member of a value declared before it and reads no memory a value before it was
//! loaded from, where without the encodings it does both.

use super::{compute, cval, elem_ptr, finish_compute, float, gid_x, int, ssbo, vec};
use crate::device::profile::DriverWorkaround;
use crate::pipeline::testing::{moltenvk_msl, AssembleError};
use crate::spirv::builder::constructs::{ArmEnd, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ir::{BinOp, MemAccess, TypedModule};
use crate::spirv::serialize::{relocate, serialize};
use crate::spirv::testing::{assert_valid, committed_profiles, is_moltenvk_capture, link_dense};
use crate::spirv_binary::{Instruction, Operand};
use crate::spv::Op;

/// `w = vectors[gid]` with `w.y + 1` inserted (outside every loop) is stored; then `v =
/// vectors[gid]` before a loop of two iterations, which stores `v` with `v.x + 1` inserted (the
/// loop does not define `v`) and then `x = vectors[gid]`, loaded in the loop, with `x.y + 1`
/// inserted (the loop defines `x`). Nothing reads `v` after the loop: a read there would make
/// SPIRV-Cross copy `v` on its own.
fn kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    // Only these types: every committed profile must accept the module.
    let u32_ = int(&mut b, 32, false);
    let f32_ = float(&mut b, 32);
    let vec2 = vec(&mut b, f32_, 2);
    let vectors = ssbo(&mut b, 0, vec2, 8);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, vectors, i);
        let one_f = cval(f, f32_, 0x3f80_0000);
        let w = f.load(at, MemAccess::default()).expect("w");
        let y = f.composite_extract(w, &[1]).expect("w.y");
        let y = f.binary(BinOp::FAdd, y, one_f).expect("w.y + 1");
        let w = f
            .composite_insert(y, w, &[1])
            .expect("insert before the loop");
        f.store(at, w, MemAccess::default())
            .expect("store before the loop");
        let v = f.load(at, MemAccess::default()).expect("v");
        let counter = f.local_var(u32_, None).expect("counter");
        let counter = f.var_ptr(counter).expect("counter pointer");
        let zero = cval(f, u32_, 0);
        f.store(counter, zero, MemAccess::default())
            .expect("counter = 0");
        let (one, two) = (cval(f, u32_, 1), cval(f, u32_, 2));
        f.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(
                move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                    let n = fb.load(counter, MemAccess::default())?;
                    let done = fb.binary(BinOp::UGreaterThanEqual, n, two)?;
                    fb.if_(
                        ctx,
                        done,
                        Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                            lc.break_(fb)?;
                            Ok(ArmEnd::Exits)
                        }),
                        None,
                    )?;
                    let n1 = fb.binary(BinOp::IAdd, n, one)?;
                    fb.store(counter, n1, MemAccess::default())?;
                    let e = fb.composite_extract(v, &[0])?;
                    let e = fb.binary(BinOp::FAdd, e, one_f)?;
                    let stepped = fb.composite_insert(e, v, &[0])?;
                    fb.store(at, stepped, MemAccess::default())?;
                    let x = fb.load(at, MemAccess::default())?;
                    let y = fb.composite_extract(x, &[1])?;
                    let y = fb.binary(BinOp::FAdd, y, one_f)?;
                    let x = fb.composite_insert(y, x, &[1])?;
                    fb.store(at, x, MemAccess::default())
                },
            ),
        )
        .expect("loop");
    });
    finish_compute(b, main)
}

fn parse(words: &[u32]) -> Vec<Instruction> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::spirv_binary::parse_bytes(&bytes)
        .expect("parse")
        .instructions
}

/// How each `OpCompositeInsert` of `words` is written, in order: whether it inserts into an
/// `OpCopyObject` written just before it, and the opcode of the instruction that defines the
/// value it inserts into (through that copy).
fn inserts(words: &[u32]) -> Vec<(bool, Op)> {
    let insts = parse(words);
    let defining = |id: u32| {
        insts
            .iter()
            .find(|i| i.result_id == Some(id))
            .map(|i| i.opcode)
            .unwrap_or_else(|| panic!("%{id} is defined"))
    };
    let composite = |i: &Instruction| match i.operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("an insert's composite: {other:?}"),
    };
    let copied_from = |i: &Instruction| match i.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("a copy's operand: {other:?}"),
    };
    insts
        .iter()
        .enumerate()
        .filter(|(_, i)| i.opcode == Op::CompositeInsert)
        .map(|(at, i)| {
            let into = composite(i);
            match at.checked_sub(1).and_then(|p| insts.get(p)) {
                Some(copy) if copy.opcode == Op::CopyObject && copy.result_id == Some(into) => {
                    (true, defining(copied_from(copy)))
                }
                _ => (false, defining(into)),
            }
        })
        .collect()
}

/// Only the MoltenVK captures take the workaround; there the loop's insert into `v` reads a copy
/// of `v` made just before it, and `v` is materialized (written as a copy of its load); the
/// insert before the loop and the loop's insert into `x`, which the loop defines, are not copied.
/// Every other profile writes the inserts as `serialize` does. Every form passes L3.
#[test]
fn an_insert_in_a_loop_that_does_not_define_its_composite_reads_a_copy_on_moltenvk() {
    let profiles = committed_profiles().expect("committed profiles");
    assert!(profiles.iter().any(|(stem, _)| is_moltenvk_capture(stem)));
    for (stem, profile) in &profiles {
        let takes = profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::CompositeInsertsWrittenInPlace);
        assert_eq!(takes, is_moltenvk_capture(stem), "{stem}");
        let words = assert_valid(kernel(), profile);
        let expected = if takes {
            [(false, Op::Load), (true, Op::CopyObject), (false, Op::Load)]
        } else {
            [(false, Op::Load), (false, Op::Load), (false, Op::Load)]
        };
        assert_eq!(inserts(&words), expected, "{stem}");
    }
}

/// The statements of `msl`'s kernel inside its first loop that write a member of a value
/// declared before the loop (`name.x = ...;`), which the loop's next iteration reads changed.
fn loop_writes_to_outer_values(msl: &str) -> Vec<String> {
    let (_, kernel) = msl.split_once("kernel void").expect("a kernel");
    let mut lines = kernel.lines().map(str::trim);
    let mut declared = Vec::new();
    for line in lines.by_ref() {
        if line.starts_with("for (") || line.starts_with("while (") || line == "do" {
            break;
        }
        if let Some((name, _)) = line
            .strip_prefix("float2 ")
            .and_then(|rest| rest.split_once(" = "))
        {
            declared.push(name.to_owned());
        }
    }
    assert!(
        !declared.is_empty(),
        "a value declared before the loop:\n{msl}"
    );
    let mut depth = 0usize;
    let mut writes = Vec::new();
    for line in lines {
        if depth > 0
            && line.contains(" = ")
            && declared
                .iter()
                .any(|name| line.starts_with(&format!("{name}.")))
        {
            writes.push(line.to_owned());
        }
        depth += line.matches('{').count();
        depth = depth.saturating_sub(line.matches('}').count());
        if depth == 0 && line.contains('}') {
            break;
        }
    }
    writes
}

/// The MSL `spirv-cross` writes for `words` with MoltenVK's options, or `None` when it is not
/// installed (the test skips visibly).
pub(super) fn msl_of(words: &[u32]) -> Option<String> {
    match moltenvk_msl(words) {
        Ok(msl) => Some(msl),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("spirv-cross: {why}"),
    }
}

/// Through the MSL `spirv-cross` writes with MoltenVK's options: without the copy, the loop
/// writes `v.x` in place (`_N.x = _N.x + 1.0;` on the `v` declared before it), so the second
/// iteration reads the first's sum; with it, the loop writes only members of values it declares.
/// (The insert before the loop may be written in place: it runs once.)
/// `spirv-cross` is external; without it the test skips visibly.
#[test]
fn moltenvk_msl_writes_no_member_of_a_value_declared_before_the_loop() {
    let profiles = committed_profiles().expect("committed profiles");
    let (_, moltenvk) = profiles
        .iter()
        .find(|(stem, _)| is_moltenvk_capture(stem))
        .expect("a MoltenVK capture");
    let copied = assert_valid(kernel(), moltenvk);
    let plain = serialize(kernel()).expect("serialize");
    let plain = relocate(&plain, &link_dense(&plain)).expect("relocate");
    let (Some(copied), Some(plain)) = (msl_of(&copied), msl_of(&plain)) else {
        return;
    };
    let writes = loop_writes_to_outer_values(&copied);
    assert!(writes.is_empty(), "{writes:#?}\n{copied}");
    let writes = loop_writes_to_outer_values(&plain);
    assert!(
        !writes.is_empty(),
        "spirv-cross no longer writes this loop's insert in place: re-measure whether MoltenVK \
         needs `CompositeInsertsWrittenInPlace`\n{plain}"
    );
}

/// `v = vectors[gid]` and its `y = v.y` before a loop of two iterations, and `u = vectors[gid]`
/// read only there (`u.x + 1` stored); the loop loads `x = vectors[gid]` and stores
/// `(v.x + y + x.y)` to both lanes of `vectors[gid]`, so a second read of the memory `v` was
/// loaded from would see the first iteration's store.
fn loads_kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    // Only these types: every committed profile must accept the module.
    let u32_ = int(&mut b, 32, false);
    let f32_ = float(&mut b, 32);
    let vec2 = vec(&mut b, f32_, 2);
    let vectors = ssbo(&mut b, 0, vec2, 8);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, vectors, i);
        let one_f = cval(f, f32_, 0x3f80_0000);
        let u = f.load(at, MemAccess::default()).expect("u");
        let ux = f.composite_extract(u, &[0]).expect("u.x");
        let ux = f.binary(BinOp::FAdd, ux, one_f).expect("u.x + 1");
        let pair = f
            .composite_construct(vec2, &[ux, ux])
            .expect("(u.x + 1, u.x + 1)");
        f.store(at, pair, MemAccess::default())
            .expect("store before the loop");
        let v = f.load(at, MemAccess::default()).expect("v");
        let y = f.composite_extract(v, &[1]).expect("v.y");
        let counter = f.local_var(u32_, None).expect("counter");
        let counter = f.var_ptr(counter).expect("counter pointer");
        let zero = cval(f, u32_, 0);
        f.store(counter, zero, MemAccess::default())
            .expect("counter = 0");
        let (one, two) = (cval(f, u32_, 1), cval(f, u32_, 2));
        f.loop_(
            &mut (),
            LoopSpec::default(),
            Box::new(
                move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                    let n = fb.load(counter, MemAccess::default())?;
                    let done = fb.binary(BinOp::UGreaterThanEqual, n, two)?;
                    fb.if_(
                        ctx,
                        done,
                        Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                            lc.break_(fb)?;
                            Ok(ArmEnd::Exits)
                        }),
                        None,
                    )?;
                    let n1 = fb.binary(BinOp::IAdd, n, one)?;
                    fb.store(counter, n1, MemAccess::default())?;
                    let x = fb.load(at, MemAccess::default())?;
                    let xy = fb.composite_extract(x, &[1])?;
                    let vx = fb.composite_extract(v, &[0])?;
                    let t = fb.binary(BinOp::FAdd, vx, y)?;
                    let t = fb.binary(BinOp::FAdd, t, xy)?;
                    let pair = fb.composite_construct(vec2, &[t, t])?;
                    fb.store(at, pair, MemAccess::default())
                },
            ),
        )
        .expect("loop");
    });
    finish_compute(b, main)
}

/// The opcodes of the instructions whose results an `OpCopyObject` of `words` copies, sorted.
pub(super) fn copied(words: &[u32]) -> Vec<Op> {
    let insts = parse(words);
    let defining = |id: u32| {
        insts
            .iter()
            .find(|i| i.result_id == Some(id))
            .map(|i| i.opcode)
            .unwrap_or_else(|| panic!("%{id} is defined"))
    };
    let mut ops: Vec<Op> = insts
        .iter()
        .filter(|i| i.opcode == Op::CopyObject)
        .map(|i| match i.operands.first() {
            Some(Operand::IdRef(id)) => defining(*id),
            other => panic!("a copy's operand: {other:?}"),
        })
        .collect();
    ops.sort_by_key(|op| *op as u32);
    ops
}

/// Only the MoltenVK captures take the load workaround; there `v` and `y`, read from memory
/// before the loop and read in it, are materialized (each written under its own id and copied);
/// `u`, read only before the loop, and `x`, loaded in the loop, are not. Every other profile writes
/// no copy. Every form passes L3.
#[test]
fn a_value_read_from_memory_before_a_loop_that_reads_it_is_materialized_on_moltenvk() {
    let profiles = committed_profiles().expect("committed profiles");
    for (stem, profile) in &profiles {
        let takes = profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::LoadsForwardedIntoLoops);
        assert_eq!(takes, is_moltenvk_capture(stem), "{stem}");
        let words = assert_valid(loads_kernel(), profile);
        let expected = if takes {
            vec![Op::Load, Op::CompositeExtract]
        } else {
            Vec::new()
        };
        let mut expected = expected;
        expected.sort_by_key(|op| *op as u32);
        assert_eq!(copied(&words), expected, "{stem}");
    }
}

/// How often the statements inside `msl`'s first loop read the memory the loop stores to: the
/// text of the store's destination on the right of an assignment.
fn loop_reads_of_its_store(msl: &str) -> usize {
    let (_, kernel) = msl.split_once("kernel void").expect("a kernel");
    let mut lines = kernel.lines().map(str::trim);
    for line in lines.by_ref() {
        if line.starts_with("for (") || line.starts_with("while (") || line == "do" {
            break;
        }
    }
    let mut depth = 0usize;
    let mut body = Vec::new();
    for line in lines {
        if depth > 0 {
            body.push(line);
        }
        depth += line.matches('{').count();
        depth = depth.saturating_sub(line.matches('}').count());
        if depth == 0 && line.contains('}') {
            break;
        }
    }
    let target = body
        .iter()
        .filter_map(|l| l.split_once(" = "))
        .map(|(lhs, _)| lhs)
        .find(|lhs| lhs.contains("._m0["))
        .unwrap_or_else(|| panic!("the loop's store:\n{msl}"));
    body.iter()
        .filter_map(|l| l.split_once(" = "))
        .map(|(_, rhs)| rhs.matches(target).count())
        .sum()
}

/// Through the MSL `spirv-cross` writes with MoltenVK's options: without the materialization the
/// loop reads the buffer element `v` and `y` were loaded from again (so the second iteration reads
/// the first's store), beside `x`'s own load; with it, the loop reads that memory only for `x`.
/// `spirv-cross` is external; without it the test skips visibly.
#[test]
fn moltenvk_msl_reads_no_memory_again_for_a_value_loaded_before_the_loop() {
    let profiles = committed_profiles().expect("committed profiles");
    let (_, moltenvk) = profiles
        .iter()
        .find(|(stem, _)| is_moltenvk_capture(stem))
        .expect("a MoltenVK capture");
    let materialized = assert_valid(loads_kernel(), moltenvk);
    let plain = serialize(loads_kernel()).expect("serialize");
    let plain = relocate(&plain, &link_dense(&plain)).expect("relocate");
    let (Some(materialized), Some(plain)) = (msl_of(&materialized), msl_of(&plain)) else {
        return;
    };
    assert_eq!(loop_reads_of_its_store(&materialized), 1, "{materialized}");
    assert!(
        loop_reads_of_its_store(&plain) > 1,
        "spirv-cross no longer forwards these loads into the loop: re-measure whether MoltenVK \
         needs `LoadsForwardedIntoLoops`\n{plain}"
    );
}
