//! A texture table copied into a stack array by `llvm.memcpy`.
//!
//! A texture table argument (`array<texture2d<float>, 3>`) is a thread pointer to its handles.
//! Metal's compiler copies such a table, as any array member of a struct, into a stack object with
//! `llvm.memcpy` and loads the handles back. Lowering follows a handle stored into a stack slot
//! and loaded back but not one moved by a byte copy, so such a kernel was refused
//! `Lowering(Access)`. Specialization now makes a copy of whole handles the handle copies it is
//! (`canon::table_copy`), and each handle loaded back is the table element it copied.
//!
//! The kernels below are our own: each copies its table `ts` whole into a private array and
//! queries the width of one element, `tc_constant` at index 1, `tc_dynamic` at the run-time
//! index `out[0]`. Each compiles, on every committed profile, into a kernel that validates and
//! binds the texture slots the element may be: `tc_constant` slot 1 alone, `tc_dynamic` all three.
//!
//! `llvm-as` assembles the module; without it the tests skip visibly. No GPU is used.

use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::sym::SlotClass;
use crate::pipeline::testing::{llvm_assemble, meter, AssembleError, Scratch};
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// The kernels (module docs). Argument 0 is the table `ts`, argument 1 `out`.
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @tc_constant(ptr readonly byval([3 x ptr addrspace(1)]) %ts, ptr addrspace(1) %out) {
entry:
  %arr = alloca [3 x ptr addrspace(1)], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %arr, ptr align 8 %ts, i64 24, i1 false)
  %e = getelementptr inbounds [3 x ptr addrspace(1)], ptr %arr, i64 0, i64 1
  %h = load ptr addrspace(1), ptr %e, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

define void @tc_dynamic(ptr readonly byval([3 x ptr addrspace(1)]) %ts, ptr addrspace(1) %out) {
entry:
  %arr = alloca [3 x ptr addrspace(1)], align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %arr, ptr align 8 %ts, i64 24, i1 false)
  %k = load i32, ptr addrspace(1) %out, align 4
  %r = urem i32 %k, 3
  %i = zext i32 %r to i64
  %e = getelementptr inbounds [3 x ptr addrspace(1)], ptr %arr, i64 0, i64 %i
  %h = load ptr addrspace(1), ptr %e, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!30, !31}
!air.version = !{!20}
!air.language_version = !{!21}

!1 = !{}
!11 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 3, !"air.read", !"air.arg_type_name", !"array<texture2d<float, read>, 3>", !"air.arg_name", !"ts"}
!12 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!22 = !{!11, !12}
!30 = !{ptr @tc_constant, !1, !22}
!31 = !{ptr @tc_dynamic, !1, !22}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`MODULE`] assembled, or `None` (skipped visibly) without `llvm-as`.
fn module() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-table-copies");
    let dir = scratch.path();
    let ll = dir.join("table_copies.ll");
    std::fs::write(&ll, MODULE).expect("write");
    let r = llvm_assemble(&ll);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// `entry` compiled on `p`.
fn compile(bc: &[u8], entry: &str, p: &DeviceProfile) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The texture slots `a` binds.
fn texture_slots(a: &FunctionArtifact) -> Vec<u32> {
    a.interface
        .slots
        .keys()
        .filter(|s| s.class == SlotClass::Texture)
        .map(|s| s.index)
        .collect()
}

/// Each kernel compiles on every committed profile, binds the slots its element may be, and
/// validates.
#[test]
fn a_texture_table_copied_into_a_stack_array_reads_its_elements() {
    let Some(bc) = module() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, slots) in [("tc_constant", &[1][..]), ("tc_dynamic", &[0, 1, 2][..])] {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert_eq!(texture_slots(&a), slots, "{stem} {entry}");
            let t = link(&[&a], &LinkInputs::standalone(), &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: link: {e}"))
                .table;
            let words = relocate(&a, &t).expect("relocate");
            check_words_valid(&words, &a.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
        }
    }
}
