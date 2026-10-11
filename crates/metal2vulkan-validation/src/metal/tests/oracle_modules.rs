//! The bound of the modules the oracle compiles, decided before any device opens. A public fixture
//! is compiled on the host apart from the row the checker classified, so its compiled module is
//! classified again by the checker's own rules. These tests run the oracle's own decision,
//! [`bound_modules`], over a checked loop-free fixture whose compiled AIR they supply: the case's
//! module and a pipeline template, each of another shape under the fixture's own interface, or of
//! another interface.

use super::*;
use crate::check::loop_free::assemble;
use crate::check::CycleEvidence;

const ENTRY: &str = "kernel_vector_function_constant";

/// The label of the template row, which `air_of` tells from the case's own.
const TEMPLATE: &str = "template";

/// The loop-free public fixture every test checks.
fn fixture() -> SourceRow {
    let ll = include_str!("../../../fixtures/public/kernel_vector_function_constant.ll");
    SourceRow {
        air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        stage: "Kernel".into(),
        entry: ENTRY.into(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "public/kernel_vector_function_constant.ll".into(),
    }
}

fn checked<'a>(scratch: &crate::ScratchDir, source: &'a SourceRow) -> CheckedCase<&'a SourceRow> {
    let mut case = crate::case::vector_function_constant_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    case.case_id = case.computed_case_id().unwrap();
    let checked = crate::check::check_case_against_source(scratch.path(), case, source).unwrap();
    assert_eq!(checked.primary.loop_class, LoopClass::LoopFree);
    checked
}

/// `checked` with the same fixture as its one pipeline template, under the row label
/// [`TEMPLATE`].
fn with_template<'a>(
    scratch: &crate::ScratchDir,
    source: &'a SourceRow,
) -> CheckedCase<&'a SourceRow> {
    let mut case = checked(scratch, source);
    let template = checked(scratch, source).primary;
    case.pipeline_templates.insert(
        TEMPLATE.into(),
        CheckedShader {
            case: template.case,
            source: Arc::new(SourceRow {
                label: TEMPLATE.into(),
                ..source.clone()
            }),
            interface: template.interface,
            reflection: template.reflection,
            companion: template.companion,
            requirements: template.requirements,
            linked_functions: template.linked_functions,
            input_sha256: template.input_sha256,
            loop_class: template.loop_class,
            fuel_scope: template.fuel_scope,
            entry_projection: template.entry_projection,
        },
    );
    case
}

/// `text` assembled with `llvm-as`, or `None` (a visible skip) without it.
fn assembled(test: &str, text: &str) -> Option<Vec<u8>> {
    match assemble(text) {
        Ok(bitcode) => Some(bitcode),
        Err(why) if why.contains("llvm-as is unavailable") => {
            eprintln!("SKIP {test}: {why}");
            None
        }
        Err(why) => panic!("{why}"),
    }
}

/// A module whose definitions are `body`, under the fixture's target, declarations, and AIR
/// metadata: the fixture's interface, compiled to other code. Assembled with `llvm-as`.
fn bitcode(test: &str, body: &str) -> Option<Vec<u8>> {
    let fixture = fixture().air_ll;
    let (declarations, rest) = fixture.split_once("define ").unwrap();
    let metadata = &rest[rest.find("!air.kernel").unwrap()..];
    assembled(test, &format!("{declarations}{body}\n{metadata}"))
}

/// The compile step, supplying `case` for the case's own row and `template` for the template's,
/// both as compiled apart from the checked row.
fn compiling<'a>(
    case: &'a [u8],
    template: &'a [u8],
) -> impl Fn(&SourceRow) -> Result<ShaderAir, String> + 'a {
    move |row| {
        Ok(ShaderAir {
            bitcode: if row.label == TEMPLATE {
                template
            } else {
                case
            }
            .to_vec(),
            checked: false,
        })
    }
}

fn not_fuel_bounded(evidence: CycleEvidence) -> ExecutorError {
    ExecutorError::Refused(ExecutionRefused::NotFuelBounded(evidence))
}

fn looping_body() -> String {
    format!(
        "define void @{ENTRY}(ptr addrspace(1) %out) {{\n\
         entry:\n  br label %loop\n\
         loop:\n  %i = phi i32 [ 0, %entry ], [ %next, %loop ]\n  \
         store i32 %i, ptr addrspace(1) %out, align 4\n  \
         %next = add i32 %i, 1\n  %done = icmp eq i32 %next, 4\n  \
         br i1 %done, label %exit, label %loop\n\
         exit:\n  ret void\n}}\n"
    )
}

/// Two structures the fuel transform's own plan reads as loop-free (it treats a call to a
/// declaration as a leaf and follows no indirect call): a call to a function no executed module
/// defines (linked by the runtime; of the `__tensorops_impl_*` declarations, only a `matmul2d`
/// call whose constant descriptor bounds it is a leaf), and recursion through a function pointer.
/// As the case's own compiled module, each is refused, typed, although the checked row is
/// loop-free.
#[test]
fn a_compiled_case_module_the_checker_refuses_is_refused() {
    let test = "a_compiled_case_module_the_checker_refuses_is_refused";
    let Some(external) = bitcode(
        test,
        &format!(
            "declare void @ext(ptr addrspace(1))\n\
             define void @{ENTRY}(ptr addrspace(1) %out) {{\n  \
             call void @ext(ptr addrspace(1) %out)\n  ret void\n}}\n"
        ),
    ) else {
        return;
    };
    let Some(recursive) = bitcode(
        test,
        &format!(
            "define void @r(ptr %f) {{\n  call void %f(ptr %f)\n  ret void\n}}\n\
             define void @{ENTRY}(ptr addrspace(1) %out) {{\n  \
             call void @r(ptr @r)\n  ret void\n}}\n"
        ),
    ) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-refused").unwrap();
    let source = fixture();
    let checked = checked(&scratch, &source);
    for (air, evidence) in [
        (
            external,
            CycleEvidence::UnresolvedCall {
                function: ENTRY.into(),
                callee: "ext".into(),
            },
        ),
        (
            recursive,
            CycleEvidence::CallCycle {
                function: "r".into(),
            },
        ),
    ] {
        let refused = bound_modules(&checked, compiling(&air, &[])).err();
        assert_eq!(refused, Some(not_fuel_bounded(evidence)));
    }
}

/// A compiled case module calling MetalPerformancePrimitives' `matmul2d` ABI with a constant
/// descriptor runs as compiled: Metal supplies the bodies at pipeline creation, and the descriptor
/// fixes their work. The same call with a dynamic inner dimension stays refused.
#[test]
fn a_compiled_case_module_calling_matmul2d_with_a_constant_descriptor_runs() {
    let test = "a_compiled_case_module_calling_matmul2d_with_a_constant_descriptor_runs";
    let body = |k: i32| {
        format!(
            "@desc = internal constant {{ i32, i32, i32, i8, i8, i8, i32 }} \
             {{ i32 16, i32 16, i32 {k}, i8 0, i8 0, i8 0, i32 0 }}\n\
             declare void @__tensorops_impl_matmul2d_op_run_dv_f32_dv_f32_dv_f32(ptr, ptr, i32, \
             ptr, i32, ptr, i32, i32) section \"air.externally_defined\"\n\
             declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n\
             define void @{ENTRY}(ptr addrspace(1) %out) {{\n  \
             %d = alloca [20 x i8], align 4\n  %t = alloca [16 x i8], align 8\n  \
             call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr @desc, i64 20, i1 false)\n  \
             call void @__tensorops_impl_matmul2d_op_run_dv_f32_dv_f32_dv_f32(ptr %d, ptr %t, \
             i32 2, ptr %t, i32 2, ptr %t, i32 2, i32 32)\n  ret void\n}}\n"
        )
    };
    let (Some(fixed), Some(dynamic)) = (bitcode(test, &body(32)), bitcode(test, &body(-1))) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-matmul2d").unwrap();
    let source = fixture();
    let checked = checked(&scratch, &source);
    let Ok(modules) = bound_modules(&checked, compiling(&fixed, &[])) else {
        panic!("a matmul2d call with a constant descriptor is refused");
    };
    assert_eq!(modules.primary.bound, OracleBound::LoopFree);
    assert_eq!(
        bound_modules(&checked, compiling(&dynamic, &[])).err(),
        Some(not_fuel_bounded(CycleEvidence::UnresolvedCall {
            function: ENTRY.into(),
            callee: "__tensorops_impl_matmul2d_op_run_dv_f32_dv_f32_dv_f32".into(),
        }))
    );
}

/// A compiled case module with a loop fuel covers runs fuel-instrumented at the case's fuel,
/// although the checked row is loop-free.
#[test]
fn a_looping_compiled_case_module_runs_under_the_case_fuel() {
    let Some(looping) = bitcode(
        "a_looping_compiled_case_module_runs_under_the_case_fuel",
        &looping_body(),
    ) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-fuel").unwrap();
    let source = fixture();
    let mut checked = checked(&scratch, &source);
    checked.primary.case.fuel = Some(5);
    let Ok(modules) = bound_modules(&checked, compiling(&looping, &[])) else {
        panic!("a looping case module fuel covers is refused");
    };
    assert_eq!(
        modules.primary.bound,
        OracleBound::Fuel { invocation_fuel: 5 }
    );
    assert_eq!(&*modules.primary.air, &looping[..]);
}

/// Templates run as compiled: one whose compiled module loops is refused though the case's own
/// module is loop-free, and one whose compiled module is loop-free runs.
#[test]
fn a_looping_compiled_template_is_refused() {
    let test = "a_looping_compiled_template_is_refused";
    let Some(looping) = bitcode(test, &looping_body()) else {
        return;
    };
    let source = fixture();
    let Some(flat) = assembled(test, &source.air_ll) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-template").unwrap();
    let checked = with_template(&scratch, &source);
    let refused = bound_modules(&checked, compiling(&flat, &looping)).err();
    assert_eq!(
        refused,
        Some(not_fuel_bounded(CycleEvidence::CfgCycle {
            function: ENTRY.into()
        }))
    );
    let Ok(modules) = bound_modules(&checked, compiling(&flat, &flat)) else {
        panic!("a loop-free template is refused");
    };
    assert_eq!(modules.primary.bound, OracleBound::LoopFree);
    assert_eq!(modules.templates[TEMPLATE].air, flat);
}

/// Compiled bytes the classifier cannot decode are refused as undecoded, for the case's own
/// module and for a template.
#[test]
fn undecodable_compiled_bytes_are_refused() {
    let source = fixture();
    let Some(flat) = assembled("undecodable_compiled_bytes_are_refused", &source.air_ll) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-undecoded").unwrap();
    let garbage = b"not AIR bitcode";
    let refused = bound_modules(&checked(&scratch, &source), compiling(garbage, &[])).err();
    assert!(
        matches!(
            &refused,
            Some(ExecutorError::Refused(ExecutionRefused::NotFuelBounded(
                CycleEvidence::Undecoded { module, .. }
            ))) if module == &source.label
        ),
        "{refused:?}"
    );
    let with_template = with_template(&scratch, &source);
    let refused = bound_modules(&with_template, compiling(&flat, garbage)).err();
    assert!(
        matches!(
            &refused,
            Some(ExecutorError::Refused(ExecutionRefused::NotFuelBounded(
                CycleEvidence::Undecoded { module, .. }
            ))) if module == TEMPLATE
        ),
        "{refused:?}"
    );
}

/// The oracle binds by the checked row's interface, so a compiled module that binds otherwise
/// fails, as the case's module and as a template. One that differs only in what the oracle does not
/// bind by runs: this host's compile of the fixture's source declares its output `air.read_write`,
/// named, with its type's size, where the committed row declares `air.write`.
#[test]
fn a_compiled_module_that_binds_otherwise_fails() {
    let test = "a_compiled_module_that_binds_otherwise_fails";
    let source = fixture();
    let variant = |from: &str, to: &str| {
        let text = source.air_ll.replace(from, to);
        assert_ne!(text, source.air_ll);
        assembled(test, &text)
    };
    let Some(flat) = assembled(test, &source.air_ll) else {
        return;
    };
    let Some(moved) = variant(
        "!\"air.location_index\", i32 0",
        "!\"air.location_index\", i32 3",
    ) else {
        return;
    };
    let Some(described) = variant(
        "!\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_name\", !\"uint*\"",
        "!\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, \
         !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"output\"",
    ) else {
        return;
    };
    let scratch = crate::ScratchDir::new("oracle-modules-binding").unwrap();
    let checked = with_template(&scratch, &source);
    for (case, template, label) in [
        (&moved, &flat, source.label.as_str()),
        (&flat, &moved, TEMPLATE),
    ] {
        let failed = bound_modules(&checked, compiling(case, template)).err();
        assert!(
            matches!(
                &failed,
                Some(ExecutorError::Failed(why)) if why.starts_with(&format!(
                    "the module compiled from {label} does not bind as its checked row: \
                     parameter 0 is Buffer(BufferArg {{ location: Slot {{ index: Const(3)"
                ))
            ),
            "{failed:?}"
        );
    }
    let Ok(modules) = bound_modules(&checked, compiling(&described, &described)) else {
        panic!("a module that differs only in descriptive metadata is refused");
    };
    assert_eq!(modules.primary.bound, OracleBound::LoopFree);
}

/// The oracle's own compile step: this host compiles the fixture to a loop-free module, which
/// runs as compiled.
#[cfg(target_os = "macos")]
#[test]
fn the_host_compiled_fixture_runs_as_compiled() {
    let scratch = crate::ScratchDir::new("oracle-modules-host").unwrap();
    let source = fixture();
    let checked = checked(&scratch, &source);
    let modules = match bound_modules(&checked, platform::shader_air) {
        Ok(modules) => modules,
        Err(error) => panic!("the host-compiled fixture is not run: {error}"),
    };
    assert_eq!(modules.primary.bound, OracleBound::LoopFree);
    assert!(!modules.primary.air.is_empty());
}
