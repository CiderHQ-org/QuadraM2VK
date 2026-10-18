//! Encoders made from reflected buffer bindings (`newArgumentEncoderWithBufferBinding:`,
//! [`ArgumentBufferLayout::from_buffer_binding`]) against Metal's, beside the function's own
//! (`newArgumentEncoderWithBufferIndex:`): `metal-answers.txt`'s `encoder_` sections over the
//! `argument_buffers` and `buffers` fixtures and its `enc:` sections over the `encoders` and
//! `encoder_types` fixtures, recorded by `probe.swift` on an M3 Max and an M4 (where the two
//! differ, the M3 Max's).

use std::collections::{BTreeMap, BTreeSet};

use super::{diff, fixture};
use crate::bind::{ArgumentBufferLayout, ArgumentLayoutError, ArgumentType};
use crate::budget::Budget;
use crate::error::{DiagName, Error, Feature};
use crate::library::{FunctionConstantValues, Library};
use crate::metal::{BindingAccess, DataType};
use crate::pipeline::testing::{fixtures_dir, AssembleError};
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
use crate::reflection::{
    Binding, BufferBinding, PointerType, StructMember, StructType, TypeDetail,
};

const FUNCTION: &str = "reflection_argument_buffers";

/// The recorded `encoder_<case>` sections, by case.
fn encoder_answers() -> BTreeMap<String, Vec<String>> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut case: Option<&str> = None;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            case = name.strip_prefix("encoder_");
            continue;
        }
        if let Some(name) = case {
            out.entry(name.to_owned())
                .or_default()
                .push(line.to_owned());
        }
    }
    out
}

/// The fixture's library binding and pipeline binding named `name`; `None` without `llvm-as`.
fn bindings(name: &str) -> Option<(Binding, Binding)> {
    let lib = Library::from_air(&fixture("argument_buffers")?).expect("library");
    let f = lib.function(FUNCTION).expect("function");
    let pick = |bindings: &[Binding]| {
        bindings
            .iter()
            .find(|b| b.name() == name)
            .unwrap_or_else(|| panic!("no binding {name}"))
            .clone()
    };
    let budget = Budget::standard();
    let library = f.reflection(&budget).expect("reflection").expect("present");
    let pipeline =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f.clone())))
            .reflection(&budget)
            .expect("pipeline reflection");
    Some((pick(library.bindings()), pick(pipeline.bindings())))
}

/// The function's encoder of its buffer `index` and the encoders of its library's and its
/// pipeline's binding `name`, labelled as the probe labels them.
fn encoders(index: u32, name: &str) -> Option<Vec<(&'static str, ArgumentBufferLayout)>> {
    let lib = Library::from_air(&fixture("argument_buffers")?).expect("library");
    let function = lib
        .function(FUNCTION)
        .expect("function")
        .argument_buffer_layout(
            index,
            &FunctionConstantValues::defaults(),
            &Budget::standard(),
        )
        .expect("the function's layout");
    let (library, pipeline) = bindings(name)?;
    let from = |b: &Binding| {
        ArgumentBufferLayout::from_buffer_binding(b.buffer().expect("a buffer")).expect("a layout")
    };
    Some(vec![
        ("function", function),
        ("library", from(&library)),
        ("pipeline", from(&pipeline)),
    ])
}

/// The probe's lines for an encoder: its length and alignment, the offset `constantDataAtIndex:`
/// answers for each of `constants` (the argument's offset, 0 for an index no argument has), and
/// each of `nested`'s nested encoder or nil.
fn encoder_lines(
    label: &str,
    l: &ArgumentBufferLayout,
    constants: &[u32],
    nested: &[u32],
) -> Vec<String> {
    let mut out = vec![format!(
        "{label} encodedLength={} alignment={}",
        l.encoded_length, l.alignment
    )];
    for &i in constants {
        let offset = l.argument(i).map_or(0, |p| p.offset);
        out.push(format!("  constant index={i} offset={offset}"));
    }
    for &i in nested {
        out.push(match l.argument(i).and_then(|p| p.nested) {
            Some(n) => format!(
                "  nested index={i} encodedLength={} alignment={}",
                n.encoded_length, n.alignment
            ),
            None => format!("  nested index={i} nil"),
        });
    }
    out
}

/// An encoder from a library's binding equals the function's own; one from a pipeline's binding
/// is shorter (its arrays of resources and of resource-only structs reflect stride 0), with the same constants and
/// nested encoders: lengths, alignments, constant offsets and nested encoders as Metal's.
#[test]
fn binding_encoders_match_metal() {
    let answers = encoder_answers();
    let mut mismatches = Vec::new();
    for (case, index, name, constants, nested) in [
        ("args", 0, "args", &[7, 8, 20, 21][..], &[2, 12, 15][..]),
        ("imp", 1, "imp", &[3][..], &[0][..]),
    ] {
        let Some(encoders) = encoders(index, name) else {
            return;
        };
        let got: Vec<String> = encoders
            .iter()
            .flat_map(|(label, l)| encoder_lines(label, l, constants, nested))
            .collect();
        mismatches.extend(diff(&format!("encoder_{case}"), &got, &answers[case]));
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// The arguments the probe sets, in its order: textures, then samplers, then buffers.
const PLACES: [(ArgumentType, &[u32]); 3] = [
    (ArgumentType::Texture, &[0, 3, 4, 5, 22, 24, 26]),
    (ArgumentType::Sampler, &[1, 16, 17, 23, 25, 27]),
    (ArgumentType::Buffer, &[2, 18, 19]),
];

/// Where each argument's value is found in a zeroed 512-byte argument buffer after the probe's
/// writes, each argument a distinct value at its place (a later write over an earlier one wins),
/// searched as the probe searches: 8 bytes at every multiple of 4.
fn placements(l: &ArgumentBufferLayout) -> BTreeMap<u32, BTreeSet<u64>> {
    let mut bytes = [0u8; 512];
    let mut values = Vec::new();
    for (ty, indices) in PLACES {
        for &i in indices {
            let place = l.argument(i).unwrap_or_else(|| panic!("no argument {i}"));
            assert_eq!(place.ty, ty, "argument {i}");
            let value = 0x1000 + u64::from(i);
            let at = place.offset as usize;
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
            values.push((i, value));
        }
    }
    values
        .into_iter()
        .map(|(i, value)| {
            let found = (0..512 - 7)
                .step_by(4)
                .filter(|&at| bytes[at..at + 8] == value.to_le_bytes())
                .map(|at| at as u64)
                .collect();
            (i, found)
        })
        .collect()
}

/// Every element and struct member lands where Metal's encoders put it: the function's and the
/// library binding's at their layout's places, the pipeline binding's every array element at
/// element 0's (the last written wins). Metal's lists can also hold another argument's place
/// (a texture's and a sampler's resource IDs can be equal), so each list must be this crate's
/// place for the argument plus only places of other arguments.
#[test]
fn binding_encoders_place_arguments_as_metal() {
    let answers = encoder_answers();
    let Some(encoders) = encoders(0, "args") else {
        return;
    };
    let mut recorded: BTreeMap<&str, (String, BTreeMap<u32, BTreeSet<u64>>)> = BTreeMap::new();
    let mut label = "";
    for line in &answers["args_places"] {
        if let Some(rest) = line.strip_prefix("  index=") {
            let (index, list) = rest.split_once(" offsets=").expect("an offsets list");
            let offsets = list
                .trim_matches(|c| c == '[' || c == ']')
                .split(", ")
                .filter(|o| !o.is_empty())
                .map(|o| o.parse().expect("an offset"))
                .collect();
            let index = index.parse().expect("an index");
            recorded.get_mut(label).unwrap().1.insert(index, offsets);
        } else {
            label = ["function", "library", "pipeline"]
                .into_iter()
                .find(|l| line.starts_with(&format!("{l} ")))
                .expect("an encoder line");
            recorded.insert(label, (line.clone(), BTreeMap::new()));
        }
    }
    for (label, l) in &encoders {
        let (line, metal) = &recorded[label];
        assert_eq!(line, &format!("{label} encodedLength={}", l.encoded_length));
        let ours = placements(l);
        assert_eq!(
            ours.keys().collect::<Vec<_>>(),
            metal.keys().collect::<Vec<_>>()
        );
        for (i, found) in &ours {
            let others: BTreeSet<u64> = ours
                .iter()
                .filter(|(j, _)| *j != i)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            assert!(
                found.is_subset(&metal[i])
                    && metal[i].difference(found).all(|o| others.contains(o)),
                "{label} index {i}: ours {found:?}, Metal's {:?}",
                metal[i]
            );
        }
    }
    // The pipeline binding's arrays overlap: texs, smps, ptrs and pairs keep their last element.
    let pipeline = placements(&encoders[2].1);
    for (lost, kept) in [(3, 5), (4, 5), (16, 17), (18, 19), (24, 26), (25, 27)] {
        assert!(pipeline[&lost].is_empty(), "{lost}");
        assert!(!pipeline[&kept].is_empty(), "{kept}");
    }
}

/// A plain buffer's binding, the library's or the pipeline's, of a pointee that is no struct makes
/// Metal's encoder of no bytes, alignment 1, which API validation lets through, while the
/// function's encoder at that buffer's index is refused, as validation stops it (Metal without
/// validation answers the same empty encoder there).
#[test]
fn a_plain_buffer_binding_makes_an_empty_encoder() {
    let answers = encoder_answers();
    for case in [
        "plain_buffer",
        "plain_buffer_pipeline",
        "plain_buffer_function",
    ] {
        assert_eq!(
            answers[case],
            ["plain encodedLength=0 alignment=1"],
            "{case}"
        );
    }
    let Some((library, pipeline)) = bindings("out") else {
        return;
    };
    for b in [&library, &pipeline] {
        let l = ArgumentBufferLayout::from_buffer_binding(b.buffer().expect("a buffer"))
            .expect("an empty layout");
        assert_eq!(
            encoder_lines("plain", &l, &[], &[]),
            ["plain encodedLength=0 alignment=1"]
        );
        assert!(l.members.is_empty());
    }
    let Some(lib) = fixture("argument_buffers").map(|bc| Library::from_air(&bc).expect("library"))
    else {
        return;
    };
    assert_eq!(
        lib.function(FUNCTION)
            .expect("function")
            .argument_buffer_layout(2, &FunctionConstantValues::defaults(), &Budget::standard()),
        Err(ArgumentLayoutError::NoArgumentBuffer { index: 2 })
    );
}

/// A plain buffer whose pointee is a struct makes the encoder Metal lays out by the plain
/// buffer's rule: each constant 8 bytes aligned to 8 at its member's offset, and index 0 the last
/// member's (`encoder_plain_struct`: the buffers fixture's `plain`, `nested` and `counter`).
#[test]
fn a_plain_struct_buffer_binding_makes_metals_encoder() {
    let answers = encoder_answers();
    let Some(lib) = fixture("buffers").map(|bc| Library::from_air(&bc).expect("library")) else {
        return;
    };
    let reflection = lib
        .function("reflection_buffers")
        .expect("function")
        .reflection(&Budget::standard())
        .expect("reflection")
        .expect("present");
    let got: Vec<String> = ["plain", "nested", "counter"]
        .into_iter()
        .flat_map(|name| {
            let b = reflection
                .bindings()
                .iter()
                .find(|b| b.name() == name)
                .and_then(Binding::buffer)
                .unwrap_or_else(|| panic!("no buffer binding {name}"));
            let l = ArgumentBufferLayout::from_buffer_binding(b)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            encoder_lines(name, &l, &[0, 1, 2], &[])
        })
        .collect();
    let mismatches = diff("encoder_plain_struct", &got, &answers["plain_struct"]);
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// Metal's `enc:<fixture>:<kernel>` sections, in order: the fixture, the kernel and the lines.
fn binding_encoder_answers() -> Vec<(String, String, Vec<String>)> {
    let path = fixtures_dir().join("reflection/metal-answers.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut out: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut open = false;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if let Some(name) = line.strip_prefix("== ") {
            open = match name.strip_prefix("enc:").and_then(|n| n.split_once(':')) {
                Some((fixture, kernel)) => {
                    out.push((fixture.to_owned(), kernel.to_owned(), Vec::new()));
                    true
                }
                None => false,
            };
        } else if open {
            out.last_mut().expect("a section").2.push(line.to_owned());
        }
    }
    out
}

/// The probe's lines for the encoder of buffer binding `name` under `label`: its length and
/// alignment and, for one of any bytes, the offset `constantDataAtIndex:` answers for indices 0
/// to 7 (the argument's offset, 0 for an index no argument has).
fn binding_encoder_lines(label: &str, name: &str, l: &ArgumentBufferLayout) -> Vec<String> {
    let mut out = vec![format!(
        "{label} {name} encodedLength={} alignment={}",
        l.encoded_length, l.alignment
    )];
    if l.encoded_length > 0 {
        let offsets: Vec<String> = (0..8)
            .map(|i| format!("{i}:{}", l.argument(i).map_or(0, |p| p.offset)))
            .collect();
        out.push(format!("  constants {}", offsets.join(" ")));
    }
    out
}

/// Every buffer binding of the encoders, encoder_types and encoder_arrays fixtures' kernels, the
/// library's and the pipeline's, makes Metal's encoder: plain buffers of a struct of each scalar,
/// vector, packed vector, matrix and atomic type, of arrays, of members in both orders and of
/// nested structs and arrays of them (each constant 8 bytes aligned to 8); argument buffers whose
/// last member is a constant of each type (its MSL size and alignment deciding the length), a
/// plain-data struct held by value and an array of them, explicit ids with a gap, and arrays of
/// structs mixing resources and constants (which a pipeline's reflection keeps at their element
/// size); and a plain buffer of a vector (no bytes). Lengths, alignments and the constants'
/// offsets, line for line.
#[test]
fn every_binding_encoder_matches_metal() {
    let answers = binding_encoder_answers();
    assert_eq!(
        answers.len(),
        9,
        "the encoders, encoder_types and encoder_arrays kernels"
    );
    let budget = Budget::standard();
    let mut libraries: BTreeMap<String, Library> = BTreeMap::new();
    let mut mismatches = Vec::new();
    let mut compared = 0;
    for (stem, kernel, want) in &answers {
        if !libraries.contains_key(stem) {
            let Some(bc) = fixture(stem) else {
                return;
            };
            let lib = Library::from_air(&bc).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            libraries.insert(stem.clone(), lib);
        }
        let f = libraries[stem]
            .function(kernel)
            .unwrap_or_else(|e| panic!("{stem} {kernel}: {e:?}"));
        let library = f
            .reflection(&budget)
            .unwrap_or_else(|e| panic!("{kernel}: {e:?}"))
            .expect("present");
        let pipeline =
            ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)))
                .reflection(&budget)
                .unwrap_or_else(|e| panic!("{kernel} pipeline: {e:?}"));
        let mut got = Vec::new();
        for (label, bindings) in [
            ("library", library.bindings()),
            ("pipeline", pipeline.bindings()),
        ] {
            for b in bindings {
                let Some(buffer) = b.buffer() else {
                    continue;
                };
                let l = ArgumentBufferLayout::from_buffer_binding(buffer)
                    .unwrap_or_else(|e| panic!("{kernel} {label} {}: {e:?}", b.name()));
                got.extend(binding_encoder_lines(label, b.name(), &l));
                compared += 1;
            }
        }
        mismatches.extend(diff(&format!("enc:{stem}:{kernel}"), &got, want));
    }
    // encoders_plain's 25 bindings, encoders_args' 11, each encoder_types_plain kernel's 22,
    // each encoder_types_tail kernel's 23 and encoder_arrays' 11, the library's and the
    // pipeline's.
    assert_eq!(compared, 2 * (25 + 11 + 3 * 22 + 3 * 23 + 11));
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

/// A plain buffer's struct member that is no constant (here a pointer) has an encoder Metal's
/// answer was not measured for: refused, typed, naming the member.
#[test]
fn a_plain_buffer_pointer_member_is_refused() {
    let pointer = PointerType::new(
        DataType::Float,
        BindingAccess::ReadOnly,
        4,
        4,
        false,
        TypeDetail::Plain,
    );
    let st = StructType::new(vec![StructMember::new(
        "p".to_owned(),
        0,
        DataType::Pointer,
        0,
        TypeDetail::Pointer(Box::new(pointer)),
    )]);
    let binding = BufferBinding::new(8, 8, DataType::Struct, Some(st), None);
    match ArgumentBufferLayout::from_buffer_binding(&binding) {
        Err(ArgumentLayoutError::Compile(Error::Unsupported(u))) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("p")));
        }
        other => panic!("{other:?}"),
    }
}

/// An argument buffer's member the AIR gives no argument index (a plain member of its struct
/// type info), which Apple's compiler emitted in no measured shape: reflection refuses it,
/// naming the member's type, rather than guess what Metal answers.
#[test]
fn an_argument_buffer_member_without_an_index_is_refused() {
    let path = fixtures_dir().join("reflection/encoders.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let from = r#"i32 8, i32 4, i32 0, !"float", !"f", !"air.indirect_argument", !68}"#;
    assert_eq!(text.matches(from).count(), 1);
    let text = text.replace(from, r#"i32 8, i32 4, i32 0, !"float", !"f"}"#);
    let bc = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("encoders_args").expect("function");
    match f.reflection(&Budget::standard()) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("float")));
        }
        other => panic!("{other:?}"),
    }
}

/// A pipeline's stride of an argument buffer's array of structs whose members are no constant
/// but hold a struct is unmeasured (encoder_arrays' `a7` holds a constant beside its struct): its
/// pipeline's reflection is refused, naming the struct, while its library's states the element
/// size.
#[test]
fn a_pipeline_array_of_structs_of_structs_is_refused() {
    let path = fixtures_dir().join("reflection/encoder_arrays.ll");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let from = r#"!"ArrTex", !"in", !"air.indirect_argument", i32 0, i32 8, i32 4, i32 0, !"float", !"c", !"air.indirect_argument", !42}"#;
    assert_eq!(text.matches(from).count(), 1);
    let text = text.replace(
        from,
        r#"!"ArrTex", !"in", !"air.indirect_argument", i32 0}"#,
    );
    let bc = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
    };
    let lib = Library::from_air(&bc).expect("library");
    let f = lib.function("encoder_arrays").expect("function");
    let budget = Budget::standard();
    let library = f.reflection(&budget).expect("reflection").expect("present");
    let a7 = library
        .bindings()
        .iter()
        .find(|b| b.name() == "a7")
        .and_then(Binding::buffer)
        .and_then(|b| b.buffer_struct_type())
        .expect("a7's struct");
    let array = a7.members()[0].array_type().expect("an array");
    assert_eq!((array.array_length(), array.stride()), (2, 16));
    match ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(f)))
        .reflection(&budget)
    {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Reflection);
            assert_eq!(u.symbol, Some(DiagName::new("ArrNestTex")));
        }
        other => panic!("{other:?}"),
    }
}
