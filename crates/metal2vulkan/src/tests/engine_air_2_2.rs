//! AIR 2.2's vertex inputs translate as today's.
//!
//! macOS 13's compiler (AIR 2.2) follows each `air.vertex_input`'s location with its attribute's
//! user key, `generated(__air_placeholder__)` for an attribute it does not name:
//! `!{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1,
//! !"generated(__air_placeholder__)", !"air.arg_type_name", ...}`. Today's compiler writes no key
//! there. A vertex input binds by its location alone, so the two spellings are the same program.
//!
//! Each public `.ll` fixture with a vertex input is respelled here in AIR 2.2's form, the key
//! inserted after every vertex input's location. Both spellings compile, entry by entry, on every
//! committed profile to the same SPIR-V words (or the same refusal).
//!
//! `llvm-as` assembles the text; without it the test skips visibly.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::device::DeviceProfile;
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_standalone, llvm_assemble, meter, AssembleError,
};
use crate::spirv::testing::committed_profiles;

/// The key AIR 2.2 writes after a vertex input's location for an attribute it does not name.
const PLACEHOLDER: &str = "!\"generated(__air_placeholder__)\"";

/// A vertex input's role and location key, as they open its attributes.
const LOCATION: &str = "!\"air.vertex_input\", !\"air.location_index\", ";

/// `line` with [`PLACEHOLDER`] after the location's two operands of each vertex input it holds,
/// and how many it held.
fn respell_line(line: &str) -> (String, usize) {
    let mut out = String::with_capacity(line.len() + 64);
    let mut rest = line;
    let mut n = 0;
    while let Some(at) = rest.find(LOCATION) {
        let operands = at + LOCATION.len();
        // `i32 <index>, i32 <count>`: the second comma after the key ends them.
        let first = operands + rest[operands..].find(", ").expect("the location's index");
        let second = first + 2 + rest[first + 2..].find([',', '}']).expect("its count");
        out.push_str(&rest[..second]);
        out.push_str(", ");
        out.push_str(PLACEHOLDER);
        rest = &rest[second..];
        n += 1;
    }
    out.push_str(rest);
    (out, n)
}

/// `text` in AIR 2.2's vertex-input form, and how many vertex inputs it holds.
fn respell(text: &str) -> (String, usize) {
    let mut out = String::with_capacity(text.len() + 1024);
    let mut n = 0;
    for line in text.lines() {
        let (line, k) = respell_line(line);
        n += k;
        out.push_str(&line);
        out.push('\n');
    }
    (out, n)
}

/// Bitcode of `text` through `llvm-as`.
fn assemble(text: &str) -> Result<Vec<u8>, AssembleError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "m2v_air_2_2_{}_{}.ll",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, text).expect("write the respelled text");
    let out = llvm_assemble(&path);
    let _ = std::fs::remove_file(&path);
    out
}

/// Every public `.ll` fixture (not the typed `.air.ll` companions).
fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for dir in [fixtures_dir(), engine_dir()] {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("fixture directory")
            .map(|e| e.expect("entry").path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                name.ends_with(".ll") && !name.ends_with(".air.ll")
            })
            .collect();
        files.sort();
        out.extend(files);
    }
    out
}

/// What entry `entry` of `bitcode` compiles to on `p`: its standalone words, or its refusal.
fn outcome(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> Result<Vec<u32>, String> {
    let a = compile(bitcode, entry, p).map_err(|e| e.to_string())?;
    link_standalone(&a, p)
        .map(|(_, w)| w)
        .map_err(|e| format!("link: {e}"))
}

#[test]
fn respelling_inserts_the_key_after_each_location() {
    let line = "!14 = !{i32 0, !\"air.vertex_input\", !\"air.location_index\", i32 0, i32 1, \
                !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"a\"}";
    assert_eq!(
        respell_line(line),
        (
            "!14 = !{i32 0, !\"air.vertex_input\", !\"air.location_index\", i32 0, i32 1, \
             !\"generated(__air_placeholder__)\", !\"air.arg_type_name\", !\"float2\", \
             !\"air.arg_name\", !\"a\"}"
                .to_owned(),
            1
        )
    );
    let gated = "!15 = !{i32 1, !\"air.function_constant\", !16, !\"air.vertex_input\", \
                 !\"air.location_index\", i32 1, i32 1}";
    assert_eq!(
        respell_line(gated),
        (
            "!15 = !{i32 1, !\"air.function_constant\", !16, !\"air.vertex_input\", \
             !\"air.location_index\", i32 1, i32 1, !\"generated(__air_placeholder__)\"}"
                .to_owned(),
            1
        )
    );
    let buffer = "!16 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1}";
    assert_eq!(respell_line(buffer), (buffer.to_owned(), 0));
}

#[test]
fn air_2_2_vertex_inputs_compile_to_the_same_words() {
    let profiles = committed_profiles().expect("committed profiles");
    let (mut respelled_fixtures, mut inputs, mut compared, mut compiled) = (0, 0, 0, 0);
    for path in fixtures() {
        let name = path.display().to_string();
        let text = std::fs::read_to_string(&path).expect("fixture text");
        let (respelled, n) = respell(&text);
        if n == 0 {
            continue;
        }
        let old = match llvm_assemble(Path::new(&path)) {
            Ok(b) => b,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED air_2_2_vertex_inputs_compile_to_the_same_words: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("{name}: llvm-as: {why}"),
        };
        let new = assemble(&respelled).unwrap_or_else(|e| panic!("{name} respelled: {e:?}"));
        let entries: Vec<String> = crate::air::decode(&old, &mut meter())
            .unwrap_or_else(|e| panic!("{name}: decode: {e}"))
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect();
        respelled_fixtures += 1;
        inputs += n;
        for entry in &entries {
            for (stem, p) in &profiles {
                let a = outcome(&old, entry, p);
                let b = outcome(&new, entry, p);
                compiled += usize::from(a.is_ok());
                assert_eq!(a, b, "{name} {entry} on {stem}");
                compared += 1;
            }
        }
    }
    eprintln!(
        "{respelled_fixtures} fixtures with {inputs} vertex inputs, {compared} cells compared, \
         {compiled} compiled"
    );
    assert!(
        respelled_fixtures >= 8,
        "only {respelled_fixtures} fixtures have a vertex input"
    );
    assert!(compiled > 0, "no respelled entry compiled");
}
