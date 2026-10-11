//! Engine support inventory over the public fixtures, one column per committed profile.
//!
//! Every `.ll` fixture at the top level of `validation/fixtures/public/` and in its `engine/`
//! directory is assembled to bitcode (`llvm-as`; a typed-pointer `.air.ll` through `xcrun
//! metal-as`), every entry point its ABI declares is compiled by the engine for every committed
//! profile under the default policy with no function constants and default lowering inputs, and
//! the outcome is recorded:
//!
//! - `ok`: compiled, linked, relocated, and passed L3 (requirements within the profile and
//!   `spirv-val` under the profile's environment; `G-val`);
//! - `Unsupported{feature; needs}`, `NeedsInput{input}`, `InvalidInput{kind}`,
//!   `LimitExceeded{limit}`: the typed refusal, without its input location.
//!
//! A fixture whose ABI does not decode has one row with entry `-`; a fixture the installed
//! assembler rejects has one row with entry `-` and `unassembled` in every column. Everything else
//! (an internal error, a transient budget or cancellation, or an `ok` artifact that fails L3) is a
//! translator bug and fails the test.
//!
//! The table must equal `public_fixture_expectations_engine.tsv` exactly. A change from `ok`
//! to a refusal needs a reviewed TSV edit whose commit states the reason. On a mismatch the test
//! prints the observed rows so a reviewed update can take them. Rows of a fixture whose assembler
//! is not installed (`metal-as` off macOS) are skipped visibly and taken from the expectations.
//!
//! The check is split so nextest runs it in parallel, and so the fixtures that need macOS are
//! apart from those that do not (one test took 136-353 s on the Mac): the table equals the
//! expectations exactly when its header and its fixtures' order are the expectations'
//! ([`inventory_lists_every_fixture_in_order`]) and each fixture's rows are its expected rows (the
//! `llvm_as` parts: the `llvm-as` fixtures; the `metal_as` parts: the typed-pointer `.air.ll`
//! ones, which skip visibly off macOS). Each part takes the fixtures whose position among its
//! assembler's fixtures is its number modulo the part count.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::device::DeviceProfile;
use crate::error::{Error, Needs};
use crate::pipeline::testing::{
    compile, engine_dir, fixtures_dir, link_and_validate, llvm_assemble, metal_assemble, meter,
    AssembleError,
};
use crate::spirv::testing::committed_profiles;

const EXPECTED: &str = include_str!("../../tests/public_fixture_expectations_engine.tsv");

/// `.ll` files directly in `dir`, sorted by name.
fn ll_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("read fixture directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "ll"))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

/// Every fixture as (inventory name, path).
fn fixtures() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for (prefix, dir) in [("", fixtures_dir()), ("engine/", engine_dir())] {
        for p in ll_files(&dir) {
            let name = p.file_name().unwrap().to_str().unwrap();
            out.push((format!("{prefix}{name}"), p));
        }
    }
    out
}

/// The bitcode of a fixture: typed-pointer `.air.ll` through `metal-as`, the rest through
/// `llvm-as`.
fn assemble(path: &Path) -> Result<Vec<u8>, AssembleError> {
    if is_metal_as(path) {
        metal_assemble(path)
    } else {
        llvm_assemble(path)
    }
}

/// The inventory cell of a refusal; `None` for an outcome that is a translator bug.
fn refusal(e: &Error) -> Option<String> {
    Some(match e {
        Error::Unsupported(u) => {
            let needs = match &u.needs {
                Needs::AnyOf(alts) => format!("AnyOf{alts:?}"),
                Needs::NotImplemented => "NotImplemented".to_owned(),
                Needs::NotRepresentable => "NotRepresentable".to_owned(),
                other => format!("{other:?}"),
            };
            format!("Unsupported{{{:?}; {needs}}}", u.feature)
        }
        Error::NeedsInput(i) => format!("NeedsInput{{{i:?}}}"),
        Error::InvalidInput(i) => format!("InvalidInput{{{:?}}}", i.kind),
        Error::LimitExceeded(l) => format!("LimitExceeded{{{l:?}}}"),
        _ => return None,
    })
}

/// The cell of `entry` of `bitcode` on profile `p`.
fn cell(bitcode: &[u8], fixture: &str, entry: &str, stem: &str, p: &DeviceProfile) -> String {
    match compile(bitcode, entry, p) {
        Ok(a) => {
            link_and_validate(&a, p)
                .unwrap_or_else(|why| panic!("{fixture} {entry} on {stem}: L3 failed: {why}"));
            "ok".to_owned()
        }
        Err(e) => refusal(&e)
            .unwrap_or_else(|| panic!("{fixture} {entry} on {stem}: translator bug: {e}")),
    }
}

/// The expected rows of each fixture, keyed by fixture name.
fn expected_rows() -> BTreeMap<&'static str, Vec<&'static str>> {
    let mut rows: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for line in EXPECTED.lines().skip(1) {
        let fixture = line.split('\t').next().unwrap_or_default();
        rows.entry(fixture).or_default().push(line);
    }
    rows
}

/// The inventory's header line: `fixture`, `entry`, and each committed profile's stem.
fn header(profiles: &[(String, DeviceProfile)]) -> String {
    let mut line = String::from("fixture\tentry");
    for (stem, _) in profiles {
        line.push('\t');
        line.push_str(stem);
    }
    line
}

/// The inventory rows of fixture `name` (each line with its newline), and whether they were taken
/// from the expectations because its assembler is not installed (a visible skip).
fn fixture_rows(
    name: &str,
    path: &Path,
    profiles: &[(String, DeviceProfile)],
    expected_rows: &BTreeMap<&str, Vec<&str>>,
) -> (String, bool) {
    let mut observed = String::new();
    let bitcode = match assemble(path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED {name}: {why}; rows taken from the expectations");
            for line in expected_rows.get(name).into_iter().flatten() {
                observed.push_str(line);
                observed.push('\n');
            }
            return (observed, true);
        }
        Err(AssembleError::Rejected(why)) => {
            eprintln!("{name}: assembler rejected the fixture: {why}");
            observed.push_str(&format!("{name}\t-"));
            for _ in profiles {
                observed.push_str("\tunassembled");
            }
            observed.push('\n');
            return (observed, false);
        }
    };
    let entries = match crate::air::decode(&bitcode, &mut meter()).map_err(Error::from) {
        Ok(d) => d
            .abi
            .entries
            .iter()
            .map(|e| e.name.to_string())
            .collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("{name}: decode: {e}");
            let c = refusal(&e).unwrap_or_else(|| panic!("{name}: decode: translator bug: {e}"));
            observed.push_str(&format!("{name}\t-"));
            for _ in profiles {
                observed.push('\t');
                observed.push_str(&c);
            }
            observed.push('\n');
            return (observed, false);
        }
    };
    assert!(!entries.is_empty(), "{name}: no entry points");
    for entry in entries {
        observed.push_str(&format!("{name}\t{entry}"));
        for (stem, p) in profiles {
            observed.push('\t');
            observed.push_str(&cell(&bitcode, name, &entry, stem, p));
        }
        observed.push('\n');
    }
    (observed, false)
}

/// The expectations' header and the order of their fixtures are the inventory's: its first line
/// is [`header`], and its rows are one block per fixture, in [`fixtures`]' order. With each
/// fixture's rows checked by a part, the observed table equals the expectations exactly.
#[test]
fn inventory_lists_every_fixture_in_order() {
    let profiles = committed_profiles().expect("committed profiles");
    let mut lines = EXPECTED.lines();
    assert_eq!(
        lines.next(),
        Some(header(&profiles).as_str()),
        "the header of tests/public_fixture_expectations_engine.tsv is not the committed \
         profiles' stems"
    );
    let mut blocks: Vec<&str> = Vec::new();
    for line in lines {
        let fixture = line.split('\t').next().unwrap_or_default();
        if blocks.last() != Some(&fixture) {
            assert!(
                !blocks.contains(&fixture),
                "{fixture}: its rows in tests/public_fixture_expectations_engine.tsv are not \
                 one block"
            );
            blocks.push(fixture);
        }
    }
    let names = fixtures().into_iter().map(|(n, _)| n).collect::<Vec<_>>();
    if blocks != names {
        let first = blocks
            .iter()
            .zip(&names)
            .position(|(b, n)| b != n)
            .unwrap_or(blocks.len().min(names.len()));
        panic!(
            "the fixtures of tests/public_fixture_expectations_engine.tsv are not the public \
             fixtures in order ({} listed vs {} fixtures); first difference at fixture {}: \
             expected {:?}, found {:?}",
            blocks.len(),
            names.len(),
            first + 1,
            blocks.get(first),
            names.get(first)
        );
    }
}

/// Part `part` of `of` of the fixtures `metal_as` selects (the `.air.ll` ones, or the others):
/// each fixture's observed rows must be its expected rows.
fn check_part(metal_as: bool, part: usize, of: usize) {
    let profiles = committed_profiles().expect("committed profiles");
    let expected_rows = expected_rows();
    let mut skipped = 0usize;
    let mut observed = String::new();
    let mut expected = String::new();
    let selected = fixtures()
        .into_iter()
        .filter(|(_, path)| is_metal_as(path) == metal_as)
        .enumerate()
        .filter(|(i, _)| i % of == part)
        .map(|(_, f)| f);
    for (name, path) in selected {
        let (rows, skip) = fixture_rows(&name, &path, &profiles, &expected_rows);
        skipped += usize::from(skip);
        observed.push_str(&rows);
        for line in expected_rows.get(name.as_str()).into_iter().flatten() {
            expected.push_str(line);
            expected.push('\n');
        }
    }
    if skipped > 0 {
        eprintln!("SKIPPED {skipped} fixture(s) whose assembler is not installed");
    }
    if observed != expected {
        eprintln!("---- observed engine inventory rows ----\n{observed}---- end ----");
        let exp: Vec<&str> = expected.lines().collect();
        let obs: Vec<&str> = observed.lines().collect();
        for (i, line) in obs.iter().enumerate() {
            if exp.get(i) != Some(line) {
                eprintln!("first difference at row {} of this part:", i + 1);
                eprintln!("  expected: {:?}", exp.get(i));
                eprintln!("  observed: {line:?}");
                break;
            }
        }
        panic!(
            "the engine inventory changed ({} observed vs {} expected rows in this part); \
             review each changed row and update tests/public_fixture_expectations_engine.tsv",
            obs.len(),
            exp.len()
        );
    }
}

/// Whether `path` is a typed-pointer `.air.ll` fixture, which `metal-as` assembles ([`assemble`]).
fn is_metal_as(path: &Path) -> bool {
    path.to_str().is_some_and(|s| s.ends_with(".air.ll"))
}

/// One test per part, so nextest runs the parts at once.
macro_rules! parts {
    ($metal_as:expr, $of:expr; $($test:ident = $part:expr),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                super::check_part($metal_as, $part, $of);
            }
        )*
    };
}

/// The fixtures `llvm-as` assembles, on every host.
mod llvm_as {
    parts!(false, 8; part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// The typed-pointer `.air.ll` fixtures `metal-as` assembles: checked on macOS, skipped visibly
/// elsewhere.
mod metal_as {
    parts!(true, 4; part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}
