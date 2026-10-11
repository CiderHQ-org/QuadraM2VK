//! The reference list and its identities, and the case facts the index keeps, over synthetic
//! inputs; and the committed list.

use super::testing::{identity, primary};
use super::*;

/// An environment's ID is produced from its identity, so identical hosts produce one ID.
#[test]
fn an_environment_id_is_produced_from_its_identity() {
    assert_eq!(
        identity("Apple M3 Max", "26.7").environment_id(),
        "apple-m3-max-macos26.7"
    );
    assert_eq!(
        identity("Apple M4", "27.0.1").environment_id(),
        "apple-m4-macos27.0.1"
    );
    let mut references = References { primary: primary() };
    references.check().unwrap();
    references.primary.environment_id = "my-mac".into();
    assert!(references
        .check()
        .unwrap_err()
        .contains("its identity produces the ID"));
}

/// A row is held to the members it records, or to them all when strict (a host that records), and
/// every row records the device and the OS.
#[test]
fn a_row_is_held_to_the_identity_it_records() {
    let declared = identity("Apple M3 Max", "26.7");
    let full = Value::Object(declared.record());
    assert!(declared.differences(&full, true).is_empty());
    // A legacy row: the device, the OS, and the architecture, nothing of the toolchain.
    let legacy = serde_json::json!({
        "device": "Apple M3 Max", "os": "26.7", "architecture": "aarch64"
    });
    assert!(declared.differences(&legacy, false).is_empty());
    assert_eq!(declared.differences(&legacy, true).len(), 4);
    // An OS or Xcode update shows in the recorded members.
    let mut updated = declared.record();
    updated.insert("os_build".into(), "25H1".into());
    let differences = declared.differences(&Value::Object(updated), false);
    assert_eq!(differences, ["os_build: \"25A1\" (\"25H1\")"]);
    // The device and the OS are always required.
    assert_eq!(
        declared.differences(&serde_json::json!({}), false),
        [
            "device: \"Apple M3 Max\" (unrecorded)",
            "os: \"26.7\" (unrecorded)"
        ]
    );
}

/// The committed reference list parses, and its reference is the Apple M3 Max.
#[test]
fn the_committed_reference_list_parses() {
    let references = References::load(&References::committed_path()).unwrap();
    assert_eq!(references.primary.environment_id, "apple-m3-max-macos26.7");
    assert_eq!(references.primary.identity.device, "Apple M3 Max");
}

/// The oracle's identity probes read what the host tools print: the AGX accelerator's GPU core
/// count, the selected Xcode's build, and the Metal toolchain's first line.
#[test]
fn the_identity_probes_read_what_the_host_tools_print() {
    let ioreg = r#"+-o AGXAcceleratorG15X  <class AGXAcceleratorG15X, id 0x10000043f, registered>
    {
      "IOMatchedAtBoot" = Yes
      "gpu-core-count" = 40
      "MetalPluginName" = "AGXMetalG15X_M1"
    }"#;
    assert_eq!(parse_gpu_cores(ioreg), Ok(40));
    assert!(parse_gpu_cores("{}").unwrap_err().contains("no AGX"));
    let twice = format!("{ioreg}\n      \"gpu-core-count\" = 10\n");
    assert!(parse_gpu_cores(&twice).unwrap_err().contains("several"));
    assert!(parse_gpu_cores("  \"gpu-core-count\" = <28000000>")
        .unwrap_err()
        .contains("states gpu-core-count as"));

    let plist = r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
	<key>CFBundleShortVersionString</key>
	<string>27.0</string>
	<key>ProductBuildVersion</key>
	<string>27A266a</string>
</dict>
</plist>"#;
    assert_eq!(parse_xcode_build(plist), Ok("27A266a".into()));
    assert!(parse_xcode_build("<dict></dict>").is_err());

    let version = "Apple metal version 32023.921 (metalfe-32023.921.6)\n\
                   Target: air64-apple-darwin25.6.0\nThread model: posix\n";
    assert_eq!(
        parse_metal_toolchain(version),
        Ok("Apple metal version 32023.921 (metalfe-32023.921.6)".into())
    );
    assert!(parse_metal_toolchain("xcrun: error: unable to find utility").is_err());
}

/// A row records the identity the oracle read, and reads back as it.
#[test]
fn a_recorded_identity_reads_back_only_when_whole() {
    let declared = identity("Apple M4", "27.0.1");
    let mut environment = declared.record();
    environment.insert("architecture".into(), "aarch64".into());
    assert_eq!(
        MetalIdentity::recorded(&Value::Object(environment.clone())),
        Some(declared)
    );
    environment.remove("xcode_build");
    assert_eq!(MetalIdentity::recorded(&Value::Object(environment)), None);
}

/// The oracle records under a declared reference environment only on a host of its identity;
/// an environment the list does not name records on any host.
#[test]
fn a_reference_environment_records_only_on_a_host_of_its_identity() {
    let references = super::testing::references();
    let primary = primary();
    admit_recording(&references, &primary.environment_id, &primary.identity).unwrap();
    // Another device under the reference's ID, and the reference after an Xcode update.
    let other = identity("Test GPU Mini", "2.0");
    let refused = admit_recording(&references, &primary.environment_id, &other).unwrap_err();
    assert!(
        refused.contains("device: \"Test GPU\" (\"Test GPU Mini\")"),
        "{refused}"
    );
    assert!(refused.contains(&other.environment_id()), "{refused}");
    let mut updated = primary.identity.clone();
    updated.xcode_build = "27B1".into();
    let refused = admit_recording(&references, &primary.environment_id, &updated).unwrap_err();
    assert!(
        refused.contains("xcode_build: \"27A1\" (\"27B1\")"),
        "{refused}"
    );
    // An environment the list does not name.
    admit_recording(&references, "scratch-env", &updated).unwrap();
}

/// A case's facts checked once are kept in the reader's index under this build's key, so the next
/// reader reads them and checks nothing; facts of another build's key are deleted when a reader
/// opens the index; and a reader given no index, or a path without one, keeps nothing (and creates
/// no index), so every reader checks again. The root and its index are a scratch directory of
/// their own: the committed public fixture's case checks against the checkout's fixtures.
#[test]
fn case_facts_are_kept_in_the_index_under_the_builds_key() {
    let committed = crate::store::CorpusStore::new(crate::source::corpus_root())
        .read_all_cases()
        .unwrap();
    let case = committed
        .iter()
        .find(|case| {
            case.public_source
                .as_deref()
                .is_some_and(|source| source.ends_with("bcu_address.ll"))
        })
        .expect("a committed case over bcu_address.ll");
    let scratch = crate::ScratchDir::new("case-facts").unwrap();
    let root = scratch.path().join("with-index");
    std::fs::create_dir_all(&root).unwrap();
    let index = crate::index::default_index_path(&root);
    {
        let connection = rusqlite::Connection::open(&index).unwrap();
        connection.execute_batch(CASE_FACTS_TABLE).unwrap();
        connection
            .execute(
                "INSERT INTO case_reference_facts (case_id, abi, facts) VALUES (?1, ?2, ?3)",
                ["another case", "case-reference-facts-v0:old", "{}"],
            )
            .unwrap();
    }
    let kept_rows = || {
        rusqlite::Connection::open(&index)
            .unwrap()
            .query_row(
                "SELECT count(*), coalesce(sum(abi = ?1), 0) FROM case_reference_facts",
                [CASE_FACTS_ABI],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap()
    };

    let first = CheckedFacts::new(&root, Some(&index));
    assert_eq!(kept_rows(), (0, 0), "another build's facts are deleted");
    let answer_gaps = first.answer_gaps(case);
    assert!(answer_gaps.is_some());
    assert_eq!(first.answer_gaps(case), answer_gaps);
    assert_eq!(first.checked(), 1, "checked once");
    drop(first);
    assert_eq!(kept_rows(), (1, 1), "kept when the reader is dropped");

    let second = CheckedFacts::new(&root, Some(&index));
    assert_eq!(second.answer_gaps(case), answer_gaps);
    assert_eq!(second.checked(), 0, "read from the index, not checked");
    drop(second);

    let absent = scratch.path().join("absent.sqlite");
    for index in [None, Some(absent.as_path())] {
        for _ in 0..2 {
            let reader = CheckedFacts::new(&root, index);
            assert_eq!(reader.answer_gaps(case), answer_gaps);
            assert_eq!(reader.checked(), 1, "nothing kept without an index");
        }
    }
    assert!(!absent.exists(), "no index is created");
}
