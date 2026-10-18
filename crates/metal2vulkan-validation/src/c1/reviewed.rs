//! C1's reviewed list.
//!
//! C1 asked the engine to translate every row the old engine translated with output that
//! passed L3 under the profile, minus this list. The list keeps only three kinds of rows: rows
//! whose device lacks what an exact translation needs ([`ListClass::DeviceLacks`]), rows whose old
//! output was wrong or guessed ([`ListClass::OldOutputWrong`]), and rows the engine translates
//! only as a recorded near-equivalent ([`ListClass::NearEquivalent`]). `needs_input` rows are not
//! list rows: C1 counted them apart. Nor are rows the launch leaves out of scope ([`OutOfScope`],
//! ray tracing), which are never a signed-off device gap.
//!
//! One entry covers one bucket of the C1 gap: a profile and the engine's typed outcome
//! (label and key, as the census records it). It records how many rows it reviewed and the digest
//! of their hashes ([`rows_digest`]), never the hashes themselves (O-3), and the census keys they
//! were taken from, for provenance only. C1's report, which derived each bucket's rows from the
//! index, is deleted with the old engine; C3 reads the list's signed-off device-capability entries
//! as its device gaps (`crate::c3`).

use crate::cutover::scope::OutOfScope;
use crate::cutover::{List, Review};
use crate::engine::{FailureKind, Outcome};
use metal2vulkan::error::ErrorClass;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeSet;

/// What a C1 gap row's engine outcome makes it, in the order C1's report listed the classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GapClass {
    /// `unsupported not_implemented`: Metal the engine does not implement yet. Work.
    Work,
    /// `device_capability_missing`: the profile lacks a capability the engine's translation
    /// needs. A list row when the device truly lacks it and no exact emulation exists; work
    /// otherwise, and a question when the old output, valid under the profile, does without.
    DeviceCapability,
    /// `unsupported not_representable`: the engine knows no Vulkan mechanism. A
    /// near-equivalent or device-capability question.
    NotRepresentable,
    /// `invalid_input` or `invalid_descriptor`: the engine reads the input or the request as
    /// malformed where the old engine translated it. A reader gap, or old output from malformed
    /// input.
    InvalidInput,
    /// `limit_exceeded` at a structural cap.
    StructuralLimit,
    /// `failed no_bitcode`: `llvm-as` rejects the row's AIR text, so no bitcode reached the new
    /// engine.
    Unassembled,
    /// A defect ([`Outcome::recorded_is_defect`]): a budget stop, a panic, an invalid module.
    Defect,
    /// An outcome the engine's census never records (old-engine or transient labels).
    Other,
    /// `needs_input`: an input the request did not supply. Counted apart.
    NeedsInput,
}

impl GapClass {
    /// The class of a recorded engine outcome, `None` for `translated`.
    pub fn of(label: &str, key: &str) -> Option<GapClass> {
        if Outcome::recorded_is_defect(label, key) {
            return Some(GapClass::Defect);
        }
        Some(match label {
            "translated" => return None,
            "needs_input" => GapClass::NeedsInput,
            "device_capability_missing" => GapClass::DeviceCapability,
            "unsupported" => match key.split_once(' ').map(|(reason, _)| reason) {
                Some("not_implemented") => GapClass::Work,
                Some("not_representable") => GapClass::NotRepresentable,
                _ => GapClass::Other,
            },
            "invalid_input" | "invalid_descriptor" => GapClass::InvalidInput,
            "limit_exceeded" => GapClass::StructuralLimit,
            "failed" if key == FailureKind::NoBitcode.as_str() => GapClass::Unassembled,
            _ => GapClass::Other,
        })
    }

    pub const ALL: [GapClass; 9] = [
        GapClass::Work,
        GapClass::DeviceCapability,
        GapClass::NotRepresentable,
        GapClass::InvalidInput,
        GapClass::StructuralLimit,
        GapClass::Unassembled,
        GapClass::Defect,
        GapClass::Other,
        GapClass::NeedsInput,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            GapClass::Work => "work",
            GapClass::DeviceCapability => "device_capability",
            GapClass::NotRepresentable => "not_representable",
            GapClass::InvalidInput => "invalid_input",
            GapClass::StructuralLimit => "structural_limit",
            GapClass::Unassembled => "unassembled",
            GapClass::Defect => "defect",
            GapClass::Other => "other",
            GapClass::NeedsInput => "needs_input",
        }
    }

    /// Whether C1 counts the class's rows (every class but `needs_input`).
    pub fn counts(self) -> bool {
        self != GapClass::NeedsInput
    }
}

/// Why the rows of an entry need not be translated by the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ListClass {
    /// The device lacks what an exact translation needs and no exact emulation exists: the new
    /// engine's typed `device_capability_missing` refusal names it.
    DeviceLacks,
    /// The old engine's output was wrong or guessed for these rows: an authored case mismatched,
    /// a driver rejected it (L4), or its structure shows it does not compute what the AIR asks.
    OldOutputWrong,
    /// The device lacks the exact mechanism and the rows translate only as a near-equivalent the
    /// artifact records.
    NearEquivalent,
}

impl ListClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ListClass::DeviceLacks => "device_lacks",
            ListClass::OldOutputWrong => "old_output_wrong",
            ListClass::NearEquivalent => "near_equivalent",
        }
    }
}

/// One reviewed bucket of one profile's C1 gap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// The committed profile (`validation/profiles/<profile>.json`).
    pub profile: String,
    /// The engine's outcome class (`engine::Outcome::label`).
    pub outcome: String,
    /// Its typed key (`engine::Outcome::key`).
    pub key: String,
    pub class: ListClass,
    /// How many rows the entry reviewed.
    pub rows: usize,
    /// [`rows_digest`] of the rows it reviewed.
    pub rows_digest: String,
    /// The engine census key the rows were taken from (provenance; not compared).
    pub new_census: String,
    /// The old engine's L3 table key for the profile (provenance; not compared).
    pub old_l3: String,
    pub review: Review,
}

/// The reviewed list.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewedList {
    pub entries: Vec<Entry>,
}

/// The committed list is `validation/cutover/c1-reviewed.json`, its entries in (profile, outcome,
/// key) order.
impl List for ReviewedList {
    const FILE: &'static str = "c1-reviewed.json";
    const NAME: &'static str = "C1 reviewed list";

    fn committed(&self) -> Cow<'_, Self> {
        let mut sorted = self.clone();
        sorted.entries.sort_by(|a, b| {
            (&a.profile, &a.outcome, &a.key).cmp(&(&b.profile, &b.outcome, &b.key))
        });
        Cow::Owned(sorted)
    }

    /// Every entry names a committed profile and a gap bucket the list may hold (not
    /// `needs_input`, which C1 counts apart; not an outcome the launch leaves out of scope, which
    /// it reports apart; not a defect, an unassembled row, or an invalid-input refusal, which are
    /// bugs or input defects to fix, except guessed old output on input the product classes as
    /// malformed, listed as `old_output_wrong`; not an outcome the census never records), is a
    /// device-capability bucket when its class says the device lacks something, reviewed at least
    /// one row with well-formed hashes, and carries its review; and no bucket is listed twice.
    fn check(&self) -> Result<(), String> {
        let profiles = crate::engine::committed_profiles()?;
        let mut buckets = BTreeSet::new();
        for entry in &self.entries {
            let at = |what: String| {
                format!(
                    "C1 entry {} `{} {}`: {what}",
                    entry.profile, entry.outcome, entry.key
                )
            };
            if !profiles.contains(&entry.profile) {
                return Err(at("not a committed profile".into()));
            }
            let class = GapClass::of(&entry.outcome, &entry.key)
                .ok_or_else(|| at("not a gap outcome".into()))?;
            if let Some(scope) = OutOfScope::of_recorded(&entry.outcome, &entry.key) {
                return Err(at(format!(
                    "{} is out of scope for the launch: its rows are reported apart, never listed",
                    scope.as_str()
                )));
            }
            match class {
                GapClass::NeedsInput => {
                    return Err(at(
                        "needs_input rows are counted apart, never listed".into()
                    ))
                }
                GapClass::Defect => {
                    return Err(at("a defect is a bug to fix, never a list row".into()))
                }
                // A reader gap is work to fix, never a list row. Input the product classes as
                // malformed has no defined translation, so old output for it was guessed: only
                // that, as `old_output_wrong`, may list such a bucket.
                GapClass::InvalidInput
                    if entry.class == ListClass::OldOutputWrong
                        && Outcome::recorded_input_class(&entry.outcome, &entry.key)
                            == Some(ErrorClass::Malformed) => {}
                GapClass::InvalidInput => {
                    return Err(at(
                        "an invalid-input refusal of a row the old engine translated is a reader gap to fix, never a list row; only guessed old output on input the product classes as malformed may be listed, as old_output_wrong"
                            .into(),
                    ))
                }
                GapClass::Unassembled => {
                    return Err(at(
                        "a row llvm-as cannot assemble is an input defect, never a list row".into(),
                    ))
                }
                GapClass::Other => {
                    return Err(at(
                        "not an outcome the engine's census records".into()
                    ))
                }
                GapClass::DeviceCapability => {}
                _ if entry.class == ListClass::DeviceLacks => return Err(at(
                    "device_lacks needs the engine's typed device_capability_missing refusal"
                        .into(),
                )),
                _ => {}
            }
            if entry.rows == 0 {
                return Err(at("reviews no row".into()));
            }
            for (field, value) in [
                ("rows_digest", &entry.rows_digest),
                ("new_census", &entry.new_census),
                ("old_l3", &entry.old_l3),
            ] {
                if !is_sha256(value) {
                    return Err(at(format!("{field} is not a lowercase SHA-256")));
                }
            }
            entry.review.check().map_err(at)?;
            if !buckets.insert((&entry.profile, &entry.outcome, &entry.key)) {
                return Err(at("listed twice".into()));
            }
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The digest an entry records for the rows it reviewed: SHA-256 over a domain tag and the rows'
/// hashes in sorted order, each ended by a newline. Order and duplicates in `rows` do not matter.
pub fn rows_digest<'a>(rows: impl IntoIterator<Item = &'a String>) -> String {
    let sorted = rows.into_iter().collect::<BTreeSet<_>>();
    let mut bytes = b"metal2vulkan-c1-reviewed-rows\n".to_vec();
    for row in sorted {
        bytes.extend_from_slice(row.as_bytes());
        bytes.push(b'\n');
    }
    crate::hash::sha256_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutover::SignOff;

    fn entry(outcome: &str, key: &str, class: ListClass) -> Entry {
        Entry {
            profile: "moltenvk-m3".into(),
            outcome: outcome.into(),
            key: key.into(),
            class,
            rows: 2,
            rows_digest: rows_digest(&["a".to_string(), "b".to_string()]),
            new_census: "1".repeat(64),
            old_l3: "2".repeat(64),
            review: Review {
                evidence: vec!["a device probe".into()],
                note: "reviewed".into(),
                sign_off: SignOff::AwaitingUser {
                    decision: "a device limit".into(),
                },
            },
        }
    }

    #[test]
    fn rows_digest_ignores_order_and_duplicates_and_sees_membership() {
        let (a, b, c) = ("a".to_string(), "b".to_string(), "c".to_string());
        assert_eq!(rows_digest([&a, &b]), rows_digest([&b, &a, &b]));
        assert_ne!(rows_digest([&a, &b]), rows_digest([&a, &b, &c]));
        assert_ne!(rows_digest([&a, &b]), rows_digest([&a]));
        assert!(is_sha256(&rows_digest([&a])));
    }

    #[test]
    fn the_list_keeps_only_the_buckets_it_may_hold() {
        let ok = entry(
            "device_capability_missing",
            "Float64 needs Feature(Float64)",
            ListClass::DeviceLacks,
        );
        let list = ReviewedList {
            entries: vec![ok.clone()],
        };
        assert_eq!(ReviewedList::parse(&list.to_json().unwrap()).unwrap(), list);
        let rejected = |entry: Entry| {
            ReviewedList {
                entries: vec![entry],
            }
            .check()
            .unwrap_err()
        };
        assert!(rejected(entry(
            "needs_input",
            "function_constant",
            ListClass::OldOutputWrong
        ))
        .contains("counted apart"));
        assert!(rejected(entry("failed", "panic", ListClass::OldOutputWrong)).contains("defect"));
        assert!(rejected(entry(
            "limit_exceeded",
            "budget WorkUnits",
            ListClass::OldOutputWrong
        ))
        .contains("defect"));
        assert!(rejected(entry(
            "unsupported",
            "not_implemented Tessellation",
            ListClass::DeviceLacks
        ))
        .contains("device_capability_missing"));
        // Ray tracing is reported apart, never listed, in any class.
        for (outcome, key, class) in [
            (
                "unsupported",
                "not_implemented RayTracing",
                ListClass::OldOutputWrong,
            ),
            (
                "device_capability_missing",
                "RayTracing needs Feature(RayQuery)+Feature(AccelerationStructure)",
                ListClass::DeviceLacks,
            ),
            (
                "unsupported",
                "not_representable RayTracing",
                ListClass::NearEquivalent,
            ),
            (
                "unsupported",
                "not_implemented IntersectionFunctionTable",
                ListClass::OldOutputWrong,
            ),
        ] {
            assert!(rejected(entry(outcome, key, class)).contains("out of scope"));
        }
        assert!(rejected(entry("translated", "", ListClass::OldOutputWrong)).contains("gap"));
        assert!(rejected(entry(
            "invalid_input",
            "Abi::BadValue",
            ListClass::OldOutputWrong
        ))
        .contains("reader gap"));
        assert!(
            rejected(entry("failed", "no_bitcode", ListClass::OldOutputWrong)).contains("assemble")
        );
        for never_recorded in [("failed", "tooling"), ("failed", "cancelled")] {
            assert!(rejected(entry(
                never_recorded.0,
                never_recorded.1,
                ListClass::NearEquivalent
            ))
            .contains("census records"));
        }
        // Work the old engine got wrong, and a near-equivalent, may be listed.
        for allowed in [
            entry(
                "unsupported",
                "not_implemented Tessellation",
                ListClass::OldOutputWrong,
            ),
            entry(
                "device_capability_missing",
                "FramebufferFetch needs Feature(RasterizationOrderColorAttachmentAccess)",
                ListClass::NearEquivalent,
            ),
        ] {
            assert!(ReviewedList {
                entries: vec![allowed]
            }
            .check()
            .is_ok());
        }
        let mut unknown = ok.clone();
        unknown.profile = "no-such-profile".into();
        assert!(rejected(unknown).contains("committed profile"));
        let mut empty = ok.clone();
        empty.rows = 0;
        assert!(rejected(empty).contains("no row"));
        let mut digest = ok.clone();
        digest.rows_digest = "ABC".into();
        assert!(rejected(digest).contains("rows_digest"));
        let mut unreviewed = ok.clone();
        unreviewed.review.note.clear();
        assert!(rejected(unreviewed).contains("note"));
        let twice = ReviewedList {
            entries: vec![ok.clone(), ok],
        };
        assert!(twice.check().unwrap_err().contains("twice"));
    }

    /// An invalid-input bucket may be listed only as guessed old output on input the product
    /// classes as malformed; a reader gap (another class), any other list class, and an invalid
    /// descriptor stay refused.
    #[test]
    fn only_guessed_old_output_on_malformed_input_lists_an_invalid_input_bucket() {
        let check = |outcome: &str, key: &str, class| {
            ReviewedList {
                entries: vec![entry(outcome, key, class)],
            }
            .check()
        };
        assert!(check(
            "invalid_input",
            "Canon::CallTypeMismatch",
            ListClass::OldOutputWrong
        )
        .is_ok());
        assert!(check(
            "invalid_input",
            "Canon::UndefCondition",
            ListClass::OldOutputWrong
        )
        .is_ok());
        for (outcome, key, class) in [
            // Malformed input, but not listed as guessed old output.
            (
                "invalid_input",
                "Canon::CallTypeMismatch",
                ListClass::DeviceLacks,
            ),
            (
                "invalid_input",
                "Canon::CallTypeMismatch",
                ListClass::NearEquivalent,
            ),
            // Guessed old output, but on input of another class: a reader gap to fix.
            (
                "invalid_input",
                "Canon::StaticInitNotEvaluable",
                ListClass::OldOutputWrong,
            ),
            // A key the product's class is not read from, and a request error.
            ("invalid_input", "Abi::BadValue", ListClass::OldOutputWrong),
            (
                "invalid_descriptor",
                "Canon::CallTypeMismatch",
                ListClass::OldOutputWrong,
            ),
        ] {
            let refused = check(outcome, key, class).unwrap_err();
            assert!(refused.contains("reader gap"), "{outcome} {key}: {refused}");
        }
    }

    #[test]
    fn the_list_is_written_in_bucket_order() {
        let later = entry(
            "unsupported",
            "not_implemented Tessellation",
            ListClass::OldOutputWrong,
        );
        let earlier = entry(
            "device_capability_missing",
            "Float64 needs Feature(Float64)",
            ListClass::DeviceLacks,
        );
        let list = ReviewedList {
            entries: vec![later.clone(), earlier.clone()],
        };
        let written = ReviewedList::parse(&list.to_json().unwrap()).unwrap();
        assert_eq!(written.entries, [earlier, later]);
    }

    #[test]
    fn the_committed_list_parses_in_its_committed_form() {
        let path = ReviewedList::committed_path();
        let list = ReviewedList::load(&path).unwrap();
        assert_eq!(
            list.to_json().unwrap(),
            std::fs::read_to_string(&path).unwrap()
        );
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut json: serde_json::Value = serde_json::from_str(
            &ReviewedList {
                entries: vec![entry(
                    "device_capability_missing",
                    "Float64 needs Feature(Float64)",
                    ListClass::DeviceLacks,
                )],
            }
            .to_json()
            .unwrap(),
        )
        .unwrap();
        json["entries"][0]["air_sha256"] = serde_json::json!(["a"]);
        assert!(ReviewedList::parse(&json.to_string()).is_err());
    }
}
