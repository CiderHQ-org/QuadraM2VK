//! The sample-rate features an entry uses: what only a multisampled render case can evidence.
//!
//! Descriptive, like `rate_map::maps_coordinates`: both closure executors run a multisampled pass
//! (`raster_sample_count`), so no tooling requirement follows from a use. Classification records
//! an entry's uses (`StructuralTriage::sample_rate`) so the focused audit selects the family
//! (`AuditTarget::SampleRate`) whatever its requirements, and gaining support never empties it.
//! Every use is read from the AIR ABI, never from a name: the `air.get_num_samples.i32` call, and
//! a fragment entry's `air.sample_id` and `air.sample_mask_in` inputs, `air.sample_mask` output
//! members, and `fragment_input` varyings sampled per sample (`air.sample`), as the product's
//! fragment metadata decodes them. A member behind an `air.function_constant` gate is a use
//! whatever its constant: a case that enables the constant draws it, so the family it belongs to
//! does not depend on the constants classification has none of.

use crate::air::meta::{parse_air_fragment_meta, FragRole, VaryingSampling};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// One sample-rate feature of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleRateUse {
    /// `air.get_num_samples.i32`: the pass's raster sample count.
    NumSamples,
    /// A `[[sample_id]]` input, which runs the function once per sample.
    SampleId,
    /// A `[[sample_mask]]` input: the coverage the rasterizer gave the invocation.
    SampleMaskIn,
    /// A `[[sample_mask]]` output member: the coverage the function writes.
    SampleMaskOut,
    /// A varying interpolated at each sample's position (`[[sample_perspective]]`,
    /// `[[sample_no_perspective]]`), which also runs the function once per sample.
    SampleVarying,
}

/// The ABI symbol of the raster sample count.
const NUM_SAMPLES: &str = "air.get_num_samples.i32";

/// The marker that opens an argument's function-constant gate in AIR metadata: the gate's
/// predicate node follows it (`!"air.function_constant", !N, `), then the argument's own role.
const GATE: &str = "!\"air.function_constant\", !";

/// `text` with every metadata argument's function-constant gate removed, so each argument reads as
/// declared whatever its constant. The product's fragment metadata keeps a gated argument only when
/// the module's own initializers enable it, which no constant-free classification can know.
fn ungated(text: &str) -> Cow<'_, str> {
    if !text.contains(GATE) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        if !line.starts_with('!') {
            out.push_str(line);
            continue;
        }
        let mut rest = line;
        while let Some(at) = rest.find(GATE) {
            let after = &rest[at + GATE.len()..];
            let digits = after.bytes().take_while(u8::is_ascii_digit).count();
            match after[digits..].strip_prefix(", ") {
                Some(tail) if digits > 0 => {
                    out.push_str(&rest[..at]);
                    rest = tail;
                }
                // Not a gate's shape: keep it as it is.
                _ => {
                    out.push_str(&rest[..at + GATE.len()]);
                    rest = after;
                }
            }
        }
        out.push_str(rest);
    }
    Cow::Owned(out)
}

/// The sample-rate features of the entry whose module text is `air_ll` (its projection onto the
/// entry) and whose stage is `stage` (`"Fragment"` for a fragment function), given its AIR call
/// counts `calls`: each use once, in order.
pub fn sample_rate_uses(
    stage: &str,
    air_ll: &str,
    calls: &BTreeMap<String, usize>,
) -> Vec<SampleRateUse> {
    let mut uses = Vec::new();
    if calls.contains_key(NUM_SAMPLES) {
        uses.push(SampleRateUse::NumSamples);
    }
    // Every role below is spelled with an `air.sample` marker, so a module without one has none
    // and its metadata is not parsed.
    if stage == "Fragment" && air_ll.contains("air.sample") {
        if let Some(meta) = parse_air_fragment_meta(&ungated(air_ll)).0 {
            for (_, role) in &meta.roles {
                match role {
                    FragRole::SampleId => uses.push(SampleRateUse::SampleId),
                    FragRole::SampleMaskIn => uses.push(SampleRateUse::SampleMaskIn),
                    _ => {}
                }
            }
            if !meta.sample_mask_members.is_empty() {
                uses.push(SampleRateUse::SampleMaskOut);
            }
            if meta
                .varying_interpolation
                .values()
                .any(|varying| !varying.flat && varying.sampling == VaryingSampling::Sample)
            {
                uses.push(SampleRateUse::SampleVarying);
            }
        }
    }
    uses.sort();
    uses.dedup();
    uses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor_contract::air_call_counts;
    use SampleRateUse as U;

    /// The uses of `entry` of the public fixture text `ll`, through its projection onto the entry
    /// as classification reads it.
    fn uses_of(ll: &str, entry: &str) -> Vec<SampleRateUse> {
        let text = crate::air::entry_text(ll, entry).unwrap();
        sample_rate_uses("Fragment", &text, &air_call_counts(&text))
    }

    /// Each entry of the owned sample-rate fixtures reports exactly the features it reads or
    /// writes, from its own projection: a sibling entry's inputs are never charged to it.
    #[test]
    fn each_sample_rate_entry_reports_its_own_uses() {
        let sample_rate = include_str!("../../fixtures/public/engine/sample_rate.ll");
        assert_eq!(
            uses_of(sample_rate, "sample_rate_num_samples"),
            [U::NumSamples]
        );
        assert_eq!(
            uses_of(sample_rate, "sample_rate_mask_in"),
            [U::SampleId, U::SampleMaskIn]
        );
        assert_eq!(
            uses_of(sample_rate, "sample_rate_coverage"),
            [U::SampleMaskIn]
        );
        assert_eq!(
            uses_of(sample_rate, "sample_rate_mask_out"),
            [U::SampleMaskOut]
        );
        assert_eq!(uses_of(sample_rate, "sample_rate_depth"), [U::SampleId]);
        let multisample = include_str!("../../fixtures/public/engine/multisample.ll");
        assert_eq!(
            uses_of(multisample, "multisample_sampled_varying"),
            [U::SampleVarying]
        );
        // A `[[stage_in]]` member keeps its place in the fragment interface whether or not the
        // function reads it, and a sample-qualified one runs the function per sample all the same:
        // each entry over `Varyings` declares its `[[sample_perspective]]` member.
        assert_eq!(
            uses_of(multisample, "multisample_center_at_sample"),
            [U::SampleId, U::SampleVarying]
        );
        assert_eq!(
            uses_of(multisample, "multisample_counter"),
            [U::SampleVarying]
        );
    }

    /// A sample-rate argument behind a function constant is a use whatever the constant, though
    /// the product's metadata, which keeps only what the module's own initializers enable, drops
    /// it; an argument without a sample-rate role stays out.
    #[test]
    fn a_gated_argument_is_a_use() {
        let sample_rate = include_str!("../../fixtures/public/engine/sample_rate.ll");
        let text = crate::air::entry_text(sample_rate, "sample_rate_mask_in").unwrap();
        let gated =
            text.replace(
                r#"!{i32 0, !"air.sample_mask_in""#,
                r#"!{i32 0, !"air.function_constant", !900, !"air.sample_mask_in""#,
            )
            .replace(
                r#"!{i32 1, !"air.sample_id""#,
                r#"!{i32 1, !"air.function_constant", !900, !"air.sample_id""#,
            ) + "!900 = !{ptr @sample_rate_gate}\n";
        assert_eq!(gated.matches(GATE).count(), 2);
        let dropped = parse_air_fragment_meta(&gated).0.unwrap();
        assert!(!dropped
            .roles
            .iter()
            .any(|(_, role)| matches!(role, FragRole::SampleId | FragRole::SampleMaskIn)));
        assert_eq!(
            sample_rate_uses("Fragment", &gated, &air_call_counts(&gated)),
            [U::SampleId, U::SampleMaskIn]
        );
        assert_eq!(ungated(&text), text, "an ungated module is left as it is");
        let not_a_gate = "!1 = !{!\"air.function_constant\", !x, !\"air.sample_id\"}\n";
        assert_eq!(ungated(not_a_gate), not_a_gate);
    }

    /// Only a fragment function's metadata is read, and a kernel's call of the count is still a
    /// use.
    #[test]
    fn other_stages_report_only_the_count_call() {
        let sample_rate = include_str!("../../fixtures/public/engine/sample_rate.ll");
        let text = crate::air::entry_text(sample_rate, "sample_rate_mask_in").unwrap();
        assert_eq!(
            sample_rate_uses("Kernel", &text, &air_call_counts(&text)),
            []
        );
        let calls = BTreeMap::from([(NUM_SAMPLES.to_string(), 1)]);
        assert_eq!(sample_rate_uses("Kernel", "", &calls), [U::NumSamples]);
    }
}
