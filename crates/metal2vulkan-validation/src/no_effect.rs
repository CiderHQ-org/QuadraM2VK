//! An output the case means to leave as authored.
//!
//! A case observes the effect of its work on one selected output. When Metal's answer is exactly
//! the selected initial bytes, that observation shows nothing the work did: the work never reached
//! the output (an early exit its parameters select, a callback that writes nothing), or it wrote
//! back what was there. A candidate that ran nothing at all would match it. Such an answer is
//! evidence only when the case declares the path intentional ([`AuthoredCase::no_effect`]), and a
//! declaration is honest only while Metal's answer is the selected initial bytes.
//!
//! The judgment is structural: the selected bytes, Metal's answer, and the declaration, never the
//! case's name. Metal qualification applies it before it records a row ([`judge`]); the committed
//! store is held to it by test ([`judge_row`]).

use crate::case::{AuthoredCase, OutputSelection};
use crate::literal::LiteralResources;
use crate::observation::{MetalObservation, MetalStatus};
use base64::Engine as _;

/// The longest declaration, in bytes: a reason, not a document.
pub const MAX_REASON_BYTES: usize = 512;

/// What is wrong with `case`'s declaration, if it declares one: an empty or padded reason, one past
/// [`MAX_REASON_BYTES`], or a case that selects no output (it observes nothing to leave as
/// authored).
pub fn declaration_errors(case: &AuthoredCase) -> Vec<String> {
    let Some(reason) = &case.no_effect else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    if reason.trim().is_empty() || reason.trim() != reason {
        errors.push("no_effect must be a reason without surrounding whitespace".into());
    }
    if reason.len() > MAX_REASON_BYTES {
        errors.push(format!(
            "no_effect is {} bytes; a reason is at most {MAX_REASON_BYTES}",
            reason.len()
        ));
    }
    if matches!(case.output, OutputSelection::None) {
        errors.push(
            "no_effect declares an output left as authored, but the case selects none".into(),
        );
    }
    errors
}

/// Whether Metal's answer `output` for `case`, whose selected output was authored as `initial`,
/// agrees with the case's declaration: an answer equal to a nonempty `initial` needs
/// [`AuthoredCase::no_effect`], and a declared case's answer must equal `initial`.
pub fn judge(case: &AuthoredCase, initial: &[u8], output: &[u8]) -> Result<(), String> {
    let unchanged = output == initial;
    match (&case.no_effect, unchanged) {
        (None, true) if !initial.is_empty() => Err(format!(
            "Metal left the selected output exactly as authored ({} bytes): the case observes no \
             effect of its work. Author inputs whose work reaches the output, or declare \
             no_effect with the reason the path is intended",
            initial.len()
        )),
        (Some(reason), false) => Err(format!(
            "the case declares no_effect ({reason}), but Metal changed the selected output"
        )),
        _ => Ok(()),
    }
}

/// [`judge`] for a recorded Metal row of `case`: a row that records no output (a reference gap)
/// has nothing to judge.
pub fn judge_row(case: &AuthoredCase, row: &MetalObservation) -> Result<(), String> {
    if !matches!(row.status, MetalStatus::Qualified) {
        return Ok(());
    }
    let initial = LiteralResources::prepare(case)?.selected_initial(&case.output)?;
    let output = base64::engine::general_purpose::STANDARD
        .decode(&row.output_b64)
        .map_err(|error| format!("Metal row of {}: output: {error}", case.case_id))?;
    judge(case, &initial, &output)
}

#[cfg(test)]
mod tests;
