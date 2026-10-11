//! `apple_only`: every Apple-only row (class A) answers typed.

use crate::{AppleOnly, AppleOnlyCall, DEFAULT_PROTECTION_OPTIONS};

/// The 12 Apple-only calls of the table, each at Metal's default where it has one and at a value
/// off it, with the runtime's answer. The getter (`protectionOptions`) is
/// `no_getter_answers_off_metals_default`'s. A texture's rotation is no longer the table's: the
/// runtime builds it.
#[test]
fn every_apple_only_call_answers_typed() {
    use AppleOnlyCall as C;
    let table: [(&str, AppleOnlyCall, Result<(), AppleOnly>); 21] = [
        // Compressed-texture controls: no compression is visible on Vulkan.
        (
            "invalidateCompressedTexture:",
            C::InvalidateCompressedTexture,
            Ok(()),
        ),
        (
            "invalidateCompressedTexture:slice:level:",
            C::InvalidateCompressedTextureSlice,
            Ok(()),
        ),
        (
            "insertCompressedTextureReinterpretationFlush",
            C::CompressedTextureReinterpretationFlush,
            Ok(()),
        ),
        // Protection: Metal's default only.
        ("setProtectionOptions:", C::SetProtectionOptions(0), Ok(())),
        (
            "setProtectionOptions:",
            C::SetProtectionOptions(1),
            Err(AppleOnly::ProtectedContent),
        ),
        // Render setters.
        (
            "setAlphaTestReferenceValue:",
            C::SetAlphaTestReferenceValue(0.5),
            Ok(()),
        ),
        ("setLineWidth:", C::SetLineWidth(1.0), Ok(())),
        (
            "setLineWidth:",
            C::SetLineWidth(2.0),
            Err(AppleOnly::LineWidth),
        ),
        (
            "setLineWidth:",
            C::SetLineWidth(0.5),
            Err(AppleOnly::LineWidth),
        ),
        (
            "setLineWidth:",
            C::SetLineWidth(f32::NAN),
            Err(AppleOnly::LineWidth),
        ),
        ("setPointSize:", C::SetPointSize(1.0), Ok(())),
        ("setPointSize:", C::SetPointSize(f32::NAN), Ok(())),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(0),
            Ok(()),
        ),
        (
            "setProvokingVertexMode:",
            C::SetProvokingVertexMode(1),
            Err(AppleOnly::ProvokingVertexMode),
        ),
        (
            "setViewportTransformEnabled:",
            C::SetViewportTransformEnabled(true),
            Ok(()),
        ),
        (
            "setViewportTransformEnabled:",
            C::SetViewportTransformEnabled(false),
            Err(AppleOnly::ViewportTransformDisabled),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode { mode: 0, value: 0 },
            Err(AppleOnly::VertexAmplificationMode),
        ),
        (
            "setVertexAmplificationMode:value:",
            C::SetVertexAmplificationMode { mode: 1, value: 2 },
            Err(AppleOnly::VertexAmplificationMode),
        ),
        (
            "setDepthCleared",
            C::SetDepthCleared,
            Err(AppleOnly::DepthCleared),
        ),
        (
            "setStencilCleared",
            C::SetStencilCleared,
            Err(AppleOnly::StencilCleared),
        ),
        (
            "setAlphaTestReferenceValue:",
            C::SetAlphaTestReferenceValue(f32::NAN),
            Ok(()),
        ),
    ];
    let mut selectors: Vec<&str> = Vec::new();
    for (selector, call, answer) in table {
        assert_eq!(call.selector(), selector, "{call:?}");
        assert_eq!(call.check(), answer, "{call:?}");
        if !selectors.contains(&selector) {
            selectors.push(selector);
        }
    }
    // The 12 calls of the table's 21 rows (some calls have several rows).
    assert_eq!(selectors.len(), 12, "{selectors:?}");
}

/// The getter answers Metal's default, which is the only value a runtime texture can hold,
/// since any other is refused.
#[test]
fn no_getter_answers_off_metals_default() {
    assert_eq!(DEFAULT_PROTECTION_OPTIONS, 0);
    assert_eq!(
        AppleOnlyCall::SetProtectionOptions(DEFAULT_PROTECTION_OPTIONS).check(),
        Ok(())
    );
}

/// Each refused Apple-only feature names why ([`AppleOnly::gap`]), and none is a device's gap:
/// a color fill packed where Metal's answer differs between processes has no one answer; the
/// provoking vertex mode, measured, is not built; every other refusal's answer on Metal is not
/// measured. The error's text names the gap.
#[test]
fn each_apple_only_refusal_names_its_gap() {
    use crate::AppleOnlyGap::{NotBuilt, Unmeasured};
    let table = [
        (AppleOnly::ProtectedContent, Unmeasured),
        (AppleOnly::LineWidth, Unmeasured),
        (AppleOnly::ProvokingVertexMode, NotBuilt),
        (AppleOnly::ViewportTransformDisabled, Unmeasured),
        (AppleOnly::VertexAmplificationMode, Unmeasured),
        (AppleOnly::DepthCleared, Unmeasured),
        (AppleOnly::StencilCleared, Unmeasured),
        (AppleOnly::SurfaceClientDirection, Unmeasured),
        (AppleOnly::FillUnmeasured, Unmeasured),
    ];
    for (feature, gap) in table {
        assert_eq!(feature.gap(), gap, "{feature:?}");
        let text = crate::Error::AppleOnly(feature).to_string();
        assert!(text.contains(&format!("({gap:?})")), "{text}");
        assert!(!text.contains("not on Vulkan"), "{text}");
    }
}
