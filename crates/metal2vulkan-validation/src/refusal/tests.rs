//! A refusal's class from the executor's typed refusal, and a refusal record's own checks.

use super::*;
use crate::coverage::{Leg, RobustBufferForm};
use crate::engine::{LimitKind, Outcome, UnsupportedReason};
use crate::requirement::ToolingRequirement;

fn device_capability_missing() -> NotSupported {
    NotSupported::Product {
        outcome: Outcome::DeviceCapabilityMissing {
            feature: "FramebufferFetch".into(),
            capability: "Feature(RasterizationOrderColorAttachmentAccess)".into(),
        },
        detail: "the device has no ordered attachment access".into(),
        at_bind: false,
    }
}

/// Every typed refusal has one class, from its type, never from its text: the device lacks what
/// the case needs, the product does not translate it, or this executor does not run it.
#[test]
fn a_refusal_is_classed_by_its_type() {
    let product = |outcome| NotSupported::Product {
        outcome,
        detail: "detail".into(),
        at_bind: false,
    };
    for (refusal, class) in [
        (device_capability_missing(), RefusalClass::Device),
        (
            NotSupported::Bind("a range past the limit".into()),
            RefusalClass::Device,
        ),
        (
            NotSupported::NotEnabled(vec!["rayQuery".into()]),
            RefusalClass::Device,
        ),
        (
            NotSupported::Uncontained([RobustBufferForm::LoadStore].into()),
            RefusalClass::Device,
        ),
        (
            NotSupported::Restriction {
                profile: "nvidia-pascal-linux".into(),
                gaps: vec![
                    "limits.max_compute_work_group_invocations 1536 beyond the device's 1024"
                        .into(),
                ],
            },
            RefusalClass::Device,
        ),
        (
            product(Outcome::Unsupported {
                reason: UnsupportedReason::NotImplemented,
                feature: "SamplerBicubicFilter".into(),
            }),
            RefusalClass::Product,
        ),
        (
            product(Outcome::Unsupported {
                reason: UnsupportedReason::NotRepresentable,
                feature: "RayTracing".into(),
            }),
            RefusalClass::Product,
        ),
        (
            product(Outcome::InvalidDescriptor {
                kind: "FunctionConstantIndex".into(),
            }),
            RefusalClass::Product,
        ),
        (
            product(Outcome::LimitExceeded {
                kind: LimitKind::Structural,
                limit: "a cap".into(),
            }),
            RefusalClass::Product,
        ),
        (
            NotSupported::NotImplemented("linked intersection functions".into()),
            RefusalClass::Executor,
        ),
        (
            NotSupported::Requirements([ToolingRequirement::RenderAccelerationStructure].into()),
            RefusalClass::Executor,
        ),
    ] {
        let of = Refusal::of(&refusal);
        assert_eq!(of.class, class, "{refusal}");
        assert_eq!(of.outcome, refusal.outcome());
        assert_eq!(of.detail, refusal.to_string());
        assert_eq!(
            of.product.is_some(),
            matches!(refusal, NotSupported::Product { .. })
        );
        assert_eq!(of.validate(), Ok(()));
    }
    let of = Refusal::of(&device_capability_missing());
    assert_eq!(
        of.product,
        Some(ProductOutcome {
            label: "device_capability_missing".into(),
            key: "FramebufferFetch needs Feature(RasterizationOrderColorAttachmentAccess)".into(),
            at_bind: Some(false),
        })
    );
}

/// A product refusal records whether the binder refused what the case binds (`at_bind`), from the
/// executor's own path: the binder's typed `Unsupported` is at bind, a compile's is not. A record
/// of a build before the field reads `None`, its phase unknown, and a record of this build keeps
/// `false` explicitly, so the two are never confused.
#[test]
fn a_product_refusal_records_whether_the_binder_refused() {
    use crate::candidate::engine::{bind_refusal, product_refusal};
    use crate::check::admission::ExecutorError;
    use metal2vulkan::error::{BindError, Error, Feature, Unsupported};
    let unsupported = Unsupported::not_implemented(Feature::Imageblocks);
    let refused = |error: ExecutorError| match error {
        ExecutorError::NotSupported(not_supported) => Refusal::of(&not_supported),
        other => panic!("{other}"),
    };
    let bound = refused(bind_refusal(&BindError::Unsupported(unsupported.clone())));
    let compiled = refused(product_refusal(&Error::Unsupported(unsupported)));
    assert_eq!(bound.class, RefusalClass::Product);
    assert_eq!(bound.outcome, compiled.outcome);
    assert_eq!(bound.product.as_ref().unwrap().at_bind, Some(true));
    assert_eq!(compiled.product.as_ref().unwrap().at_bind, Some(false));
    let written = serde_json::to_value(&compiled).unwrap();
    assert_eq!(written["product"]["at_bind"], false);
    let mut earlier = written;
    earlier["product"]
        .as_object_mut()
        .unwrap()
        .remove("at_bind");
    let earlier: Refusal = serde_json::from_value(earlier).unwrap();
    assert_eq!(earlier.product.as_ref().unwrap().at_bind, None);
    assert_eq!(earlier.validate(), Ok(()));
}

fn record(slot: &str, leg: &str) -> RefusalObservation {
    RefusalObservation {
        case_id: "aa".repeat(32),
        air_sha256: "11".repeat(32),
        input_sha256: "22".repeat(32),
        translator_fingerprint: "33".repeat(32),
        backend: Backend::Vulkan,
        environment_id: slot.into(),
        environment: serde_json::json!({"robustness": {"leg": leg}}),
        executor_abi: "vulkan-engine-v1".into(),
        refusal: Refusal::of(&device_capability_missing()),
        artifact: None,
    }
}

/// A record is an engine slot's, in the leg its environment ran, with a refusal whose class
/// agrees with the product's outcome.
#[test]
fn a_refusal_record_is_checked() {
    assert_eq!(record("env@engine", "stated").validate_content(), Ok(()));
    assert_eq!(
        record("env@engine+robustness-off", "robustness_off").validate_content(),
        Ok(())
    );
    type Edit = fn(&mut RefusalObservation);
    let edits: [(Edit, &str); 8] = [
        (
            |record| record.environment_id = "env".into(),
            "not an engine slot",
        ),
        (
            |record| record.environment_id = "env@engine+robustness-off".into(),
            "but is stored in slot",
        ),
        (
            |record| record.environment = serde_json::json!({}),
            "names no leg",
        ),
        (
            |record| record.translator_fingerprint = "fingerprint".into(),
            "translator_fingerprint",
        ),
        (|record| record.executor_abi = " ".into(), "executor ABI"),
        (
            |record| record.refusal.class = RefusalClass::Product,
            "which is device",
        ),
        (|record| record.refusal.product = None, ""),
        (|record| record.refusal.detail.clear(), "its detail"),
    ];
    for (edit, why) in edits {
        let mut record = record("env@engine", "stated");
        edit(&mut record);
        if why.is_empty() {
            // A device refusal need not name a product outcome (the binder's, the device's).
            assert_eq!(record.validate_content(), Ok(()));
            record.refusal.class = RefusalClass::Product;
            let error = record.validate_content().unwrap_err();
            assert!(error.contains("needs the product's outcome"), "{error}");
            continue;
        }
        let error = record.validate_content().unwrap_err();
        assert!(error.contains(why), "{why}: {error}");
    }
    let leg = |record: &RefusalObservation| {
        serde_json::from_value::<Leg>(record.environment["robustness"]["leg"].clone()).unwrap()
    };
    assert_eq!(leg(&record("env@engine", "stated")), Leg::Stated);
}
