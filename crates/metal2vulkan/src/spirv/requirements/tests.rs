//! Requirement derivation tests: derived requirements agree with the Vulkan registry's SPIR-V
//! enablement rules (`registry_generated.rs`, a name-keyed table generated from the same pinned
//! `vk.xml` by `scripts/regen-vulkan-requirements/registry.py --by-name`).
//!
//! [`check_against_registry`] runs on every module of the serializer's `spirv-val` sweep, so each
//! builder family's derived capabilities and extensions are checked against the registry's
//! enablement table, not only against `spirv-val`.

use std::collections::BTreeSet;

use super::registry_generated::{self as registry, Enablement, Exposure};
use super::{close_extensions, spirv_extension_name, ALL_SPIRV_EXTENSIONS};
use crate::device::enables;
use crate::device::requirements::{
    ApiVersion, FloatControl, FloatWidth, Requirements, SpirvCapability, SubgroupOp, VkExtension,
    VkFeature, VkProperty, WITHOUT_FORMAT_CAPABILITIES,
};

/// The registry's name for a derived property, when the registry has one.
fn property_name(p: &VkProperty) -> Option<String> {
    Some(match p {
        VkProperty::FloatControl(c, w) => {
            let class = match c {
                FloatControl::DenormPreserve => "DenormPreserve",
                FloatControl::DenormFlushToZero => "DenormFlushToZero",
                FloatControl::RoundingModeRte => "RoundingModeRTE",
                FloatControl::RoundingModeRtz => "RoundingModeRTZ",
                FloatControl::SignedZeroInfNanPreserve => "SignedZeroInfNanPreserve",
            };
            let width = match w {
                FloatWidth::F16 => 16,
                FloatWidth::F32 => 32,
                FloatWidth::F64 => 64,
            };
            format!("Shader{class}Float{width}")
        }
        VkProperty::SubgroupOperation(op) => {
            format!("SubgroupSupportedOperationsSubgroupFeature{op:?}Bit")
        }
        VkProperty::SubgroupQuadOperationsInAllStages => "SubgroupQuadOperationsInAllStages".into(),
        _ => return None,
    })
}

/// Check that every derived SPIR-V capability and extension has at least one registry enablement
/// alternative that the derived requirements satisfy: a core version at most the derived API
/// version, a derived device extension, a derived feature, or a derived property. By the
/// derivation's convention a feature implies its exposure (`VkFeature::exposure`, checked
/// against the registry below), so an extension alternative is also satisfied by a derived
/// feature that the extension exposes. A capability the device resolves
/// (`WITHOUT_FORMAT_CAPABILITIES`) needs nothing derived: its registry enablements are the
/// resolution's (`the_device_resolved_capabilities_are_the_registrys`).
pub(crate) fn check_against_registry(r: &Requirements) -> Result<(), String> {
    let api = r.api_version.map_or((1, 0), |v| (v.major, v.minor));
    let names = |it: &mut dyn Iterator<Item = String>| it.collect::<BTreeSet<String>>();
    let features = names(&mut r.features.iter().map(|f| format!("{f:?}")));
    let extensions = names(&mut r.extensions.iter().map(|e| format!("{e:?}")));
    let properties = names(&mut r.properties.iter().filter_map(property_name));
    let exposed: BTreeSet<String> = r
        .features
        .iter()
        .flat_map(|&f| f.exposure().extensions_at(ApiVersion::V1_2))
        .map(|e| format!("{e:?}"))
        .collect();
    let satisfied = |alts: &[Enablement]| {
        alts.iter().any(|e| match e {
            Enablement::Exposed(Exposure::Core(major, minor)) => {
                (u32::from(*major), u32::from(*minor)) <= api
            }
            Enablement::Exposed(Exposure::Extension(x)) => {
                extensions.contains(*x) || exposed.contains(*x)
            }
            Enablement::Feature(f) => features.contains(*f),
            Enablement::Property(p) => properties.contains(*p),
        })
    };
    let mut errors = Vec::new();
    for c in &r.spirv_capabilities {
        if WITHOUT_FORMAT_CAPABILITIES.iter().any(|(x, _)| x == c) {
            continue;
        }
        let name = format!("{c:?}");
        match registry::capability_enablement(&name) {
            Some(alts) if satisfied(alts) => {}
            Some(alts) => errors.push(format!("capability {name}: none of {alts:?}")),
            None => errors.push(format!("capability {name}: not in the registry")),
        }
    }
    for e in &r.spirv_extensions {
        let name = spirv_extension_name(*e);
        match registry::extension_enablement(name) {
            Some(alts) if satisfied(alts) => {}
            Some(alts) => errors.push(format!("extension {name}: none of {alts:?}")),
            None => errors.push(format!("extension {name}: not in the registry")),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[test]
fn every_spirv_extension_name_is_in_the_registry() {
    for e in ALL_SPIRV_EXTENSIONS {
        let name = spirv_extension_name(*e);
        assert!(name.starts_with("SPV_"), "{e:?}: {name}");
        assert!(
            registry::extension_enablement(name).is_some(),
            "{e:?}: {name} is not a registry extension"
        );
    }
}

#[test]
fn the_registry_check_refuses_unsatisfied_capabilities() {
    let mut r = Requirements::default();
    r.spirv_capabilities
        .insert(SpirvCapability::of(crate::spv::Capability::Shader));
    assert_eq!(check_against_registry(&r), Ok(()));
    // `Int64` is enabled only by the `shaderInt64` feature.
    r.spirv_capabilities
        .insert(SpirvCapability::of(crate::spv::Capability::Int64));
    assert!(check_against_registry(&r).is_err());
    r.features.insert(VkFeature::ShaderInt64);
    assert_eq!(check_against_registry(&r), Ok(()));
    // `StencilExportEXT` is exposure-only: its device extension must be derived.
    r.spirv_capabilities.insert(SpirvCapability::of(
        crate::spv::Capability::StencilExportEXT,
    ));
    assert!(check_against_registry(&r).is_err());
    r.extensions.insert(VkExtension::ExtShaderStencilExport);
    assert_eq!(check_against_registry(&r), Ok(()));
    // A subgroup operation class is a property alternative.
    r.spirv_capabilities.insert(SpirvCapability::of(
        crate::spv::Capability::GroupNonUniformVote,
    ));
    assert!(check_against_registry(&r).is_err());
    r.properties
        .insert(VkProperty::SubgroupOperation(SubgroupOp::Vote));
    assert_eq!(check_against_registry(&r), Ok(()));
}

/// The capabilities a module lists without a feature (`WITHOUT_FORMAT_CAPABILITIES`) are enabled,
/// per the registry, by exactly what `DeviceProfile::check` and
/// `DeviceDescription::required_enables` resolve them by: the feature, Vulkan 1.3, or
/// `VK_KHR_format_feature_flags2`.
#[test]
fn the_device_resolved_capabilities_are_the_registrys() {
    for (capability, feature) in WITHOUT_FORMAT_CAPABILITIES {
        let name = format!("{capability:?}");
        let feature_name = format!("{feature:?}");
        let flags2 = format!("{:?}", VkExtension::KhrFormatFeatureFlags2);
        let alternatives = registry::capability_enablement(&name).expect("a registry capability");
        let resolved: Vec<bool> = alternatives
            .iter()
            .map(|e| match e {
                Enablement::Feature(f) => *f == feature_name,
                Enablement::Exposed(Exposure::Core(1, 3)) => true,
                Enablement::Exposed(Exposure::Extension(x)) => *x == flags2,
                _ => false,
            })
            .collect();
        assert_eq!(resolved, vec![true; 3], "{name}: {alternatives:?}");
    }
}

/// `VkFeature::exposure` agrees with the registry's enablement table on every feature the
/// table names: a core structure at or below the 1.2 floor is `Core`, a later core version is the
/// promotion version, no core version is `Extension`, and every extension the table lists for the
/// feature is one the registry lists (the registry may list more aliases, e.g. `Nv*`).
#[test]
fn feature_exposure_agrees_with_the_registry() {
    let mut compared = 0;
    for &f in VkFeature::ALL {
        let Some(registry) = registry::feature_exposure(&format!("{f:?}")) else {
            continue;
        };
        compared += 1;
        let cores: Vec<(u8, u8)> = registry
            .iter()
            .filter_map(|x| match x {
                Exposure::Core(major, minor) => Some((*major, *minor)),
                Exposure::Extension(_) => None,
            })
            .collect();
        let exts: BTreeSet<&str> = registry
            .iter()
            .filter_map(|x| match x {
                Exposure::Extension(e) => Some(*e),
                Exposure::Core(..) => None,
            })
            .collect();
        let ours = f.exposure();
        let (core, listed): (Option<(u8, u8)>, &[VkExtension]) = match ours {
            enables::Exposure::Core(v) => (Some((v.major as u8, v.minor as u8)), &[]),
            enables::Exposure::Promoted {
                version,
                extensions,
            }
            | enables::Exposure::Implied {
                version,
                extensions,
            } => (Some((version.major as u8, version.minor as u8)), extensions),
            enables::Exposure::Extension(extensions) => (None, extensions),
        };
        match core {
            // Every core version at or below the floor exposes it on every accepted device.
            Some((1, minor)) if minor <= 2 => assert!(
                cores.iter().any(|&(_, m)| m <= 2),
                "{f:?}: ours {ours:?}, registry {registry:?}"
            ),
            Some(version) => assert_eq!(
                cores.iter().min(),
                Some(&version),
                "{f:?}: ours {ours:?}, registry {registry:?}"
            ),
            None => assert!(cores.is_empty(), "{f:?}: registry {registry:?}"),
        }
        for e in listed {
            assert!(
                exts.contains(format!("{e:?}").as_str()),
                "{f:?}: {e:?} is not among the registry's {exts:?}"
            );
        }
    }
    assert!(compared >= 60, "only {compared} features compared");
}

/// Every extension-only feature has exactly one vocabulary extension, and the derivation's
/// closing step lists it; a feature a core version exposes adds none.
#[test]
fn extension_only_features_name_their_extension() {
    for &f in VkFeature::ALL {
        let mut r = Requirements::default();
        r.features.insert(f);
        close_extensions(&mut r);
        match f.exposure() {
            enables::Exposure::Extension(list) => {
                assert_eq!(list.len(), 1, "{f:?}");
                assert_eq!(r.extensions, BTreeSet::from([list[0]]), "{f:?}");
            }
            _ => assert!(r.extensions.is_empty(), "{f:?}: {:?}", r.extensions),
        }
    }
    let mut r = Requirements::default();
    r.features.extend([
        VkFeature::FragmentShaderPixelInterlock,
        VkFeature::ComputeDerivativeGroupQuads,
        VkFeature::Maintenance8,
        VkFeature::ShaderSubgroupUniformControlFlow,
        VkFeature::ShaderMaximalReconvergence,
        VkFeature::ShaderTerminateInvocation,
    ]);
    close_extensions(&mut r);
    assert_eq!(
        r.extensions,
        BTreeSet::from([
            VkExtension::ExtFragmentShaderInterlock,
            VkExtension::KhrComputeShaderDerivatives,
            VkExtension::KhrMaintenance8,
            VkExtension::KhrShaderSubgroupUniformControlFlow,
            VkExtension::KhrShaderMaximalReconvergence,
        ])
    );
}
