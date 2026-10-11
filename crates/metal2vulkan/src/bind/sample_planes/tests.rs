//! The sample planes' copy modules: what each needs, valid on every committed profile that has
//! it (L3, `G-val`), refused on a device that does not; and the copy format of each texel size.

use super::*;
use crate::device::requirements::VkFeature;
use crate::device::{DeviceDescription, DevicePolicy};
use crate::spirv::testing::assert_valid;
use std::path::PathBuf;

/// The copy format follows the texel size of a single-texel color format that shares its
/// compatibility class, and no other format has one.
#[test]
fn the_copy_format_is_the_texel_size() {
    for (format, copy) in [
        (Format::R8_UNORM, Some(PlanesCopy::R8)),
        (Format::R8G8_UNORM, Some(PlanesCopy::R16)),
        (Format::R16_SFLOAT, Some(PlanesCopy::R16)),
        (Format::R8G8B8A8_SRGB, Some(PlanesCopy::R32)),
        (Format::B10G11R11_UFLOAT_PACK32, Some(PlanesCopy::R32)),
        (Format::R32_SFLOAT, Some(PlanesCopy::R32)),
        (Format::R16G16B16A16_SNORM, Some(PlanesCopy::Rg32)),
        (Format::R32G32B32A32_UINT, Some(PlanesCopy::Rgba32)),
        (Format::R32G32B32_SFLOAT, None),
        (Format::D32_SFLOAT, None),
        (Format::S8_UINT, None),
        (Format::BC1_RGBA_UNORM_BLOCK, None),
        // A size shared with a copy format, a class of its own: no view of it in the copy's.
        (Format::A8_UNORM, None),
        (Format::A1B5G5R5_UNORM_PACK16, Some(PlanesCopy::R16)),
        (Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16, None),
        (Format::G8B8G8R8_422_UNORM, None),
    ] {
        assert_eq!(PlanesCopy::for_format(format), copy, "{format:?}");
    }
    for copy in PlanesCopy::ALL {
        assert_eq!(
            PlanesCopy::for_format(copy.format()),
            Some(copy),
            "{copy:?}"
        );
    }
}

/// The requirements of `m` as serialized.
fn needs(m: TypedModule) -> Requirements {
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .requirements
}

/// Each module needs exactly what the module docs say: the split of an extended storage format
/// `shaderStorageImageExtendedFormats`, the merge's fragment stage `sampleRateShading`, and
/// nothing else of the device.
#[test]
fn each_module_needs_what_its_docs_say() {
    for copy in PlanesCopy::ALL {
        let r = needs(split_typed_module(copy).expect("split"));
        let extended = !matches!(copy, PlanesCopy::R32 | PlanesCopy::Rgba32);
        assert_eq!(
            r.features
                .contains(&VkFeature::ShaderStorageImageExtendedFormats),
            extended,
            "{copy:?}: {r:?}"
        );
        assert!(
            !r.features.contains(&VkFeature::SampleRateShading),
            "{copy:?}"
        );
    }
    let r = needs(merge_fragment_typed_module().expect("fragment"));
    assert!(r.features.contains(&VkFeature::SampleRateShading), "{r:?}");
    let r = needs(merge_vertex_typed_module().expect("vertex"));
    assert!(r.features.is_empty(), "{r:?}");
}

/// Every committed device (`validation/profiles/*.json`), derived with the default policy, by
/// file stem.
fn committed_devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("profiles") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let device = Device::new(desc, DevicePolicy::default()).expect("device");
        out.push((stem, device));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// On every committed device each module is valid where the device has what it needs, numbered
/// with the fixed bindings, and refused (`Unsupported`) where it does not, never given as words
/// it cannot run.
#[test]
fn the_modules_are_valid_where_the_device_has_what_they_need() {
    let mut built = 0;
    for (name, device) in committed_devices() {
        let profile = device.profile();
        for copy in PlanesCopy::ALL {
            let fits = profile
                .check(&needs(split_typed_module(copy).expect("split")))
                .is_ok();
            match planes_split_module(&device, copy) {
                Ok(module) => {
                    assert!(fits, "{name} {copy:?}");
                    assert_valid(split_typed_module(copy).expect("split"), profile);
                    assert!(!module.spirv.is_empty());
                    built += 1;
                }
                Err(Error::Unsupported(_)) => assert!(!fits, "{name} {copy:?}"),
                Err(other) => panic!("{name} {copy:?}: {other}"),
            }
        }
        let fits = profile
            .check(&needs(merge_fragment_typed_module().expect("fragment")))
            .is_ok();
        match planes_merge_modules(&device) {
            Ok(modules) => {
                assert!(fits, "{name}");
                assert_valid(merge_vertex_typed_module().expect("vertex"), profile);
                assert_valid(merge_fragment_typed_module().expect("fragment"), profile);
                assert!(!modules.vertex.is_empty() && !modules.fragment.is_empty());
            }
            Err(Error::Unsupported(_)) => assert!(!fits, "{name}"),
            Err(other) => panic!("{name}: {other}"),
        }
    }
    assert!(built > 0);
}
